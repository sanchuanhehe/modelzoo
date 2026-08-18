from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "ci" / "hil" / "target" / "hil-target-agent"


class HilTargetAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_root = Path(temporary.name)
        self.environment = {
            **os.environ,
            "HIL_TARGET_AGENT_TESTING": "1",
            "HIL_TARGET_RUN_ROOT": str(self.run_root),
        }

    def agent(
        self, *arguments: str, data: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["sh", str(AGENT), *arguments],
            input=data,
            capture_output=True,
            env=self.environment,
            timeout=15,
            check=False,
        )

    def test_version_is_machine_readable_without_run_root(self) -> None:
        result = subprocess.run(
            ["sh", str(AGENT), "--version"],
            capture_output=True,
            env=os.environ,
            timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertRegex(result.stdout.decode().strip(), r"^[0-9]+\.[0-9]+\.[0-9]+$")

    def put(self, run_id: str, name: str, data: bytes) -> None:
        digest = hashlib.sha256(data).hexdigest()
        result = self.agent("put", run_id, name, str(len(data)), digest, data=data)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def payload(self, run_id: str, run_test: bytes) -> None:
        files = {
            "main": b"target executable",
            "run-test": run_test,
            "assets/model.bin": b"model",
            "payload-manifest.json": b'{"schemaVersion":1}\n',
        }
        sums = b"".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode()
            for name, data in files.items()
        )
        for name, data in files.items():
            self.put(run_id, name, data)
        self.put(run_id, "PAYLOAD_SHA256SUMS", sums)

    def test_prepare_put_seal_run_snapshot_and_cleanup(self) -> None:
        run_id = "run-1"
        self.assertEqual(self.agent("prepare", run_id).returncode, 0)
        self.payload(
            run_id,
            b"#!/bin/sh\nset -eu\nmkdir -p output\nprintf '948,0.9\\n' > output/result.txt\n",
        )
        self.assertEqual(self.agent("seal", run_id).returncode, 0)
        executed = self.agent("run", run_id, "1", "5")
        self.assertEqual(executed.returncode, 0, executed.stderr.decode())
        iteration = self.run_root / run_id / "iterations" / "1"
        self.assertEqual((iteration / "exit-code").read_text().strip(), "0")
        self.assertTrue((iteration / "output" / "result.txt").is_file())
        snapshot = self.agent("snapshot", run_id)
        self.assertIn(b"iterations/1/exit-code", snapshot.stdout)
        fetched = self.agent("get", run_id, "iterations/1/output/result.txt")
        self.assertEqual(fetched.stdout, b"948,0.9\n")
        denied = self.agent("get", run_id, "payload/assets/model.bin")
        self.assertNotEqual(denied.returncode, 0)
        self.assertEqual(self.agent("cleanup", run_id).returncode, 0)
        self.assertFalse((self.run_root / run_id).exists())

    def test_put_rejects_traversal_and_bad_digest(self) -> None:
        self.assertEqual(self.agent("prepare", "run-2").returncode, 0)
        traversal = self.agent("put", "run-2", "../escape", "1", "0" * 64, data=b"x")
        self.assertNotEqual(traversal.returncode, 0)
        bad_digest = self.agent("put", "run-2", "main", "1", "0" * 64, data=b"x")
        self.assertNotEqual(bad_digest.returncode, 0)
        self.assertFalse((self.run_root / "escape").exists())

    def test_timeout_is_recorded_as_124(self) -> None:
        run_id = "timeout-run"
        self.assertEqual(self.agent("prepare", run_id).returncode, 0)
        self.payload(
            run_id,
            b"#!/bin/sh\nmkdir -p output\nsleep 10 &\necho $! > output/child.pid\nwait\n",
        )
        self.assertEqual(self.agent("seal", run_id).returncode, 0)
        executed = self.agent("run", run_id, "1", "1")
        self.assertEqual(executed.returncode, 124)
        iteration = self.run_root / run_id / "iterations" / "1"
        self.assertEqual((iteration / "exit-code").read_text().strip(), "124")
        self.assertTrue((iteration / "timed-out").is_file())
        child_pid = int((iteration / "output" / "child.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    def test_iterations_use_fresh_work_copies_without_mutating_payload(self) -> None:
        run_id = "stability-run"
        self.assertEqual(self.agent("prepare", run_id).returncode, 0)
        self.payload(
            run_id,
            b"#!/bin/sh\nset -eu\nmkdir -p assets/result output\n"
            b"printf 'generated\\n' > assets/result/internal.txt\n"
            b"printf '948,0.9\\n' > output/result.txt\n",
        )
        self.assertEqual(self.agent("seal", run_id).returncode, 0)
        for iteration in (1, 2):
            executed = self.agent("run", run_id, str(iteration), "5")
            self.assertEqual(executed.returncode, 0, executed.stderr.decode())
            result = (
                self.run_root
                / run_id
                / "iterations"
                / str(iteration)
                / "output"
                / "result.txt"
            )
            self.assertEqual(result.read_bytes(), b"948,0.9\n")
        self.assertFalse(
            (self.run_root / run_id / "payload" / "assets" / "result").exists()
        )
        self.assertEqual(
            list((self.run_root / run_id).glob(".work-*")),
            [],
        )

    def test_seal_rejects_unlisted_payload_file(self) -> None:
        run_id = "unlisted-run"
        self.assertEqual(self.agent("prepare", run_id).returncode, 0)
        self.payload(run_id, b"#!/bin/sh\nexit 0\n")
        (self.run_root / run_id / "payload" / "injected").write_bytes(b"not listed")
        sealed = self.agent("seal", run_id)
        self.assertNotEqual(sealed.returncode, 0)
        self.assertIn(b"unlisted", sealed.stderr)

    def test_forced_command_does_not_evaluate_shell_syntax(self) -> None:
        environment = {**self.environment, "SSH_ORIGINAL_COMMAND": "probe; touch injected"}
        result = subprocess.run(
            ["sh", str(AGENT), "--forced"],
            capture_output=True,
            env=environment,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.run_root / "injected").exists())


if __name__ == "__main__":
    unittest.main()
