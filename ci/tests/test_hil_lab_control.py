from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import pty
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ci" / "hil" / "lab_control.py"
SPEC = importlib.util.spec_from_file_location("hil_lab_control", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)
INVENTORY = ROOT / "ci" / "hil" / "inventory" / "lab.example.yaml"


class Response(io.BytesIO):
    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class HilLabControlTests(unittest.TestCase):
    def write_yaml(self, document: object) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        )
        with temporary:
            yaml.safe_dump(document, temporary, sort_keys=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def manifest(self, payload: bytes, *, state: str = "released") -> dict[str, object]:
        return {
            "apiVersion": "modelzoo.hil/v1alpha1",
            "kind": "AssetManifest",
            "metadata": {"name": "test-assets", "version": "v1", "state": state},
            "spec": {
                "model": "test",
                "engine": "svp-nnn",
                "soc": "SS928V100",
                "files": [
                    {
                        "name": "payload.bin",
                        "sourceRef": "modelzoo-public-release",
                        "immutableTag": "test-v1",
                        "releaseAsset": "payload.bin",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                        "mediaType": "application/octet-stream",
                        "access": "public",
                        "license": {
                            "status": "redistributable",
                            "reference": "LicenseRef-Test",
                        },
                    }
                ],
            },
        }

    def credentials(self, root: Path, *, token: bool = False) -> None:
        values = {
            "hil-v2-board-ssh": "private",
            "hil-board-known-hosts": "host key",
        }
        if token:
            values["github-workflow-token"] = "token"
        for name, value in values.items():
            path = root / name
            path.write_text(value, encoding="utf-8")
            path.chmod(0o600 if name != "hil-board-known-hosts" else 0o640)

    def test_fetch_uses_approved_source_and_exact_size_sha(self) -> None:
        payload = b"immutable release payload"
        manifest_path = self.write_yaml(self.manifest(payload))
        release = {
            "tag_name": "test-v1",
            "draft": False,
            "assets": [
                {
                    "name": "payload.bin",
                    "size": len(payload),
                    "url": "https://api.github.com/repos/sanchuanhehe/modelzoo/releases/assets/1",
                }
            ]
        }

        def open_url(request: object, *, timeout: int) -> Response:
            del timeout
            if request.full_url.endswith("/releases/tags/test-v1"):
                return Response(json.dumps(release).encode())
            if request.full_url.endswith("/releases/assets/1"):
                return Response(payload)
            raise AssertionError(request.full_url)

        with (
            tempfile.TemporaryDirectory() as output_directory,
            tempfile.TemporaryDirectory() as credential_directory,
        ):
            credentials = Path(credential_directory)
            self.credentials(credentials, token=True)
            result = lab.fetch_assets(
                INVENTORY,
                manifest_path,
                Path(output_directory),
                credentials,
                open_url=open_url,
            )
            self.assertEqual((Path(output_directory) / "payload.bin").read_bytes(), payload)
            self.assertEqual(result["files"][0]["source"], "modelzoo-public-release")

    def test_candidate_fetch_fails_before_network_or_credentials(self) -> None:
        candidate = (
            ROOT / "ci" / "hil" / "assets" / "resnet50-svp-nnn-v1.yaml"
        )
        candidate_document = lab.validate_config.load_document(candidate)
        candidate_document["metadata"]["state"] = "candidate"
        candidate_path = self.write_yaml(candidate_document)
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(lab.LabError, "released state"):
                lab.fetch_assets(
                    INVENTORY,
                    candidate_path,
                    Path(output),
                    Path("/missing"),
                    open_url=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("network must not be used")
                    ),
                )

    def test_restricted_source_must_be_api_verified_private(self) -> None:
        payload = b"restricted payload"
        manifest = self.manifest(payload)
        item = manifest["spec"]["files"][0]
        item["sourceRef"] = "private-model-assets"
        item["access"] = "restricted"
        item["license"]["status"] = "restricted"
        manifest_path = self.write_yaml(manifest)
        inventory = lab.load_inventory(INVENTORY)
        inventory["spec"]["assetSources"]["private-model-assets"] = {
            "kind": "github-release",
            "repository": "sanchuanhehe/private-model-assets",
            "credentialRef": "restricted-asset-token",
            "accessClass": "restricted",
            "authorizationStatus": "approved",
        }
        inventory_path = self.write_yaml(inventory)

        def open_url(request: object, *, timeout: int) -> Response:
            del timeout
            if request.full_url.endswith("/repos/sanchuanhehe/private-model-assets"):
                return Response(
                    b'{"full_name":"sanchuanhehe/private-model-assets","private":false}'
                )
            raise AssertionError("release must not be queried for a public repository")

        with (
            tempfile.TemporaryDirectory() as output_directory,
            tempfile.TemporaryDirectory() as credential_directory,
        ):
            credentials = Path(credential_directory)
            token = credentials / "restricted-asset-token"
            token.write_text("restricted-token", encoding="utf-8")
            token.chmod(0o600)
            with self.assertRaisesRegex(lab.LabError, "API-verified private"):
                lab.fetch_assets(
                    inventory_path,
                    manifest_path,
                    Path(output_directory),
                    credentials,
                    open_url=open_url,
                )

    def test_inventory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "inventory.yaml"
            link.symlink_to(INVENTORY)
            with self.assertRaisesRegex(lab.LabError, "real file"):
                lab.load_inventory(link)

    def test_structured_event_log_uses_precreated_owned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_directory = Path(directory) / "events"
            event_directory.mkdir(mode=0o750)
            payload = lab.event("target.preflight", "passed", {"temperature": 42000})
            with mock.patch.dict(
                os.environ,
                {"HIL_EVENT_LOG_DIRECTORY": str(event_directory)},
            ):
                lab.write_event_log(payload)
            records = list(event_directory.glob("*.json"))
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0].read_text()), payload)

    def test_artifact_verification_binds_sha_and_aarch64(self) -> None:
        source_sha = "1" * 40
        header = bytearray(64)
        header[:4] = b"\x7fELF"
        header[4] = 2
        header[5] = 1
        header[18:20] = (183).to_bytes(2, "little")
        main = bytes(header) + b"main"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main").write_bytes(main)
            (root / "build.log").write_text("cross build passed\n", encoding="utf-8")
            sdk_lock_path = ROOT / "ci" / "sdk-lock.json"
            sdk_lock = json.loads(sdk_lock_path.read_text(encoding="utf-8"))
            manifest = {
                "schemaVersion": 1,
                "boundary": "build-only; no board execution performed",
                "commit": source_sha,
                "workflowCommit": "2" * 40,
                "sdkReleaseTag": sdk_lock["releaseTag"],
                "sdkArtifacts": {
                    "toolchain": sdk_lock["artifacts"]["toolchain"]["archive"]["sha256"],
                    "svp-nnn": sdk_lock["artifacts"]["svp-nnn"]["archive"]["sha256"],
                },
                "engine": "svp-nnn",
                "soc": "SS928V100",
                "sample": "samples/built-in/classification/ResNet50",
                "outputs": [
                    {
                        "name": "main",
                        "size": len(main),
                        "sha256": hashlib.sha256(main).hexdigest(),
                    }
                ],
            }
            (root / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            def refresh_sums() -> None:
                (root / "SHA256SUMS").write_text(
                    "".join(
                        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
                        for name in ("main", "build.log", "build-manifest.json")
                    ),
                    encoding="ascii",
                )

            refresh_sums()
            result = lab.verify_artifact(
                root,
                source_sha,
                "2" * 40,
                sdk_lock_path,
                sample="samples/built-in/classification/ResNet50",
                engine="svp-nnn",
                soc="SS928V100",
            )
            self.assertEqual(result["main"]["sha256"], hashlib.sha256(main).hexdigest())
            manifest["sdkReleaseTag"] = "wrong-sdk"
            (root / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            refresh_sums()
            with self.assertRaisesRegex(lab.LabError, "SDK/boundary mismatch"):
                lab.verify_artifact(
                    root,
                    source_sha,
                    "2" * 40,
                    sdk_lock_path,
                    sample="samples/built-in/classification/ResNet50",
                    engine="svp-nnn",
                    soc="SS928V100",
                )
            manifest["sdkReleaseTag"] = sdk_lock["releaseTag"]
            (root / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            refresh_sums()
            with self.assertRaisesRegex(lab.LabError, "workflowCommit mismatch"):
                lab.verify_artifact(
                    root,
                    source_sha,
                    "3" * 40,
                    sdk_lock_path,
                    sample="samples/built-in/classification/ResNet50",
                    engine="svp-nnn",
                    soc="SS928V100",
                )
            (root / "expected.json").write_text('{"status":"not-run"}', encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "file set mismatch"):
                lab.verify_artifact(
                    root,
                    source_sha,
                    "2" * 40,
                    sdk_lock_path,
                    sample="samples/built-in/classification/ResNet50",
                    engine="svp-nnn",
                    soc="SS928V100",
                )

    def test_target_upload_only_uses_agent_commands(self) -> None:
        calls: list[tuple[str, bytes | None]] = []

        def runner(
            argv: list[str],
            *,
            input: bytes | None,
            capture_output: bool,
            timeout: int,
            check: bool,
        ) -> subprocess.CompletedProcess[bytes]:
            del capture_output, timeout, check
            calls.append((argv[-1], input))
            return subprocess.CompletedProcess(argv, 0, b"ok\n", b"")

        with (
            tempfile.TemporaryDirectory() as payload_directory,
            tempfile.TemporaryDirectory() as credential_directory,
        ):
            payload = Path(payload_directory)
            files = {"main": b"main", "run-test": b"#!/bin/sh\nexit 0\n"}
            for name, data in files.items():
                (payload / name).write_bytes(data)
            sums = "".join(
                f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                for name, data in files.items()
            )
            (payload / "PAYLOAD_SHA256SUMS").write_text(sums, encoding="ascii")
            credentials = Path(credential_directory)
            self.credentials(credentials)
            result = lab.target_upload(
                INVENTORY,
                "hi3403-01",
                credentials,
                "run-1",
                payload,
                runner=runner,
            )
        self.assertEqual(result["runId"], "run-1")
        commands = [command for command, _ in calls]
        self.assertEqual(commands[0], "prepare run-1")
        self.assertTrue(all(command.split()[0] in {"prepare", "put", "seal"} for command in commands))
        self.assertEqual(commands[-1], "seal run-1")

    def test_target_command_rejects_whitespace_in_arguments(self) -> None:
        inventory = lab.load_inventory(INVENTORY)
        target = lab.select_target(inventory, "hi3403-01")
        with self.assertRaisesRegex(lab.LabError, "unsafe target-agent argument"):
            lab.target_command(target, Path("/missing"), ["run", "bad id"])
        with self.assertRaisesRegex(lab.LabError, "unsafe target-agent argument"):
            lab.target_command(target, Path("/missing"), ["probe;touch"])

    def test_target_preflight_interprets_target_class_on_controller(self) -> None:
        calls: list[str] = []

        def runner(
            argv: list[str],
            *,
            input: bytes | None,
            capture_output: bool,
            timeout: int,
            check: bool,
        ) -> subprocess.CompletedProcess[bytes]:
            del input, capture_output, timeout, check
            command = argv[-1]
            calls.append(command)
            if command == "probe":
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    b"protocol=1\nagentVersion=0.2.0\nuser=root\n"
                    b"kernel=4.19.90\narchitecture=aarch64\nfreeBytes=4294967296\n"
                    b"temperatureMilliCelsius=unavailable\n",
                    b"",
                )
            if command in {
                "check-library libsvp_acl.so",
                "check-library libsvp_aicpu.so",
                "check-library libsecurec.so",
                "check-library libprotobuf-c.so.1",
                "check-library libopencv_world.so.412",
                "check-module ot_svp_npu",
            }:
                return subprocess.CompletedProcess(argv, 0, b"present\n", b"")
            if command == "check-module ot_pqp":
                return subprocess.CompletedProcess(argv, 1, b"absent\n", b"")
            raise AssertionError(command)

        with tempfile.TemporaryDirectory() as credential_directory:
            credentials = Path(credential_directory)
            self.credentials(credentials)
            result = lab.target_preflight(
                INVENTORY,
                "hi3403-01",
                credentials,
                ROOT / "ci" / "hil" / "target-classes" / "hi3403-svp-nnn.yaml",
                runner=runner,
            )
        self.assertEqual(result["probe"]["architecture"], "aarch64")
        self.assertEqual(len(result["checks"]), 7)
        self.assertEqual(result["executionMode"], "forced-command-root")
        self.assertEqual(calls[0], "probe")

    def test_local_cleanup_is_exact_and_dry_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run-1"
            run.mkdir()
            self.assertFalse(lab.local_cleanup(root, "run-1", dry_run=True)["removed"])
            self.assertTrue(run.exists())
            self.assertTrue(lab.local_cleanup(root, "run-1", dry_run=False)["removed"])
            self.assertFalse(run.exists())
            with self.assertRaisesRegex(lab.LabError, "unsafe run ID"):
                lab.local_cleanup(root, "../escape", dry_run=False)

    def test_uart_capture_is_bound_to_inventory_device_and_exact_run(self) -> None:
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        inventory = lab.load_inventory(INVENTORY)
        inventory["spec"]["targets"]["hi3403-01"]["serialDevice"] = os.ttyname(slave)
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            with mock.patch.object(lab, "load_inventory", return_value=inventory):
                started = lab.uart_start(
                    INVENTORY,
                    "hi3403-01",
                    "uart-1",
                    dry_run=False,
                    run_root_override=run_root,
                )

            def stop_if_running() -> None:
                if (run_root / "uart-1" / "uart-process.pid").exists():
                    with mock.patch.object(
                        lab, "load_inventory", return_value=inventory
                    ):
                        lab.uart_stop(
                            INVENTORY,
                            "hi3403-01",
                            "uart-1",
                            dry_run=False,
                            run_root_override=run_root,
                        )

            self.addCleanup(stop_if_running)
            os.write(master, b"boot evidence\n")
            deadline = time.monotonic() + 2
            log = run_root / "uart-1" / "uart.log"
            while time.monotonic() < deadline and (
                not log.exists() or b"boot evidence" not in log.read_bytes()
            ):
                time.sleep(0.05)
            with mock.patch.object(lab, "load_inventory", return_value=inventory):
                stopped = lab.uart_stop(
                    INVENTORY,
                    "hi3403-01",
                    "uart-1",
                    dry_run=False,
                    run_root_override=run_root,
                )
            self.assertEqual(started["device"], os.ttyname(slave))
            self.assertEqual(stopped["state"], "stopped")
            self.assertIn(b"boot evidence", log.read_bytes())

    def test_evidence_snapshot_survives_target_loss_and_redacts_credentials(self) -> None:
        # Synthetic non-secret marker used only to prove exact-value redaction.
        redaction_marker = b"unit-test-redaction-marker-12345"

        def runner(
            argv: list[str],
            *,
            input: bytes | None,
            capture_output: bool,
            timeout: int,
            check: bool,
        ) -> subprocess.CompletedProcess[bytes]:
            del input, capture_output, timeout, check
            if argv[0] == "ssh":
                return subprocess.CompletedProcess(
                    argv, 255, redaction_marker, b"offline"
                )
            return subprocess.CompletedProcess(
                argv, 0, redaction_marker + b"\n", b""
            )

        with (
            tempfile.TemporaryDirectory() as run_directory,
            tempfile.TemporaryDirectory() as credential_directory,
            tempfile.TemporaryDirectory() as input_directory,
        ):
            run_root = Path(run_directory)
            credentials = Path(credential_directory)
            self.credentials(credentials)
            (credentials / "hil-v2-board-ssh").write_bytes(redaction_marker)
            (credentials / "hil-v2-board-ssh").chmod(0o600)
            inputs = Path(input_directory)
            context = inputs / "context.json"
            context.write_text(
                json.dumps({"note": redaction_marker.decode()}), encoding="utf-8"
            )
            asset_manifest = inputs / "assets.yaml"
            asset_manifest.write_text("kind: evidence\n", encoding="utf-8")
            build_manifest = inputs / "build.json"
            build_manifest.write_text("{}\n", encoding="utf-8")
            result = lab.evidence_snapshot(
                INVENTORY,
                "hi3403-01",
                credentials,
                "evidence-1",
                context,
                asset_manifest,
                build_manifest,
                run_root_override=run_root,
                runner=runner,
            )
            evidence_root = Path(result["evidenceRoot"])
            self.assertEqual(result["targetSnapshotStatus"], "unavailable")
            self.assertIn(
                b"[REDACTED]",
                (evidence_root / "bundle" / "manifests" / "execution-context.json").read_bytes(),
            )
            self.assertNotIn(
                redaction_marker,
                b"".join(
                    path.read_bytes()
                    for path in evidence_root.rglob("*")
                    if path.is_file()
                ),
            )
            manifest = json.loads(
                (evidence_root / "evidence-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["files"]), len(result["files"]))
            events = run_root / "evidence-1" / "events"
            events.mkdir(mode=0o750)
            (events / "cleanup.json").write_text(
                '{"command":"target.cleanup","status":"failed"}\n',
                encoding="utf-8",
            )
            finalized = lab.evidence_finalize(
                run_root, "evidence-1", credentials
            )
            self.assertTrue(
                (evidence_root / "bundle" / "events" / "cleanup.json").is_file()
            )
            self.assertIn(
                "events/cleanup.json",
                {record["name"] for record in finalized["files"]},
            )

    def test_evidence_snapshot_records_missing_early_phase_inputs(self) -> None:
        def runner(
            argv: list[str],
            *,
            input: bytes | None,
            capture_output: bool,
            timeout: int,
            check: bool,
        ) -> subprocess.CompletedProcess[bytes]:
            del input, capture_output, timeout, check
            if argv[0] == "ssh":
                return subprocess.CompletedProcess(argv, 255, b"", b"unavailable")
            return subprocess.CompletedProcess(argv, 0, b"host\n", b"")

        with tempfile.TemporaryDirectory() as run_directory:
            root = Path(run_directory)
            missing = root / "never-created"
            result = lab.evidence_snapshot(
                INVENTORY,
                "hi3403-01",
                root / "missing-credentials",
                "early-failure",
                missing / "context.json",
                missing / "assets.yaml",
                missing / "build.json",
                run_root_override=root,
                runner=runner,
            )
            manifests = Path(result["evidenceRoot"]) / "bundle" / "manifests"
            self.assertTrue((manifests / "execution-context-unavailable.txt").is_file())
            self.assertTrue((manifests / "asset-manifest-unavailable.txt").is_file())
            self.assertTrue((manifests / "build-manifest-unavailable.txt").is_file())

    def test_cli_exposes_only_fixed_uart_and_evidence_operations(self) -> None:
        parser = lab.build_parser()
        uart = parser.parse_args(
            [
                "uart", "start", "--inventory", str(INVENTORY), "--target",
                "hi3403-01", "--run-id", "run-1", "--dry-run",
            ]
        )
        self.assertEqual((uart.command, uart.uart_command), ("uart", "start"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["target", "shell"])


if __name__ == "__main__":
    unittest.main()
