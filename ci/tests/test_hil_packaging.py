#!/usr/bin/env python3
"""Tests for build-only and pre-HIL package boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class HilPackagingTests(unittest.TestCase):
    def test_build_manifest_declares_build_only_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / "main"
            output = root / "build-manifest.json"
            main.write_bytes(b"aarch64-placeholder")
            environment = os.environ.copy()
            environment["GITHUB_SHA"] = "a" * 40
            completed = subprocess.run(
                [
                    "python3",
                    str(ROOT / "ci/build_manifest.py"),
                    "--engine",
                    "svp-nnn",
                    "--soc",
                    "SS928V100",
                    "--sample",
                    "samples/built-in/classification/ResNet50",
                    "--main",
                    str(main),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["boundary"], "build-only; no board execution performed")
            self.assertEqual(manifest["commit"], "a" * 40)
            self.assertEqual(manifest["outputs"][0]["name"], "main")

    def test_pre_hil_package_uses_conversion_not_run_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            conversion = root / "conversion"
            output = root / "output"
            (conversion / "input").mkdir(parents=True)
            build.mkdir()
            (build / "main").write_bytes(b"main")
            (build / "build-manifest.json").write_text(
                json.dumps(
                    {
                        "commit": "b" * 40,
                        "sdkReleaseTag": "sdk-test",
                        "sdkArtifacts": {"toolchain": "c" * 64, "svp-nnn": "d" * 64},
                        "compiler": {},
                        "engine": "svp-nnn",
                        "soc": "SS928V100",
                        "cmake": {},
                    }
                ),
                encoding="utf-8",
            )
            (conversion / "model.om").write_bytes(b"synthetic-model")
            (conversion / "expected.json").write_text(
                json.dumps({"status": "not-run", "reason": "synthetic conversion"}),
                encoding="utf-8",
            )
            (conversion / "conversion-manifest.json").write_text(
                json.dumps(
                    {
                        "commit": "b" * 40,
                        "sdkReleaseTag": "sdk-test",
                        "converter": {},
                        "conversion": {},
                    }
                ),
                encoding="utf-8",
            )
            (conversion / "input/model.onnx").write_bytes(b"onnx")
            (conversion / "input/input.bin").write_bytes(b"input")
            completed = subprocess.run(
                [
                    "python3",
                    str(ROOT / "ci/package_hil_input.py"),
                    "--build",
                    str(build),
                    "--conversion",
                    str(conversion),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            marker = json.loads((output / "expected.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "not-run")
            self.assertEqual(
                json.loads((output / "manifest.json").read_text(encoding="utf-8"))["boundary"],
                "pre-HIL; no board execution performed",
            )


if __name__ == "__main__":
    unittest.main()
