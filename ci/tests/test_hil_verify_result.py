#!/usr/bin/env python3
"""Unit tests for strict HIL evidence validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "hil" / "verify_result.py"
SPEC = importlib.util.spec_from_file_location("verify_result", MODULE_PATH)
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def expected() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "model": "resnet50",
        "engine": "svp-nnn",
        "soc": "SS928V100",
        "assetVersion": "v1",
        "input": "input/golden.jpg",
        "top1": 10,
        "top5": [10, 20, 30, 40, 50],
        "timeoutSeconds": 60,
        "resultFile": "result/txt/golden_0.txt",
    }


class ExpectedTests(unittest.TestCase):
    def test_valid_expected_and_ordered_top5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.json"
            path.write_text(json.dumps(expected()), encoding="utf-8")
            self.assertEqual(VERIFY.load_expected(path)["top1"], 10)
            result = Path(directory) / "result.txt"
            result.write_text("10,0.9\n20,0.8\n30,0.7\n40,0.6\n50,0.5\n", encoding="utf-8")
            self.assertEqual(VERIFY.read_top5(result), [10, 20, 30, 40, 50])

    def test_not_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.json"
            data = expected()
            data["status"] = "not-run"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(VERIFY.ValidationError, "not-run"):
                VERIFY.load_expected(path)

    def test_top5_order_matters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.txt"
            result.write_text("20,0.9\n10,0.8\n30,0.7\n40,0.6\n50,0.5\n", encoding="utf-8")
            self.assertNotEqual(VERIFY.read_top5(result), expected()["top5"])

    def test_path_traversal_is_rejected(self) -> None:
        for value in ("../secret", "/etc/passwd", "input/../../secret"):
            with self.subTest(value=value), self.assertRaises(VERIFY.ValidationError):
                VERIFY.safe_relative_path(value, "test")

    def test_sha256_success_mismatch_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset.bin"
            asset.write_bytes(b"asset")
            digest = hashlib.sha256(b"asset").hexdigest()
            sums = root / "SHA256SUMS"
            sums.write_text(f"{digest}  asset.bin\n", encoding="utf-8")
            self.assertEqual(VERIFY.verify_sha256s(root, sums), 1)
            sums.write_text(f"{'0' * 64}  asset.bin\n", encoding="utf-8")
            with self.assertRaisesRegex(VERIFY.ValidationError, "mismatch"):
                VERIFY.verify_sha256s(root, sums)
            sums.write_text(f"{digest}  ../asset.bin\n", encoding="utf-8")
            with self.assertRaises(VERIFY.ValidationError):
                VERIFY.verify_sha256s(root, sums)

    def test_expected_only_cli_and_missing_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.json"
            path.write_text(json.dumps(expected()), encoding="utf-8")
            validated = subprocess.run(
                [str(MODULE_PATH), "--expected", str(path), "--validate-expected-only"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertEqual(json.loads(validated.stdout)["status"], "validated")
            missing = subprocess.run(
                [str(MODULE_PATH), "--expected", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("--result is required", missing.stderr)


if __name__ == "__main__":
    unittest.main()
