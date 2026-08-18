from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_module(
    "hil_adapter_prepare",
    ROOT / "ci" / "hil" / "adapters" / "resnet50-svp-nnn" / "prepare.py",
)
VERIFY = load_module(
    "hil_adapter_verify",
    ROOT / "ci" / "hil" / "adapters" / "resnet50-svp-nnn" / "verify.py",
)


class HilAdapterV2Tests(unittest.TestCase):
    def test_prepare_does_not_send_expected_result_to_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            assets = root / "assets"
            payload = root / "payload"
            artifact.mkdir()
            assets.mkdir()
            (artifact / "main").write_bytes(b"aarch64 executable")
            manifest_path = (
                ROOT / "ci" / "hil" / "assets" / "resnet50-svp-nnn-v1.yaml"
            )
            manifest = PREPARE.validate_config.load_typed(manifest_path, "AssetManifest")
            for item in manifest["spec"]["files"]:
                path = assets / item["name"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(item["name"].encode())
            result = PREPARE.prepare(artifact, assets, manifest_path, payload)
            names = {item["name"] for item in result["files"]}
            self.assertIn("assets/resnet50.om", names)
            self.assertNotIn("assets/expected.json", names)
            self.assertFalse((payload / "assets" / "expected.json").exists())

    def test_verify_requires_exact_iteration_tree_exit_and_order(self) -> None:
        expected = {
            "schemaVersion": 1,
            "model": "resnet50",
            "engine": "svp-nnn",
            "soc": "SS928V100",
            "assetVersion": "v1",
            "input": "input/golden.jpg",
            "top1": 10,
            "top5": [10, 20, 30, 40, 50],
            "timeoutSeconds": 180,
            "resultFile": "result/txt/golden_0.txt",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_path = root / "expected.json"
            expected_path.write_text(json.dumps(expected), encoding="utf-8")
            results = root / "iterations"
            for index in (1, 2):
                output = results / str(index) / "output"
                output.mkdir(parents=True)
                (results / str(index) / "exit-code").write_text("0\n", encoding="ascii")
                (output / "result.txt").write_text(
                    "10,0.9\n20,0.8\n30,0.7\n40,0.6\n50,0.5\n",
                    encoding="utf-8",
                )
            self.assertEqual(VERIFY.verify(expected_path, results, 2)["count"], 2)
            (results / "2" / "exit-code").write_text("1\n", encoding="ascii")
            with self.assertRaisesRegex(Exception, "nonzero retained exit code"):
                VERIFY.verify(expected_path, results, 2)

    def test_payload_checksums_are_complete(self) -> None:
        data = b"payload"
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5",
        )


if __name__ == "__main__":
    unittest.main()
