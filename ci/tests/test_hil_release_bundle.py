from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ci" / "hil" / "verify_release_bundle.py"
SPEC = importlib.util.spec_from_file_location("hil_release_bundle", MODULE_PATH)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class HilReleaseBundleTests(unittest.TestCase):
    def fixture(self, root: Path, payload: bytes) -> Path:
        digest = hashlib.sha256(payload).hexdigest()
        manifest = {
            "apiVersion": "modelzoo.hil/v1alpha1",
            "kind": "AssetManifest",
            "metadata": {"name": "fixture", "version": "v1", "state": "candidate"},
            "spec": {
                "model": "fixture",
                "engine": "svp-nnn",
                "soc": "SS928V100",
                "files": [
                    {
                        "name": "input/payload.bin",
                        "sourceRef": "public-release",
                        "immutableTag": "fixture-v1",
                        "releaseAsset": "payload.bin",
                        "sha256": digest,
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
        manifest_path = root / "manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        release = root / "release"
        release.mkdir()
        (release / "payload.bin").write_bytes(payload)
        (release / "SHA256SUMS").write_text(
            f"{digest}  payload.bin\n", encoding="ascii"
        )
        (release / "LICENSE-AUDIT.md").write_text("authorized\n", encoding="utf-8")
        return manifest_path

    def test_exact_bundle_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, b"public payload")
            result = bundle.verify(manifest, root / "release")
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["tag"], "fixture-v1")

    def test_extra_file_and_checksum_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, b"public payload")
            release = root / "release"
            (release / "unexpected").write_text("no\n", encoding="utf-8")
            with self.assertRaisesRegex(bundle.BundleError, "file set mismatch"):
                bundle.verify(manifest, release)
            (release / "unexpected").unlink()
            (release / "SHA256SUMS").write_text(
                f"{'0' * 64}  payload.bin\n", encoding="ascii"
            )
            with self.assertRaisesRegex(bundle.BundleError, "does not exactly match"):
                bundle.verify(manifest, release)


if __name__ == "__main__":
    unittest.main()
