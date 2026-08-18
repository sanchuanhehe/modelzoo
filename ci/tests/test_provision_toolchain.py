from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ci" / "provision_toolchain.sh"


class ProvisionToolchainTests(unittest.TestCase):
    def test_verified_reconstructed_archive_is_reused(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        reuse = 'elif verify_file "$cann_archive" "$CANN_SOURCE_SIZE" "$CANN_SOURCE_SHA256"'
        rebuild = 'cann_tmp="$cann_archive.part"'
        self.assertIn(reuse, source)
        self.assertLess(source.index(reuse), source.index(rebuild))
        self.assertIn("Using verified reconstructed archive", source)


if __name__ == "__main__":
    unittest.main()
