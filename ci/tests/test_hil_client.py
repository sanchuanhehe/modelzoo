from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ci" / "hil" / "hilctl.py"
SPEC = importlib.util.spec_from_file_location("hil_client", MODULE_PATH)
assert SPEC and SPEC.loader
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


class HilClientTests(unittest.TestCase):
    def test_client_capabilities_are_unprivileged(self) -> None:
        self.assertEqual(client.CAPABILITIES, ("validate", "resolve", "dry-run"))
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "subprocess",
            "urllib.request",
            "socket",
            "ssh",
            "scp",
            "capture_uart",
            "os.remove",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
