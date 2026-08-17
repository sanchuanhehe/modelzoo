from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "ci" / "hil" / "provision" / "controller.sh"
TARGET = ROOT / "ci" / "hil" / "provision" / "target.sh"


class HilProvisionV2Tests(unittest.TestCase):
    def test_controller_uses_immutable_release_and_atomic_current_link(self) -> None:
        source = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("/opt/hil/control/releases", source)
        self.assertIn("INSTALL-SHA256SUMS", source)
        self.assertIn("version collision", source)
        self.assertIn('current path is not a symbolic link', source)
        self.assertIn('/usr/local/bin/lab-control is not a symbolic link', source)
        self.assertIn('command_link=/usr/local/bin/.lab-control.$$', source)
        self.assertIn('mv -Tf "$temporary_link" "$base/current"', source)

    def test_target_is_versioned_and_preserves_unrelated_keys(self) -> None:
        source = TARGET.read_text(encoding="utf-8")
        self.assertIn("hil-target-agent-$version", source)
        self.assertIn("version collision", source)
        self.assertIn("grep -v ' modelzoo-hil-target-agent$'", source)
        self.assertIn("root|hilagent", source)
        self.assertIn("/etc/hil/target-execution-user", source)
        self.assertIn("execution identity change requires explicit deprovision", source)
        self.assertIn("public key file must contain exactly one non-empty line", source)
        self.assertIn("stable target-agent path is not a symbolic link", source)
        self.assertNotIn("rm -f \"$authorized_keys\"", source)


if __name__ == "__main__":
    unittest.main()
