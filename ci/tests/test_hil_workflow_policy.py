#!/usr/bin/env python3
"""Static policy tests for the privileged HIL workflow boundary."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIL = (ROOT / ".github/workflows/hil.yml").read_text(encoding="utf-8")
CI = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
BUILD_MANIFEST = (ROOT / "ci/build_manifest.py").read_text(encoding="utf-8")


class HilWorkflowPolicyTests(unittest.TestCase):
    def test_hil_is_manual_only(self) -> None:
        trigger = HIL.split("\npermissions:", maxsplit=1)[0]
        self.assertIn("workflow_dispatch:", trigger)
        self.assertNotRegex(trigger, r"(?m)^\s+(push|pull_request|schedule):")

    def test_authorization_and_source_identity_are_enforced(self) -> None:
        for required in (
            '[[ "$ACTOR" == "$OWNER" ]]',
            ".repository.full_name == $repo",
            ".head_repository.full_name == $repo",
            ".head_sha == $sha",
            '.conclusion == "success"',
            '.name == "CI"',
            '.path == ".github/workflows/ci.yml"',
        ):
            self.assertIn(required, HIL)

    def test_privileged_job_is_isolated_and_serialized(self) -> None:
        self.assertIn("environment: hil-hi3403", HIL)
        self.assertIn(
            "runs-on: [self-hosted, linux, x64, hil, hi3403, eulerpi-v1_3, svp-nnn]",
            HIL,
        )
        self.assertIn("group: hil-hi3403-01", HIL)
        self.assertIn("cancel-in-progress: false", HIL)
        self.assertRegex(HIL, r"(?m)^permissions:\n  contents: read\n  actions: read$")

    def test_standard_build_artifact_is_build_only(self) -> None:
        package_step = CI.split(
            "- name: Verify and package HIL input boundary", maxsplit=1
        )[1].split("- uses:", maxsplit=1)[0]
        self.assertNotIn("expected.json", package_step)
        self.assertIn('"boundary": "build-only; no board execution performed"', BUILD_MANIFEST)

    def test_not_run_artifacts_remain_explicitly_rejected(self) -> None:
        self.assertIn("Synthetic expected.status=not-run is forbidden.", HIL)
        self.assertIn("actions/download-artifact@", HIL)
        action_refs = re.findall(r"uses:\s+[^\s@]+@([^\s#]+)", HIL)
        self.assertTrue(action_refs)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs))


if __name__ == "__main__":
    unittest.main()
