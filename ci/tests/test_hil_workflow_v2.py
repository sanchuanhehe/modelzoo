from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "hil-v2.yml"


class HilWorkflowV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.workflow = yaml.load(cls.text, Loader=yaml.BaseLoader)

    def test_trigger_and_control_plane_are_narrow(self) -> None:
        self.assertEqual(set(self.workflow["on"]), {"workflow_dispatch"})
        hil = self.workflow["jobs"]["hil"]
        self.assertEqual(hil["environment"], "hil-hi3403")
        self.assertEqual(
            hil["runs-on"],
            ["self-hosted", "linux", "x64", "hil", "hi3403", "eulerpi-v1_3", "svp-nnn"],
        )
        self.assertEqual(hil["concurrency"]["cancel-in-progress"], "false")
        self.assertNotIn("workflow_call", self.text)
        self.assertNotIn("pull_request", self.workflow["on"])

    def test_fixed_phase_order_and_always_evidence_cleanup(self) -> None:
        names = [step.get("name", "uses") for step in self.workflow["jobs"]["hil"]["steps"]]
        required = [
            "Resolve reviewed TestDefinition and asset policy",
            "Fetch and verify immutable assets",
            "Start external UART capture",
            "Controller preflight",
            "Target reachability and optional one-shot reset recovery",
            "Target agent preflight",
            "Prepare model-specific payload through reviewed adapter",
            "Upload sealed payload",
            "Run fixed target entrypoint",
            "Download retained outputs",
            "Verify business result through reviewed adapter",
            "Collect evidence even when the target is unavailable",
            "Clean exact target sandbox",
            "Finalize evidence after target cleanup",
            "Clean exact controller sandbox",
        ]
        positions = [names.index(name) for name in required]
        self.assertEqual(positions, sorted(positions))
        by_name = {
            step.get("name"): step for step in self.workflow["jobs"]["hil"]["steps"]
        }
        for name in (
            "Download retained outputs",
            "Stop UART capture",
            "Collect evidence even when the target is unavailable",
            "Clean exact target sandbox",
            "Finalize evidence after target cleanup",
            "Clean exact controller sandbox",
        ):
            self.assertEqual(by_name[name]["if"], "always()")
        self.assertNotIn("|| true", self.text)

    def test_workflow_does_not_use_vm_golden_cache_or_direct_ssh(self) -> None:
        self.assertNotIn("/opt/hil/models", self.text)
        self.assertNotIn("HIL_GOLDEN_DIR", self.text)
        self.assertNotIn("ssh ", self.text)
        self.assertIn("/usr/local/bin/lab-control", self.text)
        self.assertNotIn("ci/hil/assets/resnet50-svp-nnn-v1.yaml", self.text)
        self.assertIn('"ci/hil/adapters/$HIL_ADAPTER/prepare.py"', self.text)
        self.assertIn('"ci/hil/adapters/$HIL_ADAPTER/verify.py"', self.text)

    def test_early_failure_evidence_does_not_depend_on_github_env(self) -> None:
        env = self.workflow["jobs"]["hil"]["env"]
        self.assertIn("HIL_CREDENTIALS_DIRECTORY", env)
        self.assertNotIn("HIL_CREDENTIALS_DIRECTORY=", self.text)
        names = [
            step.get("name") for step in self.workflow["jobs"]["hil"]["steps"]
        ]
        self.assertLess(
            names.index("Initialize exact run and evidence context"),
            names.index("Resolve reviewed TestDefinition and asset policy"),
        )
        checkout = next(
            index
            for index, step in enumerate(self.workflow["jobs"]["hil"]["steps"])
            if step.get("uses", "").startswith("actions/checkout@")
        )
        self.assertLess(
            names.index("Initialize exact run and evidence context"), checkout
        )
        self.assertLess(
            names.index("Resolve reviewed TestDefinition and asset policy"),
            names.index("Stage runtime-only credentials"),
        )

    def test_staged_openssh_private_key_has_required_terminal_newline(self) -> None:
        self.assertIn(
            "printf '%s\\n' \"$V2_BOARD_SSH_KEY\" >\"$HIL_CREDENTIALS_DIRECTORY/hil-v2-board-ssh\"",
            self.text,
        )
        self.assertNotIn(
            "printf '%s' \"$V2_BOARD_SSH_KEY\" >\"$HIL_CREDENTIALS_DIRECTORY/hil-v2-board-ssh\"",
            self.text,
        )

    def test_root_execution_is_limited_to_master_source_artifacts(self) -> None:
        self.assertIn(
            "Root target execution accepts only CI artifacts built from master.",
            self.text,
        )
        self.assertIn(".details.executionMode", self.text)
        self.assertIn("source-branch", self.text)
        self.assertIn("workflow-sha", self.text)
        self.assertIn('--workflow-sha "$HIL_SOURCE_WORKFLOW_SHA"', self.text)

    def test_reset_recovery_is_explicit_single_shot_and_probe_only(self) -> None:
        inputs = self.workflow["on"]["workflow_dispatch"]["inputs"]
        policy = inputs["reset_policy"]
        self.assertEqual(policy["default"], "none")
        self.assertEqual(
            policy["options"], ["none", "reset-on-unreachable-once"]
        )
        self.assertEqual(self.text.count("lab-control reset pulse"), 1)
        self.assertEqual(self.text.count("lab-control target wait-ready"), 1)
        recovery = next(
            step
            for step in self.workflow["jobs"]["hil"]["steps"]
            if step.get("name")
            == "Target reachability and optional one-shot reset recovery"
        )["run"]
        self.assertIn("lab-control target probe", recovery)
        self.assertIn("probe_status=$?", recovery)
        self.assertIn("[[ $probe_status -ne 41 ]]", recovery)
        self.assertIn("reset is forbidden", recovery)
        self.assertIn("reset-on-unreachable-once", recovery)
        self.assertNotIn("target preflight", recovery)


if __name__ == "__main__":
    unittest.main()
