from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ci" / "hil" / "validate_config.py"
SPEC = importlib.util.spec_from_file_location("hil_validate_config", MODULE_PATH)
assert SPEC and SPEC.loader
config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(config)
SMOKE = ROOT / "ci" / "hil" / "definitions" / "resnet50-smoke.yaml"
INVENTORY = ROOT / "ci" / "hil" / "inventory" / "lab.example.yaml"
MANIFEST = (
    ROOT / "ci" / "hil" / "assets" / "resnet50-svp-nnn-v1.yaml"
)


class HilV2ConfigTests(unittest.TestCase):
    def write_yaml(self, document: object) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        )
        with temporary:
            yaml.safe_dump(document, temporary, sort_keys=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_committed_definitions_resolve_to_fixed_workflow(self) -> None:
        for name, iterations in (
            ("resnet50-smoke.yaml", 1),
            ("resnet50-stability.yaml", 10),
        ):
            result = config.resolve_definition(
                ROOT / "ci" / "hil" / "definitions" / name
            )
            self.assertEqual(result["iterations"], iterations)
            self.assertEqual(result["fixedPhases"], list(config.FIXED_PHASES))
            self.assertEqual(result["assetState"], "released")

    def test_definitions_cannot_supply_steps_run_or_shell(self) -> None:
        definition = config.load_document(SMOKE)
        for forbidden in ("steps", "run", "shell", "finally"):
            modified = copy.deepcopy(definition)
            modified["spec"][forbidden] = [] if forbidden in {"steps", "finally"} else "id"
            path = self.write_yaml(modified)
            with self.assertRaises(config.ConfigError):
                config.validate_schema(modified, path)

    def test_recursive_control_key_guard_is_explicit(self) -> None:
        with self.assertRaisesRegex(config.ConfigError, "workflow control key"):
            config.reject_control_keys({"metadata": {"run": "id"}})

    def test_released_manifest_is_execution_ready(self) -> None:
        result = config.resolve_definition(
            SMOKE,
            inventory_path=INVENTORY,
            target_id="hi3403-01",
            execution_ready=True,
        )
        self.assertTrue(result["executionReady"])
        self.assertEqual(result["assetState"], "released")

    def test_candidate_manifest_is_not_execution_ready(self) -> None:
        manifest = config.load_document(MANIFEST)
        manifest["metadata"]["state"] = "candidate"
        with self.assertRaisesRegex(config.ConfigError, "released state"):
            config.validate_asset_semantics(
                manifest,
                execution_ready=True,
                inventory=config.load_document(INVENTORY),
            )

    def test_inventory_keeps_topology_separate_from_target_class(self) -> None:
        target_class = config.load_document(
            ROOT / "ci" / "hil" / "target-classes" / "hi3403-svp-nnn.yaml"
        )
        serialized = yaml.safe_dump(target_class)
        for topology in ("192.168.2.88", "/dev/serial", "credentialRef"):
            self.assertNotIn(topology, serialized)
        inventory = config.load_document(INVENTORY)
        self.assertIn("192.168.2.88", yaml.safe_dump(inventory))

    def test_target_class_matches_frozen_board_inventory_requirements(self) -> None:
        target_class = config.load_typed(
            ROOT / "ci" / "hil" / "target-classes" / "hi3403-svp-nnn.yaml",
            "TargetClass",
        )["spec"]
        self.assertEqual(target_class["soc"], "SS928V100")
        self.assertEqual(target_class["engine"], "svp-nnn")
        self.assertEqual(target_class["requiredKernelModules"], ["ot_svp_npu"])
        self.assertIn("ot_pqp", target_class["forbiddenKernelModules"])
        self.assertEqual(target_class["minimumFreeBytes"], 1073741824)
        self.assertFalse(target_class["temperaturePolicy"]["required"])
        self.assertEqual(
            target_class["temperaturePolicy"]["maximumMilliCelsius"], 90000
        )
        self.assertEqual(
            set(target_class["requiredLibraries"]),
            {
                "libsvp_acl.so",
                "libsvp_aicpu.so",
                "libsecurec.so",
                "libprotobuf-c.so.1",
                "libopencv_world.so.412",
            },
        )

    def test_root_execution_requires_explicit_mode_and_rationale(self) -> None:
        inventory = config.load_document(INVENTORY)
        target = inventory["spec"]["targets"]["hi3403-01"]
        target["user"] = "root"
        target["executionMode"] = "forced-command-root"
        target.pop("executionRationale", None)
        path = self.write_yaml(inventory)
        with self.assertRaisesRegex(config.ConfigError, "executionRationale"):
            config.load_typed(path, "LabInventory")
        target["executionRationale"] = (
            "Vendor dynamic loader is root-only; UART recovery remains available."
        )
        path = self.write_yaml(inventory)
        self.assertEqual(
            config.load_typed(path, "LabInventory")["spec"]["targets"]["hi3403-01"][
                "executionMode"
            ],
            "forced-command-root",
        )

    def test_manifest_records_publication_authorization_and_logical_source(self) -> None:
        manifest = config.load_document(MANIFEST)
        om = manifest["spec"]["files"][0]
        self.assertEqual(om["sourceRef"], "modelzoo-public-release")
        self.assertNotIn("uri", om)
        self.assertEqual(om["access"], "public")
        self.assertEqual(om["license"]["status"], "redistributable")
        golden = manifest["spec"]["files"][1]
        self.assertEqual(golden["sourceRef"], "modelzoo-public-release")
        self.assertEqual(golden["access"], "public")
        self.assertEqual(golden["license"]["status"], "redistributable")

    def test_duplicate_yaml_key_is_rejected(self) -> None:
        path = self.write_yaml({"one": 1})
        path.write_text("kind: TestDefinition\nkind: TargetClass\n", encoding="utf-8")
        with self.assertRaisesRegex(config.ConfigError, "duplicate YAML key"):
            config.load_document(path)

    def test_public_unconfirmed_asset_is_rejected(self) -> None:
        manifest = config.load_document(MANIFEST)
        manifest["spec"]["files"][0]["access"] = "public"
        manifest["spec"]["files"][0]["license"]["status"] = "unconfirmed"
        with self.assertRaisesRegex(config.ConfigError, "public asset"):
            config.validate_asset_semantics(manifest, execution_ready=False)

    def test_restricted_asset_cannot_resolve_through_public_backend(self) -> None:
        manifest = config.load_document(MANIFEST)
        manifest["metadata"]["state"] = "released"
        manifest["spec"]["files"][0]["sourceRef"] = "private-model-assets"
        manifest["spec"]["files"][0]["access"] = "restricted"
        manifest["spec"]["files"][0]["license"]["status"] = "restricted"
        inventory = config.load_document(INVENTORY)
        inventory["spec"]["assetSources"]["private-model-assets"] = {
            "kind": "github-release",
            "repository": "sanchuanhehe/modelzoo",
            "credentialRef": "restricted-asset-token",
            "accessClass": "public",
            "authorizationStatus": "approved",
        }
        inventory = config.load_typed(self.write_yaml(inventory), "LabInventory")
        with self.assertRaisesRegex(config.ConfigError, "classified restricted"):
            config.validate_asset_semantics(
                manifest, execution_ready=True, inventory=inventory
            )

    def test_definition_path_traversal_is_rejected(self) -> None:
        definition = config.load_document(SMOKE)
        definition["spec"]["assetManifest"] = "../../secret.yaml"
        path = self.write_yaml(definition)
        with self.assertRaises(config.ConfigError):
            config.validate_schema(definition, path)


if __name__ == "__main__":
    unittest.main()
