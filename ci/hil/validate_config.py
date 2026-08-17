#!/usr/bin/env python3
"""Strict validation and resolution for data-only HIL v2 definitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
HIL_ROOT = ROOT / "ci" / "hil"
SCHEMA_PATHS = {
    "AssetManifest": HIL_ROOT / "schemas" / "asset-manifest.schema.json",
    "LabInventory": HIL_ROOT / "schemas" / "lab-inventory.schema.json",
    "TargetClass": HIL_ROOT / "schemas" / "target-class.schema.json",
    "TestAdapter": HIL_ROOT / "schemas" / "test-adapter.schema.json",
    "TestDefinition": HIL_ROOT / "schemas" / "test-definition.schema.json",
}
OTHER_SCHEMAS = (
    HIL_ROOT / "schemas" / "event.schema.json",
    HIL_ROOT / "schemas" / "lab-control-event.schema.json",
    HIL_ROOT / "schemas" / "evidence-manifest.schema.json",
)
FORBIDDEN_CONTROL_KEYS = {"steps", "finally", "run", "shell", "command"}
FIXED_PHASES = (
    "authorize",
    "assets.resolve-and-verify",
    "uart.start",
    "preflight.controller",
    "preflight.target",
    "adapter.prepare",
    "target.upload",
    "target.run",
    "target.download",
    "adapter.verify",
    "evidence.always",
    "cleanup.always",
)


class ConfigError(ValueError):
    """Raised when a HIL definition is malformed, unsafe, or inconsistent."""


class StrictLoader(yaml.SafeLoader):
    """Safe YAML loader which also rejects duplicate mapping keys."""


def _construct_mapping(
    loader: StrictLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_document(path: Path) -> dict[str, Any]:
    try:
        documents = list(yaml.load_all(path.read_text(encoding="utf-8"), StrictLoader))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ConfigError) as exc:
        raise ConfigError(f"cannot load {path}: {exc}") from exc
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise ConfigError(f"{path} must contain exactly one YAML object")
    return documents[0]


def load_schema(kind: str) -> dict[str, Any]:
    try:
        schema = json.loads(SCHEMA_PATHS[kind].read_text(encoding="utf-8"))
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load schema for {kind}: {exc}") from exc
    Draft202012Validator.check_schema(schema)
    return schema


def validate_all_schemas() -> None:
    for path in (*SCHEMA_PATHS.values(), *OTHER_SCHEMAS):
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot load schema {path}: {exc}") from exc
        Draft202012Validator.check_schema(schema)


def validate_schema(document: dict[str, Any], path: Path) -> str:
    kind = document.get("kind")
    if kind not in SCHEMA_PATHS:
        raise ConfigError(f"{path}: unsupported kind {kind!r}")
    errors = sorted(
        Draft202012Validator(load_schema(kind)).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ConfigError(f"{path}:{location}: {error.message}")
    return kind


def safe_relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ConfigError(f"{field} contains an unsafe path: {value!r}")
    return Path(*path.parts)


def resolve_repo_file(relative: str, expected_parent: Path) -> Path:
    candidate = ROOT / relative
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ConfigError(f"referenced configuration does not exist: {relative}") from exc
    parent = expected_parent.resolve()
    if resolved.parent != parent or candidate.is_symlink() or not resolved.is_file():
        raise ConfigError(f"unsafe configuration reference: {relative}")
    return resolved


def reject_control_keys(value: object, location: str = "<root>") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_CONTROL_KEYS:
                raise ConfigError(
                    f"workflow control key {key!r} is forbidden in data definitions at {location}"
                )
            reject_control_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_control_keys(child, f"{location}[{index}]")


def validate_asset_semantics(
    manifest: dict[str, Any],
    *,
    execution_ready: bool,
    inventory: dict[str, Any] | None = None,
) -> None:
    files = manifest["spec"]["files"]
    names = [item["name"] for item in files]
    if len(names) != len(set(names)):
        raise ConfigError("asset manifest contains duplicate file names")
    identities = [
        (item["sourceRef"], item["immutableTag"], item["releaseAsset"])
        for item in files
    ]
    if len(identities) != len(set(identities)):
        raise ConfigError("asset manifest contains duplicate immutable source identities")
    for item in files:
        status = item["license"]["status"]
        if item["access"] == "public" and status != "redistributable":
            raise ConfigError(
                f"public asset {item['name']} must have redistributable license status"
            )
        if status in {"restricted", "unconfirmed"} and item["access"] != "restricted":
            raise ConfigError(
                f"asset {item['name']} with {status} license must use restricted access"
            )
    if not execution_ready:
        return
    if manifest["metadata"]["state"] != "released":
        raise ConfigError("execution requires an asset manifest in released state")
    unconfirmed = [
        item["name"] for item in files if item["license"]["status"] == "unconfirmed"
    ]
    if unconfirmed:
        raise ConfigError(
            f"execution rejects assets with unconfirmed license: {unconfirmed}"
        )
    if inventory is None:
        raise ConfigError("execution-ready resolution requires VM LabInventory")
    sources = inventory["spec"]["assetSources"]
    for item in files:
        source = sources.get(item["sourceRef"])
        if source is None:
            raise ConfigError(f"asset sourceRef is absent from LabInventory: {item['sourceRef']}")
        if source["kind"] == "unresolved" or source["authorizationStatus"] != "approved":
            raise ConfigError(f"asset source is not authorized: {item['sourceRef']}")
        if item["access"] == "restricted" and source["accessClass"] != "restricted":
            raise ConfigError(
                f"restricted asset source is not classified restricted: {item['sourceRef']}"
            )


def load_typed(path: Path, kind: str) -> dict[str, Any]:
    document = load_document(path)
    if validate_schema(document, path) != kind:
        raise ConfigError(f"{path} is not a {kind}")
    reject_control_keys(document)
    return document


def resolve_definition(
    definition_path: Path,
    *,
    inventory_path: Path | None = None,
    target_id: str | None = None,
    execution_ready: bool = False,
) -> dict[str, Any]:
    definition = load_typed(definition_path, "TestDefinition")
    spec = definition["spec"]
    target_name = spec["targetClass"]
    target_path = resolve_repo_file(
        f"ci/hil/target-classes/{target_name}.yaml", HIL_ROOT / "target-classes"
    )
    target_class = load_typed(target_path, "TargetClass")
    if target_class["metadata"]["name"] != target_name:
        raise ConfigError("TargetClass metadata.name does not match TestDefinition")

    adapter_name = spec["adapter"]
    adapter_root = HIL_ROOT / "adapters" / adapter_name
    adapter_path = resolve_repo_file(
        f"ci/hil/adapters/{adapter_name}/adapter.yaml", adapter_root
    )
    adapter = load_typed(adapter_path, "TestAdapter")
    if adapter["metadata"]["name"] != adapter_name:
        raise ConfigError("TestAdapter metadata.name does not match TestDefinition")
    if adapter["spec"]["targetClass"] != target_name:
        raise ConfigError("TestAdapter targetClass does not match TestDefinition")
    for required in ("prepare.py", "verify.py", "target/run-test"):
        candidate = adapter_root / required
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ConfigError(f"adapter is missing conventional file: {required}") from exc
        if adapter_root.resolve() not in resolved.parents or candidate.is_symlink() or not resolved.is_file():
            raise ConfigError(f"unsafe adapter conventional file: {required}")

    manifest_path = resolve_repo_file(spec["assetManifest"], HIL_ROOT / "assets")
    manifest = load_typed(manifest_path, "AssetManifest")
    inventory: dict[str, Any] | None = None
    selected_target: dict[str, Any] | None = None
    if inventory_path is not None:
        inventory = load_typed(inventory_path, "LabInventory")
        if target_id is None:
            raise ConfigError("target ID is required when LabInventory is provided")
        selected_target = inventory["spec"]["targets"].get(target_id)
        if selected_target is None:
            raise ConfigError(f"target is absent from LabInventory: {target_id}")
        if selected_target["targetClass"] != target_name:
            raise ConfigError("LabInventory targetClass does not match TestDefinition")
    elif execution_ready:
        raise ConfigError("execution-ready resolution requires LabInventory and target ID")
    validate_asset_semantics(
        manifest, execution_ready=execution_ready, inventory=inventory
    )

    build = spec["buildArtifact"]
    expected_values = {
        "engine": target_class["spec"]["engine"],
        "soc": target_class["spec"]["soc"],
    }
    for field, expected in expected_values.items():
        if build[field] != expected or manifest["spec"][field] != expected:
            raise ConfigError(
                f"{field} mismatch across TestDefinition, TargetClass, and AssetManifest"
            )
    if adapter["spec"]["resultType"] != spec["expectation"]["type"]:
        raise ConfigError("adapter resultType does not match expectation type")

    result: dict[str, Any] = {
        "schemaVersion": 1,
        "definition": definition["metadata"]["name"],
        "targetClass": target_name,
        "adapter": adapter_name,
        "buildArtifact": build,
        "assetManifest": str(manifest_path.relative_to(ROOT)),
        "assetState": manifest["metadata"]["state"],
        "iterations": spec["iterations"],
        "timeoutSeconds": spec["timeoutSeconds"],
        "expectation": spec["expectation"],
        "fixedPhases": list(FIXED_PHASES),
        "executionReady": execution_ready,
    }
    if selected_target is not None:
        result["targetId"] = target_id
        result["concurrencyGroup"] = selected_target["concurrencyGroup"]
        result["executionMode"] = selected_target["executionMode"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--definition", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--execution-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_all_schemas()
        if args.all:
            if args.inventory or args.target or args.execution_ready:
                raise ConfigError("--all cannot be combined with execution options")
            definitions = sorted((HIL_ROOT / "definitions").glob("*.yaml"))
            if not definitions:
                raise ConfigError("no TestDefinitions found")
            results = [resolve_definition(path) for path in definitions]
            for inventory_path in sorted((HIL_ROOT / "inventory").glob("*.yaml")):
                load_typed(inventory_path, "LabInventory")
        else:
            results = [
                resolve_definition(
                    args.definition,
                    inventory_path=args.inventory,
                    target_id=args.target,
                    execution_ready=args.execution_ready,
                )
            ]
    except ConfigError as exc:
        print(f"HIL configuration validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "validated", "definitions": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
