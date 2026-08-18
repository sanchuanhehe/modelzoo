#!/usr/bin/env python3
"""Prepare the fixed ResNet50/SVP_NNN target payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci.hil import validate_config  # noqa: E402


def digest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def regular_file(root: Path, relative: str) -> Path:
    safe = validate_config.safe_relative_path(relative, "adapter payload path")
    candidate = root / safe
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"missing adapter input: {relative}") from exc
    if root.resolve() not in resolved.parents or candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"unsafe adapter input: {relative}")
    return resolved


def prepare(
    artifact_root: Path,
    asset_root: Path,
    asset_manifest: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise ValueError("adapter output must not already exist")
    manifest = validate_config.load_typed(asset_manifest, "AssetManifest")
    output.mkdir(mode=0o750, parents=False)
    payload_files: list[dict[str, object]] = []

    sources = [(regular_file(artifact_root, "main"), "main")]
    sources.extend(
        (regular_file(asset_root, item["name"]), f"assets/{item['name']}")
        for item in manifest["spec"]["files"]
        if item["name"] != "expected.json"
    )
    run_test = Path(__file__).resolve().parent / "target" / "run-test"
    sources.append((run_test, "run-test"))
    for source, relative in sources:
        destination = output / validate_config.safe_relative_path(relative, "payload path")
        destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o750 if relative in {"main", "run-test"} else 0o640)
        payload_files.append({"name": relative, **digest(destination)})

    content_manifest = {
        "schemaVersion": 1,
        "adapter": "resnet50-svp-nnn",
        "entrypoint": "run-test",
        "files": payload_files,
    }
    manifest_path = output / "payload-manifest.json"
    manifest_path.write_text(
        json.dumps(content_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload_files.append({"name": "payload-manifest.json", **digest(manifest_path)})
    sums = "".join(
        f"{item['sha256']}  {item['name']}\n" for item in payload_files
    )
    (output / "PAYLOAD_SHA256SUMS").write_text(sums, encoding="ascii")
    return content_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(
            args.artifact_root, args.asset_root, args.asset_manifest, args.output
        )
    except (OSError, ValueError, validate_config.ConfigError) as exc:
        print(f"adapter prepare failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
