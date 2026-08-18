#!/usr/bin/env python3
"""Verify an exact public HIL Release bundle without publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci.hil import validate_config  # noqa: E402

CHECKSUM_RE = re.compile(r"^(?P<sha>[0-9a-f]{64})  (?P<name>[A-Za-z0-9._-]+)$")
COMPANIONS = {"SHA256SUMS", "LICENSE-AUDIT.md"}


class BundleError(RuntimeError):
    """Release bundle violates its reviewed manifest."""


def file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify(manifest_path: Path, bundle_path: Path) -> dict[str, object]:
    manifest = validate_config.load_typed(manifest_path, "AssetManifest")
    validate_config.validate_asset_semantics(manifest, execution_ready=False)
    files = manifest["spec"]["files"]
    if any(
        item["access"] != "public"
        or item["license"]["status"] != "redistributable"
        for item in files
    ):
        raise BundleError("public Release requires redistributable public assets")
    tags = {item["immutableTag"] for item in files}
    if len(tags) != 1:
        raise BundleError("a Release bundle must use exactly one immutable tag")
    release_names = [item["releaseAsset"] for item in files]
    if len(release_names) != len(set(release_names)):
        raise BundleError("Release asset names must be unique")
    try:
        root = bundle_path.resolve(strict=True)
    except OSError as exc:
        raise BundleError("Release bundle directory is unavailable") from exc
    if bundle_path.is_symlink() or not root.is_dir():
        raise BundleError("Release bundle must be a real directory")
    actual = {path.name for path in root.iterdir()}
    expected = {*release_names, *COMPANIONS}
    if actual != expected:
        raise BundleError(
            f"Release file set mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    identities: dict[str, dict[str, object]] = {}
    for item in files:
        path = root / item["releaseAsset"]
        if path.is_symlink() or not path.is_file():
            raise BundleError(f"unsafe Release asset: {item['releaseAsset']}")
        size, digest = file_identity(path)
        if size != item["size"] or digest != item["sha256"]:
            raise BundleError(f"Release identity mismatch: {item['releaseAsset']}")
        identities[item["releaseAsset"]] = {"size": size, "sha256": digest}
    sums_path = root / "SHA256SUMS"
    audit_path = root / "LICENSE-AUDIT.md"
    if (
        sums_path.is_symlink()
        or not sums_path.is_file()
        or audit_path.is_symlink()
        or not audit_path.is_file()
        or audit_path.stat().st_size == 0
    ):
        raise BundleError("Release companion files are unsafe or empty")
    recorded: dict[str, str] = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        match = CHECKSUM_RE.fullmatch(line)
        if not match or match["name"] in recorded:
            raise BundleError("SHA256SUMS is malformed or duplicated")
        recorded[match["name"]] = match["sha"]
    expected_sums = {name: data["sha256"] for name, data in identities.items()}
    if recorded != expected_sums:
        raise BundleError("SHA256SUMS does not exactly match the AssetManifest")
    return {
        "status": "verified",
        "tag": next(iter(tags)),
        "manifest": manifest["metadata"]["name"],
        "version": manifest["metadata"]["version"],
        "assets": identities,
        "companions": sorted(COMPANIONS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.manifest, args.bundle)
    except (BundleError, OSError, UnicodeDecodeError, validate_config.ConfigError) as exc:
        print(f"HIL Release bundle verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
