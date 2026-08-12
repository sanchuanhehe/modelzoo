#!/usr/bin/env python3
"""Validate the pinned SDK release lock without network access."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> None:
    raise SystemExit(f"sdk-lock: {message}")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "ci/sdk-lock.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != 1:
        fail("unsupported schemaVersion")
    if data.get("repository") != "sanchuanhehe/modelzoo":
        fail("unexpected release repository")
    if not str(data.get("releaseTag", "")).startswith("sdk-"):
        fail("releaseTag must be an explicit SDK tag")
    for key in ("toolchain", "nnn", "svp-nnn"):
        artifact = data["artifacts"].get(key)
        if not isinstance(artifact, dict) or "archive" not in artifact:
            fail(f"missing artifact {key}")
        entries = artifact.get("parts", []) or [artifact["archive"]]
        for entry in entries:
            if not isinstance(entry.get("name"), str) or "/" in entry["name"]:
                fail(f"unsafe asset name for {key}")
            if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
                fail(f"invalid size for {entry.get('name')}")
            if not SHA256.fullmatch(str(entry.get("sha256", ""))):
                fail(f"invalid SHA-256 for {entry.get('name')}")
    svp = data["artifacts"]["svp-nnn"]
    if len(svp.get("parts", [])) < 2:
        fail("SVP_NNN must remain ordered multi-part content")
    if sum(part["size"] for part in svp["parts"]) != svp["archive"]["size"]:
        fail("SVP_NNN part sizes do not equal source archive size")
    compat = data["hostCompatibility"]["libisl19"]
    if not str(compat.get("url", "")).startswith("https://archive.ubuntu.com/"):
        fail("libisl19 must use the pinned Ubuntu archive")
    if not SHA256.fullmatch(str(compat.get("sha256", ""))):
        fail("invalid libisl19 SHA-256")
    print(f"Validated SDK lock for {data['releaseTag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
