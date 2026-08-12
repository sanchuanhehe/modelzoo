#!/usr/bin/env python3
"""Prevent generated models, SDK archives, and accidental large files in Git."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = {".om", ".onnx", ".tgz", ".tar", ".rar", ".deb", ".run"}
MAX_SIZE = 25 * 1024 * 1024
LARGE_FILE_ALLOWLIST = {
    "samples/opensource/opencv/lib/libopencv_world.a": "f6a03a6fffd1fdd6d176f24deaf5772493d89250ed714faedad7517f267e4b06",
    "samples/samples_GPL/opensource/opencv/lib/libopencv_world.a": "f6a03a6fffd1fdd6d176f24deaf5772493d89250ed714faedad7517f267e4b06",
}


def main() -> int:
    failures: list[str] = []
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in filter(None, tracked):
        path = ROOT / name
        if path.suffix.lower() in BANNED:
            failures.append(f"generated/SDK file tracked: {name}")
        if path.is_file() and path.stat().st_size > MAX_SIZE:
            expected = LARGE_FILE_ALLOWLIST.get(name)
            if expected is None:
                failures.append(f"tracked file exceeds 25 MiB: {name}")
            else:
                import hashlib
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    failures.append(f"allowlisted large file changed: {name}")
    if failures:
        print("\n".join(failures))
        return 1
    print("Repository contains no tracked SDK/model archives or files over 25 MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
