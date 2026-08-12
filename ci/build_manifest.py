#!/usr/bin/env python3
"""Write a reproducible build or HIL input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--soc", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--main", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model")
    args = p.parse_args()
    files = [pathlib.Path(args.main)] + ([pathlib.Path(args.model)] if args.model else [])
    data = {
        "schemaVersion": 1,
        "commit": os.environ.get("GITHUB_SHA") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sdkReleaseTag": json.loads(pathlib.Path("ci/sdk-lock.json").read_text())["releaseTag"],
        "engine": args.engine, "soc": args.soc, "sample": args.sample,
        "cmake": {"buildType": "Release", "toolchain": "samples/common/cmake/toolchain_aarch64_linux.cmake"},
        "outputs": [{"name": f.name, "size": f.stat().st_size, "sha256": sha(f)} for f in files],
    }
    pathlib.Path(args.output).write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
