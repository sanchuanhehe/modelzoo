#!/usr/bin/env python3
"""Combine independently built executable and model into one pre-HIL package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise SystemExit(f"missing or empty required input: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--conversion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    copies = {
        args.build / "main": args.output / "main",
        args.build / "build-manifest.json": args.output / "build-manifest.json",
        args.conversion / "model.om": args.output / "model.om",
        args.conversion / "expected.json": args.output / "expected.json",
        args.conversion / "conversion-manifest.json": args.output / "conversion-manifest.json",
        args.conversion / "input/model.onnx": args.output / "input/model.onnx",
        args.conversion / "input/input.bin": args.output / "input/input.bin",
    }
    for source, target in copies.items():
        copy_file(source, target)

    build = json.loads((args.output / "build-manifest.json").read_text())
    conversion = json.loads((args.output / "conversion-manifest.json").read_text())
    if build["commit"] != conversion["commit"]:
        raise SystemExit("build and conversion commits differ")
    if build["sdkReleaseTag"] != conversion["sdkReleaseTag"]:
        raise SystemExit("build and conversion SDK releases differ")

    payloads = [path for path in args.output.rglob("*") if path.is_file()]
    manifest = {
        "schemaVersion": 1,
        "boundary": "pre-HIL; no board execution performed",
        "commit": build["commit"],
        "sdkReleaseTag": build["sdkReleaseTag"],
        "sdkArtifacts": build["sdkArtifacts"],
        "compiler": build["compiler"],
        "converter": conversion["converter"],
        "engine": build["engine"],
        "soc": build["soc"],
        "cmake": build["cmake"],
        "conversion": conversion["conversion"],
        "outputs": [
            {"name": str(path.relative_to(args.output)), "size": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(payloads)
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksum_files = sorted(path for path in args.output.rglob("*") if path.is_file())
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.relative_to(args.output)}\n" for path in checksum_files)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
