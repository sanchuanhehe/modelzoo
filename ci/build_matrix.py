#!/usr/bin/env python3
"""Build deduplicated SS928 CI matrices from repository build_config files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(ROOT.glob("samples/**/build_config.json"))


def entries() -> list[dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for config in CONFIGS:
        for item in json.loads(config.read_text(encoding="utf-8")):
            build_def = item["buildDef"]
            if build_def not in {"SS928V100", "OPTG"}:
                continue
            sample = (config.parent / item["relativePath"]).relative_to(ROOT).as_posix()
            key = (sample, build_def)
            result[key] = {
                "id": f"{sample.removeprefix('samples/').replace('/', '-')}-{build_def}".lower(),
                "engine": "svp-nnn" if build_def == "SS928V100" else "nnn",
                "buildDef": build_def,
                "sample": sample,
            }
    return sorted(result.values(), key=lambda x: (x["engine"], x["sample"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nightly", action="store_true")
    args = parser.parse_args()
    all_entries = entries()
    if args.nightly:
        matrix = [
            {"engine": engine, "targets": [item for item in all_entries if item["engine"] == engine]}
            for engine in ("svp-nnn", "nnn")
        ]
        skipped = 91 - len(all_entries)
        print(json.dumps({"include": matrix}, separators=(",", ":")))
        print(f"supported={len(all_entries)} skipped={skipped}", file=__import__("sys").stderr)
    else:
        print(json.dumps({"include": all_entries}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
