#!/usr/bin/env python3
"""Select affected SS928 build targets from a Git diff."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from build_matrix import entries

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE = {
    ("samples/built-in/classification/ResNet50", "SS928V100"),
    ("samples/built-in/classification/ResNet50", "OPTG"),
}
GLOBAL_PREFIXES = ("samples/common/", "ci/", ".github/", "CMakeLists.txt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--representative", action="store_true")
    args = parser.parse_args()
    all_entries = entries()
    if args.representative:
        selected = [x for x in all_entries if (x["sample"], x["buildDef"]) in REPRESENTATIVE]
    else:
        command = ["git", "diff", "--name-only", f"{args.base}...{args.head}"]
        changed = subprocess.check_output(command, cwd=ROOT, text=True).splitlines()
        source_changed = [p for p in changed if not p.startswith("docs/") and p not in {"README.md"}]
        if any(p.startswith(GLOBAL_PREFIXES) for p in source_changed):
            selected = [x for x in all_entries if (x["sample"], x["buildDef"]) in REPRESENTATIVE]
        else:
            selected = [x for x in all_entries if any(p == x["sample"] or p.startswith(x["sample"] + "/") for p in source_changed)]
            if source_changed:
                selected_by_key = {(x["sample"], x["buildDef"]): x for x in selected}
                selected_by_key.update(
                    {(x["sample"], x["buildDef"]): x for x in all_entries if (x["sample"], x["buildDef"]) in REPRESENTATIVE}
                )
                selected = sorted(selected_by_key.values(), key=lambda x: (x["engine"], x["sample"]))
    print(json.dumps({"include": selected}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
