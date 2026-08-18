#!/usr/bin/env python3
"""Validate ordered ResNet50 Top-5 results for every requested iteration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci.hil import verify_result  # noqa: E402


def verify(expected_path: Path, results_root: Path, count: int) -> dict[str, object]:
    if not 1 <= count <= 100:
        raise verify_result.ValidationError("count must be in [1, 100]")
    expected = verify_result.load_expected(expected_path)
    if results_root.is_symlink() or not results_root.is_dir():
        raise verify_result.ValidationError("results root must be a real directory")
    records: list[dict[str, object]] = []
    required_iterations = [str(index) for index in range(1, count + 1)]
    actual_iterations = {path.name for path in results_root.iterdir()}
    if actual_iterations != set(required_iterations):
        raise verify_result.ValidationError(
            "iteration directory set mismatch: "
            f"expected={required_iterations}, actual={sorted(actual_iterations)}"
        )
    for index in range(1, count + 1):
        iteration_root = results_root / str(index)
        result_path = iteration_root / "output" / "result.txt"
        exit_code_path = iteration_root / "exit-code"
        if iteration_root.is_symlink() or not iteration_root.is_dir():
            raise verify_result.ValidationError(f"unsafe iteration directory: {index}")
        if exit_code_path.read_text(encoding="ascii").strip() != "0":
            raise verify_result.ValidationError(f"nonzero retained exit code: {index}")
        top5 = verify_result.read_top5(result_path)
        if top5 != expected["top5"] or top5[0] != expected["top1"]:
            raise verify_result.ValidationError(
                f"ordered Top-5 mismatch in iteration {index}: "
                f"expected={expected['top5']}, actual={top5}"
            )
        records.append({"iteration": index, "top1": top5[0], "top5": top5})
    return {"status": "passed", "count": len(records), "results": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    args = parser.parse_args()
    try:
        result = verify(args.expected, args.results_root, args.count)
    except verify_result.ValidationError as exc:
        print(f"adapter verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
