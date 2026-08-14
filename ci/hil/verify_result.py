#!/usr/bin/env python3
"""Validate immutable HIL assets and an ordered ResNet50 Top-5 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath

EXPECTED_KEYS = {
    "schemaVersion", "model", "engine", "soc", "assetVersion", "input",
    "top1", "top5", "timeoutSeconds", "resultFile",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(ValueError):
    """Raised when HIL evidence is malformed or inconsistent."""


def safe_relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValidationError(f"{field} contains an unsafe path: {value!r}")
    return Path(*path.parts)


def load_expected(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid expected JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("expected JSON must be an object")
    if data.get("status") == "not-run":
        raise ValidationError("pre-HIL expected.status=not-run is not executable evidence")
    if set(data) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS.difference(data))
        extra = sorted(set(data).difference(EXPECTED_KEYS))
        raise ValidationError(f"expected schema mismatch; missing={missing}, extra={extra}")
    if data["schemaVersion"] != 1:
        raise ValidationError("unsupported expected schemaVersion")
    for field in ("model", "engine", "soc", "assetVersion"):
        if not isinstance(data[field], str) or not data[field]:
            raise ValidationError(f"{field} must be a non-empty string")
    safe_relative_path(data["input"], "input")
    safe_relative_path(data["resultFile"], "resultFile")
    top1, top5 = data["top1"], data["top5"]
    if isinstance(top1, bool) or not isinstance(top1, int) or top1 < 0:
        raise ValidationError("top1 must be a non-negative integer")
    if (
        not isinstance(top5, list) or len(top5) != 5
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in top5)
        or len(set(top5)) != 5 or top5[0] != top1
    ):
        raise ValidationError("top5 must be five unique non-negative integers beginning with top1")
    timeout = data["timeoutSeconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 3600:
        raise ValidationError("timeoutSeconds must be an integer in [1, 3600]")
    return data


def read_top5(path: Path) -> list[int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read result file: {exc}") from exc
    ranked: list[int] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 2:
            raise ValidationError(f"result line {line_number} must be index,value")
        try:
            index, score = int(fields[0]), float(fields[1])
        except ValueError as exc:
            raise ValidationError(f"invalid result line {line_number}: {line!r}") from exc
        if index < 0 or not math.isfinite(score):
            raise ValidationError(f"invalid result line {line_number}: {line!r}")
        ranked.append(index)
    if len(ranked) < 5 or len(set(ranked[:5])) != 5:
        raise ValidationError("result must contain at least five unique ranked classes")
    return ranked[:5]


def verify_sha256s(root: Path, sums_path: Path) -> int:
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read SHA256SUMS: {exc}") from exc
    checked = 0
    root = root.resolve()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
            raise ValidationError(f"malformed SHA256SUMS line {line_number}")
        relative = safe_relative_path(parts[1].lstrip("*"), f"SHA256SUMS line {line_number}")
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValidationError(f"missing checksum target: {relative}") from exc
        if root not in resolved.parents or not resolved.is_file() or candidate.is_symlink():
            raise ValidationError(f"checksum target escapes asset root: {relative}")
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != parts[0]:
            raise ValidationError(f"checksum mismatch: {relative}")
        checked += 1
    if checked == 0:
        raise ValidationError("SHA256SUMS contains no entries")
    return checked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--validate-expected-only", action="store_true")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--sha256s", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        expected = load_expected(args.expected)
        if (args.asset_root is None) != (args.sha256s is None):
            raise ValidationError("--asset-root and --sha256s must be used together")
        checked = 0
        if args.asset_root is not None and args.sha256s is not None:
            checked = verify_sha256s(args.asset_root, args.sha256s)
        if args.validate_expected_only:
            if args.result is not None:
                raise ValidationError("--result cannot be used with --validate-expected-only")
            print(json.dumps({"status": "validated", "top1": expected["top1"], "top5": expected["top5"], "checksums": checked}))
            return 0
        if args.result is None:
            raise ValidationError("--result is required unless --validate-expected-only is used")
        actual_top5 = read_top5(args.result)
        if actual_top5 != expected["top5"]:
            raise ValidationError(f"ordered Top-5 mismatch: expected={expected['top5']}, actual={actual_top5}")
        if actual_top5[0] != expected["top1"]:
            raise ValidationError("Top-1 mismatch")
    except ValidationError as exc:
        print(f"HIL verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "top1": actual_top5[0], "top5": actual_top5, "checksums": checked}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
