#!/usr/bin/env python3
"""Validate ModelZoo metadata without model downloads or target hardware."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_CONFIGS = (
    ROOT / "samples/built-in/build_config.json",
    ROOT / "samples/samples_GPL/build_config.json",
    ROOT / "samples/contribute/build_config.json",
)
REQUIRED_BUILD_KEYS = {
    "buildTarget",
    "relativePath",
    "chip",
    "buildDef",
    "needSmoke",
    "description",
}
ALLOWED_BUILD_DEFS = {
    "Hi3516CV610": {"Hi3516CV610"},
    "Hi3591P": {"Hi3591P"},
    "SS928V100": {"OPTG", "SS928V100"},
}


class ValidationError(Exception):
    """A repository validation failure."""


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path.relative_to(ROOT)}: invalid JSON: {exc}") from exc


def validate_json_files() -> int:
    count = 0
    for path in sorted(ROOT.rglob("*.json")):
        if ".git" in path.parts:
            continue
        load_json(path)
        count += 1
    return count


def validate_build_configs() -> int:
    entry_count = 0
    for config in BUILD_CONFIGS:
        data = load_json(config)
        relative_config = config.relative_to(ROOT)
        if not isinstance(data, list):
            raise ValidationError(f"{relative_config}: top level must be a JSON array")

        sample_root = config.parent
        for index, entry in enumerate(data):
            location = f"{relative_config}[{index}]"
            if not isinstance(entry, dict):
                raise ValidationError(f"{location}: entry must be an object")

            missing = REQUIRED_BUILD_KEYS.difference(entry)
            if missing:
                raise ValidationError(f"{location}: missing keys {sorted(missing)}")

            path_value = entry["relativePath"]
            chip = entry["chip"]
            build_def = entry["buildDef"]
            if not all(isinstance(value, str) and value for value in (path_value, chip, build_def)):
                raise ValidationError(f"{location}: relativePath, chip and buildDef must be non-empty strings")

            allowed_defs = ALLOWED_BUILD_DEFS.get(chip)
            if allowed_defs is None or build_def not in allowed_defs:
                raise ValidationError(f"{location}: unsupported chip/buildDef pair {chip}/{build_def}")

            sample_dir = sample_root / path_value
            if not sample_dir.is_dir():
                raise ValidationError(f"{location}: sample directory does not exist: {sample_dir.relative_to(ROOT)}")
            cmake_entrypoints = (sample_dir / "CMakeLists.txt", sample_dir / "src/CMakeLists.txt")
            if not any(path.is_file() for path in cmake_entrypoints):
                raise ValidationError(f"{location}: sample is missing CMakeLists.txt or src/CMakeLists.txt")

            entry_count += 1

    return entry_count


def main() -> int:
    try:
        json_count = validate_json_files()
        build_count = validate_build_configs()
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Validated {json_count} JSON files and {build_count} sample build entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
