#!/usr/bin/env python3
"""Unprivileged repository client for HIL definition validation and resolution."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci.hil import validate_config  # noqa: E402

VERSION = "0.2.0"
CAPABILITIES = ("validate", "resolve", "dry-run")


def event(command: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "hilctlVersion": VERSION,
        "command": command,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "details": details,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hilctl")
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities")
    for name in CAPABILITIES:
        command = commands.add_parser(name)
        command.add_argument("--definition", required=True, type=Path)
        command.add_argument("--inventory", type=Path)
        command.add_argument("--target")
        command.add_argument("--execution-ready", action="store_true")
    return parser


def run(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    if args.command == "capabilities":
        return "capabilities", "available", {"capabilities": list(CAPABILITIES)}
    resolved = validate_config.resolve_definition(
        args.definition,
        inventory_path=args.inventory,
        target_id=args.target,
        execution_ready=args.execution_ready,
    )
    if args.command == "validate":
        return "validate", "validated", {
            "definition": resolved["definition"],
            "assetState": resolved["assetState"],
            "executionReady": resolved["executionReady"],
        }
    if args.command == "resolve":
        return "resolve", "validated", resolved
    if args.command == "dry-run":
        return "dry-run", "validated", {
            "resolved": resolved,
            "note": "No asset, UART, SSH, target, or cleanup operation was performed",
        }
    raise validate_config.ConfigError(f"unsupported command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        command, status, details = run(args)
    except validate_config.ConfigError as exc:
        print(
            json.dumps(event(args.command, "failed", {"error": str(exc)}), sort_keys=True)
        )
        return 10
    print(json.dumps(event(command, status, details), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
