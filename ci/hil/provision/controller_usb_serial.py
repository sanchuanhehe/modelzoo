#!/usr/bin/env python3
"""Install a narrow udev rule for an identity-pinned HIL USB serial device.

This provisions naming and access only.  It never opens the serial endpoint and
cannot operate a relay.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

PROVISION_ROOT = Path(__file__).resolve().parent
if str(PROVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(PROVISION_ROOT))

from host_usb_passthrough import (  # noqa: E402
    ProvisionError,
    normalize_hex_id,
    run_checked,
)


SERIAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,126}$")
SYMLINK_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,126}$")
GROUP_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


def udev_rule(
    vendor_id: str, product_id: str, serial: str, symlink: str, group: str
) -> str:
    values = (vendor_id, product_id, serial, symlink, group)
    if any('"' in value or "\\" in value for value in values):
        raise ProvisionError("udev values contain unsafe quoting")
    return (
        'ACTION=="add|change", SUBSYSTEM=="tty", '
        f'ATTRS{{idVendor}}=="{vendor_id}", '
        f'ATTRS{{idProduct}}=="{product_id}", '
        f'ATTRS{{serial}}=="{serial}", '
        'ENV{ID_MM_DEVICE_IGNORE}="1", '
        f'GROUP="{group}", MODE="0660", SYMLINK+="{symlink}"\n'
    )


def provision(
    *,
    vendor_id: str,
    product_id: str,
    serial: str,
    symlink: str,
    group: str,
    rules_directory: Path = Path("/etc/udev/rules.d"),
    dry_run: bool,
    effective_uid: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    uid = os.geteuid() if effective_uid is None else effective_uid
    if not dry_run and uid != 0:
        raise ProvisionError("controller udev provisioning must run as root")
    vendor_id = normalize_hex_id(vendor_id, "vendor ID")
    product_id = normalize_hex_id(product_id, "product ID")
    if not SERIAL_RE.fullmatch(serial):
        raise ProvisionError("unsafe USB serial number")
    if not SYMLINK_RE.fullmatch(symlink) or symlink.startswith("/") or ".." in symlink.split("/"):
        raise ProvisionError("unsafe udev symlink")
    if not GROUP_RE.fullmatch(group):
        raise ProvisionError("unsafe device group")

    content = udev_rule(vendor_id, product_id, serial, symlink, group)
    rule_path = rules_directory / "99-modelzoo-hil-usb-serial.rules"
    details: dict[str, object] = {
        "rulePath": str(rule_path),
        "deviceSymlink": f"/dev/{symlink}",
        "usbIdentity": {
            "vendorId": vendor_id,
            "productId": product_id,
            "serial": serial,
        },
        "group": group,
        "mode": "0660",
        "modemManagerIgnored": True,
        "dryRun": dry_run,
        "serialEndpointOpened": False,
        "relayCommandSent": False,
    }
    existing = None
    try:
        existing = rule_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ProvisionError(f"cannot read existing udev rule: {exc}") from exc
    if existing == content or dry_run:
        details["changed"] = False
        return details
    if not rules_directory.is_dir() or rules_directory.is_symlink():
        raise ProvisionError("udev rules directory is unavailable or unsafe")

    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".99-modelzoo-hil-usb-serial.", dir=rules_directory
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, rule_path)
        temporary_name = None
    except OSError as exc:
        raise ProvisionError(f"cannot install udev rule: {exc}") from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    run_checked(["udevadm", "control", "--reload-rules"], runner=runner)
    run_checked(
        [
            "udevadm", "trigger", "--action=change", "--subsystem-match=tty",
        ],
        runner=runner,
    )
    details["changed"] = True
    return details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision identity-pinned HIL USB serial naming and access"
    )
    parser.add_argument("--vendor-id", required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--symlink", required=True)
    parser.add_argument("--group", default="dialout")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = provision(
            vendor_id=args.vendor_id,
            product_id=args.product_id,
            serial=args.serial,
            symlink=args.symlink,
            group=args.group,
            dry_run=args.dry_run,
        )
    except ProvisionError as exc:
        print(f"controller USB serial provision failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
