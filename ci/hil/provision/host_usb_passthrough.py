#!/usr/bin/env python3
"""Safely attach one uniquely identified USB device to the HIL control VM.

This provisions transport only.  It never opens the serial endpoint and cannot
operate a relay.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Sequence


HEX_ID_RE = re.compile(r"^[0-9a-fA-F]{4}$")
DOMAIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}$")
SERIAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,126}$")


class ProvisionError(RuntimeError):
    """Expected, user-actionable provisioning failure."""


class UsbNotConnected(ProvisionError):
    """The exact expected USB identity is not currently visible."""


def normalize_hex_id(value: str, label: str) -> str:
    if not HEX_ID_RE.fullmatch(value):
        raise ProvisionError(f"{label} must contain exactly four hexadecimal digits")
    return value.lower()


def read_attribute(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def find_usb_device(
    sysfs_root: Path, vendor_id: str, product_id: str, serial: str
) -> Path:
    matches: list[Path] = []
    try:
        candidates = list(sysfs_root.iterdir())
    except OSError as exc:
        raise ProvisionError(f"USB sysfs is unavailable: {sysfs_root}") from exc
    for candidate in candidates:
        vendor = read_attribute(candidate / "idVendor")
        product = read_attribute(candidate / "idProduct")
        device_serial = read_attribute(candidate / "serial")
        if (
            vendor is not None
            and product is not None
            and vendor.lower() == vendor_id
            and product.lower() == product_id
            and device_serial == serial
        ):
            matches.append(candidate)
    if not matches:
        raise UsbNotConnected("the exact USB VID/PID/serial identity is not connected")
    if len(matches) != 1:
        raise ProvisionError("the exact USB identity is ambiguous")

    same_vid_pid = [
        candidate
        for candidate in candidates
        if (read_attribute(candidate / "idVendor") or "").lower() == vendor_id
        and (read_attribute(candidate / "idProduct") or "").lower() == product_id
    ]
    if len(same_vid_pid) != 1:
        raise ProvisionError(
            "libvirt matches USB hostdev by VID/PID; another device with the same "
            "IDs is connected"
        )
    return matches[0]


def hostdev_xml(vendor_id: str, product_id: str) -> str:
    hostdev = ET.Element(
        "hostdev", {"mode": "subsystem", "type": "usb", "managed": "yes"}
    )
    source = ET.SubElement(hostdev, "source", {"startupPolicy": "optional"})
    ET.SubElement(source, "vendor", {"id": f"0x{vendor_id}"})
    ET.SubElement(source, "product", {"id": f"0x{product_id}"})
    ET.indent(hostdev, space="  ")
    return ET.tostring(hostdev, encoding="unicode") + "\n"


def xml_has_hostdev(xml: str, vendor_id: str, product_id: str) -> bool:
    return xml_hostdev_address(xml, vendor_id, product_id) is not None


def xml_hostdev_address(
    xml: str, vendor_id: str, product_id: str
) -> tuple[int, ...] | None:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ProvisionError("virsh returned invalid domain XML") from exc
    expected_vendor = f"0x{vendor_id}"
    expected_product = f"0x{product_id}"
    for hostdev in root.findall("./devices/hostdev"):
        if hostdev.get("type") != "usb":
            continue
        source = hostdev.find("source")
        vendor = source.find("vendor") if source is not None else None
        product = source.find("product") if source is not None else None
        if (
            vendor is not None
            and product is not None
            and vendor.get("id", "").lower() == expected_vendor
            and product.get("id", "").lower() == expected_product
        ):
            address = source.find("address") if source is not None else None
            if address is None:
                return ()
            try:
                return (int(address.get("bus", "")), int(address.get("device", "")))
            except ValueError:
                return ()
    return None


def sysfs_bus_device(device: Path) -> tuple[int, int] | None:
    bus = read_attribute(device / "busnum")
    number = read_attribute(device / "devnum")
    if bus is None or number is None or not bus.isdigit() or not number.isdigit():
        return None
    return int(bus), int(number)


def run_checked(
    argv: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvisionError(f"command failed to start or timed out: {argv[0]}") from exc
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise ProvisionError(f"{argv[0]} failed: {diagnostic}")
    return result


def virsh_command(connection: str, *arguments: str) -> list[str]:
    return ["virsh", "-c", connection, *arguments]


def provision(
    *,
    domain: str,
    connection: str,
    vendor_id: str,
    product_id: str,
    serial: str,
    sysfs_root: Path = Path("/sys/bus/usb/devices"),
    dry_run: bool,
    refresh_live: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    if not DOMAIN_RE.fullmatch(domain):
        raise ProvisionError("unsafe libvirt domain name")
    if not SERIAL_RE.fullmatch(serial):
        raise ProvisionError("unsafe USB serial number")
    vendor_id = normalize_hex_id(vendor_id, "vendor ID")
    product_id = normalize_hex_id(product_id, "product ID")

    inactive_xml = run_checked(
        virsh_command(connection, "dumpxml", "--inactive", domain), runner=runner
    ).stdout
    already_configured = xml_has_hostdev(inactive_xml, vendor_id, product_id)
    state = run_checked(
        virsh_command(connection, "domstate", domain), runner=runner
    ).stdout.strip().lower()
    running = state == "running"
    device: Path | None = None
    try:
        device = find_usb_device(sysfs_root, vendor_id, product_id, serial)
    except UsbNotConnected:
        if not already_configured or refresh_live:
            raise
    live_address: tuple[int, int] | tuple[()] | None = None
    current_address = sysfs_bus_device(device) if device is not None else None
    if running:
        live_xml = run_checked(
            virsh_command(connection, "dumpxml", domain), runner=runner
        ).stdout
        live_address = xml_hostdev_address(live_xml, vendor_id, product_id)
    live_refresh_required = running and (
        live_address is None
        or (
            current_address is not None
            and live_address not in {(), current_address}
        )
    )
    details: dict[str, object] = {
        "domain": domain,
        "connection": connection,
        "usbIdentity": {
            "vendorId": vendor_id,
            "productId": product_id,
            "serial": serial,
        },
        "hostSysfsDevice": device.name if device is not None else None,
        "identityVerification": (
            "host-sysfs-exact" if device is not None else "controller-preflight-required"
        ),
        "currentHostAddress": list(current_address) if current_address else None,
        "liveHostAddress": list(live_address) if live_address else None,
        "liveRefreshRequired": live_refresh_required,
        "refreshLiveRequested": refresh_live,
        "domainRunning": running,
        "alreadyConfigured": already_configured,
        "dryRun": dry_run,
        "serialEndpointOpened": False,
        "relayCommandSent": False,
    }
    if already_configured and not (refresh_live and live_refresh_required):
        details["changed"] = False
        return details

    if already_configured and refresh_live and live_refresh_required:
        if dry_run:
            details["changed"] = False
            details["wouldRefreshLive"] = True
            return details
        if device is None or not running:
            raise ProvisionError("live refresh requires the exact connected USB device")
        xml = hostdev_xml(vendor_id, product_id)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="hil-usb-hostdev-", suffix=".xml"
        ) as descriptor:
            descriptor.write(xml)
            descriptor.flush()
            if live_address is not None:
                run_checked(
                    virsh_command(
                        connection, "detach-device", domain, descriptor.name, "--live"
                    ),
                    runner=runner,
                )
            run_checked(
                virsh_command(
                    connection, "attach-device", domain, descriptor.name, "--live"
                ),
                runner=runner,
            )
        details["changed"] = True
        details["liveRefreshed"] = True
        return details

    if dry_run:
        details["changed"] = False
        return details

    xml = hostdev_xml(vendor_id, product_id)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="hil-usb-hostdev-", suffix=".xml"
    ) as descriptor:
        descriptor.write(xml)
        descriptor.flush()
        run_checked(
            virsh_command(connection, "attach-device", domain, descriptor.name, "--config"),
            runner=runner,
        )
        if running:
            try:
                run_checked(
                    virsh_command(
                        connection, "attach-device", domain, descriptor.name, "--live"
                    ),
                    runner=runner,
                )
            except ProvisionError:
                try:
                    run_checked(
                        virsh_command(
                            connection,
                            "detach-device",
                            domain,
                            descriptor.name,
                            "--config",
                        ),
                        runner=runner,
                    )
                except ProvisionError as rollback_error:
                    raise ProvisionError(
                        "live attach failed and persistent rollback also failed: "
                        f"{rollback_error}"
                    ) from rollback_error
                raise
    details["changed"] = True
    return details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision identity-checked USB passthrough without operating it"
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument("--vendor-id", required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--connection", default="qemu:///system")
    parser.add_argument(
        "--refresh-live",
        action="store_true",
        help="refresh only a stale live attachment; persistent XML is unchanged",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = provision(
            domain=args.domain,
            connection=args.connection,
            vendor_id=args.vendor_id,
            product_id=args.product_id,
            serial=args.serial,
            dry_run=args.dry_run,
            refresh_live=args.refresh_live,
        )
    except ProvisionError as exc:
        print(f"USB passthrough provision failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
