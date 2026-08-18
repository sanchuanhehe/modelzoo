from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "ci" / "hil" / "provision" / "controller.sh"
TARGET = ROOT / "ci" / "hil" / "provision" / "target.sh"
HOST_USB = ROOT / "ci" / "hil" / "provision" / "host_usb_passthrough.py"
CONTROLLER_USB = ROOT / "ci" / "hil" / "provision" / "controller_usb_serial.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


host_usb = load_module("hil_host_usb_passthrough", HOST_USB)
controller_usb = load_module("hil_controller_usb_serial", CONTROLLER_USB)


class HilProvisionV2Tests(unittest.TestCase):
    def test_controller_uses_immutable_release_and_atomic_current_link(self) -> None:
        source = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("/opt/hil/control/releases", source)
        self.assertIn("INSTALL-SHA256SUMS", source)
        self.assertIn("version collision", source)
        self.assertIn('current path is not a symbolic link', source)
        self.assertIn('/usr/local/bin/lab-control is not a symbolic link', source)
        self.assertIn('command_link=/usr/local/bin/.lab-control.$$', source)
        self.assertIn('mv -Tf "$temporary_link" "$base/current"', source)
        self.assertIn("find \"$source_root/ci/hil/schemas\"", source)
        self.assertIn('done <"$schema_list"', source)
        self.assertNotIn('schemas/*.json; do', source)

    def test_target_is_versioned_and_preserves_unrelated_keys(self) -> None:
        source = TARGET.read_text(encoding="utf-8")
        self.assertIn("hil-target-agent-$version", source)
        self.assertIn("version collision", source)
        self.assertIn("grep -v ' modelzoo-hil-target-agent$'", source)
        self.assertIn("root|hilagent", source)
        self.assertIn("/etc/hil/target-execution-user", source)
        self.assertIn("execution identity change requires explicit deprovision", source)
        self.assertIn("public key file must contain exactly one non-empty line", source)
        self.assertIn("stable target-agent path is not a symbolic link", source)
        self.assertNotIn("rm -f \"$authorized_keys\"", source)

    def test_host_usb_passthrough_requires_exact_unique_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "1-2"
            device.mkdir()
            (device / "idVendor").write_text("0483\n", encoding="ascii")
            (device / "idProduct").write_text("5740\n", encoding="ascii")
            (device / "serial").write_text("698684C41432\n", encoding="ascii")
            calls: list[list[str]] = []

            def runner(argv, **_kwargs):
                calls.append(argv)
                if "dumpxml" in argv:
                    return subprocess.CompletedProcess(
                        argv, 0, "<domain><devices/></domain>\n", ""
                    )
                if "domstate" in argv:
                    return subprocess.CompletedProcess(argv, 0, "running\n", "")
                return subprocess.CompletedProcess(argv, 0, "", "")

            result = host_usb.provision(
                domain="hil-hi3403-01",
                connection="qemu:///system",
                vendor_id="0483",
                product_id="5740",
                serial="698684C41432",
                sysfs_root=root,
                dry_run=True,
                runner=runner,
            )
            self.assertFalse(result["changed"])
            self.assertFalse(result["serialEndpointOpened"])
            self.assertFalse(result["relayCommandSent"])
            self.assertEqual(len(calls), 3)

            duplicate = root / "1-3"
            duplicate.mkdir()
            (duplicate / "idVendor").write_text("0483\n", encoding="ascii")
            (duplicate / "idProduct").write_text("5740\n", encoding="ascii")
            (duplicate / "serial").write_text("DIFFERENT\n", encoding="ascii")
            with self.assertRaisesRegex(host_usb.ProvisionError, "another device"):
                host_usb.find_usb_device(root, "0483", "5740", "698684C41432")

    def test_host_usb_passthrough_rolls_back_config_if_live_attach_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "1-2"
            device.mkdir()
            for name, value in {
                "idVendor": "0483",
                "idProduct": "5740",
                "serial": "698684C41432",
            }.items():
                (device / name).write_text(value + "\n", encoding="ascii")
            calls: list[list[str]] = []

            def runner(argv, **_kwargs):
                calls.append(argv)
                if "dumpxml" in argv:
                    return subprocess.CompletedProcess(
                        argv, 0, "<domain><devices/></domain>\n", ""
                    )
                if "domstate" in argv:
                    return subprocess.CompletedProcess(argv, 0, "running\n", "")
                if "attach-device" in argv and "--live" in argv:
                    return subprocess.CompletedProcess(argv, 1, "", "live failed")
                return subprocess.CompletedProcess(argv, 0, "", "")

            with self.assertRaisesRegex(host_usb.ProvisionError, "live failed"):
                host_usb.provision(
                    domain="hil-hi3403-01",
                    connection="qemu:///system",
                    vendor_id="0483",
                    product_id="5740",
                    serial="698684C41432",
                    sysfs_root=root,
                    dry_run=False,
                    runner=runner,
                )
            self.assertTrue(any("detach-device" in call for call in calls))

    def test_existing_hostdev_is_idempotent_when_live_device_left_host_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def runner(argv, **_kwargs):
                if "dumpxml" in argv:
                    return subprocess.CompletedProcess(
                        argv,
                        0,
                        """<domain><devices><hostdev type="usb"><source>
                        <vendor id="0x0483"/><product id="0x5740"/>
                        </source></hostdev></devices></domain>""",
                        "",
                    )
                if "domstate" in argv:
                    return subprocess.CompletedProcess(argv, 0, "running\n", "")
                raise AssertionError(argv)

            result = host_usb.provision(
                domain="hil-hi3403-01",
                connection="qemu:///system",
                vendor_id="0483",
                product_id="5740",
                serial="698684C41432",
                sysfs_root=root,
                dry_run=True,
                runner=runner,
            )
            self.assertTrue(result["alreadyConfigured"])
            self.assertEqual(
                result["identityVerification"], "controller-preflight-required"
            )
            self.assertIsNone(result["hostSysfsDevice"])

    def test_host_usb_can_refresh_only_a_stale_live_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "1-6"
            device.mkdir()
            for name, value in {
                "idVendor": "0483",
                "idProduct": "5740",
                "serial": "698684C41432",
                "busnum": "1",
                "devnum": "13",
            }.items():
                (device / name).write_text(value + "\n", encoding="ascii")
            calls: list[list[str]] = []

            def runner(argv, **_kwargs):
                calls.append(argv)
                if "dumpxml" in argv and "--inactive" in argv:
                    xml = """<domain><devices><hostdev type="usb"><source>
                    <vendor id="0x0483"/><product id="0x5740"/>
                    </source></hostdev></devices></domain>"""
                    return subprocess.CompletedProcess(argv, 0, xml, "")
                if "dumpxml" in argv:
                    xml = """<domain><devices><hostdev type="usb"><source>
                    <vendor id="0x0483"/><product id="0x5740"/>
                    <address bus="1" device="3"/>
                    </source></hostdev></devices></domain>"""
                    return subprocess.CompletedProcess(argv, 0, xml, "")
                if "domstate" in argv:
                    return subprocess.CompletedProcess(argv, 0, "running\n", "")
                return subprocess.CompletedProcess(argv, 0, "", "")

            result = host_usb.provision(
                domain="hil-hi3403-01",
                connection="qemu:///system",
                vendor_id="0483",
                product_id="5740",
                serial="698684C41432",
                sysfs_root=root,
                dry_run=False,
                refresh_live=True,
                runner=runner,
            )
            self.assertTrue(result["liveRefreshed"])
            self.assertTrue(any("detach-device" in call for call in calls))
            self.assertTrue(any("attach-device" in call for call in calls))
            self.assertFalse(any("--config" in call for call in calls))

    def test_controller_usb_rule_is_exact_and_never_opens_serial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rules = Path(directory)
            calls: list[list[str]] = []

            def runner(argv, **_kwargs):
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, "", "")

            result = controller_usb.provision(
                vendor_id="0483",
                product_id="5740",
                serial="698684C41432",
                symlink="hil/hi3403-rst-relay",
                group="dialout",
                rules_directory=rules,
                dry_run=False,
                effective_uid=0,
                runner=runner,
            )
            rule = (rules / "99-modelzoo-hil-usb-serial.rules").read_text()
            self.assertIn('ATTRS{serial}=="698684C41432"', rule)
            self.assertIn('ENV{ID_MM_DEVICE_IGNORE}="1"', rule)
            self.assertIn('SYMLINK+="hil/hi3403-rst-relay"', rule)
            self.assertFalse(result["serialEndpointOpened"])
            self.assertFalse(result["relayCommandSent"])
            self.assertEqual([call[0] for call in calls], ["udevadm", "udevadm"])

    def test_controller_usb_rule_rejects_path_escape_and_non_root_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(controller_usb.ProvisionError, "unsafe udev"):
                controller_usb.provision(
                    vendor_id="0483",
                    product_id="5740",
                    serial="698684C41432",
                    symlink="hil/../escape",
                    group="dialout",
                    rules_directory=Path(directory),
                    dry_run=True,
                    effective_uid=1000,
                )
            with self.assertRaisesRegex(controller_usb.ProvisionError, "must run as root"):
                controller_usb.provision(
                    vendor_id="0483",
                    product_id="5740",
                    serial="698684C41432",
                    symlink="hil/hi3403-rst-relay",
                    group="dialout",
                    rules_directory=Path(directory),
                    dry_run=False,
                    effective_uid=1000,
                )


if __name__ == "__main__":
    unittest.main()
