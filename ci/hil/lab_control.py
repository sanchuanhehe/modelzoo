#!/usr/bin/env python3
"""VM-only laboratory primitives for assets, artifacts, and target transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci.hil import validate_config  # noqa: E402

VERSION = "0.2.0"
EXIT_CONFIG = 10
EXIT_ASSET = 20
EXIT_ARTIFACT = 30
EXIT_TARGET = 40
EXIT_EXECUTION = 50
EXIT_EVIDENCE = 60
EXIT_CLEANUP = 70
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAYLOAD_LINE_RE = re.compile(
    r"^(?P<sha>[0-9a-f]{64})  (?P<name>[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*)$"
)


class LabError(RuntimeError):
    """Expected laboratory operation failure with a stable exit category."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def event(command: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "labControlVersion": VERSION,
        "command": command,
        "status": status,
        "timestamp": timestamp(),
        "details": details,
    }


def hash_file(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def atomic_write(path: Path, source: BinaryIO, mode: int = 0o640) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_inventory(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"LabInventory is unavailable: {path}", EXIT_CONFIG) from exc
    if path.is_symlink() or not resolved.is_file():
        raise LabError("LabInventory must be a real file", EXIT_CONFIG)
    if resolved == Path("/etc/hil/lab-inventory.yaml"):
        metadata = resolved.stat()
        if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise LabError("runtime LabInventory ownership/mode is unsafe", EXIT_CONFIG)
    try:
        return validate_config.load_typed(resolved, "LabInventory")
    except validate_config.ConfigError as exc:
        raise LabError(str(exc), EXIT_CONFIG) from exc


def select_target(inventory: dict[str, Any], target_id: str) -> dict[str, Any]:
    target = inventory["spec"]["targets"].get(target_id)
    if target is None:
        raise LabError(f"target is absent from LabInventory: {target_id}", EXIT_CONFIG)
    return target


def credential_file(directory: Path, reference: str, *, private: bool) -> Path:
    try:
        root = directory.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"credential directory does not exist: {directory}", EXIT_CONFIG) from exc
    if directory.is_symlink() or not root.is_dir():
        raise LabError("credential directory must be a real directory", EXIT_CONFIG)
    root_stat = root.stat()
    if root_stat.st_uid != os.geteuid() or stat.S_IMODE(root_stat.st_mode) & 0o077:
        raise LabError("credential directory ownership/mode is unsafe", EXIT_CONFIG)
    candidate = root / reference
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"missing credential reference: {reference}", EXIT_CONFIG) from exc
    if resolved.parent != root or candidate.is_symlink() or not resolved.is_file():
        raise LabError(f"unsafe credential reference: {reference}", EXIT_CONFIG)
    mode = stat.S_IMODE(resolved.stat().st_mode)
    allowed = {0o400, 0o600} if private else {0o400, 0o440, 0o600, 0o640}
    if mode not in allowed:
        raise LabError(f"credential {reference} has unsafe mode {mode:o}", EXIT_CONFIG)
    return resolved


def safe_run_directory(run_root: Path, run_id: str) -> tuple[Path, Path]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise LabError("unsafe run ID", EXIT_CLEANUP)
    try:
        root = run_root.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"run root does not exist: {run_root}", EXIT_CLEANUP) from exc
    if run_root.is_symlink() or not root.is_dir():
        raise LabError("run root must be a real directory", EXIT_CLEANUP)
    run_dir = root / run_id
    if run_dir.is_symlink() or run_dir.resolve(strict=False).parent != root:
        raise LabError("unsafe run directory", EXIT_CLEANUP)
    return root, run_dir


def validate_regular_file(path: Path, *, size: int, sha256: str, name: str) -> None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"missing file: {name}", EXIT_ASSET) from exc
    if path.is_symlink() or not resolved.is_file():
        raise LabError(f"file is not regular: {name}", EXIT_ASSET)
    actual_size, actual_sha = hash_file(resolved)
    if actual_size != size:
        raise LabError(
            f"size mismatch for {name}: expected={size}, actual={actual_size}", EXIT_ASSET
        )
    if actual_sha != sha256:
        raise LabError(f"SHA-256 mismatch for {name}", EXIT_ASSET)


def load_asset_manifest(
    path: Path,
    *,
    execution_ready: bool,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        manifest = validate_config.load_typed(path, "AssetManifest")
        validate_config.validate_asset_semantics(
            manifest, execution_ready=execution_ready, inventory=inventory
        )
    except validate_config.ConfigError as exc:
        raise LabError(str(exc), EXIT_CONFIG) from exc
    return manifest


def verify_assets(manifest_path: Path, asset_root: Path) -> dict[str, Any]:
    manifest = load_asset_manifest(manifest_path, execution_ready=False)
    try:
        root = asset_root.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"asset root does not exist: {asset_root}", EXIT_ASSET) from exc
    if asset_root.is_symlink() or not root.is_dir():
        raise LabError("asset root must be a real directory", EXIT_ASSET)
    checked: list[dict[str, Any]] = []
    for item in manifest["spec"]["files"]:
        relative = validate_config.safe_relative_path(item["name"], "asset name")
        candidate = root / relative
        try:
            parent = candidate.parent.resolve(strict=True)
        except OSError as exc:
            raise LabError(f"missing asset parent: {item['name']}", EXIT_ASSET) from exc
        if parent != root and root not in parent.parents:
            raise LabError(f"asset path escapes root: {item['name']}", EXIT_ASSET)
        validate_regular_file(
            candidate, size=item["size"], sha256=item["sha256"], name=item["name"]
        )
        checked.append(
            {"name": item["name"], "size": item["size"], "sha256": item["sha256"]}
        )
    return {"root": str(root), "files": checked}


def github_request(
    request: urllib.request.Request,
    *,
    timeout: int,
    open_url: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> BinaryIO:
    try:
        return open_url(request, timeout=timeout)
    except (OSError, urllib.error.URLError) as exc:
        raise LabError(f"GitHub asset request failed: {exc}", EXIT_ASSET) from exc


def verify_restricted_github_source(
    source: dict[str, Any],
    token: str,
    *,
    open_url: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> None:
    if source["accessClass"] != "restricted":
        return
    repository = source["repository"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"modelzoo-lab-control/{VERSION}",
    }
    with github_request(
        urllib.request.Request(
            f"https://api.github.com/repos/{repository}", headers=headers
        ),
        timeout=30,
        open_url=open_url,
    ) as response:
        try:
            metadata = json.load(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LabError("restricted repository metadata is invalid JSON", EXIT_ASSET) from exc
    if metadata.get("full_name") != repository or metadata.get("private") is not True:
        raise LabError(
            "restricted asset source is not an API-verified private repository",
            EXIT_ASSET,
        )


def fetch_assets(
    inventory_path: Path,
    manifest_path: Path,
    asset_root: Path,
    credentials_dir: Path,
    *,
    open_url: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    manifest = load_asset_manifest(
        manifest_path, execution_ready=True, inventory=inventory
    )
    asset_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    root = asset_root.resolve(strict=True)
    if asset_root.is_symlink() or not root.is_dir():
        raise LabError("asset root must be a real directory", EXIT_ASSET)
    records: list[dict[str, Any]] = []
    verified_restricted_sources: set[str] = set()
    for item in manifest["spec"]["files"]:
        source = inventory["spec"]["assetSources"][item["sourceRef"]]
        token_path = credential_file(
            credentials_dir, source["credentialRef"], private=True
        )
        token = token_path.read_text(encoding="utf-8").strip()
        if not token or "\n" in token:
            raise LabError(f"invalid token credential: {item['sourceRef']}", EXIT_CONFIG)
        if (
            source["accessClass"] == "restricted"
            and item["sourceRef"] not in verified_restricted_sources
        ):
            verify_restricted_github_source(source, token, open_url=open_url)
            verified_restricted_sources.add(item["sourceRef"])
        destination = root / validate_config.safe_relative_path(item["name"], "asset name")
        destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        parent = destination.parent.resolve(strict=True)
        if parent != root and root not in parent.parents:
            raise LabError(f"asset destination escapes root: {item['name']}", EXIT_ASSET)
        if destination.exists():
            validate_regular_file(
                destination,
                size=item["size"],
                sha256=item["sha256"],
                name=item["name"],
            )
            records.append({"name": item["name"], "source": "existing"})
            continue
        repository = source["repository"]
        metadata_url = (
            f"https://api.github.com/repos/{repository}/releases/tags/"
            f"{item['immutableTag']}"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"modelzoo-lab-control/{VERSION}",
        }
        with github_request(
            urllib.request.Request(metadata_url, headers=headers),
            timeout=30,
            open_url=open_url,
        ) as response:
            try:
                release = json.load(response)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LabError("release metadata is invalid JSON", EXIT_ASSET) from exc
        if (
            release.get("tag_name") != item["immutableTag"]
            or release.get("draft") is not False
        ):
            raise LabError("release tag identity/state mismatch", EXIT_ASSET)
        matches = [
            asset
            for asset in release.get("assets", [])
            if isinstance(asset, dict) and asset.get("name") == item["releaseAsset"]
        ]
        if len(matches) != 1 or matches[0].get("size") != item["size"]:
            raise LabError(
                f"release asset identity/size mismatch: {item['releaseAsset']}", EXIT_ASSET
            )
        asset_url = matches[0].get("url")
        if not isinstance(asset_url, str) or not asset_url.startswith(
            "https://api.github.com/"
        ):
            raise LabError("release asset API URL is invalid", EXIT_ASSET)
        download_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/octet-stream",
            "User-Agent": f"modelzoo-lab-control/{VERSION}",
        }
        with github_request(
            urllib.request.Request(asset_url, headers=download_headers),
            timeout=300,
            open_url=open_url,
        ) as response:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                validate_regular_file(
                    temporary,
                    size=item["size"],
                    sha256=item["sha256"],
                    name=item["name"],
                )
                temporary.chmod(0o640)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        records.append({"name": item["name"], "source": item["sourceRef"]})
    verify_assets(manifest_path, root)
    return {"root": str(root), "files": records}


def is_aarch64_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(20)
    except OSError:
        return False
    if len(header) < 20 or header[:4] != b"\x7fELF" or header[4] != 2:
        return False
    byte_order = {1: "little", 2: "big"}.get(header[5])
    return byte_order is not None and int.from_bytes(header[18:20], byte_order) == 183


def verify_artifact(
    artifact_root: Path,
    source_sha: str,
    workflow_sha: str,
    sdk_lock_path: Path,
    *,
    sample: str,
    engine: str,
    soc: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise LabError("source SHA must be 40 lowercase hexadecimal characters", EXIT_ARTIFACT)
    if not re.fullmatch(r"[0-9a-f]{40}", workflow_sha):
        raise LabError("workflow SHA must be 40 lowercase hexadecimal characters", EXIT_ARTIFACT)
    try:
        root = artifact_root.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"artifact root does not exist: {artifact_root}", EXIT_ARTIFACT) from exc
    if artifact_root.is_symlink() or not root.is_dir():
        raise LabError("artifact root must be a real directory", EXIT_ARTIFACT)
    required_names = {"main", "build.log", "build-manifest.json", "SHA256SUMS"}
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != required_names:
        raise LabError(
            f"artifact file set mismatch: expected={sorted(required_names)}, "
            f"actual={sorted(actual_names)}",
            EXIT_ARTIFACT,
        )
    main = root / "main"
    build_log = root / "build.log"
    manifest_path = root / "build-manifest.json"
    sums_path = root / "SHA256SUMS"
    for path in (main, build_log, manifest_path, sums_path):
        if path.is_symlink() or not path.is_file():
            raise LabError(f"missing or symlinked artifact input: {path.name}", EXIT_ARTIFACT)
    expected_sum_names = {"main", "build.log", "build-manifest.json"}
    seen_sum_names: set[str] = set()
    try:
        sum_lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise LabError(f"invalid artifact SHA256SUMS: {exc}", EXIT_ARTIFACT) from exc
    for line in sum_lines:
        match = PAYLOAD_LINE_RE.fullmatch(line)
        if not match or match.group("name") not in expected_sum_names:
            raise LabError("artifact SHA256SUMS is malformed", EXIT_ARTIFACT)
        name = match.group("name")
        if name in seen_sum_names or hash_file(root / name)[1] != match.group("sha"):
            raise LabError(f"artifact checksum mismatch or duplicate: {name}", EXIT_ARTIFACT)
        seen_sum_names.add(name)
    if seen_sum_names != expected_sum_names:
        raise LabError("artifact SHA256SUMS file set mismatch", EXIT_ARTIFACT)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabError(f"invalid build manifest: {exc}", EXIT_ARTIFACT) from exc
    expected = {
        "schemaVersion": 1,
        "commit": source_sha,
        "engine": engine,
        "soc": soc,
        "sample": sample,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(field) != value for field, value in expected.items()
    ):
        raise LabError("build manifest identity mismatch", EXIT_ARTIFACT)
    if manifest.get("workflowCommit") != workflow_sha:
        raise LabError("build manifest workflowCommit mismatch", EXIT_ARTIFACT)
    try:
        sdk_lock = json.loads(sdk_lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabError(f"invalid SDK lock: {exc}", EXIT_ARTIFACT) from exc
    if sdk_lock_path.is_symlink() or not sdk_lock_path.is_file():
        raise LabError("SDK lock must be a regular non-symlink file", EXIT_ARTIFACT)
    try:
        expected_sdk = {
            "toolchain": sdk_lock["artifacts"]["toolchain"]["archive"]["sha256"],
            engine: sdk_lock["artifacts"][engine]["archive"]["sha256"],
        }
        expected_release = sdk_lock["releaseTag"]
    except (KeyError, TypeError) as exc:
        raise LabError("SDK lock is missing required artifact identities", EXIT_ARTIFACT) from exc
    if (
        manifest.get("sdkReleaseTag") != expected_release
        or manifest.get("sdkArtifacts") != expected_sdk
        or manifest.get("boundary") != "build-only; no board execution performed"
    ):
        raise LabError("build manifest SDK/boundary mismatch", EXIT_ARTIFACT)
    size, sha = hash_file(main)
    outputs = manifest.get("outputs")
    matches = [
        output
        for output in outputs if isinstance(output, dict) and output.get("name") == "main"
    ] if isinstance(outputs, list) else []
    if len(matches) != 1 or matches[0].get("size") != size or matches[0].get("sha256") != sha:
        raise LabError("main size/SHA-256 does not match build manifest", EXIT_ARTIFACT)
    if not is_aarch64_elf(main):
        raise LabError("main is not an AArch64 ELF64 executable", EXIT_ARTIFACT)
    return {"sourceSha": source_sha, "workflowCommit": workflow_sha, "main": {"size": size, "sha256": sha}}


def ssh_base(target: dict[str, Any], credentials_dir: Path) -> list[str]:
    key = credential_file(credentials_dir, target["sshCredentialRef"], private=True)
    known_hosts = credential_file(
        credentials_dir, target["knownHostsCredentialRef"], private=False
    )
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "ConnectTimeout=5", "-i", str(key), f"{target['user']}@{target['host']}",
    ]


def target_command(
    target: dict[str, Any],
    credentials_dir: Path,
    arguments: list[str],
    *,
    input_data: bytes | None = None,
    timeout: int = 30,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> subprocess.CompletedProcess[bytes]:
    for argument in arguments:
        if not re.fullmatch(r"[A-Za-z0-9._/+:-]+", argument):
            raise LabError("unsafe target-agent argument", EXIT_TARGET)
    try:
        result = runner(
            [*ssh_base(target, credentials_dir), " ".join(arguments)],
            input=input_data,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LabError(f"target-agent command failed to start: {exc}", EXIT_TARGET) from exc
    return result


def target_probe(
    inventory_path: Path,
    target_id: str,
    credentials_dir: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    result = target_command(target, credentials_dir, ["probe"], runner=runner)
    if result.returncode != 0:
        raise LabError(
            f"target probe failed with exit {result.returncode}: {result.stderr.decode(errors='replace').strip()}",
            EXIT_TARGET,
        )
    try:
        lines = result.stdout.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LabError("target probe is not UTF-8", EXIT_TARGET) from exc
    values: dict[str, str] = {}
    for line in lines:
        if line.count("=") != 1:
            raise LabError("target probe line is malformed", EXIT_TARGET)
        key, value = line.split("=", 1)
        if key in values or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", key) or not value:
            raise LabError("target probe key/value is malformed or duplicated", EXIT_TARGET)
        values[key] = value
    required = {
        "protocol",
        "agentVersion",
        "user",
        "kernel",
        "architecture",
        "freeBytes",
        "temperatureMilliCelsius",
    }
    if set(values) != required or values["protocol"] != "1":
        raise LabError("target agent protocol mismatch", EXIT_TARGET)
    temperature = values["temperatureMilliCelsius"]
    if (
        values["user"] != target["user"]
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", values["agentVersion"])
        or not values["freeBytes"].isdigit()
        or (temperature != "unavailable" and not temperature.isdigit())
    ):
        raise LabError("target probe identity/capacity is invalid", EXIT_TARGET)
    return {**values, "targetId": target_id, "targetClass": target["targetClass"]}


def target_preflight(
    inventory_path: Path,
    target_id: str,
    credentials_dir: Path,
    target_class_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    try:
        target_class = validate_config.load_typed(target_class_path, "TargetClass")
    except validate_config.ConfigError as exc:
        raise LabError(str(exc), EXIT_CONFIG) from exc
    if target_class["metadata"]["name"] != target["targetClass"]:
        raise LabError("TargetClass does not match LabInventory target", EXIT_CONFIG)
    probe = target_probe(
        inventory_path, target_id, credentials_dir, runner=runner
    )
    if probe.get("architecture") != "aarch64":
        raise LabError("target architecture is not aarch64", EXIT_TARGET)
    checks: list[dict[str, str]] = []
    spec = target_class["spec"]
    if int(probe["freeBytes"]) < spec["minimumFreeBytes"]:
        raise LabError("target has insufficient free space", EXIT_TARGET)
    temperature = probe["temperatureMilliCelsius"]
    temperature_policy = spec["temperaturePolicy"]
    if temperature == "unavailable":
        if temperature_policy["required"]:
            raise LabError("target temperature sensor is required but unavailable", EXIT_TARGET)
    elif int(temperature) > temperature_policy["maximumMilliCelsius"]:
        raise LabError("target temperature exceeds TargetClass limit", EXIT_TARGET)
    for name in spec["requiredLibraries"]:
        result = target_command(
            target, credentials_dir, ["check-library", name], runner=runner
        )
        if result.returncode != 0:
            raise LabError(f"required target library is absent: {name}", EXIT_TARGET)
        checks.append({"kind": "library", "name": name, "status": "present"})
    for path in spec["requiredDeviceNodes"]:
        result = target_command(
            target, credentials_dir, ["check-device", path], runner=runner
        )
        if result.returncode != 0:
            raise LabError(f"required target device is absent: {path}", EXIT_TARGET)
        checks.append({"kind": "device", "name": path, "status": "present"})
    required_modules = set(spec["requiredKernelModules"])
    forbidden_modules = set(spec["forbiddenKernelModules"])
    for name in sorted(required_modules | forbidden_modules):
        result = target_command(
            target, credentials_dir, ["check-module", name], runner=runner
        )
        if result.returncode not in {0, 1}:
            raise LabError(f"target module inspection failed: {name}", EXIT_TARGET)
        present = result.returncode == 0
        if name in required_modules and not present:
            raise LabError(f"required target module is absent: {name}", EXIT_TARGET)
        if name in forbidden_modules and present:
            raise LabError(f"forbidden target module is loaded: {name}", EXIT_TARGET)
        checks.append(
            {"kind": "module", "name": name, "status": "present" if present else "absent"}
        )
    return {
        "targetId": target_id,
        "executionMode": target["executionMode"],
        "executionRationale": target.get("executionRationale"),
        "probe": probe,
        "checks": checks,
    }


def controller_preflight(
    inventory_path: Path,
    target_id: str,
    credentials_dir: Path,
) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    if os.geteuid() == 0 or pwd.getpwuid(os.geteuid()).pw_name != "actions":
        raise LabError("controller primitives must run as the non-root actions user", EXIT_CONFIG)
    run_root = Path(inventory["spec"]["runner"]["runRoot"])
    try:
        resolved_root = run_root.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"controller run root is unavailable: {run_root}", EXIT_CONFIG) from exc
    root_stat = resolved_root.stat()
    if run_root.is_symlink() or not resolved_root.is_dir():
        raise LabError("controller run root must be a real directory", EXIT_CONFIG)
    if root_stat.st_uid != os.geteuid() or stat.S_IMODE(root_stat.st_mode) & 0o022:
        raise LabError("controller run root ownership/mode is unsafe", EXIT_CONFIG)
    free_bytes = shutil.disk_usage(resolved_root).free
    if free_bytes < 2 * 1024**3:
        raise LabError("controller run root has less than 2 GiB free", EXIT_CONFIG)
    serial = Path(target["serialDevice"])
    try:
        serial_resolved = serial.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"UART device is unavailable: {serial}", EXIT_CONFIG) from exc
    if not stat.S_ISCHR(serial_resolved.stat().st_mode):
        raise LabError("UART path does not resolve to a character device", EXIT_CONFIG)
    missing_commands = [name for name in ("ssh", "python3", "sha256sum") if not shutil.which(name)]
    if missing_commands:
        raise LabError(f"controller commands are missing: {', '.join(missing_commands)}", EXIT_CONFIG)
    credential_file(credentials_dir, target["sshCredentialRef"], private=True)
    credential_file(credentials_dir, target["knownHostsCredentialRef"], private=False)
    return {
        "targetId": target_id,
        "user": "actions",
        "runRoot": str(resolved_root),
        "freeBytes": free_bytes,
        "serialDevice": str(serial),
        "serialResolved": str(serial_resolved),
    }


def payload_entries(payload_root: Path) -> list[dict[str, Any]]:
    try:
        root = payload_root.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"payload root does not exist: {payload_root}", EXIT_TARGET) from exc
    if payload_root.is_symlink() or not root.is_dir():
        raise LabError("payload root must be a real directory", EXIT_TARGET)
    sums_path = root / "PAYLOAD_SHA256SUMS"
    if sums_path.is_symlink() or not sums_path.is_file():
        raise LabError("payload checksums are missing", EXIT_TARGET)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        sums_path.read_text(encoding="ascii").splitlines(), 1
    ):
        match = PAYLOAD_LINE_RE.fullmatch(line)
        if not match:
            raise LabError(f"malformed payload checksum line {line_number}", EXIT_TARGET)
        name = match.group("name")
        validate_config.safe_relative_path(name, "payload path")
        if name in seen or name == "PAYLOAD_SHA256SUMS":
            raise LabError(f"duplicate or recursive payload path: {name}", EXIT_TARGET)
        seen.add(name)
        path = root / name
        size, sha = hash_file(path) if path.is_file() and not path.is_symlink() else (-1, "")
        if sha != match.group("sha"):
            raise LabError(f"payload SHA-256 mismatch: {name}", EXIT_TARGET)
        entries.append({"name": name, "path": path, "size": size, "sha256": sha})
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected = {*seen, "PAYLOAD_SHA256SUMS"}
    if actual != expected or any(path.is_symlink() for path in root.rglob("*")):
        raise LabError("payload file set or symlink policy mismatch", EXIT_TARGET)
    sums_size, sums_sha = hash_file(sums_path)
    entries.append(
        {"name": "PAYLOAD_SHA256SUMS", "path": sums_path, "size": sums_size, "sha256": sums_sha}
    )
    return entries


def target_upload(
    inventory_path: Path,
    target_id: str,
    credentials_dir: Path,
    run_id: str,
    payload_root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise LabError("unsafe run ID", EXIT_TARGET)
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    entries = payload_entries(payload_root)
    prepared = target_command(target, credentials_dir, ["prepare", run_id], runner=runner)
    if prepared.returncode != 0:
        raise LabError("target sandbox preparation failed", EXIT_TARGET)
    uploaded: list[dict[str, Any]] = []
    for entry in entries:
        result = target_command(
            target,
            credentials_dir,
            ["put", run_id, entry["name"], str(entry["size"]), entry["sha256"]],
            input_data=entry["path"].read_bytes(),
            timeout=300,
            runner=runner,
        )
        if result.returncode != 0:
            raise LabError(
                f"target rejected payload file {entry['name']}: {result.stderr.decode(errors='replace').strip()}",
                EXIT_TARGET,
            )
        uploaded.append({key: entry[key] for key in ("name", "size", "sha256")})
    sealed = target_command(target, credentials_dir, ["seal", run_id], runner=runner)
    if sealed.returncode != 0:
        raise LabError("target payload seal failed", EXIT_TARGET)
    return {"targetId": target_id, "runId": run_id, "files": uploaded}


def target_run(
    inventory_path: Path,
    target_id: str,
    credentials_dir: Path,
    run_id: str,
    iterations: int,
    timeout_seconds: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise LabError("unsafe run ID", EXIT_EXECUTION)
    if not 1 <= iterations <= 100 or not 1 <= timeout_seconds <= 3600:
        raise LabError("iteration or timeout is out of range", EXIT_EXECUTION)
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    records: list[dict[str, int]] = []
    for iteration in range(1, iterations + 1):
        result = target_command(
            target,
            credentials_dir,
            ["run", run_id, str(iteration), str(timeout_seconds)],
            timeout=timeout_seconds + 20,
            runner=runner,
        )
        records.append({"iteration": iteration, "exitCode": result.returncode})
        if result.returncode != 0:
            raise LabError(
                f"target iteration {iteration} failed with raw exit {result.returncode}",
                EXIT_EXECUTION,
            )
    return {"targetId": target_id, "runId": run_id, "iterations": records}


def target_download(
    inventory_path: Path,
    target_id: str,
    credentials_dir: Path,
    run_id: str,
    output_root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise LabError("unsafe run ID", EXIT_EVIDENCE)
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    snapshot = target_command(target, credentials_dir, ["snapshot", run_id], runner=runner)
    if snapshot.returncode != 0:
        raise LabError("target snapshot failed", EXIT_EVIDENCE)
    output_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    if output_root.is_symlink():
        raise LabError("output root must not be a symlink", EXIT_EVIDENCE)
    (output_root / "target-snapshot.txt").write_bytes(snapshot.stdout)
    downloaded: list[str] = []
    seen: set[str] = set()
    for line in snapshot.stdout.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or not fields[1].isdigit():
            raise LabError("target snapshot line is malformed", EXIT_EVIDENCE)
        relative = fields[0]
        if relative in seen:
            raise LabError(f"duplicate target snapshot path: {relative}", EXIT_EVIDENCE)
        seen.add(relative)
        if not relative.startswith("iterations/"):
            continue
        safe = validate_config.safe_relative_path(relative, "target output path")
        result = target_command(
            target, credentials_dir, ["get", run_id, relative], timeout=60, runner=runner
        )
        if result.returncode != 0 or len(result.stdout) != int(fields[1]):
            raise LabError(f"target output download mismatch: {relative}", EXIT_EVIDENCE)
        destination = output_root / safe
        destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        with tempfile.SpooledTemporaryFile() as stream:
            stream.write(result.stdout)
            stream.seek(0)
            atomic_write(destination, stream)
        downloaded.append(relative)
    return {"targetId": target_id, "runId": run_id, "files": downloaded}


def target_cleanup(
    inventory_path: Path,
    target_id: str,
    credentials_dir: Path,
    run_id: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise LabError("unsafe run ID", EXIT_CLEANUP)
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    result = target_command(target, credentials_dir, ["cleanup", run_id], runner=runner)
    if result.returncode != 0:
        raise LabError("target cleanup failed", EXIT_CLEANUP)
    return {"targetId": target_id, "runId": run_id, "cleaned": True}


def local_cleanup(run_root: Path, run_id: str, *, dry_run: bool) -> dict[str, Any]:
    root, run_dir = safe_run_directory(run_root, run_id)
    existed = run_dir.exists()
    if existed and not run_dir.is_dir():
        raise LabError("local run path is not a directory", EXIT_CLEANUP)
    if existed and not dry_run:
        shutil.rmtree(run_dir)
    return {
        "runRoot": str(root),
        "runId": run_id,
        "existed": existed,
        "removed": existed and not dry_run,
        "dryRun": dry_run,
    }


def atomic_write_text(path: Path, value: str, mode: int = 0o640) -> None:
    with tempfile.SpooledTemporaryFile() as stream:
        stream.write(value.encode())
        stream.seek(0)
        atomic_write(path, stream, mode)


def write_event_log(payload: dict[str, Any]) -> None:
    configured = os.environ.get("HIL_EVENT_LOG_DIRECTORY")
    if not configured:
        return
    directory = Path(configured)
    try:
        resolved = directory.resolve(strict=True)
    except OSError as exc:
        raise LabError("event log directory is unavailable", EXIT_EVIDENCE) from exc
    metadata = resolved.stat()
    if (
        directory.is_symlink()
        or not resolved.is_dir()
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise LabError("event log directory ownership/mode is unsafe", EXIT_EVIDENCE)
    command = re.sub(r"[^a-z0-9.-]", "-", str(payload["command"]).lower())
    status = payload["status"]
    filename = f"{time.time_ns()}-{os.getpid()}-{command}-{status}.json"
    atomic_write_text(
        resolved / filename,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def uart_start(
    inventory_path: Path,
    target_id: str,
    run_id: str,
    *,
    timeout_seconds: int = 0,
    dry_run: bool,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    if timeout_seconds < 0 or timeout_seconds > 86400:
        raise LabError("UART timeout is out of range", EXIT_EVIDENCE)
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    run_root = run_root_override or Path(inventory["spec"]["runner"]["runRoot"])
    _, run_dir = safe_run_directory(run_root, run_id)
    log_path = run_dir / "uart.log"
    pid_file = run_dir / "uart.pid"
    process_file = run_dir / "uart-process.pid"
    details: dict[str, Any] = {
        "targetId": target_id,
        "runId": run_id,
        "device": target["serialDevice"],
        "log": str(log_path),
        "timeoutSeconds": timeout_seconds,
        "dryRun": dry_run,
    }
    if dry_run:
        return details
    if process_file.exists() or pid_file.exists():
        raise LabError("UART PID state already exists", EXIT_EVIDENCE)
    run_dir.mkdir(mode=0o750, parents=False, exist_ok=True)
    stderr_path = run_dir / "uart-capture.stderr"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "capture_uart.py"),
        "--device", target["serialDevice"],
        "--output", str(log_path),
        "--pid-file", str(pid_file),
        "--timeout", str(timeout_seconds),
    ]
    try:
        process_id = os.fork()
    except OSError as exc:
        raise LabError(f"failed to fork UART capture: {exc}", EXIT_EVIDENCE) from exc
    if process_id == 0:
        try:
            os.setsid()
            null_fd = os.open(os.devnull, os.O_RDWR)
            stderr_fd = os.open(
                stderr_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640
            )
            os.dup2(null_fd, 0)
            os.dup2(null_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(null_fd)
            os.close(stderr_fd)
            os.execv(sys.executable, command)
        except OSError as exc:
            os.write(2, f"UART exec failed: {exc}\n".encode())
        os._exit(127)
    deadline = time.monotonic() + 2
    exited = False
    while time.monotonic() < deadline and not pid_file.exists():
        exited = os.waitpid(process_id, os.WNOHANG)[0] == process_id
        if exited:
            break
        time.sleep(0.05)
    time.sleep(0.1)
    if exited or not pid_file.exists():
        try:
            os.waitpid(process_id, os.WNOHANG)
        except ChildProcessError:
            pass
        diagnostic = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        raise LabError(
            f"UART capture failed during startup: {diagnostic or 'no diagnostic'}",
            EXIT_EVIDENCE,
        )
    atomic_write_text(process_file, f"{process_id}\n")
    details["processId"] = process_id
    return details


def read_pid(path: Path) -> int:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise LabError(f"cannot read PID file {path}: {exc}", EXIT_EVIDENCE) from exc
    if not re.fullmatch(r"[1-9][0-9]{0,9}", value):
        raise LabError(f"invalid PID file: {path}", EXIT_EVIDENCE)
    return int(value)


def uart_stop(
    inventory_path: Path,
    target_id: str,
    run_id: str,
    *,
    dry_run: bool,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    select_target(inventory, target_id)
    run_root = run_root_override or Path(inventory["spec"]["runner"]["runRoot"])
    _, run_dir = safe_run_directory(run_root, run_id)
    process_file = run_dir / "uart-process.pid"
    pid_file = run_dir / "uart.pid"
    if not process_file.exists():
        return {"targetId": target_id, "runId": run_id, "state": "not-running", "dryRun": dry_run}
    pid = read_pid(process_file)
    if dry_run:
        return {"targetId": target_id, "runId": run_id, "state": "would-stop", "processId": pid, "dryRun": True}
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        command = ""
    capture_path = str(Path(__file__).resolve().parent / "capture_uart.py")
    if capture_path not in command or str(pid_file) not in command:
        raise LabError("refusing to signal unrelated PID", EXIT_EVIDENCE)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 5
    reaped = False
    while time.monotonic() < deadline and Path(f"/proc/{pid}").exists():
        try:
            reaped = os.waitpid(pid, os.WNOHANG)[0] == pid
        except ChildProcessError:
            pass
        if reaped:
            break
        time.sleep(0.05)
    if not reaped and Path(f"/proc/{pid}").exists():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    process_file.unlink(missing_ok=True)
    return {"targetId": target_id, "runId": run_id, "state": "stopped", "processId": pid, "dryRun": False}


def copy_evidence_file(source: Path, destination: Path) -> None:
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise LabError(f"missing evidence input: {source}", EXIT_EVIDENCE) from exc
    if source.is_symlink() or not resolved.is_file():
        raise LabError(f"unsafe evidence input: {source}", EXIT_EVIDENCE)
    destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    shutil.copyfile(resolved, destination)
    destination.chmod(0o640)


def copy_optional_evidence_file(source: Path, destination: Path, label: str) -> bool:
    if source.is_file() and not source.is_symlink():
        copy_evidence_file(source, destination)
        return True
    unavailable = destination.with_name(f"{destination.stem}-unavailable.txt")
    unavailable.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    unavailable.write_text(
        f"{label} was unavailable at evidence collection time\n", encoding="utf-8"
    )
    unavailable.chmod(0o640)
    return False


def redact_credentials(root: Path, credentials_dir: Path) -> None:
    secrets: list[bytes] = []
    try:
        credential_root = credentials_dir.resolve(strict=True)
    except OSError:
        return
    if credentials_dir.is_symlink() or not credential_root.is_dir():
        return
    for path in credential_root.iterdir():
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024:
            value = path.read_bytes().strip()
            if len(value) >= 8:
                secrets.append(value)
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 10 * 1024 * 1024:
            continue
        data = path.read_bytes()
        redacted = data
        for secret in secrets:
            redacted = redacted.replace(secret, b"[REDACTED]")
        if redacted != data:
            path.write_bytes(redacted)
            path.chmod(0o640)


def evidence_snapshot(
    inventory_path: Path,
    target_id: str,
    credentials_dir: Path,
    run_id: str,
    context_path: Path,
    asset_manifest_path: Path,
    build_manifest_path: Path,
    *,
    run_root_override: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    target = select_target(inventory, target_id)
    run_root = run_root_override or Path(inventory["spec"]["runner"]["runRoot"])
    _, run_dir = safe_run_directory(run_root, run_id)
    run_dir.mkdir(mode=0o750, parents=False, exist_ok=True)
    evidence_root = run_dir / "evidence"
    if evidence_root.exists() or evidence_root.is_symlink():
        raise LabError("evidence directory already exists", EXIT_EVIDENCE)
    bundle = evidence_root / "bundle"
    bundle.mkdir(mode=0o750, parents=True)
    for source, name, label in (
        (inventory_path, "lab-inventory.yaml", "LabInventory"),
        (context_path, "execution-context.json", "execution context"),
        (asset_manifest_path, "asset-manifest.yaml", "asset manifest"),
    ):
        copy_optional_evidence_file(source, bundle / "manifests" / name, label)
    copy_optional_evidence_file(
        build_manifest_path,
        bundle / "manifests" / "build-manifest.json",
        "build manifest",
    )

    host_sections: list[bytes] = []
    for command in (
        ["uname", "-a"],
        ["df", "-h", str(run_root)],
        ["ip", "-brief", "address"],
        ["ip", "route"],
    ):
        try:
            result = runner(
                command,
                input=None,
                capture_output=True,
                timeout=15,
                check=False,
            )
            host_sections.extend(
                [f"$ {' '.join(command)}\n".encode(), result.stdout, result.stderr]
            )
        except (OSError, subprocess.SubprocessError) as exc:
            host_sections.append(f"$ {' '.join(command)}\nerror={exc}\n".encode())
    (bundle / "host.txt").write_bytes(b"".join(host_sections))
    (bundle / "host.txt").chmod(0o640)

    target_status = "unavailable"
    target_note = ""
    try:
        probe = target_probe(
            inventory_path, target_id, credentials_dir, runner=runner
        )
        (bundle / "target-probe.json").write_text(
            json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (bundle / "target-probe.json").chmod(0o640)
    except LabError as exc:
        (bundle / "target-probe-unavailable.txt").write_text(
            f"target probe unavailable: {exc}\n", encoding="utf-8"
        )
        (bundle / "target-probe-unavailable.txt").chmod(0o640)
    try:
        snapshot = target_command(
            target, credentials_dir, ["snapshot", run_id], runner=runner
        )
        if snapshot.returncode == 0:
            (bundle / "target-snapshot.txt").write_bytes(snapshot.stdout)
            (bundle / "target-snapshot.txt").chmod(0o640)
            target_status = "collected"
        else:
            target_note = f"target snapshot exit={snapshot.returncode}"
    except LabError as exc:
        target_note = f"target snapshot unavailable: {exc}"
    if target_status == "unavailable":
        (bundle / "target-unavailable.txt").write_text(
            target_note + "\n", encoding="utf-8"
        )
        (bundle / "target-unavailable.txt").chmod(0o640)

    for optional in ("uart.log", "uart-capture.stderr"):
        source = run_dir / optional
        if source.is_file() and not source.is_symlink():
            copy_evidence_file(source, bundle / optional)
    target_output = run_dir / "target"
    if target_output.is_dir() and not target_output.is_symlink():
        for source in target_output.rglob("*"):
            if source.is_file() and not source.is_symlink():
                relative = source.relative_to(target_output)
                copy_evidence_file(source, bundle / "target" / relative)
    event_output = run_dir / "events"
    if event_output.is_dir() and not event_output.is_symlink():
        for source in event_output.iterdir():
            if source.is_file() and not source.is_symlink():
                copy_evidence_file(source, bundle / "events" / source.name)

    redact_credentials(bundle, credentials_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and not path.is_symlink():
            size, sha = hash_file(path)
            records.append(
                {"name": str(path.relative_to(bundle)), "size": size, "sha256": sha}
            )
    manifest = {
        "schemaVersion": 1,
        "runId": run_id,
        "targetId": target_id,
        "collectedAt": timestamp(),
        "targetSnapshotStatus": target_status,
        "files": records,
    }
    schema_path = Path(__file__).resolve().parent / "schemas" / "evidence-manifest.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot load evidence manifest schema: {exc}", EXIT_EVIDENCE) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise LabError(
            f"generated evidence manifest is invalid at {location}: {error.message}",
            EXIT_EVIDENCE,
        )
    (evidence_root / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"evidenceRoot": str(evidence_root), **manifest}


def evidence_finalize(
    run_root: Path,
    run_id: str,
    credentials_dir: Path,
) -> dict[str, Any]:
    _, run_dir = safe_run_directory(run_root, run_id)
    evidence_root = run_dir / "evidence"
    bundle = evidence_root / "bundle"
    manifest_path = evidence_root / "evidence-manifest.json"
    if (
        evidence_root.is_symlink()
        or bundle.is_symlink()
        or not bundle.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise LabError("evidence snapshot is unavailable or unsafe", EXIT_EVIDENCE)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabError("evidence manifest cannot be finalized", EXIT_EVIDENCE) from exc
    if manifest.get("runId") != run_id:
        raise LabError("evidence manifest run identity mismatch", EXIT_EVIDENCE)
    event_output = run_dir / "events"
    if event_output.is_dir() and not event_output.is_symlink():
        for source in event_output.iterdir():
            if source.is_file() and not source.is_symlink():
                copy_evidence_file(source, bundle / "events" / source.name)
    redact_credentials(bundle, credentials_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and not path.is_symlink():
            size, sha = hash_file(path)
            records.append(
                {"name": str(path.relative_to(bundle)), "size": size, "sha256": sha}
            )
    manifest["files"] = records
    schema_path = Path(__file__).resolve().parent / "schemas" / "evidence-manifest.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot load evidence manifest schema: {exc}", EXIT_EVIDENCE) from exc
    errors = list(Draft202012Validator(schema).iter_errors(manifest))
    if errors:
        raise LabError(f"finalized evidence manifest is invalid: {errors[0].message}", EXIT_EVIDENCE)
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return {"evidenceRoot": str(evidence_root), **manifest}


def add_inventory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--credentials-dir",
        type=Path,
        default=Path(
            os.environ.get("HIL_CREDENTIALS_DIRECTORY")
            or os.environ.get("CREDENTIALS_DIRECTORY")
            or "/run/hil/credentials"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab-control")
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)

    asset = commands.add_parser("asset")
    asset_commands = asset.add_subparsers(dest="asset_command", required=True)
    asset_verify = asset_commands.add_parser("verify")
    asset_verify.add_argument("--manifest", required=True, type=Path)
    asset_verify.add_argument("--asset-root", required=True, type=Path)
    asset_fetch = asset_commands.add_parser("fetch")
    asset_fetch.add_argument("--inventory", required=True, type=Path)
    asset_fetch.add_argument("--manifest", required=True, type=Path)
    asset_fetch.add_argument("--asset-root", required=True, type=Path)
    asset_fetch.add_argument(
        "--credentials-dir",
        type=Path,
        default=Path(
            os.environ.get("HIL_CREDENTIALS_DIRECTORY")
            or os.environ.get("CREDENTIALS_DIRECTORY")
            or "/run/hil/credentials"
        ),
    )

    artifact = commands.add_parser("artifact")
    artifact_commands = artifact.add_subparsers(dest="artifact_command", required=True)
    artifact_verify = artifact_commands.add_parser("verify")
    artifact_verify.add_argument("--artifact-root", required=True, type=Path)
    artifact_verify.add_argument("--source-sha", required=True)
    artifact_verify.add_argument("--workflow-sha", required=True)
    artifact_verify.add_argument("--sdk-lock", required=True, type=Path)
    artifact_verify.add_argument("--sample", required=True)
    artifact_verify.add_argument("--engine", required=True, choices=["svp-nnn"])
    artifact_verify.add_argument("--soc", required=True, choices=["SS928V100"])

    target = commands.add_parser("target")
    target_commands = target.add_subparsers(dest="target_command", required=True)
    probe = target_commands.add_parser("probe")
    add_inventory_arguments(probe)
    preflight = target_commands.add_parser("preflight")
    add_inventory_arguments(preflight)
    preflight.add_argument("--target-class", required=True, type=Path)
    upload = target_commands.add_parser("upload")
    add_inventory_arguments(upload)
    upload.add_argument("--run-id", required=True)
    upload.add_argument("--payload-root", required=True, type=Path)
    run = target_commands.add_parser("run")
    add_inventory_arguments(run)
    run.add_argument("--run-id", required=True)
    run.add_argument("--iterations", required=True, type=int)
    run.add_argument("--timeout", required=True, type=int)
    download = target_commands.add_parser("download")
    add_inventory_arguments(download)
    download.add_argument("--run-id", required=True)
    download.add_argument("--output-root", required=True, type=Path)
    cleanup = target_commands.add_parser("cleanup")
    add_inventory_arguments(cleanup)
    cleanup.add_argument("--run-id", required=True)

    controller = commands.add_parser("controller")
    controller_commands = controller.add_subparsers(
        dest="controller_command", required=True
    )
    controller_preflight_parser = controller_commands.add_parser("preflight")
    add_inventory_arguments(controller_preflight_parser)

    local = commands.add_parser("local")
    local_commands = local.add_subparsers(dest="local_command", required=True)
    local_cleanup_parser = local_commands.add_parser("cleanup")
    local_cleanup_parser.add_argument("--run-root", required=True, type=Path)
    local_cleanup_parser.add_argument("--run-id", required=True)
    local_cleanup_parser.add_argument("--dry-run", action="store_true")

    uart = commands.add_parser("uart")
    uart_commands = uart.add_subparsers(dest="uart_command", required=True)
    uart_start_parser = uart_commands.add_parser("start")
    uart_start_parser.add_argument("--inventory", required=True, type=Path)
    uart_start_parser.add_argument("--target", required=True)
    uart_start_parser.add_argument("--run-id", required=True)
    uart_start_parser.add_argument("--timeout", type=int, default=0)
    uart_start_parser.add_argument("--dry-run", action="store_true")
    uart_stop_parser = uart_commands.add_parser("stop")
    uart_stop_parser.add_argument("--inventory", required=True, type=Path)
    uart_stop_parser.add_argument("--target", required=True)
    uart_stop_parser.add_argument("--run-id", required=True)
    uart_stop_parser.add_argument("--dry-run", action="store_true")

    evidence = commands.add_parser("evidence")
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    evidence_snapshot_parser = evidence_commands.add_parser("snapshot")
    add_inventory_arguments(evidence_snapshot_parser)
    evidence_snapshot_parser.add_argument("--run-id", required=True)
    evidence_snapshot_parser.add_argument("--context", required=True, type=Path)
    evidence_snapshot_parser.add_argument("--asset-manifest", required=True, type=Path)
    evidence_snapshot_parser.add_argument("--build-manifest", required=True, type=Path)
    evidence_finalize_parser = evidence_commands.add_parser("finalize")
    evidence_finalize_parser.add_argument("--run-root", required=True, type=Path)
    evidence_finalize_parser.add_argument("--run-id", required=True)
    evidence_finalize_parser.add_argument("--credentials-dir", required=True, type=Path)
    return parser


def run_cli(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.command == "asset" and args.asset_command == "verify":
        return "asset.verify", verify_assets(args.manifest, args.asset_root)
    if args.command == "asset" and args.asset_command == "fetch":
        return "asset.fetch", fetch_assets(
            args.inventory,
            args.manifest,
            args.asset_root,
            args.credentials_dir,
        )
    if args.command == "artifact" and args.artifact_command == "verify":
        return "artifact.verify", verify_artifact(
            args.artifact_root,
            args.source_sha,
            args.workflow_sha,
            args.sdk_lock,
            sample=args.sample,
            engine=args.engine,
            soc=args.soc,
        )
    if args.command == "target" and args.target_command == "probe":
        return "target.probe", target_probe(
            args.inventory, args.target, args.credentials_dir
        )
    if args.command == "target" and args.target_command == "preflight":
        return "target.preflight", target_preflight(
            args.inventory,
            args.target,
            args.credentials_dir,
            args.target_class,
        )
    if args.command == "target" and args.target_command == "upload":
        return "target.upload", target_upload(
            args.inventory,
            args.target,
            args.credentials_dir,
            args.run_id,
            args.payload_root,
        )
    if args.command == "target" and args.target_command == "run":
        return "target.run", target_run(
            args.inventory,
            args.target,
            args.credentials_dir,
            args.run_id,
            args.iterations,
            args.timeout,
        )
    if args.command == "target" and args.target_command == "download":
        return "target.download", target_download(
            args.inventory,
            args.target,
            args.credentials_dir,
            args.run_id,
            args.output_root,
        )
    if args.command == "target" and args.target_command == "cleanup":
        return "target.cleanup", target_cleanup(
            args.inventory, args.target, args.credentials_dir, args.run_id
        )
    if args.command == "controller" and args.controller_command == "preflight":
        return "controller.preflight", controller_preflight(
            args.inventory, args.target, args.credentials_dir
        )
    if args.command == "local" and args.local_command == "cleanup":
        return "local.cleanup", local_cleanup(
            args.run_root, args.run_id, dry_run=args.dry_run
        )
    if args.command == "uart" and args.uart_command == "start":
        return "uart.start", uart_start(
            args.inventory,
            args.target,
            args.run_id,
            timeout_seconds=args.timeout,
            dry_run=args.dry_run,
        )
    if args.command == "uart" and args.uart_command == "stop":
        return "uart.stop", uart_stop(
            args.inventory, args.target, args.run_id, dry_run=args.dry_run
        )
    if args.command == "evidence" and args.evidence_command == "snapshot":
        return "evidence.snapshot", evidence_snapshot(
            args.inventory,
            args.target,
            args.credentials_dir,
            args.run_id,
            args.context,
            args.asset_manifest,
            args.build_manifest,
        )
    if args.command == "evidence" and args.evidence_command == "finalize":
        return "evidence.finalize", evidence_finalize(
            args.run_root, args.run_id, args.credentials_dir
        )
    raise LabError("unsupported command", EXIT_CONFIG)


def main() -> int:
    args = build_parser().parse_args()
    command = args.command
    try:
        command, details = run_cli(args)
    except (LabError, validate_config.ConfigError) as exc:
        exit_code = exc.exit_code if isinstance(exc, LabError) else EXIT_CONFIG
        payload = event(command, "failed", {"error": str(exc)})
        try:
            write_event_log(payload)
        except LabError as log_error:
            print(f"event logging failed: {log_error}", file=sys.stderr)
            exit_code = EXIT_EVIDENCE
        print(json.dumps(payload, sort_keys=True))
        return exit_code
    payload = event(command, "passed", details)
    try:
        write_event_log(payload)
    except LabError as exc:
        print(json.dumps(event(command, "failed", {"error": str(exc)}), sort_keys=True))
        return EXIT_EVIDENCE
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
