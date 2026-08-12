#!/usr/bin/env bash
set -euo pipefail

DIRECTORY=""
MANIFEST=""
OUTPUT=""
while (($#)); do
    case "$1" in
        --directory) DIRECTORY=${2:?}; shift 2 ;;
        --manifest) MANIFEST=${2:?}; shift 2 ;;
        --output) OUTPUT=${2:?}; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -d "$DIRECTORY" && -f "$MANIFEST" ]] || { printf 'Directory and manifest are required\n' >&2; exit 2; }
OUTPUT=${OUTPUT:-"$DIRECTORY/reconstructed"}
mkdir -p "$OUTPUT"

python3 - "$DIRECTORY" "$MANIFEST" "$OUTPUT" <<'PY'
import hashlib
import json
import pathlib
import sys

root, manifest_path, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def check(path, entry):
    if not path.is_file():
        raise SystemExit(f"missing release asset: {path.name}")
    if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
        raise SystemExit(f"size/SHA-256 mismatch: {path.name}")

for key in ("toolchain", "nnn"):
    entry = manifest["artifacts"][key]["archive"]
    check(root / entry["name"], entry)
svp = manifest["artifacts"]["svp-nnn"]
for part in svp["parts"]:
    check(root / part["name"], part)
target = output / svp["archive"]["name"]
with target.open("wb") as merged:
    for part in svp["parts"]:
        with (root / part["name"]).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                merged.write(block)
check(target, svp["archive"])
print(f"Verified {len(svp['parts']) + 2} assets; reconstructed {target}")
PY
