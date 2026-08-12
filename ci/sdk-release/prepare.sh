#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: prepare.sh --output DIR --toolchain FILE --nnn FILE --svp-nnn FILE

Creates byte-for-byte release assets and metadata. The SVP_NNN archive is split
below GitHub's 2 GiB per-asset limit; source archives are never recompressed.
EOF
}

OUTPUT=""
TOOLCHAIN=""
NNN=""
SVP_NNN=""
while (($#)); do
    case "$1" in
        --output) OUTPUT=${2:?}; shift 2 ;;
        --toolchain) TOOLCHAIN=${2:?}; shift 2 ;;
        --nnn) NNN=${2:?}; shift 2 ;;
        --svp-nnn) SVP_NNN=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
for value in OUTPUT TOOLCHAIN NNN SVP_NNN; do
    [[ -n ${!value} ]] || { printf '%s is required\n' "$value" >&2; exit 2; }
done
for source in "$TOOLCHAIN" "$NNN" "$SVP_NNN"; do
    [[ -f "$source" ]] || { printf 'Missing source: %s\n' "$source" >&2; exit 1; }
done

VERSION=ss928v100-r001c02spc022-ci.1
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
cp "$TOOLCHAIN" "$OUTPUT/${VERSION}-aarch64-mix210-linux.tgz"
cp "$NNN" "$OUTPUT/${VERSION}-NNN_PC.tgz"
split -b 1900000000 -d -a 3 \
    "$SVP_NNN" "$OUTPUT/${VERSION}-SVP_NNN_PC_V1.0.2.17.tgz.part-"

python3 - "$OUTPUT" "$VERSION" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
version = sys.argv[2]

def record(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"name": path.name, "size": path.stat().st_size, "sha256": h.hexdigest()}

toolchain = record(root / f"{version}-aarch64-mix210-linux.tgz")
nnn = record(root / f"{version}-NNN_PC.tgz")
parts = [record(path) for path in sorted(root.glob(f"{version}-SVP_NNN_PC_V1.0.2.17.tgz.part-*"))]
manifest = {
    "schemaVersion": 1,
    "releaseTag": "sdk-ss928v100-r001c02spc022-ci.1",
    "artifacts": {
        "toolchain": {"archive": toolchain},
        "nnn": {"archive": nnn},
        "svp-nnn": {
            "archive": {
                "name": "SVP_NNN_PC_V1.0.2.17.tgz",
                "size": 2398084680,
                "sha256": "3d492cea67c46c2a35bc3136f9a1c3f88c615e8662b7292905c3682ceb54de2e",
            },
            "parts": parts,
        },
    },
}
expected = {
    "toolchain": (260213042, "935a86894cba0d6434915e0bae832e11e124dcee1ad848ee289fccf996ba8332"),
    "nnn": (843961029, "a6a1063388887594e17c9ba3acc6dfd4905251b3d1cc9794ea876d127f7fcceb"),
}
for key, (size, digest) in expected.items():
    actual = manifest["artifacts"][key]["archive"]
    if (actual["size"], actual["sha256"]) != (size, digest):
        raise SystemExit(f"{key} source does not match the approved size/SHA-256")
(root / "sdk-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

(
    cd "$OUTPUT"
    shasum -a 256 \
        "${VERSION}-aarch64-mix210-linux.tgz" \
        "${VERSION}-NNN_PC.tgz" \
        "${VERSION}-SVP_NNN_PC_V1.0.2.17.tgz.part-"* \
        sdk-manifest.json > SHA256SUMS
)
cat > "$OUTPUT/PROVENANCE.md" <<'EOF'
# SDK CI asset provenance

- Release family: SS928 V100R001C02SPC022
- Packaging: original bytes; no source archive was recompressed or modified.
- `aarch64-mix210-linux.tgz`: vendor SDK package supplied by the repository owner.
- `NNN_PC.tgz`: extracted byte-for-byte from the vendor `SVP_PC` multi-volume archive.
- `SVP_NNN_PC_V1.0.2.17.tgz`: extracted byte-for-byte from the same vendor archive and split only to satisfy GitHub's per-asset size limit.
- Integrity: `sdk-manifest.json` records sizes, SHA-256 digests, and ordered parts. `SHA256SUMS` covers every uploaded machine-consumed asset.
EOF
cat > "$OUTPUT/REDISTRIBUTION-NOTICE.md" <<'EOF'
# Redistribution notice

These binary SDK and toolchain assets are published by the repository owner solely as pinned CI inputs for building this ModelZoo fork. They are not covered by the repository's Apache-2.0 source license. Vendor notices and restrictions, if any, continue to apply. No warranty or additional rights are granted by this packaging.
EOF

"$(dirname "$0")/verify.sh" --directory "$OUTPUT" --manifest "$OUTPUT/sdk-manifest.json"
printf 'Prepared release assets in %s\n' "$OUTPUT"
