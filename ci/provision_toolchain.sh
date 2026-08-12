#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ci/provision_toolchain.sh \
  --engine <svp-nnn|nnn> \
  --cann-url <url> --cann-sha256 <sha256> \
  --toolchain-url <url> --toolchain-sha256 <sha256>

Downloads and verifies the CANN package and a relocatable
aarch64-mix210-linux toolchain archive, then writes paths to GITHUB_OUTPUT.
EOF
}

ENGINE=""
CANN_URL=""
CANN_SHA256=""
TOOLCHAIN_URL=""
TOOLCHAIN_SHA256=""

while (($#)); do
    case "$1" in
        --engine) ENGINE=${2:?}; shift 2 ;;
        --cann-url) CANN_URL=${2:?}; shift 2 ;;
        --cann-sha256) CANN_SHA256=${2:?}; shift 2 ;;
        --toolchain-url) TOOLCHAIN_URL=${2:?}; shift 2 ;;
        --toolchain-sha256) TOOLCHAIN_SHA256=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$ENGINE" in
    svp-nnn|nnn) ;;
    *) printf 'Unsupported engine: %s\n' "$ENGINE" >&2; exit 2 ;;
esac

for value in CANN_URL CANN_SHA256 TOOLCHAIN_URL TOOLCHAIN_SHA256; do
    if [[ -z ${!value} ]]; then
        printf '%s is required\n' "$value" >&2
        exit 2
    fi
done

DOWNLOAD_ROOT=${MODELZOO_DOWNLOAD_CACHE:-"$HOME/.cache/modelzoo/downloads"}
WORK_ROOT=${MODELZOO_TOOLCHAIN_WORK:-"$RUNNER_TEMP/modelzoo-toolchains"}
ENGINE_ROOT="$WORK_ROOT/$ENGINE"
CANN_ARCHIVE="$DOWNLOAD_ROOT/cann-${CANN_SHA256}.tgz"
TOOLCHAIN_ARCHIVE="$DOWNLOAD_ROOT/toolchain-${TOOLCHAIN_SHA256}.tar"
CANN_ROOT="$ENGINE_ROOT/cann"
TOOLCHAIN_ROOT="$ENGINE_ROOT/toolchain"
mkdir -p "$DOWNLOAD_ROOT" "$ENGINE_ROOT"

download_and_verify() {
    local url=$1
    local sha256=$2
    local destination=$3
    if [[ ! -f "$destination" ]] || ! printf '%s  %s\n' "$sha256" "$destination" | sha256sum -c - >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "$destination.part" "$url"
        printf '%s  %s\n' "$sha256" "$destination.part" | sha256sum -c -
        mv "$destination.part" "$destination"
    fi
}

download_and_verify "$CANN_URL" "$CANN_SHA256" "$CANN_ARCHIVE"
download_and_verify "$TOOLCHAIN_URL" "$TOOLCHAIN_SHA256" "$TOOLCHAIN_ARCHIVE"

rm -rf "$CANN_ROOT" "$TOOLCHAIN_ROOT"
mkdir -p "$CANN_ROOT" "$TOOLCHAIN_ROOT"
tar -xf "$CANN_ARCHIVE" -C "$CANN_ROOT"
tar -xf "$TOOLCHAIN_ARCHIVE" -C "$TOOLCHAIN_ROOT"

compiler=$(find "$TOOLCHAIN_ROOT" -type f -name aarch64-mix210-linux-gcc -print -quit)
if [[ -z "$compiler" ]]; then
    printf 'Toolchain archive does not contain aarch64-mix210-linux-gcc\n' >&2
    exit 1
fi
compiler_bin=$(dirname "$compiler")
include_path=$(find "$CANN_ROOT" -type d -path '*/acllib/include/acl' -print -quit)
lib_path=$(find "$CANN_ROOT" -type d -path '*/acllib/lib64/stub' -print -quit)

if [[ -z "$include_path" || -z "$lib_path" ]]; then
    printf 'CANN archive is missing an installed acllib/include/acl or acllib/lib64/stub tree\n' >&2
    exit 1
fi

printf '%s\n' "$compiler_bin" >> "$GITHUB_PATH"
{
    printf 'npu-include-path=%s\n' "$include_path"
    printf 'npu-lib-path=%s\n' "$lib_path"
} >> "$GITHUB_OUTPUT"
