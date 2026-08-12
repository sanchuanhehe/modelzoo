#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ci/provision_toolchain.sh \
  --engine <svp-nnn|nnn> \
  --cann-url <url> --cann-sha256 <sha256> \
  --toolchain-url <url> --toolchain-sha256 <sha256>

Downloads and verifies the original CANN/toolchain packages (or already
installed relocatable archives), then writes paths to GITHUB_OUTPUT.
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
COMPAT_ROOT="$ENGINE_ROOT/host-compat"
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

compiler=$(find -L "$TOOLCHAIN_ROOT" -type f -name aarch64-mix210-linux-gcc -print -quit)
if [[ -z "$compiler" ]]; then
    nested_toolchain=$(find "$TOOLCHAIN_ROOT" -type f -name 'aarch64-mix210-linux.tar.bz2' -print -quit)
    if [[ -n "$nested_toolchain" ]]; then
        tar -xjf "$nested_toolchain" -C "$TOOLCHAIN_ROOT"
        compiler=$(find -L "$TOOLCHAIN_ROOT" -type f -name aarch64-mix210-linux-gcc -print -quit)
    fi
fi
if [[ -z "$compiler" ]]; then
    printf 'Toolchain archive does not contain aarch64-mix210-linux-gcc\n' >&2
    exit 1
fi
compiler_bin=$(dirname "$compiler")

mapfile -t host_executables < <(
    find -L "$TOOLCHAIN_ROOT" -type f \
        \( -path '*/libexec/gcc/*/cc1' -o -path '*/libexec/gcc/*/cc1plus' \) -print
)
host_executables+=("$compiler")

unresolved_host_libraries() {
    local executable
    for executable in "${host_executables[@]}"; do
        LD_LIBRARY_PATH="${compat_lib_path:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            ldd "$executable" 2>/dev/null | awk '/not found/ {print $1}'
    done | sort -u
}

missing_libraries=$(unresolved_host_libraries)
if grep -qx 'libisl.so.19' <<< "$missing_libraries"; then
    libisl_url=https://archive.ubuntu.com/ubuntu/pool/main/i/isl/libisl19_0.19-1_amd64.deb
    libisl_sha256=07a0827aba14140b1833ca19ced3f939b2d075646094926d43678a0d19cc942f
    libisl_package="$DOWNLOAD_ROOT/libisl19_${libisl_sha256}.deb"
    download_and_verify "$libisl_url" "$libisl_sha256" "$libisl_package"
    rm -rf "$COMPAT_ROOT"
    mkdir -p "$COMPAT_ROOT"
    dpkg-deb --extract "$libisl_package" "$COMPAT_ROOT"
    compat_lib_path="$COMPAT_ROOT/usr/lib/x86_64-linux-gnu"
    if [[ -n ${GITHUB_ENV:-} ]]; then
        printf 'LD_LIBRARY_PATH=%s\n' "$compat_lib_path" >> "$GITHUB_ENV"
    else
        export LD_LIBRARY_PATH="$compat_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

remaining_libraries=$(unresolved_host_libraries)
if [[ -n "$remaining_libraries" ]]; then
    printf 'Toolchain compiler has unresolved host libraries:\n%s\n' "$remaining_libraries" >&2
    exit 1
fi

find_cann_paths() {
    local header library preferred_library
    if [[ "$ENGINE" == svp-nnn ]]; then
        header=svp_acl.h
        library=libsvp_acl.so
        preferred_library='*/acllib/lib64_aarch64-mix210-linux/stub/libsvp_acl.so'
    else
        header=acl.h
        library=libascendcl.so
        preferred_library='*/runtime/lib64/stub/libascendcl.so'
    fi

    include_file=$(find -L "$CANN_ROOT" -type f -name "$header" -print -quit)
    library_file=$(find -L "$CANN_ROOT" -type f -path "$preferred_library" -print -quit)
    if [[ -z "$library_file" ]]; then
        library_file=$(find -L "$CANN_ROOT" -type f -name "$library" -print -quit)
    fi
    include_path=${include_file%/*}
    lib_path=${library_file%/*}
}

find_cann_paths
if [[ -z "$include_path" || -z "$lib_path" ]]; then
    installer=$(find "$CANN_ROOT" -type f -name 'Ascend-cann-toolkit*.run' -print -quit)
    if [[ -z "$installer" ]]; then
        printf 'CANN archive has neither an installed development tree nor an Ascend toolkit installer\n' >&2
        exit 1
    fi

    chmod +x "$installer"
    payload_root="$CANN_ROOT/installer-payload"
    "$installer" --noexec --extract="$payload_root" >/dev/null

    if [[ "$ENGINE" == svp-nnn ]]; then
        target_installer=$(find "$payload_root" -type f -name 'Ascend-acllib-*.run' -print -quit)
    else
        target_installer=$(find "$payload_root" -type f -name 'CANN-runtime-*-lmixlinux*.run' -print -quit)
    fi
    if [[ -z "$target_installer" ]]; then
        printf 'CANN toolkit does not contain the target development package for %s\n' "$ENGINE" >&2
        exit 1
    fi

    chmod +x "$target_installer"
    "$target_installer" --quiet --devel --install-path="$CANN_ROOT/installed"
    find_cann_paths
fi

if [[ -z "$include_path" || -z "$lib_path" ]]; then
    printf 'CANN package is missing the %s target headers or stub library\n' "$ENGINE" >&2
    exit 1
fi

LD_LIBRARY_PATH="${compat_lib_path:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$compiler" --version | head -n 1
printf 'CANN include path: %s\n' "$include_path"
printf 'CANN library path: %s\n' "$lib_path"

printf '%s\n' "$compiler_bin" >> "$GITHUB_PATH"
{
    printf 'npu-include-path=%s\n' "$include_path"
    printf 'npu-lib-path=%s\n' "$lib_path"
} >> "$GITHUB_OUTPUT"
