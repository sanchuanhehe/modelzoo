#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ci/provision_toolchain.sh --engine <svp-nnn|nnn> [--lock ci/sdk-lock.json]

Downloads only SHA-locked GitHub Release assets, reconstructs split archives,
and installs the target development package into the runner temporary directory.
EOF
}

ENGINE=""
LOCK=ci/sdk-lock.json
while (($#)); do
    case "$1" in
        --engine) ENGINE=${2:?}; shift 2 ;;
        --lock) LOCK=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$ENGINE" in svp-nnn|nnn) ;; *) printf 'Unsupported engine: %s\n' "$ENGINE" >&2; exit 2 ;; esac
python3 ci/validate_sdk_lock.py "$LOCK"

eval "$(python3 - "$LOCK" "$ENGINE" <<'PY'
import json, shlex, sys
d=json.load(open(sys.argv[1], encoding='utf-8')); engine=sys.argv[2]
def emit(k,v): print(f'{k}={shlex.quote(str(v))}')
emit('REPOSITORY', d['repository']); emit('RELEASE_TAG', d['releaseTag'])
for prefix, entry in [('TOOLCHAIN', d['artifacts']['toolchain']['archive']), ('CANN_SOURCE', d['artifacts'][engine]['archive'])]:
    for key in ('name','size','sha256'): emit(f'{prefix}_{key.upper()}', entry[key])
parts=d['artifacts'][engine].get('parts', [d['artifacts'][engine]['archive']])
emit('CANN_PARTS', '\n'.join(f"{p['name']}|{p['size']}|{p['sha256']}" for p in parts))
c=d['hostCompatibility']['libisl19']
for key in ('url','size','sha256'): emit(f'LIBISL_{key.upper()}', c[key])
PY
)"

DOWNLOAD_ROOT=${MODELZOO_DOWNLOAD_CACHE:-"$HOME/.cache/modelzoo/downloads"}
WORK_ROOT=${MODELZOO_TOOLCHAIN_WORK:-"${RUNNER_TEMP:-/tmp}/modelzoo-toolchains"}
ENGINE_ROOT="$WORK_ROOT/$ENGINE"
CANN_ROOT="$ENGINE_ROOT/cann"
TOOLCHAIN_ROOT="$ENGINE_ROOT/toolchain"
COMPAT_ROOT="$ENGINE_ROOT/host-compat"
mkdir -p "$DOWNLOAD_ROOT" "$ENGINE_ROOT"

verify_file() {
    local path=$1 size=$2 sha256=$3
    [[ -f "$path" && $(wc -c < "$path" | tr -d ' ') == "$size" ]] &&
        printf '%s  %s\n' "$sha256" "$path" | sha256sum -c - >/dev/null 2>&1
}

download_asset() {
    local url=$1 size=$2 sha256=$3 destination=$4
    if verify_file "$destination" "$size" "$sha256"; then return; fi
    rm -f "$destination.part"
    curl --fail --location --retry 10 --retry-all-errors --retry-delay 5 \
        --retry-max-time 600 --connect-timeout 30 --continue-at - \
        --output "$destination.part" "$url"
    verify_file "$destination.part" "$size" "$sha256" || {
        printf 'Size/SHA-256 mismatch: %s\n' "$destination.part" >&2; exit 1;
    }
    mv "$destination.part" "$destination"
}

release_url=${MODELZOO_RELEASE_BASE_URL:-"https://github.com/$REPOSITORY/releases/download/$RELEASE_TAG"}
toolchain_archive="$DOWNLOAD_ROOT/$TOOLCHAIN_NAME"
download_asset "$release_url/$TOOLCHAIN_NAME" "$TOOLCHAIN_SIZE" "$TOOLCHAIN_SHA256" "$toolchain_archive"

cann_archive="$DOWNLOAD_ROOT/$CANN_SOURCE_NAME"
if [[ "$CANN_PARTS" == "$CANN_SOURCE_NAME|$CANN_SOURCE_SIZE|$CANN_SOURCE_SHA256" ]]; then
    download_asset "$release_url/$CANN_SOURCE_NAME" "$CANN_SOURCE_SIZE" "$CANN_SOURCE_SHA256" "$cann_archive"
elif verify_file "$cann_archive" "$CANN_SOURCE_SIZE" "$CANN_SOURCE_SHA256"; then
    printf 'Using verified reconstructed archive: %s\n' "$cann_archive"
else
    cann_tmp="$cann_archive.part"
    rm -f "$cann_tmp"
    while IFS='|' read -r name size sha256; do
        part="$DOWNLOAD_ROOT/$name"
        download_asset "$release_url/$name" "$size" "$sha256" "$part"
        cat "$part" >> "$cann_tmp"
    done <<< "$CANN_PARTS"
    verify_file "$cann_tmp" "$CANN_SOURCE_SIZE" "$CANN_SOURCE_SHA256" || {
        printf 'Reconstructed CANN archive failed size/SHA-256 validation\n' >&2; exit 1;
    }
    mv "$cann_tmp" "$cann_archive"
fi

rm -rf "$CANN_ROOT" "$TOOLCHAIN_ROOT" "$COMPAT_ROOT"
mkdir -p "$CANN_ROOT" "$TOOLCHAIN_ROOT" "$COMPAT_ROOT"
tar -xf "$cann_archive" -C "$CANN_ROOT"
tar -xf "$toolchain_archive" -C "$TOOLCHAIN_ROOT"
nested_toolchain=$(find "$TOOLCHAIN_ROOT" -type f -name 'aarch64-mix210-linux.tar.bz2' -print -quit)
[[ -z "$nested_toolchain" ]] || tar -xjf "$nested_toolchain" -C "$TOOLCHAIN_ROOT"
compiler=$(find -L "$TOOLCHAIN_ROOT" -type f -name aarch64-mix210-linux-gcc -print -quit)
[[ -n "$compiler" ]] || { printf 'Toolchain compiler not found\n' >&2; exit 1; }
compiler_bin=$(dirname "$compiler")

mapfile -t host_executables < <(find -L "$TOOLCHAIN_ROOT" -type f \
    \( -path '*/libexec/gcc/*/cc1' -o -path '*/libexec/gcc/*/cc1plus' \) -print)
host_executables+=("$compiler")
unresolved_host_libraries() {
    local executable
    for executable in "${host_executables[@]}"; do
        LD_LIBRARY_PATH="${compat_lib_path:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            ldd "$executable" 2>/dev/null | awk '/not found/ {print $1}'
    done | sort -u
}
missing_libraries=$(unresolved_host_libraries)
if grep -qx libisl.so.19 <<< "$missing_libraries"; then
    libisl_package="$DOWNLOAD_ROOT/libisl19_${LIBISL_SHA256}.deb"
    download_asset "$LIBISL_URL" "$LIBISL_SIZE" "$LIBISL_SHA256" "$libisl_package"
    if command -v dpkg-deb >/dev/null 2>&1; then
        dpkg-deb --extract "$libisl_package" "$COMPAT_ROOT"
    else
        command -v ar >/dev/null 2>&1 || { printf 'dpkg-deb and ar are both unavailable\n' >&2; exit 1; }
        deb_member=$(ar t "$libisl_package" | awk '/^data\.tar\./ {print; exit}')
        [[ -n "$deb_member" ]] || { printf 'Debian package has no data archive\n' >&2; exit 1; }
        case "$deb_member" in
            *.tar.xz) ar p "$libisl_package" "$deb_member" | tar -xJf - -C "$COMPAT_ROOT" ;;
            *.tar.gz) ar p "$libisl_package" "$deb_member" | tar -xzf - -C "$COMPAT_ROOT" ;;
            *.tar.bz2) ar p "$libisl_package" "$deb_member" | tar -xjf - -C "$COMPAT_ROOT" ;;
            *.tar.zst) ar p "$libisl_package" "$deb_member" | tar --zstd -xf - -C "$COMPAT_ROOT" ;;
            *) printf 'Unsupported Debian data archive: %s\n' "$deb_member" >&2; exit 1 ;;
        esac
    fi
    compat_lib_path="$COMPAT_ROOT/usr/lib/x86_64-linux-gnu"
    [[ -z ${GITHUB_ENV:-} ]] || printf 'LD_LIBRARY_PATH=%s\n' "$compat_lib_path" >> "$GITHUB_ENV"
    export LD_LIBRARY_PATH="$compat_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
remaining_libraries=$(unresolved_host_libraries)
[[ -z "$remaining_libraries" ]] || { printf 'Unresolved host libraries:\n%s\n' "$remaining_libraries" >&2; exit 1; }

find_cann_paths() {
    if [[ "$ENGINE" == svp-nnn ]]; then
        include_file=$(find -L "$CANN_ROOT" -type f -path '*/acllib/include/acl/svp_acl.h' -print -quit)
        library_file=$(find -L "$CANN_ROOT" -type f -path '*/acllib/lib64_aarch64-mix210-linux/stub/libsvp_acl.so' -print -quit)
    else
        include_file=$(find -L "$CANN_ROOT" -type f -path '*/runtime/include/acl/acl.h' -print -quit)
        library_file=$(find -L "$CANN_ROOT" -type f -path '*/runtime/lib64/stub/libascendcl.so' -print -quit)
    fi
    include_path=${include_file%/*}; lib_path=${library_file%/*}
}
find_cann_paths
if [[ -z "$include_path" || -z "$lib_path" ]]; then
    installer=$(find "$CANN_ROOT" -type f -name 'Ascend-cann-toolkit*.run' -print -quit)
    [[ -n "$installer" ]] || { printf 'CANN toolkit installer not found\n' >&2; exit 1; }
    chmod +x "$installer"
    payload_root="$CANN_ROOT/installer-payload"
    "$installer" --noexec --extract="$payload_root" >/dev/null
    if [[ "$ENGINE" == svp-nnn ]]; then
        target_installer=$(find "$payload_root" -type f -name 'Ascend-acllib-*.run' -print -quit)
    else
        target_installer=$(find "$payload_root" -type f -name 'CANN-runtime-*-lmixlinux*.run' -print -quit)
    fi
    [[ -n "$target_installer" ]] || { printf 'Target development installer not found\n' >&2; exit 1; }
    chmod +x "$target_installer"
    "$target_installer" --noexec --extract="$CANN_ROOT/installed" >/dev/null
    rm -rf "$payload_root"
    find_cann_paths
fi
[[ -n "$include_path" && -n "$lib_path" ]] || { printf 'Target headers/stub library not found\n' >&2; exit 1; }

"$compiler" --version | head -n 1
printf '%s\n' "$compiler_bin" >> "${GITHUB_PATH:-/dev/null}"
{
    printf 'npu-include-path=%s\n' "$include_path"
    printf 'npu-lib-path=%s\n' "$lib_path"
    printf 'compiler=%s\n' "$compiler"
    printf 'sdk-release-tag=%s\n' "$RELEASE_TAG"
    printf 'cann-sha256=%s\n' "$CANN_SOURCE_SHA256"
    printf 'toolchain-sha256=%s\n' "$TOOLCHAIN_SHA256"
} >> "${GITHUB_OUTPUT:-/dev/null}"
