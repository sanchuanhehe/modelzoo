#!/usr/bin/env bash
set -euo pipefail

LOCK=${1:-ci/sdk-lock.json}
python3 ci/validate_sdk_lock.py "$LOCK"
eval "$(python3 - "$LOCK" <<'PY'
import json,shlex,sys
d=json.load(open(sys.argv[1])); a=d['artifacts']['svp-nnn']; t=d['artifacts']['toolchain']['archive']
for k,v in {'REPOSITORY':d['repository'],'TAG':d['releaseTag'],'ARCHIVE_NAME':a['archive']['name'],'ARCHIVE_SIZE':a['archive']['size'],'ARCHIVE_SHA':a['archive']['sha256'],'PARTS':'\n'.join(f"{p['name']}|{p['size']}|{p['sha256']}" for p in a['parts'])}.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"
DOWNLOAD_ROOT=${MODELZOO_DOWNLOAD_CACHE:-"$HOME/.cache/modelzoo/downloads"}
WORK_ROOT=${MODELZOO_CONVERTER_WORK:-"${RUNNER_TEMP:-/tmp}/modelzoo-converter"}
base=${MODELZOO_RELEASE_BASE_URL:-"https://github.com/$REPOSITORY/releases/download/$TAG"}
mkdir -p "$DOWNLOAD_ROOT" "$WORK_ROOT"
verify() { [[ -f $1 && $(wc -c < "$1" | tr -d ' ') == "$2" ]] && printf '%s  %s\n' "$3" "$1" | sha256sum -c - >/dev/null 2>&1; }
download() {
    local name=$1 size=$2 sha=$3 path="$DOWNLOAD_ROOT/$1"
    if ! verify "$path" "$size" "$sha"; then
        curl --fail --location --retry 10 --retry-all-errors --retry-delay 5 \
            --retry-max-time 600 --connect-timeout 30 --continue-at - \
            --output "$path.part" "$base/$name"
        verify "$path.part" "$size" "$sha" || { printf 'converter asset mismatch: %s\n' "$name" >&2; exit 1; }
        mv "$path.part" "$path"
    fi
}
archive="$DOWNLOAD_ROOT/$ARCHIVE_NAME"
if ! verify "$archive" "$ARCHIVE_SIZE" "$ARCHIVE_SHA"; then
    rm -f "$archive.part"
    while IFS='|' read -r name size sha; do download "$name" "$size" "$sha"; cat "$DOWNLOAD_ROOT/$name" >> "$archive.part"; done <<< "$PARTS"
    verify "$archive.part" "$ARCHIVE_SIZE" "$ARCHIVE_SHA" || { printf 'converter archive reconstruction failed\n' >&2; exit 1; }
    mv "$archive.part" "$archive"
fi
rm -rf "$WORK_ROOT/package" "$WORK_ROOT/payload" "$WORK_ROOT/installed"
mkdir -p "$WORK_ROOT/package"
tar -xf "$archive" -C "$WORK_ROOT/package"
outer=$(find "$WORK_ROOT/package" -type f -name 'Ascend-cann-toolkit*.run' -print -quit)
chmod +x "$outer"; "$outer" --noexec --extract="$WORK_ROOT/payload" >/dev/null
atc=$(find "$WORK_ROOT/payload" -type f -name 'Ascend-atc-*.run' -print -quit)
toolkit=$(find "$WORK_ROOT/payload" -type f -name 'Ascend-toolkit-*.run' -print -quit)
for installer in "$atc" "$toolkit"; do chmod +x "$installer"; "$installer" --quiet --devel --pylocal --install-path="$WORK_ROOT/installed"; done
rm -rf "$WORK_ROOT/payload" "$WORK_ROOT/package"
atc_bin=$(find "$WORK_ROOT/installed" -type f -path '*/bin/atc' -print -quit)
[[ -x "$atc_bin" ]] || { printf 'ATC not installed\n' >&2; exit 1; }
atc_root=$(dirname "$(dirname "$atc_bin")")
printf '%s\n' "$(dirname "$atc_bin")" >> "${GITHUB_PATH:-/dev/null}"
[[ -z ${GITHUB_ENV:-} ]] || {
  printf 'LD_LIBRARY_PATH=%s:%s\n' "$atc_root/lib64" "$atc_root/third_party_lib" >> "$GITHUB_ENV"
  printf 'PYTHONPATH=%s/python/site-packages\n' "$atc_root" >> "$GITHUB_ENV"
}
printf 'atc=%s\nsdk-release-tag=%s\ncann-sha256=%s\n' "$atc_bin" "$TAG" "$ARCHIVE_SHA" >> "${GITHUB_OUTPUT:-/dev/null}"
