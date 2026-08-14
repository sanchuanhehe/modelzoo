#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    printf 'usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

profile_path=${HIL_BOARD_PROFILE_PATH:-/opt/hil/board-profile.json}
profile_name=${HIL_BOARD_PROFILE:-hi3403-svp-nnn-01}
board_host=${HIL_BOARD_HOST:-192.168.2.88}
board_user=${HIL_BOARD_USER:-root}
ssh_key=${HIL_SSH_KEY:-/opt/hil/keys/board_actions_ed25519}
known_hosts=${HIL_KNOWN_HOSTS:-/opt/hil/keys/known_hosts}
ssh_bin=${HIL_SSH_BIN:-ssh}

if $dry_run; then
    printf 'dry-run board preflight: profile=%s target=%s@%s\n' "$profile_name" "$board_user" "$board_host"
    exit 0
fi

jq -e --arg name "$profile_name" '.name == $name and .engine == "svp-nnn" and .soc == "SS928V100"' \
    "$profile_path" >/dev/null
remote_env=$(jq -r '
    @sh "PROFILE_NAME=\(.name) PROFILE_SOC=\(.soc) PROFILE_ENGINE=\(.engine) PROFILE_SSH_USER=\(.sshUser) TEMP_SENSOR_POLICY=\(.temperatureSensors // "required") MIN_FREE_KIB=\(.minimumFreeKiB // 1048576) MAX_TEMP_MC=\(.maximumTemperatureMilliCelsius // 90000) REQUIRED_LIBS=\((.requiredLibraries // ["libsvp_acl.so", "libsecurec.so"]) | join("|")) REQUIRED_NODES=\((.requiredDeviceNodes // []) | join("|"))"
' "$profile_path")

"$ssh_bin" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile="$known_hosts" -o ConnectTimeout=5 -i "$ssh_key" \
    "$board_user@$board_host" "$remote_env sh -s" <<'REMOTE'
set -eu
fail() { printf 'board preflight failed: %s\n' "$*" >&2; exit 1; }

test "$PROFILE_NAME" = hi3403-svp-nnn-01 || fail "unexpected profile"
test "$PROFILE_SOC" = SS928V100 || fail "unexpected SoC"
test "$PROFILE_ENGINE" = svp-nnn || fail "unexpected engine"
test "$(id -un)" = "$PROFILE_SSH_USER" || fail "SSH session user does not match profile"
if test -r /etc/os-release; then
    :
elif test -r /proc/umap/sys && grep -q 'SS928V100' /proc/umap/sys; then
    :
else
    fail "missing trusted OS/SoC identity source"
fi
test -r /proc/meminfo || fail "missing /proc/meminfo"
test -d /opt/hil/runs || fail "missing /opt/hil/runs"
lsmod | grep -q '^ot_svp_npu ' || fail "SVP_NNN kernel module is not loaded"
if lsmod | grep -q '^ot_pqp '; then fail "mutually exclusive ot_pqp module is loaded"; fi

old_ifs=$IFS
IFS='|'
for library in $REQUIRED_LIBS; do
    found=false
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -Fq "$library"; then
        found=true
    elif find /usr /opt -type f -name "$library*" -print -quit 2>/dev/null | grep -q .; then
        found=true
    fi
    "$found" || fail "missing runtime library $library"
done
for node in $REQUIRED_NODES; do
    test -e "$node" || fail "missing device node $node"
done
IFS=$old_ifs

available_kib=$(df -Pk /opt/hil/runs | awk 'NR == 2 {print $4}')
test "$available_kib" -ge "$MIN_FREE_KIB" || fail "insufficient free disk"

max_temp=0
sensor_count=0
for sensor in /sys/class/thermal/thermal_zone*/temp; do
    test -r "$sensor" || continue
    value=$(cat "$sensor")
    case $value in *[!0-9]*) continue ;; esac
    sensor_count=$((sensor_count + 1))
    test "$value" -le "$MAX_TEMP_MC" || fail "temperature $value exceeds $MAX_TEMP_MC"
    test "$value" -le "$max_temp" || max_temp=$value
done
if test "$sensor_count" -eq 0; then
    test "$TEMP_SENSOR_POLICY" = unavailable || fail "no readable temperature sensor"
    max_temp=unavailable
fi
printf 'board preflight passed: profile=%s kernel=%s free_kib=%s max_temp_mc=%s\n' \
    "$PROFILE_NAME" "$(uname -r)" "$available_kib" "$max_temp"
REMOTE
