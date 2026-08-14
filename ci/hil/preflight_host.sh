#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    printf 'usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

board_host=${HIL_BOARD_HOST:-192.168.2.88}
board_user=${HIL_BOARD_USER:-root}
board_profile=${HIL_BOARD_PROFILE:-hi3403-svp-nnn-01}
profile_path=${HIL_BOARD_PROFILE_PATH:-/opt/hil/board-profile.json}
serial_device=${HIL_SERIAL_DEVICE:-/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0}
ssh_key=${HIL_SSH_KEY:-/opt/hil/keys/board_actions_ed25519}
known_hosts=${HIL_KNOWN_HOSTS:-/opt/hil/keys/known_hosts}
ssh_bin=${HIL_SSH_BIN:-ssh}

fail() {
    printf 'host preflight failed: %s\n' "$*" >&2
    exit 1
}

for command in bash file python3 jq ip ping ssh scp timeout sha256sum realpath stat; do
    command -v "$command" >/dev/null || fail "missing command: $command"
done

[[ $board_host =~ ^[0-9a-fA-F:.]+$ ]] || fail "unsafe board host"
[[ $board_user =~ ^[a-z_][a-z0-9_-]*$ ]] || fail "unsafe board user"
[[ $board_profile =~ ^[a-z0-9._-]+$ ]] || fail "unsafe board profile"

if $dry_run; then
    printf 'dry-run host preflight: profile=%s target=%s@%s serial=%s\n' \
        "$board_profile" "$board_user" "$board_host" "$serial_device"
    exit 0
fi

[[ $(id -un) == actions ]] || fail "runner must execute as actions, got $(id -un)"
[[ -r $profile_path ]] || fail "missing board profile: $profile_path"
jq -e --arg name "$board_profile" '.name == $name and .engine == "svp-nnn" and .soc == "SS928V100"' \
    "$profile_path" >/dev/null || fail "profile identity/engine/soc mismatch"
jq -e --arg user "$board_user" \
    '.sshUser == $user and (.requiresRoot == ($user == "root"))' "$profile_path" >/dev/null || \
    fail "profile SSH user/root requirement mismatch"

[[ -c $serial_device && -r $serial_device && -w $serial_device ]] || \
    fail "serial device is not an accessible character device: $serial_device"
serial_real=$(realpath "$serial_device")
[[ $serial_real == /dev/ttyUSB* ]] || fail "unexpected serial target: $serial_real"

ip -4 address show dev hilboard | grep -Fq '192.168.2.3/24' || fail "hilboard is not 192.168.2.3/24"
if ip route show default | grep -q 'dev hilboard'; then
    fail "hilboard must not be a default route"
fi
ping -c 1 -W 2 "$board_host" >/dev/null || fail "board is unreachable at $board_host"

[[ -f $ssh_key && ! -L $ssh_key ]] || fail "missing or symlinked SSH key: $ssh_key"
key_mode=$(stat -c '%a' "$ssh_key")
[[ $key_mode == 600 ]] || fail "SSH key mode must be 600, got $key_mode"
[[ -f $known_hosts && ! -L $known_hosts ]] || fail "missing pinned known_hosts"

available_kib=$(df -Pk /opt/hil/runs | awk 'NR == 2 {print $4}')
[[ $available_kib =~ ^[0-9]+$ && $available_kib -ge 5242880 ]] || fail "less than 5 GiB free in /opt/hil/runs"

"$ssh_bin" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile="$known_hosts" -o ConnectTimeout=5 -i "$ssh_key" \
    "$board_user@$board_host" true || fail "board SSH key login failed"

printf 'host preflight passed: profile=%s target=%s@%s serial=%s\n' \
    "$board_profile" "$board_user" "$board_host" "$serial_real"
