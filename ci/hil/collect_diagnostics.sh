#!/usr/bin/env bash
set -u

run_id=${1:-}
[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || { printf 'unsafe run id\n' >&2; exit 2; }
run_root=${HIL_RUN_ROOT:-/opt/hil/runs}
run_dir=$(realpath -m -- "$run_root/$run_id")
root_real=$(realpath -m -- "$run_root")
[[ $run_dir == "$root_real/"* && $run_dir != "$root_real" ]] || { printf 'unsafe run directory\n' >&2; exit 1; }
mkdir -p "$run_dir/diagnostics"

profile_path=${HIL_BOARD_PROFILE_PATH:-/opt/hil/board-profile.json}
golden_dir=${HIL_GOLDEN_DIR:-/opt/hil/models/resnet50-svp-nnn/v1}
for evidence in "$profile_path" "$golden_dir/manifest.json" "$golden_dir/expected.json" \
    "$golden_dir/SHA256SUMS"; do
    if [[ -f $evidence && ! -L $evidence ]]; then
        cp "$evidence" "$run_dir/diagnostics/"
    fi
done

board_host=${HIL_BOARD_HOST:-192.168.2.88}
board_user=${HIL_BOARD_USER:-root}
ssh_key=${HIL_SSH_KEY:-/opt/hil/keys/board_actions_ed25519}
known_hosts=${HIL_KNOWN_HOSTS:-/opt/hil/keys/known_hosts}
ssh_bin=${HIL_SSH_BIN:-ssh}
remote_dir=/opt/hil/runs/$run_id
ssh_options=(-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile="$known_hosts" -o ConnectTimeout=5 -i "$ssh_key")

{
    printf 'collected_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'runner_user=%s\n' "$(id -un)"
    uname -a
    df -h /opt/hil/runs
    ip -brief address
    ip route
} >"$run_dir/diagnostics/host.txt" 2>&1

"$ssh_bin" "${ssh_options[@]}" "$board_user@$board_host" "REMOTE_DIR='$remote_dir' sh -s" \
    >"$run_dir/diagnostics/board.txt" 2>&1 <<'REMOTE'
set +e
date -Iseconds 2>/dev/null || date
uname -a
cat /etc/os-release 2>/dev/null || true
cat /proc/umap/sys
cat /proc/meminfo
df -h
if command -v ip >/dev/null 2>&1; then
    ip address show
    ip route show
else
    ifconfig -a
    route -n
fi
sensor_count=0
for sensor in /sys/class/thermal/thermal_zone*/temp; do
    test -r "$sensor" || continue
    printf '%s=' "$sensor"
    cat "$sensor"
    sensor_count=$((sensor_count + 1))
done
test "$sensor_count" -ne 0 || printf 'temperature_sensors=unavailable\n'
if test -d "$REMOTE_DIR"; then
    ls -la "$REMOTE_DIR"
    for file in "$REMOTE_DIR"/exit-code-*; do
        test -f "$file" || continue
        printf '%s=' "$file"
        cat "$file"
    done
else
    printf 'remote_run_directory=not-created\n'
fi
exit 0
REMOTE
board_rc=$?

if [[ -f ${HIL_UART_LOG:-} ]]; then
    cp "${HIL_UART_LOG}" "$run_dir/diagnostics/uart.log"
fi
find "$run_dir" -maxdepth 2 -type f -printf '%P\t%s bytes\n' | sort >"$run_dir/diagnostics/files.txt"
exit "$board_rc"
