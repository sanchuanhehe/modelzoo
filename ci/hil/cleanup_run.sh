#!/usr/bin/env bash
set -euo pipefail

run_id=${1:-}
[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || { printf 'unsafe run id\n' >&2; exit 2; }
run_root=${HIL_RUN_ROOT:-/opt/hil/runs}
run_dir=$(realpath -m -- "$run_root/$run_id")
root_real=$(realpath -m -- "$run_root")
[[ $run_dir == "$root_real/"* && $run_dir != "$root_real" ]] || { printf 'unsafe run directory\n' >&2; exit 1; }

board_host=${HIL_BOARD_HOST:-192.168.2.88}
board_user=${HIL_BOARD_USER:-root}
ssh_key=${HIL_SSH_KEY:-/opt/hil/keys/board_actions_ed25519}
known_hosts=${HIL_KNOWN_HOSTS:-/opt/hil/keys/known_hosts}
ssh_bin=${HIL_SSH_BIN:-ssh}
remote_dir=/opt/hil/runs/$run_id

"$ssh_bin" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile="$known_hosts" -o ConnectTimeout=5 -i "$ssh_key" \
    "$board_user@$board_host" "case '$remote_dir' in /opt/hil/runs/*) rm -rf '$remote_dir' ;; *) exit 1 ;; esac" || true

if [[ ${HIL_KEEP_LOCAL_RUN:-false} != true && -d $run_dir ]]; then
    rm -rf --one-file-system "$run_dir"
fi
