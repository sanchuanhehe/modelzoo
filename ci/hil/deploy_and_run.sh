#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

run_id=
artifact_dir=
golden_dir=${HIL_GOLDEN_DIR:-/opt/hil/models/resnet50-svp-nnn/v1}
test_level=smoke
dry_run=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id) run_id=$2; shift 2 ;;
        --artifact-dir) artifact_dir=$2; shift 2 ;;
        --golden-dir) golden_dir=$2; shift 2 ;;
        --test-level) test_level=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || { printf 'unsafe run id\n' >&2; exit 2; }
[[ -n $artifact_dir ]] || { printf 'missing --artifact-dir\n' >&2; exit 2; }
[[ $test_level == smoke || $test_level == stability ]] || { printf 'invalid test level\n' >&2; exit 2; }

run_root=${HIL_RUN_ROOT:-/opt/hil/runs}
run_dir=$(realpath -m -- "$run_root/$run_id")
root_real=$(realpath -m -- "$run_root")
[[ $run_dir == "$root_real/"* && $run_dir != "$root_real" ]] || { printf 'unsafe run directory\n' >&2; exit 1; }

board_host=${HIL_BOARD_HOST:-192.168.2.88}
board_user=${HIL_BOARD_USER:-root}
ssh_key=${HIL_SSH_KEY:-/opt/hil/keys/board_actions_ed25519}
known_hosts=${HIL_KNOWN_HOSTS:-/opt/hil/keys/known_hosts}
ssh_bin=${HIL_SSH_BIN:-ssh}
scp_bin=${HIL_SCP_BIN:-scp}
source_sha=${HIL_SOURCE_SHA:-}
expected_om_sha=c9c3f12d4e0f1b856d1fd7db639d0947abd006416914204adbbeeb568cbf9c5d
expected_input_sha=0565d0b330f11ecaae92412ff5931e6c1e4bf1636f7924928ea0fa5a97468d67
remote_dir=/opt/hil/runs/$run_id
iterations=1
[[ $test_level == stability ]] && iterations=10

ssh_options=(
    -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes
    -o UserKnownHostsFile="$known_hosts" -o ConnectTimeout=10 -i "$ssh_key"
)
scp_options=(-O "${ssh_options[@]}")
target=$board_user@$board_host

if $dry_run; then
    printf 'dry-run deploy: run=%s artifact=%s assets=%s target=%s:%s iterations=%s\n' \
        "$run_id" "$artifact_dir" "$golden_dir" "$target" "$remote_dir" "$iterations"
    exit 0
fi

for path in "$artifact_dir/main" "$artifact_dir/build-manifest.json" "$golden_dir/resnet50.om" \
    "$golden_dir/input/golden.jpg" "$golden_dir/file_list.json" "$golden_dir/acl.json" \
    "$golden_dir/expected.json" "$golden_dir/manifest.json" "$golden_dir/SHA256SUMS"; do
    [[ -f $path && ! -L $path ]] || { printf 'missing or symlinked input: %s\n' "$path" >&2; exit 1; }
done
if [[ -f $artifact_dir/expected.json ]] && jq -e '.status == "not-run"' "$artifact_dir/expected.json" >/dev/null; then
    printf 'refusing pre-HIL expected.status=not-run artifact\n' >&2
    exit 1
fi
[[ $(sha256sum "$golden_dir/resnet50.om" | awk '{print $1}') == "$expected_om_sha" ]] || { printf 'OM SHA mismatch\n' >&2; exit 1; }
[[ $(sha256sum "$golden_dir/input/golden.jpg" | awk '{print $1}') == "$expected_input_sha" ]] || { printf 'golden input SHA mismatch\n' >&2; exit 1; }
(cd "$golden_dir" && sha256sum --check --strict SHA256SUMS)
python3 "$script_dir/verify_result.py" --expected "$golden_dir/expected.json" \
    --validate-expected-only --asset-root "$golden_dir" --sha256s "$golden_dir/SHA256SUMS"
jq -e --arg sha "$source_sha" '
    .schemaVersion == 1 and .commit == $sha and .engine == "svp-nnn"
    and .soc == "SS928V100" and .sample == "samples/built-in/classification/ResNet50"
    and ([.outputs[] | select(.name == "main")] | length == 1)
' "$artifact_dir/build-manifest.json" >/dev/null || { printf 'build manifest mismatch\n' >&2; exit 1; }
main_sha=$(sha256sum "$artifact_dir/main" | awk '{print $1}')
jq -e --arg sha "$main_sha" '.outputs[] | select(.name == "main" and .sha256 == $sha)' \
    "$artifact_dir/build-manifest.json" >/dev/null || { printf 'main SHA does not match build manifest\n' >&2; exit 1; }
file "$artifact_dir/main" | grep -q 'ARM aarch64' || { printf 'main is not AArch64 ELF\n' >&2; exit 1; }

mkdir -p "$run_dir/results"
cp "$artifact_dir/build-manifest.json" "$run_dir/"
cp "$golden_dir/manifest.json" "$run_dir/asset-manifest.json"
"$ssh_bin" "${ssh_options[@]}" "$target" "install -d -m 0750 '$remote_dir/assets/input' '$remote_dir/result/txt'"
"$scp_bin" "${scp_options[@]}" "$artifact_dir/main" "$target:$remote_dir/main"
"$scp_bin" "${scp_options[@]}" "$golden_dir/resnet50.om" "$golden_dir/acl.json" \
    "$golden_dir/file_list.json" "$golden_dir/expected.json" "$golden_dir/manifest.json" \
    "$golden_dir/SHA256SUMS" "$target:$remote_dir/assets/"
"$scp_bin" "${scp_options[@]}" "$golden_dir/input/golden.jpg" "$target:$remote_dir/assets/input/golden.jpg"

remote_sums=$(mktemp)
trap 'rm -f "$remote_sums"' EXIT
{
    printf '%s  main\n' "$main_sha"
    sed 's#  #  assets/#' "$golden_dir/SHA256SUMS"
} >"$remote_sums"
"$scp_bin" "${scp_options[@]}" "$remote_sums" "$target:$remote_dir/DEPLOYED_SHA256SUMS"
"$ssh_bin" "${ssh_options[@]}" "$target" "cd '$remote_dir' && sha256sum -c DEPLOYED_SHA256SUMS && chmod 0750 main"

timeout_seconds=$(jq -r '.timeoutSeconds' "$golden_dir/expected.json")
result_file=$(jq -r '.resultFile' "$golden_dir/expected.json")
overall_rc=0
for iteration in $(seq 1 "$iterations"); do
    set +e
    "$ssh_bin" "${ssh_options[@]}" "$target" \
        "REMOTE_DIR='$remote_dir' ITERATION='$iteration' TIMEOUT_SECONDS='$timeout_seconds' sh -s" <<'REMOTE'
set -u
cd "$REMOTE_DIR/assets" || exit 1
rm -rf result
mkdir -p result/txt || exit 1
LD_LIBRARY_PATH=/opt/hil/runtime/svp_nnn/lib:/opt/hil/runtime/svp_nnn/lib/svp_npu:/opt/lib \
    ../main --model resnet50.om --acl acl.json --input file_list.json \
    >"../stdout-$ITERATION.log" 2>"../stderr-$ITERATION.log" &
main_pid=$!
(
    elapsed=0
    while kill -0 "$main_pid" 2>/dev/null && test "$elapsed" -lt "$TIMEOUT_SECONDS"; do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    kill -0 "$main_pid" 2>/dev/null || exit 0
    kill -TERM "$main_pid" 2>/dev/null || exit 0
    grace=0
    while kill -0 "$main_pid" 2>/dev/null && test "$grace" -lt 10; do
        sleep 1
        grace=$((grace + 1))
    done
    kill -KILL "$main_pid" 2>/dev/null || true
) &
watchdog_pid=$!
wait "$main_pid"
rc=$?
wait "$watchdog_pid" 2>/dev/null || true
printf '%s\n' "$rc" >"../exit-code-$ITERATION"
exit "$rc"
REMOTE
    remote_rc=$?
    set -e
    "$scp_bin" "${scp_options[@]}" "$target:$remote_dir/stdout-$iteration.log" "$run_dir/stdout-$iteration.log" || true
    "$scp_bin" "${scp_options[@]}" "$target:$remote_dir/stderr-$iteration.log" "$run_dir/stderr-$iteration.log" || true
    "$scp_bin" "${scp_options[@]}" "$target:$remote_dir/exit-code-$iteration" "$run_dir/exit-code-$iteration" || true
    if [[ $remote_rc -ne 0 ]]; then
        printf 'board inference iteration %s failed with exit %s\n' "$iteration" "$remote_rc" >&2
        overall_rc=$remote_rc
        break
    fi
    "$scp_bin" "${scp_options[@]}" "$target:$remote_dir/assets/$result_file" "$run_dir/results/$iteration.txt"
    python3 "$script_dir/verify_result.py" --expected "$golden_dir/expected.json" \
        --result "$run_dir/results/$iteration.txt" --asset-root "$golden_dir" --sha256s "$golden_dir/SHA256SUMS"
done
exit "$overall_rc"
