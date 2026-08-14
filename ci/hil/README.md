# Hi3403 HIL runner operations

This directory implements the manually dispatched Linux/SVP_NNN ResNet50 HIL boundary. Normal pushes and pull requests never target the self-hosted runner. The workflow accepts only a successful `CI` run from this repository, an exact source SHA, the pinned ResNet50 AArch64 artifact name, and a dispatch by the repository owner.

## Controlled state outside Git

The VM service runs as `actions`. Keep the following owned by `actions:actions` and mode `0750` unless a stricter mode is stated:

- `/opt/actions-runner`: GitHub runner `hil-hi3403-01`.
- `/opt/hil/runs`: ephemeral local evidence and the runner work directory.
- `/opt/hil/models/resnet50-svp-nnn/v1`: licensed local model and golden assets; never commit these.
- `/opt/hil/board-profile.json`: board inventory contract.
- `/opt/hil/keys/board_actions_ed25519`: board SSH private key, mode `0600`.
- `/opt/hil/keys/known_hosts`: UART-verified board host key.

The fixed local OM SHA-256 is `c9c3f12d4e0f1b856d1fd7db639d0947abd006416914204adbbeeb568cbf9c5d`. The fixed generated JPEG SHA-256 is `0565d0b330f11ecaae92412ff5931e6c1e4bf1636f7924928ea0fa5a97468d67`.

## Dispatch

Run `CI` on the desired commit first. In **Actions → Hi3403 HIL → Run workflow**, select an Environment-allowed branch and supply the successful CI run ID, its exact 40-character SHA, `hi3403-svp-nnn-01`, and `smoke` or `stability`. Stability performs ten sequential inferences under one concurrency lock.

GitHub only exposes a new `workflow_dispatch` file after that file exists on the default branch.
Before this workflow is merged, use **Actions → CI → Run workflow** on the feature branch and
set `hil_source_run_id` and `hil_source_sha`; the optional bootstrap job calls this same reusable
HIL workflow. It is guarded by `github.event_name == 'workflow_dispatch'` plus a nonempty run ID,
so pushes and pull requests cannot schedule the self-hosted runner. After merge, use the direct
Hi3403 HIL dispatch entry above.

## UART and recovery

The DEBUG connector is board port 14, the Ethernet cable uses port 3 (`eth0`), and UART is 115200 8N1. The workflow starts UART capture before any network or SSH preflight. If SSH fails, the job uploads the UART log and reports failure; it never reflashes or claims recovery.

For manual bring-up, start `capture_uart.py` as `actions`, then press the board's RST button once. Do not power-cycle while ToolPlatform explicitly asks for reset during a recovery burn. Password login is disabled after key provisioning; use the serial console for recovery.

The vendor Linux image is root-only at the ABI level: `/bin/busybox` is mode `0750` and
`/lib/ld-2.29.so` is mode `0700`. The VM runner still runs only as `actions`, while the pinned
board key logs in as `root` because an unprivileged board process cannot execute the dynamic
loader. The board profile must explicitly declare `"sshUser": "root"` and
`"requiresRoot": true`; preflight rejects any mismatch. Password SSH remains disabled.

## Failure handling

- `expected.status=not-run`, the synthetic `1x3x4x4` model package, wrong SHA, wrong board profile, non-AArch64 `main`, missing input, and unordered Top-5 all fail before or during deployment.
- Each board invocation records stdout, stderr, and the raw exit code. A timeout is nonzero and receives a ten-second kill grace period.
- Cleanup accepts only a single normalized child of `/opt/hil/runs`; diagnostics are uploaded before cleanup.
- Hard reset, power removal, and firmware flashing always require an operator. If firmware recovery is required, use the official Windows ToolPlatform and the DDR/eMMC-matched image and rank file.

For local logic checks, run:

```bash
python3 -m unittest discover -s ci/tests -p 'test_*.py' -v
bash -n ci/hil/*.sh
shellcheck --severity=error ci/hil/*.sh
python3 ci/hil/verify_result.py --help
python3 ci/hil/capture_uart.py --help
```
