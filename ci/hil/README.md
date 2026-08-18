# Hi3403 HIL operations

The repository currently keeps two isolated paths:

- `Hi3403 HIL` is the proven v1 rollback workflow. It uses VM-local golden
  assets and the existing root SSH boundary.
- `Hi3403 HIL v2` is the replacement architecture. It is intentionally blocked
  until its reviewed public Release and promoted AssetManifest are present on
  the branch from which it runs.

Normal pushes and pull requests schedule neither HIL workflow. v2 is manual,
owner-only, master-only, protected by the `hil-hi3403` Environment, fixed runner
labels, and one concurrency group.

## v2 interface

Run the normal `CI` workflow first, then dispatch `Hi3403 HIL v2` with:

- `source_run_id`: successful same-repository `CI` run;
- `source_sha`: exact 40-character SHA built by that run;
- `target`: reviewed LabInventory ID, currently `hi3403-01`;
- `test_definition`: reviewed pure-data definition, currently
  `resnet50-smoke` or `resnet50-stability`.
- `reset_policy`: `none` by default, or `reset-on-unreachable-once`. The latter
  performs exactly one reviewed reset pulse only if the initial target-agent
  probe fails, waits up to the LabInventory boot timeout, and then runs the full
  target preflight. It never retries a library, ABI, temperature, or inference
  failure and never loops resets.

Adding a test means adding a schema-valid `TestDefinition` and a conventional
adapter directory with `adapter.yaml`, `prepare.py`, `verify.py`, and the fixed
target `run-test`. A definition cannot add workflow phases or supply `steps`,
`run`, `shell`, or `command`. GitHub Actions retains the only phase ordering.

The control VM runs the GitHub Runner as `actions`. Each run receives an exact
`/opt/hil/runs/<run-id>` sandbox and short-lived credentials. `lab-control`
performs only asset verification, UART, controller/target probes, verified
target transport/execution, evidence, and exact cleanup. See
`provision/README.md` for reconstruction. External reset transport is pinned by
USB VID/PID/serial and exposed only through a fixed `USB Relay (TC, 2, Opto)`
channel-1 pulse primitive. The normally-open `NO1`/`COM1` wiring releases reset
if relay or VM power is lost. Neither TestDefinition nor workflow input can
provide serial bytes, channel selection, or pulse duration.

The board SSH identity is explicit in VM LabInventory and always uses a forced
command. Prefer the dedicated `hilagent` account; a recorded forced-command
`root` exception is available only when the vendor image's root-only loader
prevents unprivileged execution. The board does not contain a GitHub Runner,
repository, workflow parser, TestDefinition, GitHub token, Release token, or
private key. Model/engine meaning remains in the repository adapter; the board
agent exposes only generic sandbox and inspection operations.

## Asset and evidence policy

An executable AssetManifest requires an immutable tag, exact release asset
name, size, SHA-256, approved LabInventory source, and confirmed license state
for every file. Public redistributable input/metadata and the restricted OM may
resolve through different logical sources. The OM is not uploaded anywhere by
this change.

The current candidate records the known OM digest
`c9c3f12d4e0f1b856d1fd7db639d0947abd006416914204adbbeeb568cbf9c5d`
and observed JPEG digest
`0565d0b330f11ecaae92412ff5931e6c1e4bf1636f7924928ea0fa5a97468d67`.
The repository maintainer explicitly authorized public publication of the OM
and JPEG on 2026-08-17. Release `hil-assets-resnet50-svp-nnn-v1` was created and
read back with matching tag, names, sizes, GitHub digests, and locally computed
SHA-256 values before the manifest was promoted to `released`.

Every run stops UART and collects a schema-validated, SHA-indexed evidence
bundle even if SSH is unavailable. The bundle contains execution context,
manifests, structured lab-control events, host snapshot, a final target probe
(including temperature/free-space when available), UART, retained raw exit
codes/stdout/stderr/results, and credential-value redaction. A reset event
records the pinned device identity, duration, and verified released state but
never arbitrary command bytes. Automatic workflow recovery is disabled until
the first physical pulse is accepted; power-cycle and flashing remain manual.

For an operator-authorized reset, start UART capture first, then run:

```sh
/usr/local/bin/lab-control reset pulse \
  --inventory /etc/hil/lab-inventory.yaml \
  --target hi3403-01 \
  --run-id <existing-run-id> \
  --dry-run
```

Remove `--dry-run` only after the dry-run identity check passes. If release
verification fails, disconnect relay USB to return the normally-open contact to
its safe state; do not loop resets.

## Verification

```bash
python3 -m unittest discover -s ci/tests -p 'test_*.py'
python3 ci/hil/validate_config.py --all
python3 ci/hil/verify_release_bundle.py --help
ruff check ci
shellcheck --severity=error ci/hil/target/hil-target-agent \
  ci/hil/adapters/resnet50-svp-nnn/target/run-test ci/hil/provision/*.sh
```

Before v2 replaces v1, it still requires authorized asset migration, VM/target
provisioning, v1/v2 smoke comparison, three identical golden results, ten-run
stability, failure/security tests, and a clean VM rebuild. A target outage never
triggers automatic flashing or a false recovery result.
