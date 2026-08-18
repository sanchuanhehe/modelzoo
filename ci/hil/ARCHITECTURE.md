# Hi3403 HIL v2 architecture

HIL v2 deliberately separates GitHub orchestration, laboratory control, target
execution, and model-specific test semantics.

## Trust boundaries

- GitHub Actions on the control VM is the only GitHub and laboratory control
  plane. It owns authorization, asset resolution, UART, recovery coordination,
  evidence upload, Environment protection, runner labels, and concurrency.
- The GitHub workflow has a fixed phase order. A repository data file cannot add
  a phase or provide `steps`, `run`, `shell`, SSH commands, host paths, or secrets.
- `TestDefinition` is data only: target class, adapter, build artifact, immutable
  asset manifest, iteration count, timeout, and expectation.
- A repository `TestAdapter` owns model/engine semantics. Adapter layout is
  conventional and validated; YAML cannot select an arbitrary entrypoint.
- VM `lab-control` owns small laboratory primitives. It consumes a resolved
  execution request and VM-local `LabInventory`; it does not interpret workflow
  YAML.
- Hi3403 runs only `hil-target-agent`. It has no GitHub token, repository,
  Release credential, workflow parser, or TestDefinition parser. It accepts a
  content manifest, operates only in one run sandbox, invokes the allowlisted
  `run-test` filename under timeout, returns raw outputs, and cleans that sandbox.
- Unprivileged `hilagent` execution is preferred. A LabInventory may record a
  forced-command root exception for vendor images with a root-only loader; that
  mode accepts only source CI artifacts built from `master` because the tested
  executable itself is otherwise a root-code boundary.
- UART, loss of SSH, kernel panic, reset, and flashing remain outside the board
  agent. The VM owns one identity-pinned, fixed-protocol, normally-open reset
  primitive; power-cycle and flashing remain operator recovery actions.

## Fixed workflow

The only supported order is:

1. authorize source CI run and SHA;
2. resolve and verify immutable assets;
3. start external UART capture;
4. run controller and target preflight;
5. prepare the adapter payload;
6. create target sandbox and transfer verified files;
7. execute the allowlisted target entry under timeout;
8. download and validate results through the repository adapter;
9. always collect evidence;
10. always clean the exact local and target run sandboxes.

The workflow does not accept arbitrary recovery steps. A reset may only invoke
the reviewed `lab-control reset pulse` primitive for the target's VM-local
`resetControllerRef`. Its model, USB identity, channel, 115200-baud driver,
normally-open contact, fixed pulse duration, and state-verification policy come
from root-owned LabInventory. Command frames are compiled into the reviewed
driver and cannot be supplied by TestDefinition, workflow input, or inventory.
The only workflow recovery choice is `reset-on-unreachable-once`: after UART
and controller preflight, one failed target-agent reachability probe may cause
one pulse and one bounded wait. Full target preflight failures, test failures,
and repeated outages never trigger another reset or flashing.

## Asset policy

Asset manifests use a logical `sourceRef`, immutable tag, exact release asset
name, size, and SHA-256. `LabInventory` maps the logical source to an authorized
backend and a runtime credential reference.

Redistributable metadata/input and restricted model binaries may use different
sources. An asset with unconfirmed redistribution rights cannot be execution
ready and must not be uploaded to a public or third-party backend merely to make
the pipeline pass. A restricted asset must use an approved backend explicitly
classified `restricted`; pointing it at an otherwise valid public Release
backend is rejected. For the built-in GitHub Release backend, `lab-control`
also verifies through the GitHub API that a restricted source repository is
actually private before accepting or downloading any cached asset.

## Compatibility and rollback

The existing v1 workflow remains untouched until v2 passes dry-run, negative
tests, v1/v2 smoke comparison, three-result golden consistency, ten-run
stability, and VM rebuild validation. Rollback is selecting the existing v1
workflow; v2 never mutates v1 golden state during bring-up.
