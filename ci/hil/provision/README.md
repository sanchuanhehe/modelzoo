# Rebuilding the HIL v2 execution layers

Provisioning installs code and directory ownership only. It does not copy model
assets, SDKs, firmware, GitHub tokens, SSH private keys, or repository content to
the target.

## Control VM

After reviewing a specific repository commit, run as root:

```sh
ci/hil/provision/controller.sh /absolute/path/to/reviewed/modelzoo
```

This installs the root-owned `lab-control` implementation under an immutable
`/opt/hil/control/releases/<version>` directory, verifies its generated
`INSTALL-SHA256SUMS`, atomically switches `/opt/hil/control/current`, creates
`/opt/hil/runs` for `actions`, and exposes a root-owned
`/usr/local/bin/lab-control` symlink. Reusing a version with different bytes is
rejected. Install Python, PyYAML,
jsonschema, OpenSSH client, and the `actions` Runner through the VM image or
configuration management before this step. Store the real LabInventory at
`/etc/hil/lab-inventory.yaml`, owned by root and not writable by group/other;
runtime loading rejects a symlink or unsafe ownership/mode. The committed
inventory is deliberately only an example.

Every workflow run writes its board key, pinned `known_hosts`, and short-lived
GitHub token into its mode-0700 run directory. Evidence redaction runs before
the exact directory is removed. Secrets are never installed into the control
package.

## Hi3403 target

After reviewing the agent and providing only the matching public key, run on the
board as root:

```sh
ci/hil/provision/target.sh \
  /absolute/path/to/hil-target-agent \
  /absolute/path/to/controller-key.pub \
  hilagent
```

Use a dedicated v2 keypair. Do not reuse the v1 unrestricted/root recovery key;
the workflow receives only the `HIL_V2_BOARD_SSH_KEY` Environment secret and
LabInventory refers to `hil-v2-board-ssh`.

The third argument is mandatory and must match LabInventory. Prefer `hilagent`:
it creates a dedicated unprivileged account. The vendor default image has also
been observed with a root-only dynamic loader/runtime; if read-only inventory
proves that the tested binary cannot execute as `hilagent`, use the explicit
`root` argument instead. Root mode remains constrained by the same
forced-command agent but has a larger impact if that agent or the reviewed
adapter is compromised, so record this exception and rationale in the device
inventory. The fixed workflow additionally rejects non-`master` source CI
artifacts whenever `executionMode` is `forced-command-root`.

The agent is installed under a versioned root-owned filename and switched by a
symlink; different bytes cannot reuse an installed version. Provisioning
preserves unrelated `authorized_keys` entries and replaces only the line ending
in `modelzoo-hil-target-agent`. It records the selected account in the
root-owned `/etc/hil/target-execution-user` marker and refuses an implicit
root/`hilagent` switch; changing identity requires a separately reviewed
deprovision step after all run sandboxes and the old managed key are handled.
The target receives no repository, workflow
definition, GitHub credential, Release credential, private key, SDK, or
firmware.

The target agent accepts version/probe, generic library/module inspection,
exact sandbox preparation, verified upload, seal, fixed entrypoint execution,
snapshot/get, and exact cleanup. It cannot run an arbitrary shell command and
does not implement reset or flashing.
