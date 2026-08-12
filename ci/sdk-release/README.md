# SDK CI Release tooling

`prepare.sh` copies the approved toolchain and NNN archive byte-for-byte, splits the oversized SVP_NNN archive, and produces integrity/provenance metadata. `verify.sh` checks every asset and reconstructs SVP_NNN in declared part order.

The generated directory is release material, never repository source. Do not add binary SDK assets to Git.

```bash
ci/sdk-release/prepare.sh \
  --output /private/tmp/modelzoo-sdk-release \
  --toolchain /path/to/aarch64-mix210-linux.tgz \
  --nnn /path/to/NNN_PC.tgz \
  --svp-nnn /path/to/SVP_NNN_PC_V1.0.2.17.tgz
```
