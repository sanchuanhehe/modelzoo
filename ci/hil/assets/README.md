# HIL asset promotion gate

`resnet50-svp-nnn-v1.yaml` is the authoritative released identity record.
Creating a new version or changing an existing identity requires all of the
following:

1. retained written publication authorization for every public file (received
   from the repository maintainer on 2026-08-17);
2. immutable source tags and exact asset names matching the recorded sizes and
   SHA-256 values;
3. a VM LabInventory mapping each logical `sourceRef` to the approved backend
   and runtime-only credential reference;
4. retained three-run golden and ten-run stability evidence, followed by a v2
   smoke comparison after publication;
5. review of the complete diff, staged Release hashes, tests, and device risk
   before upload;
6. verification of the published Release identities before a new manifest is
   promoted to `state: released`.

The OM remains forbidden from Git history and workflow logs. A public Release
upload is permitted by the recorded authorization but must still pass the
explicit pre-publication approval gate above.

Before creating the Release, verify the flat upload directory (the five runtime
assets plus `SHA256SUMS` and `LICENSE-AUDIT.md`) with:

```bash
python3 ci/hil/verify_release_bundle.py \
  --manifest ci/hil/assets/resnet50-svp-nnn-v1.yaml \
  --bundle /absolute/path/to/release-staging
```

The verifier requires one immutable tag, an exact file set, public/
redistributable policy, and matching size/SHA-256 identities in both the
AssetManifest and `SHA256SUMS`. It performs no upload.
