# ResNet50 HIL asset license audit

Status: **publication authorized by the repository maintainer on 2026-08-17**.
The authorization source is the maintainer's explicit statement in the HIL
implementation task that all five golden assets may be public. This record
captures that project authorization; it is not independent legal verification
of upstream model-weight rights.

## Evidence reviewed

- The handoff document requires recording model provenance and redistribution
  permission, and explicitly says not to publish a model whose permission is
  unclear.
- The Euler Pi material `HiEuler_PI_ModelZoo使用说明.md` points to the HiSpark
  and HiEuler ModelZoo projects and states that the package provides converted
  models for use. It does not state that the converted binaries may be
  redistributed.
- The adjacent `model说明.txt` says the included OM files can be used to get
  started. That is a usage statement, not a redistribution grant.
- `model.zip` contains `model/resnet50.om` with size `26074852` and SHA-256
  `c9c3f12d4e0f1b856d1fd7db639d0947abd006416914204adbbeeb568cbf9c5d`.
  No file whose name contains `LICENSE`, `NOTICE`, `COPYRIGHT`, `README`, `许可`,
  `授权`, or `协议` is present in that archive.
- A source repository license cannot by itself be assumed to license a
  separately supplied converted model binary or its original weights.

## Authorization decision

Record `resnet50.om`, `input/golden.jpg`, and the three JSON files as
`access: public` with `license.status: redistributable` and reference
`LicenseRef-User-Authorized-Publication-2026-08-17`. The OM remains excluded
from Git history; publication is through the hash-pinned Release asset boundary.

This authorization resolves the license/provenance policy gate but does not by
itself authorize the operational publication step. Per the HIL change-control
boundary, the Release upload still waits for the complete diff, exact staged
hashes, tests, and device risks to be shown and explicitly approved.

## Golden input follow-up

The frozen v1 manifest labels `input/golden.jpg` as `original-hil-asset`; the
observed file is a 512-by-512 JFIF JPEG with SHA-256
`0565d0b330f11ecaae92412ff5931e6c1e4bf1636f7924928ea0fa5a97468d67`.
The durable bundle does not contain its generation prompt/tool record. The
maintainer's explicit publication authorization is therefore the durable
project decision used by the manifest; the missing generation record remains a
documented provenance limitation rather than an unreported assumption.
