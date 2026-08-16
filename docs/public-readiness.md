# Public-readiness assessment

Snapshot date: 2026-08-16.

## Cold assessment

VettedMesh is useful today as an **experimental orchestration and audit layer
for developers already operating local 2D/3D workers**. Its typed D0–D10 state
machine, hash-bound human decisions, downstream invalidation, worker protocol,
and model-lineage enforcement solve real integration and traceability problems.

It is **not yet useful as a dependable end-user text-to-3D product**. The
repository has no completed, reproducible live D0–D10 asset, the first 8 GB
qualification corpus is unrun, several optional workers have unresolved or
territory-limited terms, and the animated path is research-grade. A large test
suite proves the coordinator and contracts; it cannot prove generator quality.

Accordingly, the defensible public label is **developer preview**, not stable,
production-ready, one-click text-to-3D, or a generator model. The repository
becomes broadly useful when an independent user can install it, reproduce at
least 8/10 acceptable static props on declared hardware, inspect every final
artifact in Studio, and verify its lineage from published evidence.

## Closest public repositories

These are upstream generator or workflow projects, not exact substitutes. Star
counts are a volatile GitHub API snapshot from the date above; they measure
attention, not output quality.

| Repository | GitHub stars | Primary documented scope | Relationship to VettedMesh |
|---|---:|---|---|
| [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | 14,500 | Text/image-to-shape and texture generation | A possible D2 backend; much closer to a generator product |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | 13,450 | Image/text-conditioned 3D asset generation | A possible D2 backend; VettedMesh adds gates and packaging around it |
| [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) | 8,855 | Text/image-to-3D research implementation | Direct generation research rather than lifecycle orchestration |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | 6,852 | Fast single-image 3D reconstruction | A possible reconstruction backend, not a full gated asset pipeline |
| [InstantMesh](https://github.com/TencentARC/InstantMesh) | 4,498 | Sparse-view large-reconstruction model | A possible D2 backend with a narrower output contract |
| [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack) | 3,837 | ComfyUI nodes for multiple 3D techniques | The closest workflow-oriented peer; node graphs rather than an auditable release state machine |
| [stable-fast-3d](https://github.com/Stability-AI/stable-fast-3d) | 1,787 | Fast single-image mesh reconstruction | A direct generator with separate model terms |
| [T3Bench](https://github.com/THU-LYJ-Lab/T3Bench) | 1,099 | Text-to-3D benchmark | A relevant evaluation precedent, not an asset compiler |

Repository license metadata is not enough to clear model weights or generated
outputs. VettedMesh therefore records the exact backend and model revision and
fails release closed when policy is unresolved; see
[`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

## Objective release gates

All gates are binary. A partial pass is a failed public-release gate.

| Gate | Pass condition | Current status |
|---|---|---|
| Identity | Neutral name used in product paths; no dependency on a consuming game/project | Implemented; legal trademark clearance remains owner responsibility |
| License | Repository license, notices, contribution terms, security policy, and third-party matrix present | Implemented; not a legal opinion |
| Secrets/privacy | All reachable commits pass a secret scan; owner accepts every historical author email becoming public | Secret scan passed locally; email consent unresolved |
| Reproducible core install | Hash-locked clean install passes on Windows, Ubuntu, and macOS | Windows and bare Ubuntu passed locally; macOS/hosted evidence pending |
| Package | Wheel installs outside the checkout and can load 13 manifests, resolve config, and run the demo | Passed locally from a clean environment; hosted evidence pending |
| Container | Compose validates and the pinned, non-root image builds and becomes healthy | Passed locally as UID 10001 |
| Deterministic behavior | Full randomized tests and deterministic D0–D10 demo pass | Passed locally: 355 tests, one environment skip, demo completed |
| Security automation | Dependency audit, full-history secret scan, and CodeQL are green | Local audit has no known vulnerabilities; hosted checks pending |
| Live usefulness | Exactly ten real 8 GB runs attempted; at least eight pass the checked-in recognition/geometry/surface/features rubric | **Failed: no live corpus evidence exists** |
| Release integrity | Tag matches package version; archives, wheel/sdist, checksums, SPDX SBOMs, and attestations publish | Workflow implemented; must not tag before preceding gates pass |

## Execution order

1. Land and run the deterministic, packaging, security, and cross-platform CI
   work on a private repository.
2. Resolve every CI failure and verify a fresh launcher install on each named
   operating system.
3. Have the owner decide whether the historical email exposure is acceptable
   and obtain an actual trademark/name review if commercial use is planned.
4. On a physically observed 8 GB machine, run every prompt in
   `golden/static-props.json` with real workers and exact model revisions.
5. Record the human assessments, then run `vettedmesh golden evaluate`; publish
   the portable gallery only if all ten were attempted and at least eight pass.
6. Re-run the dependency/legal matrix against the exact chosen models. A
   territory-limited or unresolved backend cannot support a globally cleared
   release.
7. Rename the private GitHub repository to `vettedmesh`, enable private
   vulnerability reporting and protected-branch checks, merge the reviewed
   change, and create signed tag `v0.2.0-rc.1`.
8. Confirm the release job produced checksums, both SBOMs, attestations, source
   archives, wheel, and sdist. Only then change repository visibility to public.

Until gates 3–6 pass, publishing code for review may be reasonable, but
claiming a qualified text-to-3D release is not.
