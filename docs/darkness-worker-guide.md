# Darkness worker guide

Detailed, historical worker-by-worker reference for the `darkness` compiler package. For an overview of the whole
tool, start with the repo root [README](../README.md). This document predates the repo's extraction from a game
project and preserves the original R&D log of qualifying each worker; some referenced design documents from that
project are not included here.

Darkness's orchestration, contracts, optimizer, workers, and QA are independent of any particular asset. The
qualified D8 adapter deliberately reuses the small, tested `assetforge` ComfyUI multiview client and Blender view
baker; Darkness wraps those components with its own provenance, selected-master re-render, black-atlas rejection,
Qwen/referee, and downstream hash gates.

The implemented foundation provides:

- a schema-validated model/worker registry with explicit priority and licensing state;
- recursive artifact-lineage checks that permit research while failing closed for release;
- bounded, schema-validating LocalDeploy structured-output retries;
- digest-pinned runtime qualification records that separate transport, schema, and semantic evidence;
- a qualification gate for any legacy component proposed for reuse.
- isolated subprocess workers with allowlisted paths, logs, GPU leasing, timeout, and cancellation;
- an operation allowlist, candidate branching, deterministic regression guards, immutable artifacts, and a
  hash-chained resumable event log;
- a qualified Blender 4.5.11 LTS background worker with strict analyze/repair/render/export operations,
  source-preserving checkpoints, fixed-view visual comparison, and fail-closed GLB export re-import validation;
- a bounded Blender quad-dominant repair that removes only a declared face budget, never moves source vertices, stages
  a 99%+ quad editable checkpoint, and validates a separately triangulated GLB export copy;
- an AutoRemesher 1.0.0 D2b research worker with immutable-source hashing, bounded CLI parameters, polygon-aware OBJ
  analysis, and separate structural/all-quad promotion gates; the first real TripoSG result is correctly rejected;
- a deterministic Instant Meshes D2b second-opinion worker pinned by source/binary digest; its pure-quad TripoSG result
  is also correctly rejected because it is open and fragmented;
- a MIT TripoSG 1.5B image-to-3D adapter (code only right now — see below) that requires caller-owned RGBA masking
  and excludes BRIA RMBG, kept as the native-Windows offline fallback (demoted 2026-07-12: real corpus output had
  17 disconnected surface components — not a topology quality bar to build on);
- a generic Windows -> WSL2 worker launcher (`adapters/wsl_launch.py`) that translates the native-Windows
  request/response file contract across the WSL2 DrvFS boundary, plus a MIT TRELLIS.2-4B adapter
  (`adapters/trellis2_worker.py`) on top of it as the new primary geometry candidate. WSL2 (Ubuntu-24.04) is being
  installed on this host now; see `environments/trellis2-wsl2-cu124.setup.md` for the remaining manual steps
  (CUDA toolkit, conda, `setup.sh --flash-attn`) and the qualification gate this worker must still pass before its
  manifest can leave `lifecycle: "discovered"`;
- an original project-owned short-biped fallback that emits one connected watertight skinned GLB, two lower LODs,
  a 21-joint rig contract, five embedded clips, sockets, collider/hitbox contracts, deformation reports, and
  four-direction sprite sheets;
- fail-closed research/release packaging with per-file SHA-256 and qualification/license state.

The first live optimizer record is
`qualifications/qwen3.6-27b_ollama_rtx5090.json`. It qualifies two-image semantic review and four-image transport for
the exact recorded Qwen/Ollama digest. Eight-image use is not qualified: the tested backend returned an empty completion
inside a successful HTTP response, which the Darkness client correctly rejected.

Model weights and generated artifacts do not belong in this directory. Store them in external or gitignored
workspaces and record exact digests in run lineage.

Hunyuan3D and HY-Motion are parked as lineage-gated research candidates: still real R&D options, but their
community license excludes the EU/UK/South Korea from output use, so they cannot clear a global release export.
They are not production dependencies. R-DMesh remains blocked. The open-source-first geometry order, reevaluated
2026-07-12 (`DESIGN/asset_forge_darkness_master_plan.md` Section 17), is: stand up the WSL2 boundary, qualify
TRELLIS.2-4B as primary and Direct3D-S2 for high-resolution detail, add an automatic AutoRemesher/Instant Meshes
retopology sub-stage (D2b) to fix the blobby/disconnected marching-cubes output generically, and keep TripoSG only
as the no-WSL2 fallback. Host GPU confirmed via `nvidia-smi` on 2026-07-12: one RTX 5090, 32,607 MiB, driver 610.74.

This machine's only Python is now 3.12 (3.10 was uninstalled 2026-07-12 along with the 14 GB local TripoSG
venv/weights it was pinned to, since TripoSG is the demoted fallback candidate, not a top-priority worker; run
`darkness workers` and rebuild that venv against 3.12 if TripoSG is ever needed again). `canonical.short_biped`'s
mesh dependencies (`numpy`, `trimesh`, `scikit-image`) were reinstalled directly under 3.12 so it no longer needs
TripoSG's venv to run.

The current vertical slice is deliberately marked `partial`: the qualified D7–D10 sprite-target chain now retargets
four CC0 motions, builds one ComfyUI-assisted stable-UV painted master, renders/packages 16 directional sheets, and
validates 144 sprites/16 clips in a standalone Unity project. It is still a human-review candidate, not a release:
canonical landmark approval, corrective shoulder deformation, D0–D6 orchestration, and owner sign-off remain open.
The optional live-3D PBR/FBX prefab adapter is also not implemented.

Useful commands (from the repo root, package installed via `pip install -e .`):

```powershell
python -m darkness workers
python -m darkness demo --workspace C:/AssetForgeDarknessRuns/demo
python -m darkness run-worker --worker-id canonical.short_biped --request C:/path/request.json
python -m darkness mesh-check --input C:/path/triangular.obj --output C:/path/new-evidence-directory
python -m darkness glb-component-audit --input C:/path/candidate.glb --output C:/path/new-audit-directory
python adapters/build_blender_request.py `
  --input C:/path/candidate.glb --output-directory C:/path/blender-evidence `
  --out C:/path/blender-request.json --job-id blender.example.v1 --operation-id blender.repair
python -m darkness run-worker --worker-id blender --request C:/path/blender-request.json
python adapters/build_blender_request.py `
  --input C:/path/autoremesher-candidate.obj --output-directory C:/path/repaired-evidence `
  --out C:/path/repair-request.json --job-id blender.retopology.repair.v1 `
  --operation-id blender.repair_retopology --minimum-quad-fraction 0.99
python -m darkness run-worker --worker-id blender --request C:/path/repair-request.json
python adapters/build_retopology_request.py `
  --input C:/path/normalized-source.obj --output-directory C:/path/retopology-evidence `
  --out C:/path/retopology-request.json --job-id autoremesher.example.v1 --target-quads 50000
python -m darkness run-worker --worker-id autoremesher --request C:/path/retopology-request.json
python adapters/build_instant_meshes_request.py `
  --input C:/path/normalized-source.obj --output-directory C:/path/instant-meshes-evidence `
  --out C:/path/instant-meshes-request.json --job-id instant.meshes.example.v1 --field-faces 12500
python -m darkness run-worker --worker-id instant_meshes.retopology --request C:/path/instant-meshes-request.json
python -m darkness package --package-id goblin.v1 --candidate-id darkness-canonical-short-biped-v1 `
  --source C:/path/validated-output --output C:/path/package `
  --qualification qualifications/canonical-short-biped-v1.json --mode research
python adapters/run_motion_candidate_pipeline.py `
  --target C:/path/approved-rig.blend --motion-source C:/path/UAL1_Standard.glb `
  --output-root C:/AssetForgeDarknessRuns/goblin-painted-review `
  --blender C:/path/blender.exe --unity C:/path/Unity.exe
```

`mesh-check`, `glb-component-audit`, and the Blender worker never overwrite their source or an existing evidence
directory. The Blender worker emits source/candidate checkpoints, fixed-view comparisons, a separate GLB, and a
re-import validation report, but it does not grant the required human approval or promote the candidate. For the
current TripoSG goblin, the original generated geometry is the locked visual source; the generic procedural fitted-v2
package is negative technical evidence, not the visual master to continue.

AutoRemesher 1.0.0 accepts normalized OBJ input, not the source GLB directly. Its first real candidate preserved the
broad goblin silhouette and reached 99.15% quads, but failed closed on 12 non-manifold edges and 346 non-quad faces.
Repeated identical jobs also differed, so this worker is not claimed deterministic. No D2b candidate is approved yet.

Instant Meshes is the independent second opinion. With deterministic mode and one thread, repeated outputs were
bit-identical and pure quad, but the calibrated ~49k-face result had 30 components and 454 boundary edges. It is also
rejected. AutoRemesher's branch was the stronger bounded-repair starting point. Blender's bounded pass now stages a
one-component, closed, 99.37%-quad checkpoint with unchanged source vertex coordinates and a structurally valid GLB.
Automatic topology gates pass, but human identity and canonical-fit approval are still required before production
rigging or promotion.

The staged `blender.propose_short_biped_rig` operation now continues that same editable checkpoint into a research
deformation probe: 26 proportional landmarks, a 21-bone semantic rig, Blender bone-heat weights pruned to four
influences, neutral identity comparison, shoulder/elbow and hip/knee stress renders, and skinned GLB re-import. Its
automatic gates pass, but visible shoulder/armpit compression, canonical fitting, corrective weights, and human
approval remain open; this operation does not promote the generated topology.

Machine-specific worker commands belong in ignored `config.local.json`; use `config.example.json` as the template.

Run tests from the repo root with a Python environment containing the project dependencies:

```powershell
python -m pytest tests -q
```
