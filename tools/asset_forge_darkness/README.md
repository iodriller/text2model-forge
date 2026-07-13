# Asset Forge Darkness

This is the independent implementation described by
`DESIGN/asset_forge_darkness_master_plan.md`. It does not import or depend on the legacy
`tools/asset_forge` package.

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
TRELLIS.2-4B as primary and Direct3D-S2 for high-resolution detail, add a deterministic AutoRemesher/Instant Meshes
retopology sub-stage (D2b) to fix the blobby/disconnected marching-cubes output generically, and keep TripoSG only
as the no-WSL2 fallback. Host GPU confirmed via `nvidia-smi` on 2026-07-12: one RTX 5090, 32,607 MiB, driver 610.74.

This machine's only Python is now 3.12 (3.10 was uninstalled 2026-07-12 along with the 14 GB local TripoSG
venv/weights it was pinned to, since TripoSG is the demoted fallback candidate, not a top-priority worker; run
`darkness workers` and rebuild that venv against 3.12 if TripoSG is ever needed again). `canonical.short_biped`'s
mesh dependencies (`numpy`, `trimesh`, `scikit-image`) were reinstalled directly under 3.12 so it no longer needs
TripoSG's venv to run.

The current vertical slice is deliberately marked `partial`: TripoSG produced a coherent candidate on the RTX 5090,
coarse target-envelope fitting reduced mean relative extent error from 24.7% to 5.6%, all fitted canonical LOD skin
stress gates pass, and Blender validates a source-preserving 17-to-1 component cleanup of the original TripoSG goblin.
Fine-detail/semantic retopology, UV/PBR work, pose/deformation validation, and Unity import/playmode validation are not
implemented. A research package can be built; a release package fails closed.

Useful commands (from the EmberDefense root):

```powershell
$env:PYTHONPATH = (Resolve-Path tools/asset_forge_darkness).Path
python -m darkness workers
python -m darkness demo --workspace C:/AssetForgeDarknessRuns/demo
python -m darkness run-worker --worker-id canonical.short_biped --request C:/path/request.json
python -m darkness mesh-check --input C:/path/triangular.obj --output C:/path/new-evidence-directory
python -m darkness glb-component-audit --input C:/path/candidate.glb --output C:/path/new-audit-directory
python tools/asset_forge_darkness/adapters/build_blender_request.py `
  --input C:/path/candidate.glb --output-directory C:/path/blender-evidence `
  --out C:/path/blender-request.json --job-id blender.example.v1 --operation-id blender.repair
python -m darkness run-worker --worker-id blender --request C:/path/blender-request.json
python -m darkness package --package-id goblin.v1 --candidate-id darkness-canonical-short-biped-v1 `
  --source C:/path/validated-output --output C:/path/package `
  --qualification tools/asset_forge_darkness/qualifications/canonical-short-biped-v1.json --mode research
```

`mesh-check`, `glb-component-audit`, and the Blender worker never overwrite their source or an existing evidence
directory. The Blender worker emits source/candidate checkpoints, fixed-view comparisons, a separate GLB, and a
re-import validation report, but it does not grant the required human approval or promote the candidate. For the
current TripoSG goblin, the original generated geometry is the locked visual source; the generic procedural fitted-v2
package is negative technical evidence, not the visual master to continue.

Machine-specific worker commands belong in ignored `config.local.json`; use `config.example.json` as the template.

Run tests from the EmberDefense root with a Python environment containing the project dependencies:

```powershell
python -m pytest tools/asset_forge_darkness/tests -q
```
