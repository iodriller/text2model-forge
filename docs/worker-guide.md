# Worker guide

Text2Model Forge keeps model runtimes outside the compiler process. Each worker
is declared by a portable manifest, invoked through a typed request/response
file contract, and evaluated against separate qualification evidence.

## Sources of truth

- `resources/workers/*.json` — identity, lifecycle, command template, licenses,
  hardware assumptions, and blockers.
- `resources/qualifications/*.json` — observed versions, hashes, hardware,
  transport results, semantic findings, and unresolved limits.
- `machine.example.toml` — portable configuration template.
- ignored `config.local.toml` — executable paths and service URLs for one
  machine.
- `src/text2model_forge/schemas.py` — worker request/response contracts.

README claims must not upgrade a worker beyond its manifest and qualification
records.

## Inspect this machine

```powershell
python -m text2model_forge workers
python -m text2model_forge doctor --deep
python -m text2model_forge config show --profile advanced
```

`workers` reports both the declared lifecycle and current local readiness. A
worker can be `research_ready` in its manifest and still be unavailable because
its executable or service is not configured on this machine.

## Lifecycle meanings

| Lifecycle | Meaning |
|---|---|
| `discovered` | Contract or adapter exists, but execution/quality evidence is incomplete |
| `research_ready` | Bounded research use has recorded evidence and explicit blockers |
| `available` | The declared portable contract is usable within its stated limits |
| `blocked` | License, safety, or technical uncertainty prevents use |

None of these labels alone authorizes a release export. Export policy also
checks the complete artifact lineage and territory/output terms.

## Request lifecycle

1. Resolve the manifest and machine binding.
2. Validate input paths and the typed request.
3. Acquire the required GPU lease, if any.
4. Execute in an isolated subprocess with bounded time and captured logs.
5. Validate the response and every declared artifact.
6. Apply hard structural gates.
7. Present evidence for AI analysis and human review where required.
8. Promote only after the applicable gate and lineage policy pass.

Adapters under `resources/adapters/` must not import the Studio coordinator.
They accept the external worker contract and write a response file; orchestration
depends on them, never the reverse.

## Running a worker

Use a request builder where one exists, then invoke the configured worker:

```powershell
python resources/adapters/build_blender_request.py `
  --input C:/input/candidate.glb `
  --output-directory C:/evidence/blender `
  --out C:/requests/blender.json `
  --job-id blender.example.v1 `
  --operation-id blender.repair

python -m text2model_forge run-worker `
  --worker-id blender `
  --request C:/requests/blender.json
```

Other maintained builders include:

- `resources/adapters/build_retopology_request.py`
- `resources/adapters/build_instant_meshes_request.py`
- `resources/adapters/build_smoke_request.py`
- `resources/adapters/build_unity_smoke_bundle.py`

Diagnostic compiler commands are non-destructive:

```powershell
python -m text2model_forge mesh-check --input C:/input/model.obj --output C:/evidence/mesh
python -m text2model_forge glb-component-audit --input C:/input/model.glb --output C:/evidence/components
```

## Current backend roles

The exact lifecycle and blockers remain in the manifests. At a high level:

| Role | Implemented candidates | Important boundary |
|---|---|---|
| 2D generation/review | ComfyUI, Qwen, SDXL, Ollama-compatible reviewer | Checkpoint revisions and terms must be recorded |
| Image-to-3D | TRELLIS.2, InstantMesh, TripoSG, Hunyuan3D candidates | Live quality is incomplete; some backends have license or platform constraints |
| Geometry repair | Deterministic mesh kernel, Blender, AutoRemesher, Instant Meshes | Structural success is not proof of visual identity |
| Canonical fallback | Repository-owned short-biped worker | Research baseline, not a universal reconstruction model |
| Motion | Donor retargeting and research candidates | Donor license and deformation review are mandatory |
| Surface/sprites | ComfyUI multiview plus Blender renderer | Approved geometry remains the identity source |
| Runtime validation | Optional isolated Unity smoke project | Engine validation is not required for core glTF checks |

The WSL2 setup notes for compatible geometry workers are maintained under
`docs/environments/`. They are environment-specific supplements, not promises
that a worker is available.

## Qualification rules

A qualification record should contain:

- exact source revision and executable/package hashes;
- exact model/checkpoint revision and relevant license terms;
- operating system, GPU, driver, runtime, and dependency versions;
- input artifact hashes;
- transport/schema outcome and observed timing;
- structural metrics and human-visible semantic findings;
- reproducibility evidence or an explicit statement that it is unproven;
- blockers that prevent broader use.

Never infer visual quality from a lower connected-component count, a successful
subprocess exit, or a fake-provider test. Never remove a blocker merely to make
a worker selectable.

## Adding a worker

1. Add or extend a typed request/response field in the shared schema.
2. Create one adapter in `resources/adapters/`.
3. Add a manifest under `resources/workers/` with a conservative lifecycle.
4. Add machine binding examples without committing real paths.
5. Add deterministic contract tests, including failure and cancellation paths.
6. Run a real smoke qualification and add a separate evidence record.
7. Update preflight with every new cross-stage assumption and a concrete remedy.
8. Update third-party/license documentation before enabling release use.

Model weights and generated artifacts remain outside Git. A successful local
experiment is evidence to record, not a reason to weaken lineage or promotion
gates.
