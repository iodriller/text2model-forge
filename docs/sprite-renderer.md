# Directional sprite renderer

`text2model-sprites` is the optional, lower-level renderer inside Text2Model
Forge. It turns an owned or properly licensed 3D master into deterministic
directional frames, sheets, review boards, and delivery packages. The main
D0–D10 compiler uses parts of this subsystem at D8 and D9; most users should
start in Studio rather than call it directly.

## Boundary

The renderer owns camera consistency, frame packing, alpha, deterministic QA,
hash-bound approvals, and export. It does not create a production-ready master
from an arbitrary image, silently accept unlicensed inputs, or let diffusion
change motion and silhouette ground truth.

Portable example contracts live under:

- `resources/characters/`
- `resources/creatures/`
- `resources/creatures/families/`
- `resources/genesis/families/`
- `resources/assets/`
- `resources/configs/`

Copy examples into a separate workspace before editing them. Generated frames,
models, approvals, and reports belong in that workspace, not in this repository.

## CLI

Install the project, then inspect the current command contract:

```powershell
text2model-sprites --help
text2model-sprites <command> --help
```

The commands are grouped by purpose:

| Purpose | Commands |
|---|---|
| Workspace and briefs | `init`, `new`, `status`, `doctor` |
| Local 2D generation | `prompt`, `workflow`, `generate`, `register-model` |
| Motion prototypes | `create-unit`, `poses`, `animate`, `pack-sheets` |
| Mesh and creature contracts | `mesh-seed`, `validate-morphology`, `morphology-proof`, `validate-genesis-spec`, `validate-genesis-target`, `validate-family-contract` |
| Blender master review | `audit-master`, `validate-canonical-master`, `canonical-review`, `check-deformation` |
| Surface treatment | `overpaint`, `repair-overpaint`, `bake-master` |
| QA and delivery | `critique`, `qa-sheets`, `approve`, `package-production`, `export`, `compliance` |
| Local browser | `serve` |

Typical exploration starts with an isolated workspace:

```powershell
text2model-sprites init --help
text2model-sprites new --help
text2model-sprites prompt --help
text2model-sprites workflow --help
text2model-sprites generate --help
text2model-sprites status --help
```

The CLI help is authoritative for arguments. This guide intentionally avoids
duplicating every flag, because stale copied command lines previously diverged
from the tested parser.

## Production invariants

- Every production source must be original or have recorded redistribution and
  output rights.
- A fixed camera and frame scale own directional consistency.
- Rigged geometry owns silhouette, equipment placement, alpha, and timing.
- Optional diffusion may change surface appearance only within recorded masks
  and depth constraints.
- Approvals bind to artifact SHA-256 values. Editing an artifact makes the
  approval stale.
- Packaging fails when mechanical QA, provenance, compliance, or a required
  human approval is absent.
- Source files are never overwritten as an implicit repair step.

## Surface workflows

Two surface paths exist:

1. `overpaint` treats rendered frames in bounded chunks, restores protected
   pixels, and records the model, workflow, masks, inputs, and outputs.
2. `bake-master` paints canonical views once, projects them back onto the 3D
   surface, and renders all later frames from the same textured master.

The baked-master route is preferred when available because identity remains a
property of one surface rather than a statistical agreement between frames.

## Qualification

Mechanical tests can prove dimensions, hashes, alpha, frame counts, baseline
stability, and contract behavior. They cannot prove that motion looks good or
that a generated character is commercially usable. Human review follows
[`docs/review-rubric.md`](review-rubric.md), and live model claims require the
same qualification discipline as the main worker system.
