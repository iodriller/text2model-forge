# Third-party tools, models, and assets

The Apache-2.0 license in `LICENSE` covers VettedMesh's repository-authored
source and documentation. It does **not** relicense software, model weights,
checkpoints, donor assets, or generated outputs supplied by other parties.

VettedMesh does not commit model weights, Blender, ComfyUI, Ollama, Unity, or
the Quaternius animation library. The launchers download optional components
only after an explicit setup choice and, where required, explicit acceptance
of the component's own terms. Every generated artifact retains backend and
license lineage; release packaging fails closed when that lineage is blocked or
unresolved.

## Integration matrix

| Component | Repository role | Upstream terms / release policy |
|---|---|---|
| Blender | External D3–D10 process | GPL-2.0-or-later; installed separately |
| ComfyUI | External workflow server | Upstream license and each selected custom node/checkpoint apply separately |
| Ollama | Optional local reviewer server | Installed separately under its upstream license |
| SDXL checkpoints | Optional D1/D8 models | The exact checkpoint's Open RAIL or other model terms apply; never infer one checkpoint's terms from another |
| Qwen image/reviewer models | Optional D0/D1/review models | The exact model card and weight license apply; record the model identifier and digest |
| Hunyuan3D-2 | Optional D2 backend | Tencent Hunyuan 3D 2.0 Community License; excluded in the EU, UK, and South Korea and subject to notice/use restrictions. It is never a globally cleared default |
| TRELLIS.2 | Optional D2 backend | Core code/weights are MIT; DINO encoders and NVIDIA rendering dependencies carry separate terms. Native texture paths are not globally/commercially cleared by VettedMesh |
| TripoSG | Optional D2 backend | MIT code and recorded MIT weights; caller-owned foreground masking is required to avoid non-commercial background-removal weights |
| InstantMesh | Optional D2 backend | Apache-2.0 code; checkpoint licensing remains review-required where the model card is not explicit |
| UniRig | Optional D4–D6 backend | MIT; installed separately. Current upstream release status and checkpoint terms must still be recorded |
| Instant Meshes | Optional D3 tool | BSD-3-Clause; installed separately |
| AutoRemesher | Optional D3 tool | MIT; installed separately |
| Quaternius Universal Animation Library | Optional D7 donor | CC0-1.0; downloaded separately and recorded in `motion_library/catalog.example.json` |
| Unity | Optional D10 validator | Unity's separate editor/software terms apply; no Unity binaries or packages are distributed here |

The authoritative machine-readable lifecycle and release decision for workers
lives under `workers/`; observed hardware evidence lives under
`qualifications/`. A worker marked `review_required` or `blocked` must not be
promoted merely because its command executes successfully.

## Shipped binary fixtures

The two PNG files under `fixtures/concepts/` are repository-owner qualification
fixtures. They are included for deterministic testing and documentation, not as
production art. To the extent the repository owner holds rights in those files,
they are made available under CC0-1.0. Their presence does not grant rights in
any model used during the original qualification run.

## Output rights

VettedMesh does not guarantee that a generated output is non-infringing or
commercially usable. Users are responsible for prompts, input rights, model
terms, applicable law, and human review. The lineage/export-policy code is a
technical enforcement mechanism, not legal advice.
