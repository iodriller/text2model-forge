# Legacy sprite pipeline (assetforge)

This is the original directional-sprite renderer this repo grew from, predating the `darkness` compiler. It is still
used by the D8/D9 surface and sprite stages. For an overview of the whole tool, start with the repo root
[README](../README.md). Examples below use the generic `example_biped` fixture shipped in this repo; swap in your
own character configs and Blender masters under `sources/`.

Asset Forge turns owned Blender masters into deterministic sprites for a 2.5D battlefield. A character rig
locks identity and motion across animation frames; the same renderer also handles unrigged buildings, equipment, and
props. Blender fixes camera, light, scale, and baseline. AI image tools are optional concept/texture helpers, never the
animation source of truth.

## Asset Forge Studio quick start

The reusable product layer is game-neutral. Create a workspace and an asset card:

```powershell
.\scripts\setup_asset_forge.ps1
.\scripts\forge.ps1 init --workspace workspaces/my-game --project-id my_game --name "My Game"
.\scripts\forge.ps1 new --workspace workspaces/my-game --asset-id iron_guard --name "Iron Guard" --kind character --role "shield tank readable at phone size"
.\scripts\forge.ps1 prompt --workspace workspaces/my-game --asset-id iron_guard
.\scripts\forge.ps1 serve --workspace workspaces/my-game
```

Studio opens at `http://127.0.0.1:8765`. It binds only to loopback, queues one GPU job at a time, and uses the same
service layer as the CLI.

Set up the isolated local concept engine after reviewing the model license:

```powershell
.\scripts\setup_local_ai.ps1 detect
.\scripts\setup_local_ai.ps1 install-comfyui
.\scripts\setup_local_ai.ps1 download-sdxl -AcceptSdxlLicense
.\scripts\start_asset_forge_comfyui.ps1
.\scripts\forge.ps1 doctor
.\scripts\forge.ps1 register-model --workspace workspaces/my-game --model-id sdxl-base-1.0 --file runtime/comfyui/models/checkpoints/sd_xl_base_1.0.safetensors --reviewer <name>
```

Generate a provenance-tracked candidate:

```powershell
.\scripts\forge.ps1 generate --workspace workspaces/my-game --asset-id iron_guard --mode turnaround --seed 1001
```

The core workflow uses only built-in ComfyUI nodes. Custom nodes are optional and may not enter a production workflow
until their source, version, license, and dependency snapshot are recorded.

Approve a selected concept by hash, optionally create a mesh seed, then audit/render the owned Blender master:

```powershell
.\scripts\forge.ps1 approve --workspace workspaces/my-game --asset-id iron_guard --stage concept --artifact <selected.png> --reviewer <name>
.\scripts\setup_local_ai.ps1 install-triposr -AcceptTripoSrLicense
.\scripts\forge.ps1 mesh-seed --workspace workspaces/my-game --asset-id iron_guard --input <selected.png>
.\scripts\forge.ps1 audit-master --config <character-config.json> --report <master-audit.json>
.\scripts\asset_forge.ps1 render -CharacterConfig <character-config.json>
.\scripts\forge.ps1 qa-sheets --config <character-config.json> --report <sheet-qa.json> --contact-sheet <review.png>
```

Export is blocked until sheet and license artifacts have current hash-bound approvals. Existing engine files with a
different hash are never overwritten silently.

Before the license-stage approval, run `forge.ps1 compliance`. It rejects unregistered model filenames, changed or
missing generated outputs, incomplete non-original input records, and assets whose license state is not
`original-confirmed` or `mixed-reviewed`.

## Creature DNA: goblins, ogres, and dragons

Do not solve a creature by repainting a human frame sequence.  Creature anatomy is a
versioned Blender input and must pass a geometry-only proof before texture baking:

```powershell
.\scripts\create_creature_proof.ps1 `
  -CharacterConfig characters/example_biped.character.json
```

The command builds a disposable master under `artifacts/asset-forge-morphology/`,
validates all semantic landmarks and action-wide posture coverage, renders only idle
and attack, and creates `<unit>_morphology_proof.png`.  It does not alter Unity and
does not consume ComfyUI/GPU diffusion time.

After a full build, rerender a tuned action without discarding all other frames:

```powershell
.\scripts\asset_forge.ps1 render `
  -CharacterConfig characters/example_biped.character.json `
  -Actions attack
```

The incremental path deletes and replaces only the named action folders, then
repacks and validates the complete sheet set.  A normal render without `-Actions`
still performs a clean deterministic rebuild.

Species profiles live in `creatures/`; family contracts live in
`creatures/families/`.  Ogre-like bipeds reuse
`biped_humanoid_v1` with different proportions.  Dragons must use the separate
`winged_quadruped_v1` contract and cannot publish while that family remains marked
`contract_only`.  See `DESIGN/creature_dna_pipeline.md`.

## Genesis: create anatomy without a donor pack

Genesis is the second mesh-acquisition branch. It generates a semantic, watertight
shape target from a versioned specification while explicitly preventing that voxel
mesh from being mistaken for production topology:

```powershell
.\scripts\asset_forge.ps1 genesis-target `
  -GenesisSpec genesis/families/short_biped_v1.family.json
```

The command validates the spec, fuses its volumes in Blender, writes a preview and a
hash-bound mesh report under `artifacts/asset-forge-genesis/<id>/`, and verifies that
the target is one watertight component. The report remains
`production_eligible=false` and names `canonical_topology_fit` as the required next
stage. The target describes anatomy; it is never skinned, animated, or published.

Build and validate the complete canonical engineering proof for either registered
family with the same command:

```powershell
.\scripts\asset_forge.ps1 genesis-build `
  -GenesisSpec genesis/families/short_biped_v1.family.json

.\scripts\asset_forge.ps1 genesis-build `
  -GenesisSpec genesis/families/winged_quadruped_v1.family.json
```

This adds a family contract, watertight UV master, semantic rig, normalized weights,
required actions, foot/grip/reach/intersection audits, and a four-panel review board.
The canonical validator remains `production_eligible=false` until surface bake, sheet
QA, license review, and owner art approval pass.

See `DESIGN/asset_forge_genesis_plan.md` for the verified two-branch architecture,
motion contracts, GPU/tool decisions, and staged implementation plan.

## Human motion storyboard prototype (not production art)

The pose-map/ComfyUI path is retained only for exploring humanoid poses. Testing showed that reference
img2img still changes silhouette, equipment, and anatomy between frames and cannot provide the
four genuine directions required by the portrait battlefield. Its manifest is therefore marked
`production.eligible=false`; the CLI and Unity importer refuse to promote or copy it as final art.
See `DESIGN/asset_forge_animation_plan.md` for the failure record and production decision.

```powershell
.\scripts\setup_local_ai.ps1 download-animation-models -AcceptSdxlLicense
.\scripts\forge.ps1 register-model --workspace <ws> --model-id dreamshaper-xl-v2-turbo --file runtime/comfyui/models/checkpoints/dreamshaper_xl_v2_turbo.safetensors --reviewer <name>
.\scripts\forge.ps1 register-model --workspace <ws> --model-id controlnet-openpose-sdxl-xinsir --file runtime/comfyui/models/controlnet/controlnet_openpose_sdxl_xinsir.safetensors --reviewer <name>
.\scripts\forge.ps1 register-model --workspace <ws> --model-id controlnet-depth-sdxl-xinsir --file runtime/comfyui/models/controlnet/controlnet_depth_sdxl_xinsir.safetensors --reviewer <name>
.\scripts\forge.ps1 poses --workspace <ws> --preview          # editable pose pack + control-image previews
.\scripts\forge.ps1 animate --workspace <ws> --asset-id footman --actions idle,walk,attack,hit,death --reference <approved-concept.png> --seed 3001
.\scripts\forge.ps1 pack-sheets --workspace <ws> --asset-id footman   # prototype review only
```

For this repository, `download-animation-models` registers all three installed hashes automatically when the Asset
Forge environment and workspace already exist. The explicit commands remain useful for a portable/new workspace.

Use these outputs to discuss timing or pose ideas, then author the motion on one owned Blender
master. `create-unit --to-unity` intentionally fails with migration instructions.

## One-time setup

From the repository root:

```powershell
.\scripts\setup_asset_forge.ps1
.\scripts\asset_forge.ps1 doctor
```

The setup creates `.venv` and installs only Pillow. Blender 5.x is discovered from `PATH` or the
standard Windows install folders. ComfyUI is optional and is not installed by this script.

## One-command production character

Copy a `.factory.json` and `.character.json`, adjust proportions, palette, equipment, actions,
sample frames, contact phases, and output paths, then run:

```powershell
.\scripts\asset_forge.ps1 build -CharacterConfig characters/example_biped.character.json
```

The command creates the original rigged `.blend`, renders every configured action in genuine
`north/south/east/west`, packs fixed-camera sheets without per-frame scaling, rejects clipping,
static attacks, missed hit poses, bad death silhouettes, and missing phases, creates review GIFs,
and hash-locks the master/QA/config in a staged schema-2 manifest. It does not replace Unity art.

After the owner visually accepts the generated board/GIF, bind approval to those exact bytes and publish:

```powershell
.\scripts\forge.ps1 approve --workspace asset_sources/ember-defense --asset-id footman --stage sheets --artifact asset_sources/ember-defense/assets/footman/reports/production-review/footman_acceptance_board.png --reviewer <name>
.\scripts\asset_forge.ps1 publish -CharacterConfig characters/example_biped.character.json
```

`publish` refuses stale approvals by checking the acceptance-board SHA-256. The editor preparation workflow applies
published production units after the legacy placeholder roster.

Current review examples:

- `asset_sources/ember-defense/assets/footman/reports/production-review/footman_all_actions.gif`
- `asset_sources/ember-defense/assets/footman/reports/production-review/footman_gameplay_transition.gif`
- `asset_sources/ember-defense/assets/goblin/reports/production-review/goblin_all_actions.gif`
- `asset_sources/ember-defense/assets/goblin/reports/production-review/goblin_gameplay_transition.gif`

### Deterministic surface-overpaint contract

Per-frame diffusion is never allowed to own motion, silhouette, equipment, alpha, or heraldry. When an
`overpaint` block is enabled, Blender renders beauty, true depth, and equipment-protection passes. Asset Forge then:

1. paints action frames together with one persistent hash-recorded style anchor;
2. constrains the optional surface treatment with the real Blender depth pass;
3. restores every protected sword, shield, club, and emblem pixel from the authored master;
4. writes chunks to a resumable staging area, leaving source frames unchanged after interruption;
5. promotes the complete set atomically only when every required mask and protected pixel passes;
6. records model, anchor, inputs, workflow, protection coverage, and result in schema-3 provenance;
7. blocks production packaging if that provenance is missing, old, or failing.

This is the zero-drift boundary: AI can suggest painted surface character, while the rigged master remains the
authority for identity-critical geometry. Re-running the direct `assetforge overpaint` command resumes matching
staged chunks. Running `asset_forge.ps1 build` intentionally starts a clean deterministic render.

### Texture master bake (paint once, render forever)

When a character config has an enabled `texture_master` block, the paint moves from frames to the surface and
per-frame diffusion is retired for that unit:

1. `bake-master` renders 6 orbit + 2 elevated canonical views of the master (flat neutral light, true depth);
2. all views plus the unit's persistent style anchor ride ONE depth-controlled SDXL img2img call as a grid;
3. Blender projects the painted views back onto every mesh (facing^2 x depth-visibility x alpha weighting over a
   freshly smart-unwrapped `AssetForgeBake` UV) and bakes a texture atlas per mesh; texels no view can see fall
   back to a diffuse-color bake of the original authored materials;
4. the result is saved as `<master>_baked.blend` with packed atlases plus unit icon/portrait renders, and
   `asset_forge.ps1 build` renders all sheets from it directly — no overpaint stage, no depth/equipment passes,
   no protection machinery, because drift is structurally impossible when every frame samples one painted surface.

Consistency this way is mathematical, not statistical, and a full 4-direction rebuild costs minutes, which is what
makes automated iteration (below) affordable.

```powershell
.\scripts\asset_forge.ps1 bake -CharacterConfig characters/example_biped.character.json   # force rebake
.\scripts\asset_forge.ps1 build -CharacterConfig characters/example_biped.character.json  # bake if stale + full chain
```

### Automated art direction (critic)

Every `build` must pass `assetforge critique` after mechanical QA: masked tone inside the dark-fantasy band, muted
saturation, interior edge energy at 96 px gameplay size, hue coherence between consecutive frames, and palette
identity of every sheet against idle/south. Thresholds live in `assetforge/critic.py` (override per unit via a
`critic` block). The judgment half of the standard lives in `RUBRIC.md`; the reviewing agent
applies it to each acceptance board before the owner ever sees a candidate, and every owner rejection is appended
there as a permanent rule.

## Prove the pipeline

```powershell
.\scripts\asset_forge.ps1 fixture
```

This creates a small animated Blender fixture, renders `north/south/east/west`, packs sheets, and validates every
output. Results go to `artifacts/asset-forge-fixture/` and are not game art.

## Render the repository-authored humanoid fixture

```powershell
.\scripts\asset_forge.ps1 template
```

This generates `artifacts/asset-forge-template/source/humanoid_template.blend` from repository-owned Python, then
renders and validates its four-direction idle, walk, attack, shoot, hit, and death actions. Its body is made from
separate rigid bone-parented objects: it proves cameras, action naming, sheet packing, and rendering, but it is **not**
an animation-safe continuous/skinned canonical topology. Do not use it as the Genesis family template or claim its
2D render proof validates joint deformation.

## Render a real character

1. Put a rigged `.blend`, `.fbx`, or `.glb` under `sources/`.
2. Copy `characters/example_biped.character.json` and set `source`, `animation_object`, and the action aliases.
3. Confirm the actions exist in Blender: idle, walk, attack/shoot, hit, and death.
4. Run:

```powershell
.\scripts\asset_forge.ps1 render -CharacterConfig characters/example_biped.character.json
```

The command renders intermediate frames, packs Unity-ready horizontal strips, and validates dimensions, alpha,
clipping, frame count, and baseline stability. To activate the current heroic sprite set in Unity:

```powershell
.\scripts\asset_forge.ps1 activate-unity
```

For a static prop/building, copy `assets/static_prop.asset.example.json`, omit `animation_object`, use one animation
entry with `"actions": []` and `"frames": 1`, and choose one or four directions. Animated VFX or props use the same
named-action path as characters. UI frames, buttons, bars, and the Ember icon are separately generated at runtime by
`UiTheme`, so they have no atlas dependency.

## Source ownership

The required path is an original Ember Defense `.blend` master. No character pack, Mixamo account, cloud API, or AI
model is required by the renderer. External rigging, animations, CC0 components, commissioned art, or image-to-3D may
accelerate a master, but each is optional, must be reviewed, and must be logged before use.

Approved masters are source assets and must be committed (use Git LFS if repository size becomes a problem). Every
imported model, animation, texture, or generated sheet must be recorded in `assets_manifest/license_log.md`.

See `DESIGN/asset_forge_product_plan.md` for the product contract, stage gates, workspace layout, and future-game
adapter boundary.

## Art direction

Use “original chunky heroic dark fantasy” in production prompts: broad readable silhouettes, tarnished steel, worn
leather, muted cool colors, sparse ember-orange accents, painterly/cel-shaded materials, and a slightly elevated 3/4
camera. “Warcraft” is only a high-level taste reference from the project owner; prompts and assets must not copy its
characters, armor, factions, symbols, shapes, or textures.

## Optional local concept workflow

ComfyUI's official Windows portable build supports NVIDIA RTX cards. On this 8 GB GPU, keep concept generations near
1024 px and use low-VRAM settings. Generate a four-view turnaround concept first, then build/texture the 3D source.
Do not generate independent animation frames with diffusion; they drift in identity and timing.

Prompt scaffold:

```text
Use case: stylized-concept
Asset type: four-view game character turnaround used to build a rigged 3D model
Primary request: <unit role and equipment>
Subject: the same full-body character in north/back, south/front, east, and west views
Style/medium: original chunky heroic dark fantasy; painterly cel-shaded game character
Composition/framing: orthographic turnaround, neutral stance, equal scale, feet on one baseline
Lighting/mood: top-left key light, deep readable shadows, restrained ember highlights
Constraints: identical armor, proportions, equipment, palette, and silhouette in every view; no logos; no text;
  no copyrighted characters; no copied game-IP symbols; plain neutral background
```
