# Ember Defense Asset Forge

Asset Forge turns owned Blender masters into deterministic sprites for the portrait 2.5D battlefield. A character rig
locks identity and motion across animation frames; the same renderer also handles unrigged buildings, equipment, and
props. Blender fixes camera, light, scale, and baseline. AI image tools are optional concept/texture helpers, never the
animation source of truth.

## Asset Forge Studio quick start

The reusable product layer is game-neutral. Create a workspace and an asset card:

```powershell
.\scripts\setup_asset_forge.ps1
.\scripts\forge.ps1 init --workspace tools/asset_forge/workspaces/my-game --project-id my_game --name "My Game"
.\scripts\forge.ps1 new --workspace tools/asset_forge/workspaces/my-game --asset-id iron_guard --name "Iron Guard" --kind character --role "shield tank readable at phone size"
.\scripts\forge.ps1 prompt --workspace tools/asset_forge/workspaces/my-game --asset-id iron_guard
.\scripts\forge.ps1 serve --workspace tools/asset_forge/workspaces/my-game
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
.\scripts\forge.ps1 register-model --workspace tools/asset_forge/workspaces/my-game --model-id sdxl-base-1.0 --file tools/asset_forge/runtime/comfyui/models/checkpoints/sd_xl_base_1.0.safetensors --reviewer <name>
```

Generate a provenance-tracked candidate:

```powershell
.\scripts\forge.ps1 generate --workspace tools/asset_forge/workspaces/my-game --asset-id iron_guard --mode turnaround --seed 1001
```

The core workflow uses only built-in ComfyUI nodes. Custom nodes are optional and may not enter a production workflow
until their source, version, license, and dependency snapshot are recorded.

Approve a selected concept by hash, optionally create a mesh seed, then audit/render the owned Blender master:

```powershell
.\scripts\forge.ps1 approve --workspace tools/asset_forge/workspaces/my-game --asset-id iron_guard --stage concept --artifact <selected.png> --reviewer <name>
.\scripts\setup_local_ai.ps1 install-triposr -AcceptTripoSrLicense
.\scripts\forge.ps1 mesh-seed --workspace tools/asset_forge/workspaces/my-game --asset-id iron_guard --input <selected.png>
.\scripts\forge.ps1 audit-master --config <character-config.json> --report <master-audit.json>
.\scripts\asset_forge.ps1 render -CharacterConfig <character-config.json>
.\scripts\forge.ps1 qa-sheets --config <character-config.json> --report <sheet-qa.json> --contact-sheet <review.png>
```

Export is blocked until sheet and license artifacts have current hash-bound approvals. Existing engine files with a
different hash are never overwritten silently.

Before the license-stage approval, run `forge.ps1 compliance`. It rejects unregistered model filenames, changed or
missing generated outputs, incomplete non-original input records, and assets whose license state is not
`original-confirmed` or `mixed-reviewed`.

## One-time setup

From the repository root:

```powershell
.\scripts\setup_asset_forge.ps1
.\scripts\asset_forge.ps1 doctor
```

The setup creates `tools/asset_forge/.venv` and installs only Pillow. Blender 5.x is discovered from `PATH` or the
standard Windows install folders. ComfyUI is optional and is not installed by this script.

## Prove the pipeline

```powershell
.\scripts\asset_forge.ps1 fixture
```

This creates a small animated Blender fixture, renders `north/south/east/west`, packs sheets, and validates every
output. Results go to `artifacts/asset-forge-fixture/` and are not game art.

## Start from the owned humanoid rig

```powershell
.\scripts\asset_forge.ps1 template
```

This generates `artifacts/asset-forge-template/source/humanoid_template.blend` from repository-owned Python, then
renders and validates its four-direction idle, walk, attack, shoot, hit, and death actions. Open that file in Blender,
replace or reshape the neutral construction mesh, adjust equipment/materials, and save the approved master under
`tools/asset_forge/sources/<unit>.blend`. This removes the blank-file, rig, action-naming, camera, and export setup work.

## Render a real character

1. Put a rigged `.blend`, `.fbx`, or `.glb` under `tools/asset_forge/sources/`.
2. Copy `characters/footman.character.json` and set `source`, `animation_object`, and the action aliases.
3. Confirm the actions exist in Blender: idle, walk, attack/shoot, hit, and death.
4. Run:

```powershell
.\scripts\asset_forge.ps1 render -CharacterConfig tools/asset_forge/characters/footman.character.json
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
