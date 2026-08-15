# Running the whole pipeline free, locally, on an 8 GB GPU

Every model here is free and open source, and the whole chain fits on a
consumer laptop GPU. No API keys, no subscriptions, no cloud.

## Why this stack

The pipeline was originally qualified against an RTX 5090 (32 GB). On an 8 GB
card the original D2 worker cannot run at all — TRELLIS.2-4B needs 16–24 GB.
That single mismatch is what made "text to 3D" unreachable, not any bug in the
orchestration.

| Stage | Tool | VRAM | License |
|---|---|---|---|
| D1 text → 2D | SDXL / DreamShaper in ComfyUI | ~6 GB | open weights |
| **D2 2D → 3D** | **Hunyuan3D-2 (native ComfyUI)** | **~5 GB mini, ~6 GB std** | Tencent community licence — see below |
| D4–D6 rig + skin | [UniRig](https://github.com/VAST-AI-Research/UniRig) | ~8 GB | MIT |
| D7 motion | Donor retarget (Quaternius/Mixamo CC0) | CPU + Blender | CC0 |
| D8–D9 surface, sprites | Blender | CPU/GPU | GPL |

Total peak VRAM is set by whichever single stage is running, not the sum —
stages run one at a time, and the Studio releases ComfyUI's models between
them (`Free ComfyUI memory` on the run page).

## 1. ComfyUI

Install ComfyUI and start it on the default port:

```powershell
python main.py --listen 127.0.0.1 --port 8188 --lowvram
```

`--lowvram` matters on 8 GB. The Studio expects `http://127.0.0.1:8188`.

## 2. Models

**Concept art (D1)** — any SDXL checkpoint into `ComfyUI/models/checkpoints/`.
The default expects `dreamshaper_xl_v2_turbo.safetensors`; change
`[studio].checkpoint` in a profile if you use another.

**Image-to-3D (D2)** — Hunyuan3D-2, also into
`ComfyUI/models/checkpoints/`:

- `hunyuan3d-dit-v2-mini.safetensors` — ~5 GB, the default here
- `hunyuan3d-dit-v2.safetensors` — ~6 GB, better shapes, used by the
  `advanced` profile

ComfyUI supports these natively. There is no custom node to install, and no
`--extra-model-paths` juggling. Meshes land in `ComfyUI/output/mesh` and the
Studio downloads them automatically.

Verify ComfyUI sees them:

```powershell
python -m darkness config show --profile simple    # what the run will ask for
```

## 3. Rigging (optional at first)

UniRig is MIT-licensed and rigs an arbitrary mesh. It needs its own
environment; see its README. Until it is installed, D4 falls back to the
procedural short-biped worker, which produces a valid rig but ignores your
mesh's actual proportions.

## 4. Point the Studio at it

```powershell
copy machine.example.toml config.local.toml
```

Fill in your Blender path. `workspace_root` is where runs are written.

```powershell
python -m darkness studio --workspace C:/AssetForgeRuns --open-browser
```

## What you get

Describe your asset, then:

1. **D1** renders two concept candidates. Retry, edit the prompt, or pin a
   seed until one is right. This is cheap and local — iterate here.
2. **Approve one.** That exact image becomes the 3D input. Nothing is
   re-rendered behind your back.
3. **D2** feeds it to Hunyuan3D and returns a mesh, then gates it on real
   geometry (vertex/face counts, watertightness, connected components).
4. **D3 onward** cleans up, rigs, skins, animates, paints, and packages,
   pausing at D4, D7, D8 and D10 for your approval.

## Honest limits

- **Hunyuan3D's licence is a community licence, not MIT/Apache.** It excludes
  some territories (EU/UK/South Korea) from *output* use. Fine for personal
  and local work; check it before shipping commercially. The lineage system
  records this, and `export_policy.py` will block a release export that
  depends on it. That refusal is deliberate, not a bug.
- **ComfyUI's native Hunyuan3D support is shape-only.** Texture generation is
  not part of it; the full shape+texture path needs ~12 GB. Surface work
  happens at D8 instead, which is where this pipeline already does it.
- **8 GB is the real ceiling.** The concept model and the 3D model do not fit
  in VRAM simultaneously. Use the run page's memory-release control between
  stages if ComfyUI starts swapping.
- **Language-model stages want a bigger model than 8 GB fits.** The qualified
  reviewer is a 27B model. Smaller local models fail the typed contracts
  loudly rather than silently — which is correct, but it means D0 and the
  review stages want either a larger machine or a hosted endpoint.
