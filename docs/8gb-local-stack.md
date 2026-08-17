# Experimental local 8 GB path

This guide records a real 8 GB-class text-to-concept-to-geometry run and the
settings derived from it. It does **not** qualify the complete D0–D10 pipeline
on 8 GB, and it does not mean every optional model is open source. The control
plane is local and needs no API key; each selected model has separate terms.

## Automated setup on Windows, Linux, and macOS

The repository-root launcher automates the core environment, Ollama reviewer,
ComfyUI runtime, Qwen Image or SDXL concept path, Hunyuan3D, Blender
discovery/installation, local config, smoke check, and Studio startup. Windows
uses the official ComfyUI portable runtime:

```powershell
.\text2model-forge.ps1
```

Linux and macOS use ComfyUI's supported manual layout in its own isolated
environment:

```bash
bash ./text2model-forge.sh
```

Choose the recommended Qwen stack in the prompt. For an unattended first
install, explicitly accept the separate SDXL and Hunyuan model terms:

```powershell
.\text2model-forge.ps1 -Action install -AiStack qwen -NonInteractive `
  -AcceptSdxlLicense -AcceptHunyuanLicense
.\text2model-forge.ps1 -AiStack existing
```

```bash
bash ./text2model-forge.sh install --ai-stack qwen --non-interactive \
  --accept-sdxl-license --accept-hunyuan-license
bash ./text2model-forge.sh start --ai-stack existing
```

After installation, the normal launcher is idempotent: it skips a current
environment and starts the installed services plus Studio. Use `doctor` for
readiness, `repair` for bounded reinstall attempts, or the platform's
no-elevation option to prohibit its final administrator fallback. The
no-browser option keeps both service and Studio tabs closed. Existing
`config.local.toml` is preserved. Run `docker compose up --build` for the
containerized Studio + reviewer path; the root README describes its persistent
volumes, NVIDIA overlay, loopback binding, and host-native ComfyUI connection.
The manual sections below remain useful for custom locations, specialized
workers, and troubleshooting.

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
| D7 motion | Donor retarget (Quaternius CC0 catalog entry) | CPU + Blender | CC0-1.0 |
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
`ComfyUI/models/checkpoints/`. Take it from
[`Comfy-Org/hunyuan3D_2.0_repackaged`](https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged),
which packages the DiT, CLIP-Vision and VAE into the single file ComfyUI's
`ImageOnlyCheckpointLoader` expects — the `tencent/*` repos ship those parts
separately and will not load:

```powershell
curl -L -o ComfyUI/models/checkpoints/hunyuan3d-dit-v2_fp16.safetensors `
  https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2_fp16.safetensors
```

That is 4.9 GB. Note the lowercase `D` in `hunyuan3D_2.0_repackaged`; the
capitalised spelling is a different, gated repo that returns 401.

**Verified on this hardware:** RTX 3080 Laptop, 8.6 GB. A 768x1024 concept
became a 182,019-vertex knight in about 70 seconds, with the SDXL concept
render taking a further ~26 seconds.

ComfyUI supports these natively. There is no custom node to install, and no
`--extra-model-paths` juggling. Meshes land in `ComfyUI/output/mesh` and the
Studio downloads them automatically.

Verify ComfyUI sees them:

```powershell
python -m text2model_forge config show --profile simple    # what the run will ask for
```

## 3. Rigging (optional at first)

UniRig is MIT-licensed and rigs an arbitrary mesh. It needs its own
environment; see its README. Until it is installed, D4 can use the procedural
short-biped research fixture. That fixture proves contracts and deformation
checks; because it does not preserve the generated mesh's fine identity, it is
not a production fallback.

## 4. Point the Studio at it

```powershell
copy machine.example.toml config.local.toml
```

Fill in your Blender path. `workspace_root` is where runs are written.

```powershell
python -m text2model_forge studio --workspace C:/Text2ModelForgeRuns --open-browser
```

## What you get

Describe your asset, then:

1. **D1** renders two concept candidates. Retry, edit the prompt, or pin a
   seed until one is right. This is cheap and local — iterate here.
2. **Approve one.** That exact image becomes the 3D input. Nothing is
   re-rendered behind your back.
3. **D2** feeds it to Hunyuan3D and returns a mesh, then gates it on real
   geometry (vertex/face counts, watertightness, connected components).
4. **D3 onward** is the experimental half of this profile: cleanup, rigging,
   skinning, motion, surface work, sprites, and runtime validation still need a
   complete live qualification on this hardware.

## Honest limits

- **Hunyuan3D's licence is a community licence, not MIT/Apache.** It excludes
  the EU, UK, and South Korea and imposes additional use and distribution
  conditions. Do not download or run it where its terms do not grant you
  permission. The lineage system records this, and `export_policy.py` blocks a
  globally cleared release export that depends on it.
- **ComfyUI's native Hunyuan3D support is shape-only.** Texture generation is
  not part of it; the full shape+texture path needs ~12 GB. Surface work
  happens at D8 instead, which is where this pipeline already does it.
- **8 GB is the real ceiling.** The concept model and the 3D model do not fit
  in VRAM simultaneously. Use the run page's memory-release control between
  stages if ComfyUI starts swapping. Start ComfyUI with `--lowvram`.
- **The mesh is dense and unwelded.** A real run produced 182k vertices /
  469k faces across many components, because "surface net" emits an
  unmerged surface. That is expected and is exactly what D3's cleanup stage
  exists to fix; it is not a sign the generation failed.
- **Your concept must sit on a plain backdrop.** D2 isolates the subject by
  growing the background inward from the image border, then flattens it to
  white. A busy or cluttered background leaves fragments that the 3D model
  will faithfully reconstruct as slabs around your asset.
- **D0 needs `spec_strategy = "chunked"` on a small model.** The qualified
  reviewer is a 27B model, and the default one-shot spec compile assumes it.
  On 8 GB, see "Compiling the spec on a 7–8B model" below — this is solved,
  not merely a warning.
- **The review stages still want a bigger model than 8 GB fits.** D0 is
  handled by chunking; the *vision* review gates (D1's critic, D2/D3's
  assessors) have not been given the same treatment yet and remain the
  weakest link on a small local model.

## Compiling the spec on a 7–8B model

D0 turns your description into a typed `StudioAssetSpec`: seventeen fields,
two arrays of nested objects, six enums, in one grammar-constrained call.
A 27B model does that fine. Measured here on an RTX 3080 Laptop, an 8B model
does not — `qwen3-vl:8b-instruct` and `gemma3:12b` both took ~119 s and
returned schema-valid JSON with `equipment: []`, which fails the handedness
contract and stops the run at D0.

That is not a knowledge problem, and it is fixable without a bigger model.
Two published results explain why:

- [*Let Me Speak Freely?*](https://aclanthology.org/2024.emnlp-industry.91/)
  (EMNLP 2024) measures that format restriction degrades reasoning, and that
  tighter constraints degrade it further. One giant schema is close to the
  worst case.
- [llama.cpp/Ollama compile the schema to a GBNF grammar](https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output)
  and mask invalid tokens. That guarantees *shape*, never *content* — an
  empty array is perfectly grammatical, which is exactly what came back.

`src/text2model_forge/chunked_spec.py` implements the alternative. Set:

```toml
[studio_defaults]          # in config.local.toml
spec_strategy = "chunked"
llm_timeout_seconds = 600
```

It works in four moves:

1. **Decompose** — one small call per field group, so no call is near the
   model's structured-output limit.
2. **Reason first, constrain second** — judgement-heavy chunks answer in free
   prose, then a second constrained call extracts from that prose. This is
   the paper's own mitigation, and it is what moved a supply crate from
   `environment` to `prop`.
3. **Make emptiness ungrammatical** — `minItems` in the chunk schema survives
   into the grammar, so `[]` becomes unreachable rather than retried. Use the
   *minimum viable* floor (1), never an aspirational one: asked for two
   silhouette phrases when it had one, the model padded the slot with
   `"%20%20%20…"`.
4. **Never make the model spell the contract** — it reports
   `held_in="right_hand"`; Python maps that to
   `side/socket/grip = right / hand_right.grip / palm_and_fingers`. Ids and
   clip names are slugified in code too.

Measured on the same hardware and model, compiling the same knight:

| | monolithic | chunked |
|---|---|---|
| knight | ~119 s, **failed** (`equipment: []`) | **~18 s**, sword right / shield left correct |
| goblin | timed out | ~14 s |
| static crate | — | ~9 s |
| hinged gate | — | ~19 s, 4 movable parts |

The result is validated by the same `StudioAssetSpec` and the same
`_validate_explicit_handedness` check the 27B path uses. It is a different
route to the contract, not a lower bar.
