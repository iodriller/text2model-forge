# WSL2 + geometry-worker setup runbook (2026-07-12)

Everything below was discovered live, once, on this machine (RTX 5090, Windows 11). It replaces
`trellis2-wsl2-cu124.setup.md` as the authoritative record — that file's title is now wrong (12.4 does **not**
actually work for compute; see §3). Follow this file top to bottom on a fresh machine and every failure recorded
here should already be avoided.

## 0. Before you start

- Confirm the real GPU with `nvidia-smi` on the **Windows** side first. Do this before touching WSL2 — an earlier
  session wasted time chasing a phantom "RTX 6090" that was just a typo for the actual RTX 5090.
- If you have exactly one other Python version installed system-wide (e.g. an old 3.10 next to 3.12), decide up
  front whether to keep it. We removed a stray Python 3.10 install because nothing on the priority list needed it;
  don't repeat that removal unless you've checked nothing (like an old TripoSG venv) depends on it first.

## 1. Install WSL2 (needs your manual action — cannot be scripted from here)

```powershell
wsl --install -d Ubuntu-24.04
```

Run this in an **elevated** PowerShell. It needs a reboot. After reboot, run `wsl -d Ubuntu-24.04` yourself in an
**interactive** terminal — first boot prompts for a UNIX username/password and there is no way to answer that
prompt from a non-interactive script/agent. Pick a username and password and keep them; every command below assumes
they already exist.

Verify GPU passthrough works with zero extra driver install (the existing Windows NVIDIA driver is enough):

```bash
nvidia-smi   # run inside the WSL2 shell; should list your GPU with no separate Linux driver installed
```

**Known trap:** WSL2 auto-shuts-down the VM after a short idle period, and `/tmp` is tmpfs — it is wiped on every
restart. Anything you need to survive between separate `wsl.exe` invocations (constraints files, staged clones)
must live under `~` (the WSL2 home directory), not `/tmp`.

## 2. CUDA toolkit — install 12.8, not 12.4

**Do not stop at CUDA 12.4.** It compiles but silently produces binaries with no Blackwell (`sm_120`) code path, and
even a version of torch that *reports* CUDA as available will fail every real kernel launch (see §5). Install both
if you want the smaller 12.4 footprint for something that doesn't need Blackwell, but you need 12.8 for anything
that will actually run inference on an RTX 5090.

```bash
sudo apt update
sudo apt install -y build-essential git wget
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

**Known trap:** the full `cuda-toolkit-12-8` meta-package pulls in `nsight-systems`, which depends on `libtinfo5` —
a package Ubuntu 24.04's repos no longer carry. Install the minimal build components directly instead; they are
everything a PyTorch CUDA extension actually needs:

```bash
sudo apt install -y cuda-minimal-build-12-8 cuda-libraries-dev-12-8 cuda-nvml-dev-12-8 cuda-nvtx-12-8
```

Add to `~/.profile` (not `~/.bashrc` — see the note in §7):

```bash
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/usr/lib/wsl/lib:$LIBRARY_PATH"
```

**Known trap:** `/usr/lib/wsl/lib` (WSL2's GPU passthrough shim, containing `libcuda.so`, the CUDA *driver* library)
is not on the default linker search path. Any CUDA extension that links `-lcuda` fails with `cannot find -lcuda`
without `LIBRARY_PATH` pointed at it. This is separate from `LD_LIBRARY_PATH` (needed at runtime) — you need both.

## 3. The one finding that matters most: verify real compute, not `cuda_available`

```python
import torch
torch.cuda.is_available()          # True even when this GPU cannot run a single kernel
torch.ones(4, device='cuda') * 2   # THIS is the real test
```

We hit `RuntimeError: CUDA error: no kernel image is available for execution on the device` with **both**
`torch==2.1.0` and `torch==2.6.0+cu124`, despite `is_available()` reporting `True` for both. Only
`torch==2.7.1+cu128` actually executes a kernel on this GPU. **Always run the real tensor op, not just
`is_available()`, before trusting any new environment.**

If you already built CUDA extensions against an older torch and then upgrade: **rebuild every one of them.** They
are compiled against torch's C++ ABI and silently produce `undefined symbol` errors otherwise (see §6).

## 4. Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
conda init bash
```

**Known trap:** first use of `conda create`/`conda install` prompts to accept Anaconda's channel Terms of Service,
which blocks non-interactive scripts. Accept it once, explicitly, before scripting anything else:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 5. Credentials this project needed (store locally, never in a repo file)

- **Hugging Face token**, for `facebook/dinov3-vitl16-pretrain-lvd1689m` (TRELLIS.2's image encoder, gated by Meta).
  You must (a) accept the license on the model page while logged in, (b) generate a read-scoped token, (c) run
  `huggingface_hub.login(token=...)` inside WSL2 (stores it in `~/.cache/huggingface/token`, WSL2-local — this is
  the correct place for it, not a Windows env var). **Meta's access approval is a real manual review that can take
  up to several days** — budget for that lead time on a fresh machine; it is not a technical step you can skip or
  script around.

## 6. TRELLIS.2 environment (conda env `trellis2`)

```bash
mkdir -p ~/AssetForgeDarknessWorkers && cd ~/AssetForgeDarknessWorkers
git clone https://github.com/microsoft/TRELLIS.2.git
cd TRELLIS.2
git submodule update --init --recursive   # needed for o-voxel/third_party/eigen; easy to forget

conda create -y -n trellis2 python=3.10
conda activate trellis2

# The confirmed-working pair on this GPU. Do NOT use the cu124 pair the
# project's own setup.sh installs by default -- see §3.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

cat > ~/trellis2_constraints.txt << 'EOF'
torch==2.7.1
torchvision==0.22.1
EOF

pip install psutil                          # flash-attn's setup.py needs it; setup.sh doesn't install it
pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir -c ~/trellis2_constraints.txt

mkdir -p /tmp/extensions
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast --no-build-isolation -c ~/trellis2_constraints.txt

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install /tmp/extensions/nvdiffrec --no-build-isolation -c ~/trellis2_constraints.txt

git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
pip install /tmp/extensions/CuMesh --no-build-isolation -c ~/trellis2_constraints.txt

git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
pip install /tmp/extensions/FlexGEMM --no-build-isolation -c ~/trellis2_constraints.txt

cp -r o-voxel /tmp/extensions/o-voxel   # from the TRELLIS.2 checkout root; needs the eigen submodule above
pip install /tmp/extensions/o-voxel --no-build-isolation -c ~/trellis2_constraints.txt
```

### Known traps specific to TRELLIS.2

1. **`--no-build-isolation` alone is not enough for flash-attn.** `pip install flash-attn==2.7.3
   --no-build-isolation` can still fail with `ModuleNotFoundError: No module named 'psutil'` — its `setup.py` needs
   `psutil` installed in the target env first (`setup.sh` does not install it).
2. **Stale wheel cache survives a torch upgrade.** If you rebuild flash-attn after already upgrading torch, pip may
   silently reuse a wheel it cached while building against the *old* torch, producing
   `undefined symbol: _ZN3c105Error...` at import time. Force `--no-cache-dir` whenever you rebuild an extension
   after a torch version change.
3. **`transformers`'s newer major versions break TRELLIS.2's own code.** `setup.sh` installs `transformers`
   unpinned. Version 5.13.1 restructured `DINOv3ViTModel` so its transformer blocks live at `model.model.layer`
   instead of the flat `model.layer` TRELLIS.2's `image_feature_extractor.py` expects, raising `AttributeError:
   'DINOv3ViTModel' object has no attribute 'layer'`. Darkness's `adapters/trellis2_worker.py` monkeypatches
   `DinoV3FeatureExtractor.extract_features` with a shim that checks both layouts — re-check this if you ever
   upgrade `transformers` further.
4. **TRELLIS.2 eagerly constructs a gated, non-commercial background-removal model even when it will never be
   called.** `Trellis2ImageTo3DPipeline.from_pretrained()` unconditionally builds `pipeline.rembg_model` (a
   `BiRefNet` pointed at `briaai/RMBG-2.0` in this pipeline's config), which tries to download RMBG-2.0's config
   immediately — even though `preprocess_image()` only ever calls that object when the input image *lacks* real
   alpha. Fix: pass the input image through with its alpha channel intact (do **not** composite it onto an opaque
   background first — that was our own first mistake and it silently forced the RMBG path), and monkeypatch
   `trellis2.pipelines.rembg.BiRefNet` to a no-op stub before calling `from_pretrained()`. Both are implemented in
   `adapters/trellis2_worker.py`.
5. **`glb.export(..., extension_webp=True)` — the README's own example — can fail** with `module 'PIL._webp' has no
   attribute 'HAVE_WEBPANIM'` (a Pillow/libwebp build mismatch). Export with `extension_webp=False` (plain PNG
   textures, bigger file, same quality) until that's root-caused.
6. **`mesh.split()` can `MemoryError` on TRELLIS.2's dense output.** Its o_voxel-based meshes can be ~1M
   vertices/~900K faces with many duplicate/unwelded vertices. `trimesh`'s `.split(only_watertight=False)` is too
   memory-heavy for that; compute connected components via `networkx` on `mesh.face_adjacency` directly instead,
   and call `mesh.merge_vertices()` first if you want a meaningful (not duplicate-vertex-inflated) count.

## 7. InstantMesh environment (conda env `instantmesh`)

```bash
cd ~/AssetForgeDarknessWorkers
git clone https://github.com/TencentARC/InstantMesh.git
cd InstantMesh

conda create -y -n instantmesh python=3.10
conda activate instantmesh
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128   # NOT the README's 2.1.0/cu121 -- see §3

cat > ~/instantmesh_constraints.txt << 'EOF'
torch==2.7.1
torchvision==0.22.1
setuptools<70
accelerate==0.24.1
EOF

pip install "setuptools<70"                 # see trap 2 below
grep -v nvdiffrast requirements.txt > /tmp/instantmesh_requirements_no_nvdiffrast.txt
pip install -r /tmp/instantmesh_requirements_no_nvdiffrast.txt -c ~/instantmesh_constraints.txt
pip install git+https://github.com/NVlabs/nvdiffrast/ --no-build-isolation --no-cache-dir -c ~/instantmesh_constraints.txt
pip install onnxruntime -c ~/instantmesh_constraints.txt   # see trap 4 below
pip install "accelerate==0.24.1" -c ~/instantmesh_constraints.txt   # see trap 5 below
```

### Known traps specific to InstantMesh

1. **The README's pinned stack (`torch==2.1.0+cu121`, `xformers==0.0.22.post7`) has zero Blackwell support at all**,
   not even the "reports available but fails" behavior — just don't install it. `xformers` is not actually
   referenced anywhere in InstantMesh's inference code path (only used internally by `diffusers` if explicitly
   enabled, which `run.py` never does); it's safe to drop entirely rather than chase a newer compatible version.
2. **`pip install -r requirements.txt` alone will silently upgrade torch out from under you.**
   `pytorch-lightning==2.1.2`'s loose `torch>=1.12.0` constraint lets pip's resolver grab the latest torch during a
   plain `pip install -r requirements.txt`, even though a compatible pinned torch is already installed. Always pass
   `-c constraints.txt` pinning `torch`/`torchvision` explicitly on every `pip install` call in this environment,
   not just the first one.
3. **`nvdiffrast`'s git-based install needs `--no-build-isolation` too** — same as TRELLIS.2's — because
   `requirements.txt` lists it as a plain `git+https://...` line, and a single `pip install -r requirements.txt`
   cannot apply `--no-build-isolation` to just one line. Filter it out of `requirements.txt` and install it
   separately.
4. **A much newer `setuptools` breaks torch 2.x's own `cpp_extension.py`** with a swallowed
   `ModuleNotFoundError: No module named 'pkg_resources'` — surfaces as the generic, misleading nvdiffrast error
   `"Cannot compile nvdiffrast CUDA extension... 1. You have PyTorch installed 2. --no-build-isolation"` even when
   both of those conditions are already true. The real fix is `pip install "setuptools<70"`; add it to the
   constraints file for future installs in this env.
5. **`run.py` imports `rembg` unconditionally at module load**, even though Darkness always passes `--no_rembg` and
   never actually invokes it. If `onnxruntime` (one of `rembg`'s own dependencies) isn't installed, the bare
   `import rembg` line crashes before `--no_rembg` is ever consulted. Install `onnxruntime` even though it's never
   actually used at runtime.
6. **`pip install -r requirements.txt` can also silently pull in an `accelerate` version too new for the
   `huggingface_hub` version the rest of the pinned stack (`transformers==4.34.1`) resolves to**, raising
   `ImportError: cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub'` deep inside
   `diffusers`' import chain. Pin `accelerate==0.24.1` explicitly (contemporaneous with `transformers==4.34.1`).

## 8. Darkness-side wiring (Windows side, once both environments above exist)

- `config.local.toml` needs a `command_prefix` entry per worker pointing at
  `adapters/wsl_launch.py`, which bridges the Windows-side `--request`/`--response` paths across the WSL2 DrvFS
  boundary. It is a **generic** launcher — the same script works for any WSL2-hosted worker; only the
  `--wsl-python`/`--wsl-script`/`--env` arguments differ per worker. See `machine.example.toml` for the exact shape.
- Smoke-test a worker with `adapters/build_smoke_request.py` (builds a schema-valid request against a real image)
  and `python -m darkness run-worker --worker-id <id> --request <path>`.
- If a worker fails, check `<workspace_root>/worker_logs/<job_id>.std{out,err}.log` — that's where
  `WorkerManager` actually captures the subprocess output, not just the top-level `ExternalWorkerContractError`
  message, which is often a generic wrapper.
- Job IDs are immutable by design — a second attempt with the same `--job-id` fails with `FileExistsError` before
  even reaching the worker. Use a fresh job ID per retry (`build_smoke_request.py --job-id trellis2.smoke.v2`, etc.).

## 9. Reproducing the exact result recorded 2026-07-12

```powershell
python adapters/build_smoke_request.py `
  --image fixtures/concepts/short_biped_fixture_rgba_v1.png `
  --output-directory C:/AssetForgeDarknessRuns/smoke/trellis2/output `
  --job-id trellis2.smoke.v1 --run-id trellis2.smoke.run.v1 `
  --artifact-id concept.goblin.smoke.v1 `
  --out C:/AssetForgeDarknessRuns/smoke/trellis2/request.json --seed 42
python -m darkness run-worker --worker-id trellis2.4b --request C:/AssetForgeDarknessRuns/smoke/trellis2/request.json --timeout 1800
```

Swap `--worker-id instantmesh.large` for the InstantMesh path. Results, hashes, and environment versions from this
exact run are recorded in `qualifications/trellis2-4b_wsl2_rtx5090.json` and
`qualifications/instantmesh-large_wsl2_rtx5090.json`.
