# TRELLIS.2-4B on WSL2 — setup record

> **Superseded 2026-07-12.** This file's title is wrong: CUDA 12.4 does **not** actually run kernels on an RTX 5090
> (torch reports `cuda_available: True` but every real op fails with "no kernel image is available"). The
> authoritative, corrected, and now-verified-working setup record is
> [`wsl2_setup_runbook.md`](./wsl2_setup_runbook.md) — it covers CUDA 12.8, both the TRELLIS.2 and InstantMesh
> environments, and every bug hit along the way. Kept below only as a historical record of the original plan.

Status as of 2026-07-12 on this host: **WSL2 is not installed.** `wsl --status` reports no distro, and enabling it
requires administrator elevation this automated session does not have, plus a reboot. The steps in Section 1 must be
run manually, once, by the machine owner in an elevated PowerShell.

Observed host GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.74 (`nvidia-smi`, 2026-07-12). TRELLIS.2's official
minimum is 24 GB VRAM, so this card qualifies with headroom.

## 1. Install WSL2 (manual, one-time, requires reboot)

Run in an **elevated** ("Run as Administrator") PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

This enables the `Microsoft-Windows-Subsystem-Linux` and `VirtualMachinePlatform` optional features, installs the WSL2
kernel, and installs Ubuntu 24.04. Reboot when prompted. On first launch, Ubuntu asks for a UNIX username/password —
record the distro name exactly as it appears in `wsl -l -v` (default `Ubuntu-24.04`); Darkness's worker binding needs
that exact string.

NVIDIA's WSL2 CUDA support does **not** require a separate Linux NVIDIA driver install — the existing Windows driver
(610.74, already installed) exposes `/usr/lib/wsl/lib/libcuda.so` inside WSL2 automatically. Verify after reboot:

```bash
nvidia-smi   # run inside the WSL2 Ubuntu shell; should list the RTX 5090
```

If `nvidia-smi` is not found inside WSL2, the Windows GPU driver is too old for WSL2 CUDA passthrough and needs
updating from NVIDIA first (driver 470.76+ historically required; 610.74 is far newer and should already work).

## 2. Install CUDA toolkit + build tooling inside WSL2

```bash
sudo apt update && sudo apt install -y build-essential git wget
# CUDA 12.4 toolkit per TRELLIS.2's documented recommendation; use NVIDIA's
# WSL-Ubuntu network repo, not the generic Linux one, to avoid pulling in a
# conflicting kernel driver package.
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-4
```

## 3. Install Miniconda and TRELLIS.2

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate

git clone https://github.com/microsoft/TRELLIS.2.git
cd TRELLIS.2
# Must be SOURCED (`. ./setup.sh`, not `bash setup.sh`) because it calls
# `conda activate trellis2` in the current shell. On CUDA it pins
# torch==2.6.0+cu124 (matches the CUDA 12.4 toolkit installed in Section 2)
# and installs flash-attn==2.7.3 via pip (not a from-source build on CUDA;
# that path is ROCm-only) -- but pip may still build it locally if no wheel
# matches this exact torch/cuda/python ABI, so budget real time regardless.
# The --basic block also runs `sudo apt install libjpeg-dev`, so prime the
# sudo credential cache (`sudo -v`) immediately before sourcing this if
# running non-interactively.
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

Confirmed cloned commit on this host (2026-07-12): `75fbf0183001ed9876c8dbb35de6b68552ee08bd`.

Record the exact commit this checks out (`git rev-parse HEAD`) and pin it into `workers/trellis2.json`
(`pinned_revision`) once the environment exists — the manifest currently has `pinned_revision: null` and an explicit
blocker for this.

## 4. Locate the interpreter and script paths Darkness needs

```bash
conda activate trellis2
which python   # -> the --wsl-python value for the worker binding
```

`--wsl-script` is the absolute Linux path to
`tools/asset_forge_darkness/adapters/trellis2_worker.py` as seen from inside WSL2 (this repository lives under
`C:\Users\oneye\Documents\for fun\EmberDefense`, which WSL2's default DrvFS mount exposes at
`/mnt/c/Users/oneye/Documents/for fun/EmberDefense/tools/asset_forge_darkness/adapters/trellis2_worker.py`).

`DARKNESS_TRELLIS2_WEIGHTS` may stay as the default `microsoft/TRELLIS.2-4B` Hugging Face repo id (downloaded and
cached on first run) or point at a local snapshot directory once one is pinned.

## 5. Wire the binding into `config.local.json`

See `config.example.json` for the `trellis2.4b` entry. Darkness always calls the generic
`adapters/wsl_launch.py` on the Windows side; that script translates the request/response file paths across the
WSL2 DrvFS boundary and forwards to `trellis2_worker.py` on the Linux side. No worker-specific Windows-side code is
needed beyond that config binding.

## 6. First smoke test

```powershell
$env:PYTHONPATH = (Resolve-Path tools/asset_forge_darkness).Path
python -m darkness run-worker --worker-id trellis2.4b --request C:/path/to/request.json
```

Until this has actually been run once and produced a valid GLB, `workers/trellis2.json` must stay at
`lifecycle: "discovered"`, not `"research_ready"` — see Section 7.1 of `DESIGN/asset_forge_darkness_master_plan.md`.
Record the result (peak VRAM, elapsed time, mesh health) as a `WorkerRuntimeQualification` next to
`qualifications/triposg-1.5b_windows_rtx5090.json`.

## Known gaps carried forward from this environment change

- **Cross-boundary cancellation is unsolved.** `WorkerManager.cancel()` on Windows can only terminate the direct
  child (`wsl.exe`), not the Linux process tree it started (`_terminate_tree` on Windows already documents this
  limitation for local workers; the WSL2 boundary adds a second hop). Until the GPU scheduler owns WSL process trees
  directly (master plan Phase 2), a cancelled or timed-out TRELLIS.2 job may leave orphaned GPU memory inside WSL2.
  Check `nvidia-smi` inside WSL2 after a cancellation and `wsl.exe -d <distro> -- pkill -f trellis2_worker.py` if
  needed.
- **DrvFS I/O is slower than native ext4.** Darkness's workspace_root stays on the Windows drive by design (the rest
  of the pipeline is native-Windows Python), so job request/response/output files cross the DrvFS boundary. Model
  weights and the conda environment should stay inside WSL2's own filesystem (`~/`, not `/mnt/c/...`) to avoid paying
  that cost on every inference step, not just the final artifact write.
