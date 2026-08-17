"""What this machine actually is, and what stack it can actually run.

Every blocker in the 8 GB bring-up was the same bug wearing a different hat:
a constant that was correct for the qualification machine (RTX 5090, 32 GB,
TRELLIS.2, a 27B reviewer, Blender 4.x) and silently wrong here. They were
found one at a time, each costing a full pipeline run, because nothing in
the system could state what it assumed.

This module answers the first half of that -- what the hardware is -- and
`recommend_stack()` answers the second -- what settings that hardware
implies. `src/text2model_forge/preflight.py` then checks the answers against reality
before a run instead of discovering them three stages in.

Detection deliberately degrades rather than fails. In order:

1. LocalDeploy's HTTP ``GET /system/hardware``, if a LocalDeploy is running.
   It is the best source (multi-GPU, per-adapter free VRAM, driver, unified
   memory) and it is the user's own tool, so prefer it when present.
2. LocalDeploy's ``detect_hardware()`` imported directly, if the package
   happens to be importable.
3. ``nvidia-smi``.

System RAM is enriched separately through psutil or the host utility. An
accelerator without one of the three sources remains unknown rather than being
guessed from system memory.

Text2Model Forge must run standalone, so LocalDeploy is never a dependency --
only an enrichment. A machine with none of the above still gets a
HardwareProfile with ``detected=False``, and every caller treats that as
"unknown, do not pretend", never as "zero".
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Literal

from pydantic import Field

from .schemas import StrictModel


# Measured on this project's own 8 GB bring-up: with ComfyUI started and its
# models unloaded via /free, the process still held ~2.2 GB of CUDA context
# and allocator pools. That is the number that made an 8B reviewer spill to
# CPU and turned every vision review into a >600 s timeout, so it is the
# single most important constant here. It is a floor, not a guess: freeing
# both services returned ~7 GB of an 8 GB card.
COMFY_RESIDENT_VRAM_GB = 2.2

# Rough resident cost of a Q4_K_M GGUF reviewer, by parameter count. Ollama
# reports 6.24 GB for an 8.8B qwen3-vl and 2.7 GB for a 3B, which is where
# these come from -- weights plus KV cache plus the vision tower.
REVIEWER_VRAM_GB: dict[str, float] = {
    "3b": 2.7,
    "4b": 3.6,
    "7b": 5.6,
    "8b": 6.3,
    "12b": 8.9,
    "27b": 17.0,
}


def reviewer_vram_requirement(model: str) -> float | None:
    """Best documented resident estimate inferable from a model identifier."""

    normalized = model.lower()
    size = next(
        (key for key in sorted(REVIEWER_VRAM_GB, key=len, reverse=True) if key in normalized),
        None,
    )
    return REVIEWER_VRAM_GB[size] if size is not None else None

# The smallest feature that must survive D2->D3 as its own connected piece.
# D7's grip builder searches the mesh near the hand for distinct finger
# branches; voxelisation fuses anything thinner than about two voxels. Human
# fingers are ~15 mm, so a voxel above ~7.5 mm silently produces a mitt and
# D7 fails several stages later with an error that names none of this.
FINGER_WIDTH_M = 0.015


class GpuInfo(StrictModel):
    name: str = Field(min_length=1)
    vendor: str = ""
    backend: str = ""
    vram_total_gb: float | None = Field(default=None, ge=0)
    vram_free_gb: float | None = Field(default=None, ge=0)
    driver_version: str | None = None
    # True when the figure was inferred rather than read from the driver, so
    # a caller can refuse to make a confident recommendation on a guess.
    estimated: bool = False


class HardwareProfile(StrictModel):
    detected: bool = False
    source: str = "none"
    gpus: list[GpuInfo] = Field(default_factory=list)
    system_ram_gb: float | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)

    @property
    def primary(self) -> GpuInfo | None:
        """The accelerator a run would actually use: the CUDA/ROCm device with
        the most VRAM. Integrated adapters are ignored -- this machine reports
        an AMD Radeon with 512 MB alongside the RTX 3080, and picking that one
        would produce a nonsense recommendation."""
        usable = [
            gpu
            for gpu in self.gpus
            if (gpu.vram_total_gb or 0) >= 2.0
            and gpu.backend.upper() in {"CUDA", "ROCM", "HIP", "METAL", ""}
        ]
        if not usable:
            return None
        return max(usable, key=lambda gpu: gpu.vram_total_gb or 0)

    @property
    def vram_total_gb(self) -> float | None:
        gpu = self.primary
        return gpu.vram_total_gb if gpu else None


def _from_localdeploy_http(base_url: str, timeout: float = 4.0) -> dict[str, Any] | None:
    import urllib.request

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(root + "/system/hardware", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if payload.get("gpus") is not None else None
    except Exception:
        return None


def _from_localdeploy_import() -> dict[str, Any] | None:
    try:
        from localdeploy.control.hardware import detect_hardware as _detect  # type: ignore

        return _detect()
    except Exception:
        return None


def _parse_localdeploy(payload: dict[str, Any], source: str) -> HardwareProfile:
    gpus = []
    for entry in payload.get("gpus") or []:
        total = entry.get("vram_total_mb")
        free = entry.get("vram_free_mb")
        gpus.append(
            GpuInfo(
                name=str(entry.get("name") or "unknown"),
                vendor=str(entry.get("vendor") or ""),
                backend=str(entry.get("backend") or ""),
                vram_total_gb=round(total / 1024, 2) if total else None,
                vram_free_gb=round(free / 1024, 2) if free else None,
                driver_version=entry.get("driver_version"),
                estimated=bool(entry.get("vram_estimated")),
            )
        )
    return HardwareProfile(detected=bool(gpus), source=source, gpus=gpus)


def _from_nvidia_smi() -> HardwareProfile | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except Exception:
        return None
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    name=parts[0],
                    vendor="NVIDIA",
                    backend="CUDA",
                    vram_total_gb=round(float(parts[1]) / 1024, 2),
                    vram_free_gb=round(float(parts[2]) / 1024, 2),
                    driver_version=parts[3],
                )
            )
        except ValueError:
            continue
    return HardwareProfile(detected=bool(gpus), source="nvidia-smi", gpus=gpus) if gpus else None


def _system_ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        for token in result.stdout.split():
            if token.isdigit():
                return round(int(token) / 1024**3, 1)
    except Exception:
        pass
    return None


def detect_hardware(localdeploy_url: str | None = None) -> HardwareProfile:
    """Best available picture of this machine, never raising."""
    profile: HardwareProfile | None = None
    if localdeploy_url:
        payload = _from_localdeploy_http(localdeploy_url)
        if payload:
            profile = _parse_localdeploy(payload, "localdeploy-http")
    if profile is None:
        payload = _from_localdeploy_import()
        if payload:
            profile = _parse_localdeploy(payload, "localdeploy-import")
    if profile is None:
        profile = _from_nvidia_smi()
    if profile is None:
        profile = HardwareProfile(
            detected=False,
            source="none",
            notes=[
                "No GPU could be detected. Install nvidia-smi, or run LocalDeploy, "
                "or pass --vram-gb to state the budget explicitly."
            ],
        )
    profile.system_ram_gb = _system_ram_gb()
    return profile


class StackRecommendation(StrictModel):
    """A coherent set of settings for one machine, plus why each was chosen.

    Every field here is one of the constants that had to be discovered the
    hard way. Emitting them together, derived from one measured VRAM budget,
    is the point: they are not independent knobs, they are a single choice
    about which stack you are on.
    """

    profile: Literal["8gb", "12gb", "24gb", "unknown"] = "unknown"
    vram_total_gb: float | None = None
    vram_budget_for_reviewer_gb: float | None = None
    reviewer_size: str = "unknown"
    spec_strategy: Literal["monolithic", "chunked"] = "chunked"
    vram_handoff: bool = True
    manifold_policy: Literal["weld_only", "if_needed", "always"] = "if_needed"
    voxel_fraction: float = 0.003
    llm_timeout_seconds: int = 900
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def recommended_voxel_fraction(asset_height_m: float, *, needs_grip: bool = True) -> float:
    """Voxel size as a fraction of the asset's longest axis.

    Derived from the feature that must survive, not from taste: to keep two
    fingers topologically separate the voxel has to be at most about half a
    finger width. Returned as a fraction because D3 sizes the voxel against
    the asset's bounding box, so the same number works for a 0.9 m chair and
    a 12 m statue.
    """
    if not needs_grip:
        return 0.006
    height = max(float(asset_height_m), 0.05)
    fraction = (FINGER_WIDTH_M / 2.0) / height
    # Never coarser than 0.003 (the measured knight value) and never so fine
    # that the mesh explodes past a few million faces.
    return max(0.0008, min(0.003, round(fraction, 5)))


def recommend_stack(
    hardware: HardwareProfile,
    *,
    comfy_resident: bool = True,
    asset_height_m: float = 1.8,
    needs_grip: bool = True,
) -> StackRecommendation:
    """Turn a measured VRAM budget into the whole coherent stack.

    `comfy_resident` is the clause that matters and the one that is easy to
    forget: the reviewer does not get the card, it gets what is left after
    ComfyUI. Sizing a model against total VRAM is exactly the mistake that
    produced the >600 s vision reviews.
    """
    total = hardware.vram_total_gb
    reasons: list[str] = []
    warnings: list[str] = []

    if total is None:
        return StackRecommendation(
            profile="unknown",
            reasons=["No GPU detected, so no budget could be computed."],
            warnings=[
                "Recommendations below are the conservative 8 GB defaults. "
                "State the real budget with --vram-gb to get a tuned answer."
            ],
            reviewer_size="4b",
            voxel_fraction=recommended_voxel_fraction(asset_height_m, needs_grip=needs_grip),
        )

    overhead = COMFY_RESIDENT_VRAM_GB if comfy_resident else 0.0
    # A safety margin for the display, the compositor, and CUDA's own
    # workspace. Without it a model that "just fits" spills under load.
    margin = 0.6
    budget = round(total - overhead - margin, 2)
    if comfy_resident:
        reasons.append(
            f"{total:.1f} GB total, minus {overhead:.1f} GB ComfyUI holds resident even after "
            f"/free, minus {margin:.1f} GB headroom = {budget:.1f} GB for the reviewer."
        )
    else:
        reasons.append(f"{total:.1f} GB total, minus {margin:.1f} GB headroom = {budget:.1f} GB.")

    affordable = [size for size, need in REVIEWER_VRAM_GB.items() if need <= budget]
    reviewer = max(affordable, key=lambda size: REVIEWER_VRAM_GB[size]) if affordable else "3b"
    if not affordable:
        warnings.append(
            f"Even a 3B reviewer needs {REVIEWER_VRAM_GB['3b']:.1f} GB and only {budget:.1f} GB "
            "is free. Run ComfyUI and the reviewer on separate machines, or stop ComfyUI "
            "between stages."
        )
    reasons.append(
        f"Largest reviewer that fits entirely on the GPU: {reviewer} "
        f"(~{REVIEWER_VRAM_GB[reviewer]:.1f} GB). A model that does not fit spills to CPU, "
        "which is ~100x slower for a vision tower, not marginally slower."
    )

    if total <= 9:
        profile: Any = "8gb"
    elif total <= 13:
        profile = "12gb"
    else:
        profile = "24gb"

    # A 27B satisfies the whole StudioAssetSpec in one call; anything smaller
    # measurably does not, and needs the chunked compiler instead.
    chunked = REVIEWER_VRAM_GB[reviewer] < REVIEWER_VRAM_GB["27b"]
    if chunked:
        reasons.append(
            f"spec_strategy=chunked: a {reviewer} model returns `equipment: []` from the "
            "single-call spec compile and fails D0's handedness contract."
        )
    else:
        reasons.append("spec_strategy=monolithic: a 27B-class reviewer handles the whole spec.")

    voxel = recommended_voxel_fraction(asset_height_m, needs_grip=needs_grip)
    if needs_grip:
        reasons.append(
            f"voxel_fraction={voxel}: on a {asset_height_m:.2f} m asset that is "
            f"{voxel * asset_height_m * 1000:.1f} mm, fine enough to keep ~{FINGER_WIDTH_M * 1000:.0f} mm "
            "fingers separate for D7's grip builder."
        )

    return StackRecommendation(
        profile=profile,
        vram_total_gb=total,
        vram_budget_for_reviewer_gb=budget,
        reviewer_size=reviewer,
        spec_strategy="chunked" if chunked else "monolithic",
        vram_handoff=comfy_resident and total < 16,
        manifold_policy="if_needed",
        voxel_fraction=voxel,
        llm_timeout_seconds=900 if total < 16 else 300,
        reasons=reasons,
        warnings=warnings,
    )
