"""Tests for hardware detection, stack recommendation, and preflight.

The value of this module is not that it exercises some code -- it is that
each test below encodes one failure that actually cost a full pipeline run
during the 8 GB bring-up. If a check regresses, the corresponding failure
becomes silently discoverable-only-by-running again.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from text2model_forge.hardware import (
    COMFY_RESIDENT_VRAM_GB,
    FINGER_WIDTH_M,
    GpuInfo,
    HardwareProfile,
    detect_hardware,
    recommend_stack,
    recommended_voxel_fraction,
)
from text2model_forge.preflight import (
    Check,
    check_comfy_nodes,
    check_donor_motion,
    check_equipment_conformance_capability,
    check_reviewer_fits,
    check_spec_strategy,
    check_voxel_vs_grip,
)


def _card(total_gb: float, *, name: str = "Test GPU", backend: str = "CUDA") -> HardwareProfile:
    return HardwareProfile(
        detected=True,
        source="test",
        gpus=[GpuInfo(name=name, backend=backend, vram_total_gb=total_gb)],
    )


def test_detect_hardware_never_raises_and_always_answers():
    """Called on every doctor run and on a machine that may have no GPU at
    all; it must degrade to detected=False rather than throw."""
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.source in {"localdeploy-http", "localdeploy-import", "nvidia-smi", "none"}


def test_integrated_adapter_does_not_win_the_primary_slot():
    """Regression: this machine reports an AMD Radeon with 512 MB alongside
    the RTX 3080. Picking the integrated adapter would compute a nonsense
    VRAM budget and recommend a 3B model on a machine that fits more."""
    profile = HardwareProfile(
        detected=True,
        source="test",
        gpus=[
            GpuInfo(name="RTX 3080 Laptop", backend="CUDA", vram_total_gb=8.0),
            GpuInfo(name="AMD Radeon(TM) Graphics", backend="Vulkan", vram_total_gb=0.5),
        ],
    )
    assert profile.primary is not None
    assert "3080" in profile.primary.name
    assert profile.vram_total_gb == 8.0


def test_eight_gigabyte_card_is_told_to_use_a_small_reviewer():
    """The headline finding: the reviewer's budget is what is left AFTER
    ComfyUI, not the size of the card. Sizing against total VRAM is what
    produced >600 s vision reviews."""
    recommendation = recommend_stack(_card(8.0), comfy_resident=True)
    assert recommendation.profile == "8gb"
    assert recommendation.reviewer_size in {"3b", "4b"}
    assert recommendation.spec_strategy == "chunked"
    assert recommendation.vram_handoff is True
    # the budget must actually subtract ComfyUI's resident cost
    assert recommendation.vram_budget_for_reviewer_gb < 8.0 - COMFY_RESIDENT_VRAM_GB + 0.01


def test_the_same_card_affords_more_when_comfy_is_not_resident():
    with_comfy = recommend_stack(_card(8.0), comfy_resident=True)
    without = recommend_stack(_card(8.0), comfy_resident=False)
    assert without.vram_budget_for_reviewer_gb > with_comfy.vram_budget_for_reviewer_gb


def test_a_large_card_keeps_the_original_monolithic_behaviour():
    """The 8 GB work must not have quietly redefined the qualified stack."""
    recommendation = recommend_stack(_card(32.0), comfy_resident=True)
    assert recommendation.reviewer_size == "27b"
    assert recommendation.spec_strategy == "monolithic"
    assert recommendation.vram_handoff is False


def test_no_gpu_gives_conservative_defaults_not_a_crash():
    recommendation = recommend_stack(HardwareProfile(detected=False, source="none"))
    assert recommendation.profile == "unknown"
    assert recommendation.warnings, "an undetected GPU must be flagged, not silently assumed"


def test_voxel_fraction_is_derived_from_finger_width_not_taste():
    """The D7 grip builder needs fingers to survive as separate branches."""
    for height in (1.2, 1.8, 2.4):
        fraction = recommended_voxel_fraction(height, needs_grip=True)
        voxel_m = fraction * height
        assert voxel_m <= FINGER_WIDTH_M / 2 + 1e-6, f"{height} m asset fuses fingers"
    # a static prop with no hands does not pay for that resolution
    assert recommended_voxel_fraction(1.8, needs_grip=False) > recommended_voxel_fraction(1.8)


# ---- preflight: each of these replays a real failure from the bring-up


def test_preflight_catches_an_oversized_reviewer():
    check = check_reviewer_fits(_card(8.0), {"model": "qwen3-vl:8b-instruct"})
    assert check.status == "fail"
    assert "spill" in check.detail.lower()
    assert check.remedy, "a failing check must say what to do"


def test_preflight_accepts_a_reviewer_that_fits():
    check = check_reviewer_fits(_card(8.0), {"model": "qwen3-vl:4b-instruct"})
    assert check.status == "ok"


def test_preflight_catches_monolithic_spec_on_a_small_model():
    check = check_spec_strategy({"model": "qwen3-vl:8b-instruct", "spec_strategy": "monolithic"})
    assert check.status == "fail"
    assert "chunked" in check.remedy


def test_preflight_accepts_chunked_on_a_small_model():
    assert (
        check_spec_strategy({"model": "qwen3-vl:4b-instruct", "spec_strategy": "chunked"}).status
        == "ok"
    )


def test_preflight_warns_when_reviewer_is_not_vision_capable():
    """The D1 equipment-duplication guard and check_equipment_conformance()
    both call visual_presence unconditionally on the real StudioQwen -- a
    text-only reviewer does not raise, it just answers image questions it
    cannot see, so the exact duplicate-equipment defect this mechanism
    exists to catch would go uncaught with no error anywhere."""
    check = check_equipment_conformance_capability({"model": "llama3.1:8b"})
    assert check.status == "warn"
    assert "vision" in check.detail.lower()
    assert check.remedy.strip()


def test_preflight_accepts_a_vision_model_by_name():
    check = check_equipment_conformance_capability({"model": "qwen3-vl:4b-instruct"})
    assert check.status == "ok"


def test_equipment_conformance_capability_check_is_skipped_with_no_model():
    assert check_equipment_conformance_capability({}).status == "skip"


def test_preflight_catches_a_voxel_that_would_fuse_fingers():
    """Regression: voxel_fraction 0.006 is 10.8 mm on a 1.8 m character and
    fused the knight's fingers, which surfaced four stages later at D7 as
    'grip topology did not resolve the expected digit branches: []'."""
    check = check_voxel_vs_grip({"D3": {"manifold_policy": "if_needed", "voxel_fraction": 0.006}})
    assert check.status == "fail"
    assert "fuse" in check.detail
    assert "voxel_fraction" in check.remedy


def test_voxel_check_is_skipped_when_no_remesh_runs():
    check = check_voxel_vs_grip({"D3": {"manifold_policy": "weld_only", "voxel_fraction": 0.02}})
    assert check.status == "skip"


def test_preflight_catches_an_unreachable_comfy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("text2model_forge.preflight._http_json", lambda *a, **k: None)
    check = check_comfy_nodes({"comfy_url": "http://127.0.0.1:9"}, {"D2": {"backend": "hunyuan3d"}})
    assert check.status == "fail"
    assert "Start ComfyUI" in check.remedy


def test_preflight_catches_a_comfy_without_hunyuan3d_nodes(monkeypatch: pytest.MonkeyPatch):
    """Regression: an Oct-2024 ComfyUI runs fine and serves object_info, but
    has none of the Hunyuan3D nodes. 'Is ComfyUI installed' never caught it."""
    monkeypatch.setattr(
        "text2model_forge.preflight._http_json",
        lambda *a, **k: {"CheckpointLoaderSimple": {}, "SaveGLB": {}},
    )
    check = check_comfy_nodes({"comfy_url": "http://x"}, {"D2": {"backend": "hunyuan3d"}})
    assert check.status == "fail"
    assert "Hunyuan3Dv2Conditioning" in check.detail


def test_comfy_node_check_skips_a_subprocess_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("text2model_forge.preflight._http_json", lambda *a, **k: {})
    check = check_comfy_nodes({"comfy_url": "http://x"}, {"D2": {"backend": "trellis2.4b"}})
    assert check.status == "skip"


def test_preflight_catches_a_missing_donor_motion(tmp_path: Path):
    check = check_donor_motion({"D7": {"donor_motion_id": "nope"}}, tmp_path)
    assert check.status == "fail"
    assert check.remedy


def test_every_failing_check_carries_a_remedy():
    """A check that reports a problem without a fix just relocates the
    guessing, which is the thing this whole module exists to remove."""
    failures = [
        check_reviewer_fits(_card(8.0), {"model": "qwen3-vl:8b-instruct"}),
        check_spec_strategy({"model": "llama3.1:8b", "spec_strategy": "monolithic"}),
        check_voxel_vs_grip({"D3": {"manifold_policy": "always", "voxel_fraction": 0.01}}),
    ]
    for check in failures:
        assert isinstance(check, Check)
        assert check.status == "fail"
        assert check.remedy.strip(), f"{check.name} has no remedy"
