from __future__ import annotations

from pathlib import Path

from darkness.config import load_local_config
from darkness.manifests import load_manifests, preflight


def test_worker_manifests_are_strict_unique_and_keep_rdmesh_blocked() -> None:
    manifests = load_manifests()
    assert manifests["mesh.deterministic"].lifecycle == "available"
    assert manifests["blender"].lifecycle == "research_ready"
    assert manifests["blender"].capability.candidate_id == "blender-4.5-lts"
    assert manifests["blender"].capability.supports_cancel is False
    assert manifests["autoremesher"].lifecycle == "research_ready"
    assert manifests["autoremesher"].capability.candidate_id == "autoremesher-1.0.0"
    assert manifests["autoremesher"].capability.supports_cancel is False
    assert manifests["autoremesher"].capability.deterministic is False
    assert manifests["optimizer.qwen3.6-27b"].capability.candidate_id == "qwen3.6-27b"
    assert manifests["triposg.1.5b"].license_gate == "permitted"
    assert manifests["hunyuan3d.2.1"].license_gate == "review_required"
    assert manifests["r-dmesh"].lifecycle == "blocked"
    assert manifests["r-dmesh"].license_gate == "blocked"


def test_builtin_worker_preflight_is_ready() -> None:
    manifest = load_manifests()["mesh.deterministic"]
    result = preflight(manifest)
    assert result["ready"] is True


def test_example_local_config_is_strict() -> None:
    path = Path(__file__).resolve().parents[1] / "config.example.json"
    config = load_local_config(path)
    assert config is not None
    assert config.workers["triposg.1.5b"].command_prefix
