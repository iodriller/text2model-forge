from __future__ import annotations

from pathlib import Path

from darkness.lineage import evaluate_release, evaluate_research
from darkness.registry import candidates_in_priority_order, load_registry
from darkness.schemas import ArtifactLineage, ModelRuntimeQualification


ROOT = Path(__file__).resolve().parents[1]


def artifact(artifact_id: str, producer: str, parents: list[str] | None = None) -> ArtifactLineage:
    return ArtifactLineage(
        artifact_id=artifact_id,
        artifact_sha256="a" * 64,
        stage="fixture",
        producer_candidate_ids=[producer],
        parent_artifact_ids=parents or [],
    )


def test_registry_prioritizes_permissive_open_models_and_logs_later_candidates() -> None:
    registry = load_registry()
    order = candidates_in_priority_order(registry)
    assert order.index("trellis2-4b") < order.index("hunyuan3d-2.1")
    assert order.index("hunyuan3d-2.1") < order.index("r-dmesh")
    assert {item.candidate_id for item in registry.candidates} >= {"r-dmesh", "roblox-cubepart", "spar3d"}


def test_hunyuan_is_allowed_for_research_but_blocked_for_unreviewed_release() -> None:
    registry = load_registry()
    artifacts = {"mesh": artifact("mesh", "hunyuan3d-2.1")}
    assert evaluate_research("mesh", artifacts, registry).allowed is True
    release = evaluate_release("mesh", artifacts, registry)
    assert release.allowed is False
    assert "review_required" in " ".join(release.reasons)


def test_rdmesh_is_blocked_until_license_and_vram_are_resolved() -> None:
    registry = load_registry()
    artifacts = {"motion": artifact("motion", "r-dmesh")}
    research = evaluate_research("motion", artifacts, registry)
    assert research.allowed is False
    assert "not approved even for research" in " ".join(research.reasons)


def test_release_checks_recursive_parents() -> None:
    registry = load_registry()
    artifacts = {
        "target": artifact("target", "hunyuan3d-omni"),
        "retopo": artifact("retopo", "qwen3.6-27b", ["target"]),
    }
    decision = evaluate_release("retopo", artifacts, registry)
    assert decision.allowed is False
    assert decision.traversed_artifact_ids == ["target", "retopo"]


def test_runtime_qualification_record_is_strict_and_digest_pinned() -> None:
    path = ROOT / "qualifications" / "qwen3.6-27b_ollama_rtx5090.json"
    record = ModelRuntimeQualification.model_validate_json(path.read_text(encoding="utf-8"))

    assert record.status == "partial"
    assert record.profile_max_images == 4
    assert record.transport_qualified_max_images == 4
    assert record.semantic_qualified_max_images == 2
    assert record.model_sha256.startswith("sha256:")
    assert any(probe.image_count == 8 and not probe.schema_valid for probe in record.probes)
