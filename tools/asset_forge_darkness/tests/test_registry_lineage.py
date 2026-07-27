from __future__ import annotations

from pathlib import Path

from darkness.lineage import evaluate_release, evaluate_research
from darkness.registry import candidates_in_priority_order, load_registry
from darkness.schemas import ArtifactLineage, ModelRuntimeQualification, WorkerRuntimeQualification


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
    assert order.index("triposg-1.5b") < order.index("hunyuan3d-2.1")
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


def test_triposg_qualification_records_permissive_lineage_and_exclusions() -> None:
    path = ROOT / "qualifications" / "triposg-1.5b_windows_rtx5090.json"
    record = WorkerRuntimeQualification.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.code_license == "MIT"
    assert record.weights_license == "MIT"
    assert record.observed["background_removal_model"] == "none"
    assert record.observed["connected_surface_components"] == 17
    assert any("RMBG-1.4" in item for item in record.excluded_dependencies)


def test_blender_qualification_records_real_source_preserving_cleanup() -> None:
    path = ROOT / "qualifications" / "blender-4.5.11_windows.json"
    record = WorkerRuntimeQualification.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.status == "passed"
    assert record.environment["worker_sha256"] == "4a5bfb222ded9bb41534e103a63b1b2847fe05e016d0604de6659f1ab9d9a710"
    assert record.observed["source_sha256"] == "d95e921e72b47575d3a7f091b5b986154a7e47dada559526c2a2e4ea3905414a"
    assert record.observed["source_components"] == 17
    assert record.observed["candidate_components"] == 1
    assert record.observed["candidate_watertight"] is True
    assert record.observed["automatic_visual_gate_passed"] is True
    assert record.observed["human_approved"] is False


def test_autoremesher_qualification_records_rejected_real_candidate() -> None:
    path = ROOT / "qualifications" / "autoremesher-1.0.0_windows.json"
    record = WorkerRuntimeQualification.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.status == "partial"
    assert record.observed["source_components"] == 1
    assert record.observed["candidate_components"] == 1
    assert record.observed["candidate_quad_fraction"] > 0.99
    assert record.observed["structural_gate_passed"] is False
    assert record.observed["all_quad_gate_passed"] is False
    assert record.observed["automatic_promotion_gate_passed"] is False
    assert record.observed["repeat_outputs_identical"] is False
    assert record.observed["source_overwritten"] is False


def test_instant_meshes_qualification_records_deterministic_rejected_candidate() -> None:
    path = ROOT / "qualifications" / "instant-meshes-7b316086_windows.json"
    record = WorkerRuntimeQualification.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.status == "partial"
    assert record.observed["candidate_quad_fraction"] == 1.0
    assert record.observed["all_quad_gate_passed"] is True
    assert record.observed["structural_gate_passed"] is False
    assert record.observed["automatic_promotion_gate_passed"] is False
    assert record.observed["repeat_outputs_identical"] is True
    assert record.observed["candidate_components"] == 30
    assert record.observed["source_overwritten"] is False


def test_blender_retopology_repair_qualification_records_staged_gate_pass() -> None:
    path = ROOT / "qualifications" / "blender-4.5.11_retopology_repair_windows.json"
    record = WorkerRuntimeQualification.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.status == "passed"
    assert record.observed["quad_fraction"] > 0.99
    assert record.observed["connected_components"] == 1
    assert record.observed["boundary_edges"] == 0
    assert record.observed["non_manifold_edges"] == 0
    assert record.observed["vertex_coordinates_unchanged"] is True
    assert record.observed["automatic_promotion_gate_passed"] is True
    assert record.observed["human_approved"] is False
    assert record.observed["source_overwritten"] is False


def test_blender_triposg_rig_probe_records_staged_deformation_evidence() -> None:
    path = ROOT / "qualifications" / "blender-4.5.11_triposg_rig_probe_windows.json"
    record = WorkerRuntimeQualification.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.status == "passed"
    assert record.observed["landmarks"] == 26
    assert record.observed["bones"] == 21
    assert record.observed["maximum_influences"] == 4
    assert record.observed["unweighted_vertices"] == 0
    assert record.observed["vertex_coordinates_unchanged"] is True
    assert record.observed["topology_unchanged"] is True
    assert record.observed["stress_collapsed_faces"] == 0
    assert record.observed["automatic_rig_probe_gate_passed"] is True
    assert record.observed["human_approved"] is False
    assert record.observed["source_overwritten"] is False


def test_canonical_qualification_records_passing_skin_stress_and_missing_fit() -> None:
    path = ROOT / "qualifications" / "canonical-short-biped-v1.json"
    record = WorkerRuntimeQualification.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.observed["components"] == 1
    assert record.observed["skinning_hard_failures"] == []
    assert record.observed["lod1_skinning_hard_failures"] == []
    assert record.observed["lod2_skinning_hard_failures"] == []
    assert record.observed["lod1_faces"] < record.observed["faces"]
    assert record.observed["lod2_faces"] < record.observed["lod1_faces"]
    assert record.observed["sprite_sheets"] == 20
    assert record.observed["body_colliders"] == 3
    assert record.observed["target_fit"] == "cross_section_envelope_v1"
    assert record.observed["target_fit_mean_relative_extent_error_after"] < record.observed[
        "target_fit_mean_relative_extent_error_before"
    ]
    assert record.status == "partial"
