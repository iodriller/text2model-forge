from __future__ import annotations

from pathlib import Path

from text2model_forge.cli import _demo_brief
from text2model_forge.compiler import Text2ModelCompiler
from text2model_forge.fake_worker import build_fake_registry
from text2model_forge.mesh import MeshRepairDecision
from text2model_forge.mesh_worker import evaluate_mesh_repair, register_mesh_operations
from text2model_forge.regression import evaluate_candidate
from text2model_forge.schemas import AssetStage, ExternalWorkerOutput, ExternalWorkerResponse, MetricResult


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mesh"


def _compiler_at_geometry(tmp_path: Path) -> tuple[Text2ModelCompiler, str, str]:
    run_id = "goblin.mesh.cleanup"
    registry = build_fake_registry()
    register_mesh_operations(registry)
    compiler = Text2ModelCompiler(tmp_path / "workspace", registry)
    brief = compiler.create_run(run_id, _demo_brief())
    brief_evaluation = evaluate_candidate(
        evaluation_id="eval.brief.mesh",
        candidate_artifact_id=brief.artifact_id,
        baseline_artifact_id=None,
        metrics=[MetricResult(name="schema", value=True, passed=True, hard_failure=True)],
        technical_score=1,
    )
    compiler.record_evaluation(run_id, brief_evaluation)
    compiler.approve(run_id, artifact_id=brief.artifact_id, approved_by="fixture-human")
    compiler.advance(run_id)
    concept = compiler.execute_candidate(
        run_id,
        operation_id="fake.advance",
        artifact_id="goblin.concept.mesh.v1",
        input_artifact_ids=[brief.artifact_id],
        parameters={"target_stage": AssetStage.concept.value},
    )
    concept_evaluation = evaluate_candidate(
        evaluation_id="eval.concept.mesh",
        candidate_artifact_id=concept.artifact_id,
        baseline_artifact_id=brief.artifact_id,
        metrics=[MetricResult(name="fixture", value=True, passed=True, hard_failure=True)],
        technical_score=1,
    )
    compiler.record_evaluation(run_id, concept_evaluation)
    compiler.approve(run_id, artifact_id=concept.artifact_id, approved_by="fixture-human")
    compiler.advance(run_id)
    return compiler, run_id, concept.artifact_id


def test_mesh_repair_is_staged_evaluated_and_promoted_without_overwriting_source(tmp_path: Path) -> None:
    compiler, run_id, concept_id = _compiler_at_geometry(tmp_path)
    source_path = FIXTURES / "five_defects.obj"
    source_bytes = source_path.read_bytes()
    response = ExternalWorkerResponse(
        job_id="geometry.fixture",
        status="succeeded",
        outputs=[ExternalWorkerOutput(path=str(source_path), media_type="model/obj", role="geometry_candidate")],
    )
    imported = compiler.import_external_response(
        run_id,
        response=response,
        candidate_id="seeded-defect-corpus",
        parent_artifact_ids=[concept_id],
        artifact_ids_by_role={"geometry_candidate": "goblin.geometry.seeded.v1"},
        primary_role="geometry_candidate",
        source_license_ids=["project-owned"],
    )
    source = imported["geometry_candidate"]
    repaired = compiler.execute_candidate(
        run_id,
        operation_id="mesh.repair_structural",
        artifact_id="goblin.geometry.repaired.v1",
        input_artifact_ids=[source.artifact_id],
        parameters={},
    )
    decision = MeshRepairDecision.model_validate(repaired.metadata["repair_decision"])
    evaluation = evaluate_mesh_repair(
        evaluation_id="eval.geometry.repair.v1",
        candidate_artifact_id=repaired.artifact_id,
        baseline_artifact_id=source.artifact_id,
        decision=decision,
    )
    compiler.record_evaluation(run_id, evaluation)

    state = compiler.log(run_id).state()
    assert evaluation.accepted
    assert state.promoted_by_stage[AssetStage.geometry.value] == repaired.artifact_id
    assert compiler.artifacts.read_bytes(source.artifact_id) == source_bytes
    assert compiler.artifacts.read_bytes(repaired.artifact_id) != source_bytes
    assert repaired.lineage.parent_artifact_ids == [source.artifact_id]


def test_destructive_mesh_repair_evaluation_is_rejected() -> None:
    from text2model_forge.mesh import TriangleMesh

    source = TriangleMesh.from_obj((FIXTURES / "clean_tetrahedron.obj").read_text(encoding="utf-8"))
    _, decision = source.guarded_repair(minimum_component_faces=5)
    evaluation = evaluate_mesh_repair(
        evaluation_id="eval.geometry.destructive.v1",
        candidate_artifact_id="goblin.geometry.destructive.v1",
        baseline_artifact_id="goblin.geometry.clean.v1",
        decision=decision,
    )

    assert not evaluation.accepted
    assert "repair_branch_accepted" in evaluation.reasons[0]
