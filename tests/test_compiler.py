from __future__ import annotations

from darkness.cli import run_demo
from darkness.compiler import DarknessCompiler
from darkness.fake_worker import build_fake_registry
from darkness.regression import evaluate_candidate
from darkness.schemas import AssetStage, ExternalWorkerOutput, ExternalWorkerResponse, MetricResult


def test_demo_runs_all_stages_and_is_resumable_and_verifiable(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    state = run_demo(workspace, "goblin.demo")

    assert state["status"] == "completed"
    assert state["stage"] == AssetStage.export.value
    assert len(state["promoted_by_stage"]) == len(AssetStage)

    resumed = DarknessCompiler(workspace, build_fake_registry())
    reduced = resumed.log("goblin.demo").state()
    assert reduced.status == "completed"
    assert len(resumed.log("goblin.demo").read(verify=True)) == reduced.sequence
    for artifact_id in resumed.artifacts.all_records():
        resumed.artifacts.read_bytes(artifact_id, verify=True)


def test_hard_failure_rejects_staged_candidate_without_overwriting_baseline(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    compiler = DarknessCompiler(workspace, build_fake_registry())
    from darkness.cli import _demo_brief

    brief = compiler.create_run("goblin.reject", _demo_brief())
    evaluation = evaluate_candidate(
        evaluation_id="eval.brief.bad",
        candidate_artifact_id=brief.artifact_id,
        baseline_artifact_id=None,
        metrics=[MetricResult(name="locked_identity", value=False, passed=False, hard_failure=True)],
        technical_score=0.95,
    )
    compiler.record_evaluation("goblin.reject", evaluation)

    state = compiler.log("goblin.reject").state()
    assert state.promoted_by_stage == {}
    assert state.rejected_candidates == [brief.artifact_id]
    assert compiler.artifacts.read_bytes(brief.artifact_id)


def test_external_worker_output_enters_staging_with_recursive_lineage(tmp_path) -> None:
    from darkness.cli import _demo_brief

    compiler = DarknessCompiler(tmp_path / "workspace", build_fake_registry())
    brief = compiler.create_run("goblin.external", _demo_brief())
    brief_eval = evaluate_candidate(
        evaluation_id="eval.brief.external",
        candidate_artifact_id=brief.artifact_id,
        baseline_artifact_id=None,
        metrics=[MetricResult(name="schema", value=True, passed=True, hard_failure=True)],
        technical_score=1,
    )
    compiler.record_evaluation("goblin.external", brief_eval)
    compiler.approve("goblin.external", artifact_id=brief.artifact_id, approved_by="fixture-human")
    compiler.advance("goblin.external")
    concept = compiler.execute_candidate(
        "goblin.external",
        operation_id="fake.advance",
        artifact_id="goblin.concept.v1",
        input_artifact_ids=[brief.artifact_id],
        parameters={"target_stage": AssetStage.concept.value},
    )
    concept_eval = evaluate_candidate(
        evaluation_id="eval.concept.external",
        candidate_artifact_id=concept.artifact_id,
        baseline_artifact_id=brief.artifact_id,
        metrics=[MetricResult(name="fixture", value=True, passed=True, hard_failure=True)],
        technical_score=1,
    )
    compiler.record_evaluation("goblin.external", concept_eval)
    compiler.approve("goblin.external", artifact_id=concept.artifact_id, approved_by="fixture-human")
    compiler.advance("goblin.external")

    output = tmp_path / "candidate.glb"
    output.write_bytes(b"glTF research fixture")
    response = ExternalWorkerResponse(
        job_id="triposg.fixture",
        status="succeeded",
        outputs=[ExternalWorkerOutput(path=str(output), media_type="model/gltf-binary", role="geometry_candidate")],
    )
    imported = compiler.import_external_response(
        "goblin.external",
        response=response,
        candidate_id="triposg-1.5b",
        parent_artifact_ids=[concept.artifact_id],
        artifact_ids_by_role={"geometry_candidate": "goblin.geometry.triposg.v1"},
        primary_role="geometry_candidate",
        source_license_ids=["MIT"],
    )
    record = imported["geometry_candidate"]
    state = compiler.log("goblin.external").state()
    assert state.staged_candidates == [record.artifact_id]
    assert record.lineage.parent_artifact_ids == [concept.artifact_id]
    assert record.lineage.producer_candidate_ids == ["triposg-1.5b"]
