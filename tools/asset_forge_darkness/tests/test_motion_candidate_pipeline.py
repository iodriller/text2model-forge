import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "adapters/run_motion_candidate_pipeline.py"


def test_motion_candidate_pipeline_compiles_and_has_resumable_stage_markers() -> None:
    source = PIPELINE.read_text(encoding="utf-8")
    compile(source, str(PIPELINE), "exec")
    for stage in (
        "retarget",
        "retarget_evidence",
        "retarget_qwen_review",
        "sprite_render",
        "sprite_package",
        "sprite_qwen_review",
        "unity_smoke_bundle",
        "unity_candidate_validation",
    ):
        assert f'"{stage}"' in source
    assert "resume refused" in source
    assert "partial-" in source
    assert "ready_for_external_execution" in source
    assert "human_approval_required" in source
    assert "required only when a retarget or sprite-render stage must run" in source
    assert '_require_executable(blender, "retarget")' in source
    assert '_require_executable(blender, "sprite_render")' in source
    assert "ready_for_external_unity_smoke" in source
    assert "_approved_automatic_candidate_manifest" in source
    assert 'manifest.get("automatic_gate_passed") is True' in source
    assert "build_unity_smoke_bundle.py" in source
    assert "ingest_unity_smoke_result.py" in source
    assert "Darkness motion candidate — human review" in source


def test_failed_sprite_manifest_is_never_resumed_as_an_approved_candidate(tmp_path: Path) -> None:
    approved_manifest = runpy.run_path(str(PIPELINE))["_approved_automatic_candidate_manifest"]
    manifest_path = tmp_path / "candidate_manifest.json"

    failed = {
        "automatic_gate_passed": False,
        "hard_failures": ["edge clipping"],
        "human_approval_required": True,
        "human_approved": False,
    }
    manifest_path.write_text(json.dumps(failed), encoding="utf-8")
    assert approved_manifest(manifest_path) is False

    passed = {
        "automatic_gate_passed": True,
        "hard_failures": [],
        "human_approval_required": True,
        "human_approved": False,
    }
    manifest_path.write_text(json.dumps(passed), encoding="utf-8")
    assert approved_manifest(manifest_path) is True
