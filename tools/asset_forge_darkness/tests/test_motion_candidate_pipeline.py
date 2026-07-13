from pathlib import Path


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
        "unity_candidate_validation",
    ):
        assert f'"{stage}"' in source
    assert "resume refused" in source
    assert "partial-" in source
    assert "waiting_for_unity_editor" in source
    assert "human_approval_required" in source
    assert "Darkness motion candidate — human review" in source
