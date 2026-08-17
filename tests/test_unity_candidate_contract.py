from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "resources/unity_smoke_template/Assets/Editor/Text2ModelCandidateValidator.cs"
REVIEW_WINDOW = ROOT / "resources/unity_smoke_template/Assets/Editor/Text2ModelCandidateReviewWindow.cs"


def test_unity_candidate_validator_is_non_promoting_and_hash_bound() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "TEXT2MODEL_FORGE_CANDIDATE_PACKAGE" in source
    assert "source_master_sha256" in source
    assert "sheet hash mismatch" in source.lower()
    assert "live_game_assets_modified = false" in source
    assert "human_approved = false" in source
    assert 'project_kind = "text2model_standalone_unity_smoke"' in source
    assert "TEXT2MODEL_FORGE_BUNDLE_MANIFEST" in source
    assert "package-relative source master" in source
    assert "ConfigureUnitPrefabs" not in source
    assert "GetComponent<UnitViewRegistry>" not in source


def test_unity_candidate_validator_exercises_sprite_and_clip_creation() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "Sprite.Create" in source
    assert "AnimationUtility.SetObjectReferenceCurve" in source
    assert "RenderCapture" in source
    assert "no_edge_clipping" in source


def test_unity_smoke_template_is_a_minimal_pinned_project() -> None:
    version = (ROOT / "resources/unity_smoke_template/ProjectSettings/ProjectVersion.txt").read_text(encoding="utf-8")
    packages = (ROOT / "resources/unity_smoke_template/Packages/manifest.json").read_text(encoding="utf-8")
    assert "6000.5.1f1" in version
    assert "0d9463e84828" in version
    assert '"dependencies": {}' in packages


def test_unity_human_review_window_plays_all_required_motions_without_promotion() -> None:
    source = REVIEW_WINDOW.read_text(encoding="utf-8")
    assert '"idle", "walk", "attack", "death"' in source
    assert '"north", "south", "east", "west"' in source
    assert "GUI.DrawTextureWithTexCoords" in source
    assert "Auto-tour all four motions" in source
    assert "human_approval_required" in source
    assert "human_approved" in source
    assert "ConfigureUnitPrefabs" not in source
