from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT.parents[1] / "unity/EmberDefenseUnity/Assets/Editor/DarknessCandidateValidator.cs"


def test_unity_candidate_validator_is_non_promoting_and_hash_bound() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "DARKNESS_CANDIDATE_PACKAGE" in source
    assert "source_master_sha256" in source
    assert "sheet hash mismatch" in source.lower()
    assert "live_game_assets_modified = false" in source
    assert "human_approved = false" in source
    assert "ConfigureUnitPrefabs" not in source
    assert "GetComponent<UnitViewRegistry>" not in source


def test_unity_candidate_validator_exercises_sprite_and_clip_creation() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "Sprite.Create" in source
    assert "AnimationUtility.SetObjectReferenceCurve" in source
    assert "RenderCapture" in source
    assert "no_edge_clipping" in source
