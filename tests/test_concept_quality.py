from pathlib import Path

from PIL import Image

from text2model_forge.concept_quality import assess_concept_image


FIXTURES = Path(__file__).parent / "fixtures" / "concepts"


def test_known_geometry_ready_fixture_passes_deterministic_quality_gate() -> None:
    result = assess_concept_image(
        FIXTURES / "short_biped_fixture_chroma_v1.png",
        FIXTURES / "short_biped_fixture_rgba_v1.png",
        minimum_score=0.4,
    )
    assert result.hard_requirements_satisfied is True
    assert result.metrics["meaningful_alpha"] is True
    assert result.metrics["border_touch_fraction"] <= 0.08


def test_missing_alpha_fails_before_semantic_review() -> None:
    result = assess_concept_image(
        FIXTURES / "short_biped_fixture_chroma_v1.png",
        None,
        minimum_score=0.1,
    )
    assert result.hard_requirements_satisfied is False
    assert any("background mask" in reason for reason in result.reasons)


def test_cropped_foreground_fails_layout_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    alpha = tmp_path / "alpha.png"
    Image.new("RGB", (512, 512), "#8f3344").save(source)
    Image.new("RGBA", (512, 512), (143, 51, 68, 255)).save(alpha)
    result = assess_concept_image(source, alpha, minimum_score=0.1)
    assert result.hard_requirements_satisfied is False
    assert result.metrics["border_touch_fraction"] == 1.0
