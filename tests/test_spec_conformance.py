"""Tests for the D0-equipment-contract conformance checker.

A scripted fake answers `visual_presence` deterministically by matching on
which crop (expected-side vs opposite-side) and equipment id the question is
about, so these assert the *routing and scoping logic* -- which crop gets
asked what, and for which grip type -- without needing a live model.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from text2model_forge.spec_conformance import check_equipment_conformance
from text2model_forge.studio_models import StudioAssetSpec, StudioEquipment


def _spec(equipment: list[StudioEquipment]) -> StudioAssetSpec:
    return StudioAssetSpec(
        asset_id="test_knight",
        title="Test Knight",
        description="A knight for spec-conformance testing.",
        creative_direction="Testing only.",
        asset_kind="character",
        behavior="deformable_animated",
        anatomy_family="humanoid",
        silhouette=["broad shoulders"],
        materials=["steel"],
        equipment=equipment,
        locked_features=["right-hand sword"],
        negative_constraints=["no copied designs"],
        gameplay_readability=["reads at sprite scale"],
    )


def _sword(side: str = "right") -> StudioEquipment:
    return StudioEquipment(
        equipment_id="longsword",
        category="weapon",
        side=side,
        socket=f"hand_{side}.grip",
        grip="palm_and_fingers",
        description="a straight longsword",
    )


def _shield(side: str = "left") -> StudioEquipment:
    return StudioEquipment(
        equipment_id="buckler",
        category="shield",
        side=side,
        socket=f"forearm_{side}.shield",
        grip="forearm_strap",
        description="a round iron buckler",
    )


class ScriptedPresence:
    """Answers by matching on which crop temp-path index this is (expected vs
    opposite) per call, in strict call order -- and records the paths it was
    given so a test can assert they were cleaned up afterward."""

    def __init__(self, answers: list[bool]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[Path, str]] = []

    def visual_presence(self, image_path: Path, question: str, *, max_tokens: int = 40) -> bool:
        self.calls.append((image_path, question))
        if not self.answers:
            raise AssertionError("more visual_presence calls than the test scripted answers for")
        return self.answers.pop(0)


def test_non_character_assets_are_not_checked_at_all():
    spec = _spec([_sword()])
    spec = spec.model_copy(update={"asset_kind": "prop"})
    provider = ScriptedPresence([])
    report = check_equipment_conformance(provider, Path("unused.png"), spec)
    assert report.conforms is True
    assert report.items == []
    assert provider.calls == []


def test_a_character_with_no_equipment_declared_is_not_checked():
    spec = _spec([])
    provider = ScriptedPresence([])
    report = check_equipment_conformance(provider, Path("unused.png"), spec)
    assert report.conforms is True
    assert provider.calls == []


def test_correctly_equipped_candidate_conforms(tmp_path):
    image = tmp_path / "candidate.png"
    Image.new("RGB", (400, 600), "green").save(image)
    spec = _spec([_sword(), _shield()])
    # sword: present=True, no duplicate check (palm_and_fingers)
    # shield: present=True, duplicate check runs -> False (no duplicate)
    provider = ScriptedPresence([True, True, False])
    report = check_equipment_conformance(provider, image, spec)
    assert report.conforms is True
    assert report.violations == []
    assert len(provider.calls) == 3


def test_missing_equipment_is_a_violation(tmp_path):
    image = tmp_path / "candidate.png"
    Image.new("RGB", (400, 600), "green").save(image)
    spec = _spec([_sword()])
    provider = ScriptedPresence([False])  # sword not visible
    report = check_equipment_conformance(provider, image, spec)
    assert report.conforms is False
    assert "not visibly present" in report.violations[0]
    assert report.items[0].present_at_expected_side is False


def test_duplicated_shield_on_the_opposite_side_is_a_violation(tmp_path):
    """The exact real-world case: a shield present at its own socket AND a
    second one detected on the opposite side."""
    image = tmp_path / "candidate.png"
    Image.new("RGB", (400, 600), "green").save(image)
    spec = _spec([_shield()])
    provider = ScriptedPresence([True, True])  # present=True, duplicate=True
    report = check_equipment_conformance(provider, image, spec)
    assert report.conforms is False
    assert any("second" in v and "duplicated" in v for v in report.violations)
    assert report.items[0].duplicate_on_opposite_side is True


def test_hand_held_weapons_never_get_the_duplicate_check(tmp_path):
    """A weapon the prompt asks to be 'extended away from the body' routinely
    crosses the image midline -- a real render's axe head does exactly this
    -- so asking a duplicate question about the opposite half would false-
    positive on a correctly posed weapon. Only one call (presence) may
    happen for a palm_and_fingers item."""
    image = tmp_path / "candidate.png"
    Image.new("RGB", (400, 600), "green").save(image)
    spec = _spec([_sword()])
    provider = ScriptedPresence([True])
    report = check_equipment_conformance(provider, image, spec)
    assert len(provider.calls) == 1
    assert report.items[0].duplicate_on_opposite_side is None


def test_center_side_equipment_is_recorded_but_never_queried(tmp_path):
    image = tmp_path / "candidate.png"
    Image.new("RGB", (400, 600), "green").save(image)
    backpack = StudioEquipment(
        equipment_id="pack",
        category="attachment",
        side="center",
        socket="spine.back",
        grip="none",
        description="a travel pack",
    )
    spec = _spec([backpack])
    provider = ScriptedPresence([])
    report = check_equipment_conformance(provider, image, spec)
    assert provider.calls == []
    assert report.items[0].present_at_expected_side is True
    assert report.conforms is True


def test_crop_temp_files_are_cleaned_up_after_the_check(tmp_path):
    image = tmp_path / "candidate.png"
    Image.new("RGB", (400, 600), "green").save(image)
    spec = _spec([_shield()])
    provider = ScriptedPresence([True, True])
    check_equipment_conformance(provider, image, spec)
    for path, _question in provider.calls:
        assert not path.exists(), f"crop {path} was not cleaned up"


def test_crop_temp_files_are_cleaned_up_even_when_a_call_raises(tmp_path):
    image = tmp_path / "candidate.png"
    Image.new("RGB", (400, 600), "green").save(image)
    spec = _spec([_shield()])

    class RaisingProvider:
        def __init__(self) -> None:
            self.seen: list[Path] = []

        def visual_presence(self, image_path: Path, question: str, *, max_tokens: int = 40) -> bool:
            self.seen.append(image_path)
            raise RuntimeError("simulated model failure")

    provider = RaisingProvider()
    try:
        check_equipment_conformance(provider, image, spec)
    except RuntimeError:
        pass
    for path in provider.seen:
        assert not path.exists(), f"crop {path} leaked after an exception"


def test_studio_qwen_visual_presence_sends_one_bounded_question(tmp_path):
    """Integration with the real StudioQwen surface: a scripted sender proves
    the method builds a single-image, boolean-schema request and returns the
    parsed boolean, without needing a live model."""
    import json

    from text2model_forge.localdeploy import LocalDeployStructuredClient
    from text2model_forge.studio_qwen import StudioQwen

    image = tmp_path / "crop.png"
    Image.new("RGB", (200, 200), "grey").save(image)

    captured = {}

    def sender(payload):
        captured.update(payload)
        return {"choices": [{"message": {"content": json.dumps({"present": True})}}]}

    qwen = StudioQwen(base_url="http://unused/v1", model="test-model")
    qwen.client = LocalDeployStructuredClient(sender=sender)

    result = qwen.visual_presence(image, "Does this crop show a shield?")
    assert result is True
    schema = captured["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"present"}
    content = captured["messages"][0]["content"]
    assert len(content) == 2
    assert content[0]["text"] == "Does this crop show a shield?"
    assert content[1]["type"] == "image_url"
