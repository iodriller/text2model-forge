"""Tests for the chunked D0 spec compiler.

These drive `ChunkedSpecCompiler` against a scripted sender rather than a live
model, so they assert the *method* -- decomposition, deterministic mapping,
filler rejection -- deterministically and with no GPU. The live behaviour the
method exists for is recorded in darkness/chunked_spec.py's module docstring
and was measured separately against qwen3-vl:8b-instruct.
"""
from __future__ import annotations

import json

import pytest

from darkness.chunked_spec import (
    ChunkedSpecCompiler,
    _clean_phrases,
    _explicit_articulation_demanded,
    _explicit_equipment_demanded,
    _slug,
)
from darkness.localdeploy import LocalDeployStructuredClient
from darkness.studio_qwen import StudioQwen


class ScriptedSender:
    """Answers each call by matching on what the schema asks for.

    Keyed on schema shape rather than call order so a test does not silently
    pass when the compiler reorders or skips a chunk -- an unmatched call
    raises instead of returning the previous answer.
    """

    def __init__(self, **overrides: dict) -> None:
        self.calls: list[dict] = []
        self.overrides = overrides

    def __call__(self, payload: dict) -> dict:
        self.calls.append(payload)
        schema = (
            payload.get("response_format", {}).get("json_schema", {}).get("schema")
        )
        if schema is None:  # the free-form reasoning call
            return self._wrap("It is a living armoured figure with a bendable skeleton.")
        properties = set(schema.get("properties", {}))
        name = self._name_for(properties)
        if name in self.overrides:
            return self._wrap(json.dumps(self.overrides[name]))
        return self._wrap(json.dumps(self._default_for(name, schema)))

    @staticmethod
    def _name_for(properties: set[str]) -> str:
        if {"asset_kind", "behavior"} <= properties:
            return "classification"
        if "anatomy_family" in properties:
            return "anatomy"
        if {"title", "creative_direction"} <= properties:
            return "identity"
        if "height_m" in properties:
            return "physical"
        if "answer" in properties:
            return "yesno"
        if {"silhouette", "materials"} <= properties:
            return "appearance"
        if {"locked_features", "gameplay_readability"} <= properties:
            return "production"
        if "items" in properties:
            return "items"
        raise AssertionError(f"unexpected chunk schema: {sorted(properties)}")

    @staticmethod
    def _default_for(name: str, schema: dict) -> dict:
        return {
            "classification": {"asset_kind": "character", "behavior": "deformable_animated"},
            "anatomy": {"anatomy_family": "humanoid"},
            "identity": {"title": "Iron Vanguard", "creative_direction": "Stylized defender."},
            "physical": {"height_m": 1.8, "width_m": 0.7, "depth_m": 0.5},
            "yesno": {"answer": True},
            "appearance": {"silhouette": ["broad shoulders"], "materials": ["steel"]},
            "production": {
                "locked_features": ["right-hand sword"],
                "gameplay_readability": ["reads at sprite scale"],
            },
            "items": {
                "items": [
                    {
                        "name": "Straight Longsword",
                        "kind": "weapon",
                        "held_in": "right_hand",
                        "description": "A straight arming sword.",
                    },
                    {
                        "name": "Heater Shield",
                        "kind": "shield",
                        "held_in": "left_forearm",
                        "description": "A strapped heater shield.",
                    },
                ]
            },
        }[name]

    @staticmethod
    def _wrap(content: str) -> dict:
        return {"choices": [{"message": {"content": content}}]}


KNIGHT = (
    "An original armored knight with a straight longsword in his right hand and a "
    "heater shield strapped to his left forearm."
)


def _compiler(sender: ScriptedSender) -> ChunkedSpecCompiler:
    client = LocalDeployStructuredClient(sender=sender)
    return ChunkedSpecCompiler(client, "test-model")


def test_placement_is_mapped_to_the_contract_by_code_not_the_model():
    """The heart of the method: the model says "right_hand"/"left_forearm" and
    never has to produce `hand_right.grip` or `forearm_strap`. Those exact
    values are what _validate_explicit_handedness enforces, so producing them
    in Python is what makes an 8B model able to satisfy the contract."""
    sender = ScriptedSender()
    spec = _compiler(sender).compile(KNIGHT)

    sword = next(item for item in spec.equipment if item.category == "weapon")
    shield = next(item for item in spec.equipment if item.category == "shield")
    assert (sword.side, sword.socket, sword.grip) == ("right", "hand_right.grip", "palm_and_fingers")
    assert (shield.side, shield.socket, shield.grip) == (
        "left",
        "forearm_left.shield",
        "forearm_strap",
    )
    # and the model was never asked for those strings in any chunk
    sent = json.dumps(sender.calls)
    assert "hand_right.grip" not in sent
    assert "forearm_strap" not in sent


def test_the_compiled_spec_passes_the_same_handedness_check_as_the_27b_path():
    """Chunking must reach the existing contract, not a softened one."""
    spec = _compiler(ScriptedSender()).compile(KNIGHT)
    StudioQwen._validate_explicit_handedness(KNIGHT, spec)  # must not raise


def test_no_single_chunk_asks_for_the_whole_spec():
    """The premise of the method. If any one call requested most of
    StudioAssetSpec we would be back to the monolithic failure mode."""
    sender = ScriptedSender()
    _compiler(sender).compile(KNIGHT)
    constrained = [
        call for call in sender.calls if "response_format" in call
    ]
    assert len(constrained) >= 6, "the spec should be composed from several small calls"
    for call in constrained:
        properties = call["response_format"]["json_schema"]["schema"].get("properties", {})
        assert len(properties) <= 4, f"chunk got too broad: {sorted(properties)}"


def test_equipment_chunk_is_grammar_forced_to_be_non_empty():
    """`equipment: []` is what the monolithic call returned. minItems is what
    makes it unreachable, so it must actually be in the schema sent."""
    sender = ScriptedSender()
    _compiler(sender).compile(KNIGHT)
    items_call = next(
        call
        for call in sender.calls
        if "response_format" in call
        and "items" in call["response_format"]["json_schema"]["schema"].get("properties", {})
    )
    array = items_call["response_format"]["json_schema"]["schema"]["properties"]["items"]
    assert array["minItems"] >= 1


def test_a_static_prop_gets_no_animations_and_no_equipment_chunk():
    sender = ScriptedSender(
        classification={"asset_kind": "prop", "behavior": "static"},
        identity={"title": "Supply Crate", "creative_direction": "Plain wooden crate."},
        appearance={"silhouette": ["square box"], "materials": ["oak"]},
        production={"locked_features": ["iron brackets"], "gameplay_readability": ["reads as a crate"]},
    )
    spec = _compiler(sender).compile("A plain original wooden supply crate, static scenery.")
    assert spec.behavior == "static"
    assert spec.animations == []  # StudioAssetSpec forbids clips on a static asset
    assert spec.equipment == []
    assert spec.anatomy_family is None


def test_classification_defines_a_standalone_object_as_a_prop_not_an_environment():
    """Regression from the first live golden run: the 4B model labelled one
    workshop stool as an environment because the enum was supplied without a
    taxonomy. Both the reasoning and constrained calls need the distinction;
    otherwise either call can undo the other one's correct interpretation.
    """
    sender = ScriptedSender(
        classification={"asset_kind": "prop", "behavior": "static"},
        identity={"title": "Workshop Stool", "creative_direction": "Simple worn furniture."},
        appearance={"silhouette": ["round three-legged seat"], "materials": ["wood"]},
        production={"locked_features": ["three legs"], "gameplay_readability": ["reads as a stool"]},
    )
    _compiler(sender).compile("A single three-legged wooden workshop stool.")

    free_call = next(call for call in sender.calls if "response_format" not in call)
    classification_call = next(
        call
        for call in sender.calls
        if "response_format" in call
        and {
            "asset_kind",
            "behavior",
        }
        <= set(call["response_format"]["json_schema"]["schema"].get("properties", {}))
    )
    for call in (free_call, classification_call):
        prompt = call["messages"][0]["content"]
        assert "prop = one self-contained object" in prompt
        assert "environment = a place, terrain, room, or scene composed of multiple objects" in prompt
        assert "Classify the requested deliverable, not its presentation background" in prompt


def test_a_rigid_asset_gets_movable_parts_and_open_close():
    sender = ScriptedSender(
        classification={"asset_kind": "architecture", "behavior": "rigid_articulated"},
        identity={"title": "Iron Gate", "creative_direction": "Hinged stone gate."},
        appearance={"silhouette": ["arched frame"], "materials": ["iron"]},
        production={"locked_features": ["two door leaves"], "gameplay_readability": ["open state clear"]},
        items={
            "items": [
                {
                    "name": "Left Door Leaf",
                    "connection": "hinged to the stone frame",
                    "description": "Swinging left leaf.",
                }
            ]
        },
    )
    spec = _compiler(sender).compile("A weathered original stone gate with two hinged iron doors.")
    assert spec.behavior == "rigid_articulated"
    assert [item.component_id for item in spec.components] == ["left_door_leaf"]
    assert spec.components[0].motion == "rigid"
    assert spec.animations == ["open", "close"]


def test_rigid_articulation_requires_an_explicit_moving_state():
    assert _explicit_articulation_demanded("A door with two hinged leaves that opens outward.") is True
    assert _explicit_articulation_demanded("A sealed fire hydrant with two capped outlets.") is False
    assert _explicit_articulation_demanded("A workshop stool with no moving parts.") is False

    sender = ScriptedSender(
        classification={"asset_kind": "prop", "behavior": "rigid_articulated"},
        identity={"title": "Fire Hydrant", "creative_direction": "Compact municipal ironwork."},
        appearance={"silhouette": ["domed compact body"], "materials": ["cast iron"]},
        production={"locked_features": ["two capped outlets"], "gameplay_readability": ["reads as hydrant"]},
        items={
            "items": [
                {
                    "name": "Outlet Cap",
                    "connection": "threaded onto the body",
                    "description": "A removable side cap.",
                }
            ]
        },
    )
    compiler = _compiler(sender)
    spec = compiler.compile("A compact red fire hydrant with two capped side outlets.")
    assert spec.behavior == "static"
    assert spec.components == []
    assert spec.animations == []
    assert any(item["chunk"] == "deterministic_behavior_guard" for item in compiler.trace)


def test_grammar_forced_filler_is_rejected_before_it_reaches_the_spec():
    """Regression: a minItems floor obliges the model to fill every slot, and
    qwen3-vl:8b filled a spare silhouette slot with "%20%20%20...". That is
    grammar-valid and would otherwise land in StudioAssetSpec verbatim."""
    sender = ScriptedSender(
        appearance={
            "silhouette": ["broad shouldered knight", "%20%20%20%20%20%20%20", "..."],
            "materials": ["steel"],
        }
    )
    spec = _compiler(sender).compile(KNIGHT)
    assert spec.silhouette == ["broad shouldered knight"]


def test_clean_phrases_falls_back_rather_than_emitting_nothing():
    # StudioAssetSpec requires min_length=1 on these lists, so filtering
    # everything out must still leave a valid spec.
    assert _clean_phrases(["%20%20", "---", ".."], "fallback") == ["fallback"]
    assert _clean_phrases(["Iron plate", "iron plate"], "fallback") == ["Iron plate"]


def test_ids_are_slugified_in_code():
    assert _slug("Straight Longsword", "x") == "straight_longsword"
    assert _slug("!!!", "fallback") == "fallback"
    assert _slug("Rusty  --  Dagger", "x") == "rusty_dagger"


def test_explicit_equipment_demand_matches_what_the_validator_enforces():
    assert _explicit_equipment_demanded(KNIGHT) is True
    assert _explicit_equipment_demanded("A plain wooden crate.") is False


def test_studio_qwen_rejects_an_unknown_spec_strategy():
    with pytest.raises(ValueError, match="spec_strategy"):
        StudioQwen(base_url="http://localhost/v1", model="m", spec_strategy="nonsense")
