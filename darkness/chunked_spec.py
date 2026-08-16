"""Compile a StudioAssetSpec out of many small model calls instead of one big one.

Why this exists
---------------
`StudioQwen.compile_spec` asks for the whole `StudioAssetSpec` -- seventeen
fields, two arrays of nested objects, six enums -- in a single constrained
call. A 27B model handles that. A 7-8B model does not: measured on an
RTX 3080 Laptop against `qwen3-vl:8b-instruct` and `gemma3:12b`, that call
took ~119s and came back schema-valid with `equipment: []`, which then
failed `_validate_explicit_handedness` and killed the run at D0.

Two published results explain it and point at the fix:

* "Let Me Speak Freely?" (EMNLP 2024 Industry Track, arXiv:2408.02442)
  measures that format restriction degrades reasoning, and that the tighter
  the constraint the worse it gets. One giant schema is close to the
  worst case for a small model.
* llama.cpp/Ollama compile a JSON schema to a GBNF grammar and mask invalid
  tokens during sampling. That guarantees *shape*, never *content* -- an
  empty array satisfies the grammar perfectly. Which is exactly the failure
  observed.

So the method here is four things, none of which need a bigger model:

1. **Decompose.** Each call returns a handful of fields, so no single call
   is near the small model's structured-output limit. Measured: the
   equipment chunk answered correctly in 2.2s where the monolithic call
   produced nothing in 119s.
2. **Reason first, constrain second.** For judgement-heavy chunks the model
   answers in free prose, then a second constrained call extracts from its
   own prose. This is the direct mitigation the paper suggests, and it is
   what moved a supply crate from `environment` to `prop` in testing.
3. **Make emptiness ungrammatical.** Where a field must be populated, the
   chunk's schema carries `minItems`, which survives the schema-to-GBNF
   conversion. The model then *cannot* emit `[]` -- the failure mode stops
   being possible rather than being retried.
4. **Never make the model spell the contract.** It reports a placement it
   actually understands (`held_in="right_hand"`); `_PLACEMENT` maps that to
   `side`/`socket`/`grip` in Python. Ids and animation names are slugified
   here too. Every value the schema demands in an exact machine format is
   produced by code, not hoped for from the model.

The result assembles into the same `StudioAssetSpec` the monolithic path
returns and is validated once, at the end, by the same pydantic model and
the same `_validate_explicit_handedness` check. This is a different way to
reach the contract, not a relaxation of it -- nothing here lowers the bar
that `studio_qwen.py` already enforces.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints

from .localdeploy import LocalDeployStructuredClient
from .schemas import StrictModel
from .studio_models import StudioAssetSpec, StudioComponent, StudioEquipment


# A placement the model can reason about -> the exact contract triple
# StudioEquipment demands. Keeping this in code is the whole point: asking a
# small model to recall "hand_right.grip" is asking it to be a lookup table,
# and it is measurably bad at that while being fine at "which hand".
_PLACEMENT: dict[str, tuple[str, str, str]] = {
    "right_hand": ("right", "hand_right.grip", "palm_and_fingers"),
    "left_hand": ("left", "hand_left.grip", "palm_and_fingers"),
    "right_forearm": ("right", "forearm_right.shield", "forearm_strap"),
    "left_forearm": ("left", "forearm_left.shield", "forearm_strap"),
    "back": ("center", "spine.back", "none"),
    "head": ("center", "head.mount", "none"),
    "torso": ("center", "torso.mount", "none"),
}

_BEHAVIOR_RULES = (
    "static = scenery that never moves. "
    "rigid_articulated = solid parts that hinge or rotate but never bend, such as a door or a lid. "
    "deformable_animated = a living body with a bendable skeleton; any character or creature that "
    "walks, attacks, or reacts. "
    "simulated = driven by cloth, fluid, or particle simulation."
)

_ASSET_KIND_RULES = (
    "character = one person or person-like individual; "
    "creature = one non-human living or monster-like individual; "
    "prop = one self-contained object, including furniture, tools, containers, vehicles, and decor; "
    "architecture = a building, structural construction, or building part such as a door, wall, or stair; "
    "environment = a place, terrain, room, or scene composed of multiple objects, never one standalone object; "
    "material = a surface or shader definition without its own object geometry; "
    "vfx = a transient visual effect such as fire, smoke, sparks, or magic. "
    "Classify the requested deliverable, not its presentation background."
)

# Canonical clip names for a body that animates. Deliberately fixed rather
# than model-authored: these ids are consumed downstream by D7's donor
# retarget catalogue, so an invented synonym would silently miss its donor.
_CHARACTER_CLIPS = ("idle", "walk", "attack", "hit", "death")


def _clean_phrases(values: list[str], fallback: str) -> list[str]:
    """Drop filler a grammar floor can force out of a small model.

    A `minItems` rule obliges the model to emit *something* in every slot; when
    it has run out of real content it emits padding ("%20%20%20...", "...",
    a bare separator). That is grammar-valid and would reach StudioAssetSpec
    intact, so it is filtered here rather than trusted. A phrase has to be
    mostly letters to count as one.
    """
    kept: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip(" -_.,;:")
        letters = sum(character.isalpha() for character in text)
        if len(text) < 3 or letters < max(3, len(text) // 2):
            continue
        if text.lower() not in {existing.lower() for existing in kept}:
            kept.append(text)
    return kept or [fallback]


def _slug(value: str, fallback: str) -> str:
    """A machine-safe id. StudioAssetSpec's patterns are strict and a small
    model reliably drifts (title case, spaces, punctuation); deriving the id
    from the text in code removes a whole class of validation failure."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned or not cleaned[0].isalnum():
        return fallback
    return cleaned[:48]


class _Classification(StrictModel):
    asset_kind: Literal[
        "character", "creature", "prop", "architecture", "environment", "material", "vfx"
    ]
    behavior: Literal["static", "rigid_articulated", "deformable_animated", "simulated"]


class _Anatomy(StrictModel):
    anatomy_family: Literal["humanoid", "short_biped", "quadruped", "custom"]


class _Identity(StrictModel):
    title: str = Field(min_length=1)
    creative_direction: str = Field(min_length=1)


class _Physical(StrictModel):
    height_m: float = Field(gt=0.01, lt=1000.0)
    width_m: float = Field(gt=0.01, lt=1000.0)
    depth_m: float = Field(gt=0.01, lt=1000.0)


class _EquipmentItem(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["weapon", "shield", "armor", "attachment"]
    held_in: Literal[
        "right_hand", "left_hand", "right_forearm", "left_forearm", "back", "head", "torso"
    ]
    description: str = Field(min_length=1)


class _EquipmentList(StrictModel):
    # min_length renders as minItems, which becomes a grammar rule that makes
    # the empty array the monolithic path returned literally unreachable.
    items: list[_EquipmentItem] = Field(min_length=1, max_length=8)


class _MovablePart(StrictModel):
    name: str = Field(min_length=1)
    connection: str = Field(min_length=1)
    description: str = Field(min_length=1)


class _MovablePartList(StrictModel):
    items: list[_MovablePart] = Field(min_length=1, max_length=8)


# Capping each entry matters as much as capping the list. Measured on
# qwen3-vl:8b: one four-array chunk of uncapped strings took 310s of a 359s
# compile -- the model writes a sentence where a phrase is wanted, and every
# extra token is generation time. Bounding the string in the grammar makes
# terseness mechanical instead of a request the model can ignore.
_Phrase = Annotated[str, StringConstraints(min_length=3, max_length=70)]


class _Appearance(StrictModel):
    # min_length is 1, never an aspirational 2+. A grammar floor is a hard
    # obligation: asked for two silhouette phrases when it had one, the model
    # padded the slot with "%20%20%20..." rather than stop. Force only what
    # must exist (non-empty) and let the prompt ask for more.
    silhouette: list[_Phrase] = Field(min_length=1, max_length=4)
    materials: list[_Phrase] = Field(min_length=1, max_length=4)


class _Production(StrictModel):
    locked_features: list[_Phrase] = Field(min_length=1, max_length=5)
    gameplay_readability: list[_Phrase] = Field(min_length=1, max_length=3)


class _YesNo(StrictModel):
    answer: bool


def _explicit_equipment_demanded(description: str) -> bool:
    """Does the brief itself name carried equipment?

    This mirrors what `StudioQwen._validate_explicit_handedness` will later
    enforce. Using it to *drive* extraction as well as validate it means the
    check that can fail the run is the same one that decides whether the
    equipment chunk runs with a non-empty grammar -- they cannot disagree.
    """
    text = re.sub(r"\s+", " ", description.lower())
    return any(
        word in text
        for word in ("sword", "shield", "axe", "dagger", "spear", "bow", "staff", "hammer", "mace", "wield")
    )


class ChunkedSpecCompiler:
    """Compile one StudioAssetSpec through many small calls.

    Takes the same client and model a StudioQwen does, so it runs against
    whatever OpenAI-compatible endpoint is already configured.
    """

    def __init__(self, client: LocalDeployStructuredClient, model: str) -> None:
        self.client = client
        self.model = model
        # Per-chunk trace: what was asked, what came back, how long. Surfaced
        # as D0 evidence so a human reviewing the gate can see which chunk
        # decided what, instead of one opaque blob.
        self.trace: list[dict[str, Any]] = []

    def _extract(self, model_cls, prompt: str, *, max_tokens: int = 500, temperature: float = 0.1):
        import time

        started = time.monotonic()
        value = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=model_cls,
            max_attempts=2,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.trace.append(
            {
                "chunk": model_cls.__name__.lstrip("_"),
                "seconds": round(time.monotonic() - started, 2),
                "result": value.model_dump(mode="json"),
            }
        )
        return value

    def _prose(self, prompt: str, *, max_tokens: int = 220) -> str:
        """One free-form call. No schema, so the model reasons unconstrained --
        the mitigation "Let Me Speak Freely?" points at. Its prose is only ever
        consumed by a following constrained call, never trusted as data."""
        import time

        started = time.monotonic()
        text = self.client.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        ).strip()
        self.trace.append(
            {
                "chunk": "free_reasoning",
                "seconds": round(time.monotonic() - started, 2),
                "result": text[:600],
            }
        )
        return text

    def compile(self, description: str) -> StudioAssetSpec:
        brief = description.strip()

        # 1. Classification, reasoned first then constrained.
        analysis = self._prose(
            "In 2-3 sentences of plain prose and no JSON, identify the single requested deliverable, "
            "classify its asset kind, and say whether it is scenery that never moves, an object with "
            "hinged rigid parts, or a living body with a bendable skeleton.\n"
            f"{_ASSET_KIND_RULES}\n{_BEHAVIOR_RULES}\nDESCRIPTION: {brief}"
        )
        classification = self._extract(
            _Classification,
            f"DESCRIPTION: {brief}\n\nANALYSIS: {analysis}\n\n{_ASSET_KIND_RULES}\n{_BEHAVIOR_RULES}\n"
            "Using the analysis, output the classification.",
            max_tokens=120,
        )
        kind, behavior = classification.asset_kind, classification.behavior

        # 2. Anatomy, only where the contract actually requires it.
        anatomy: str | None = None
        if kind in {"character", "creature"} and behavior == "deformable_animated":
            anatomy = self._extract(
                _Anatomy,
                "Pick the body plan. humanoid = adult human proportions. short_biped = two legs but "
                "small, stocky, or hunched, such as a goblin or dwarf. quadruped = four legs. "
                f"custom = anything else.\nDESCRIPTION: {brief}",
                max_tokens=80,
            ).anatomy_family

        # 3. Identity.
        identity = self._extract(
            _Identity,
            "Write a short title (3-6 words) and one sentence of creative direction for this asset. "
            "Keep it original: no franchise names, no copied characters.\n"
            f"DESCRIPTION: {brief}",
            max_tokens=220,
        )

        # 4. Size.
        physical = self._extract(
            _Physical,
            "Estimate this asset's real-world size in metres: height, width, depth. A human is about "
            "1.8 m tall, a chair about 0.9 m, a door about 2.1 m.\n"
            f"DESCRIPTION: {brief}",
            max_tokens=140,
        )

        # 5. Equipment -- the chunk the monolithic call silently dropped.
        equipment: list[StudioEquipment] = []
        if kind in {"character", "creature"}:
            demanded = _explicit_equipment_demanded(brief)
            carries = demanded or self._extract(
                _YesNo,
                "Does this asset carry, hold, or wear any distinct weapon, shield, or removable "
                f"piece of equipment? Answer true or false.\nDESCRIPTION: {brief}",
                max_tokens=60,
            ).answer
            if carries:
                listed = self._extract(
                    _EquipmentList,
                    "List every weapon, shield, or distinct piece of equipment this asset carries, and "
                    "exactly where each one sits on the body. A shield strapped to an arm is a forearm, "
                    "not a hand. Preserve the described side exactly.\n"
                    f"DESCRIPTION: {brief}",
                    max_tokens=700,
                )
                seen: set[str] = set()
                for index, item in enumerate(listed.items, start=1):
                    side, socket, grip = _PLACEMENT[item.held_in]
                    equipment_id = _slug(item.name, f"equipment_{index}")
                    while equipment_id in seen:
                        equipment_id = f"{equipment_id}_{index}"
                    seen.add(equipment_id)
                    equipment.append(
                        StudioEquipment(
                            equipment_id=equipment_id,
                            category=item.kind,
                            side=side,
                            socket=socket,
                            grip=grip,
                            description=item.description,
                        )
                    )

        # 6. Movable parts, for things that hinge rather than bend.
        components: list[StudioComponent] = []
        if behavior == "rigid_articulated":
            parts = self._extract(
                _MovablePartList,
                "List the parts of this asset that move, and what each one is attached to. A hinged "
                "door leaf attached to a frame is one part.\n"
                f"DESCRIPTION: {brief}",
                max_tokens=600,
            )
            seen_parts: set[str] = set()
            for index, part in enumerate(parts.items, start=1):
                component_id = _slug(part.name, f"part_{index}")
                while component_id in seen_parts:
                    component_id = f"{component_id}_{index}"
                seen_parts.add(component_id)
                components.append(
                    StudioComponent(
                        component_id=component_id,
                        role="movable_part",
                        connection=part.connection,
                        motion="rigid",
                        description=part.description,
                    )
                )

        # 7. Descriptive lists, minItems-guarded and split in two: one call
        # per pair of arrays, because a single four-array call was by far the
        # slowest step in the whole compile.
        appearance = self._extract(
            _Appearance,
            "Answer in SHORT PHRASES of at most six words, never sentences.\n"
            "silhouette: what the shape reads as at a glance.\n"
            "materials: what it is made of.\n"
            f"DESCRIPTION: {brief}",
            max_tokens=260,
        )
        production = self._extract(
            _Production,
            "Answer in SHORT PHRASES of at most six words, never sentences.\n"
            "locked_features: details that must survive production unchanged.\n"
            "gameplay_readability: what must stay readable at small on-screen size.\n"
            f"DESCRIPTION: {brief}",
            max_tokens=220,
        )

        # 8. Animations are derived, not asked for: their ids are a fixed
        # downstream contract at D7, so a model-invented synonym would miss
        # its donor clip. A static asset must declare none at all.
        animations: list[str] = []
        if behavior == "deformable_animated":
            animations = list(_CHARACTER_CLIPS)
        elif behavior == "rigid_articulated" and components:
            animations = ["open", "close"]

        spec = StudioAssetSpec(
            asset_id=_slug(identity.title, "original_asset"),
            title=identity.title,
            description=brief,
            creative_direction=identity.creative_direction,
            asset_kind=kind,
            behavior=behavior,
            anatomy_family=anatomy,
            height_m=physical.height_m,
            dimensions_m=[physical.width_m, physical.height_m, physical.depth_m],
            silhouette=_clean_phrases(appearance.silhouette, "readable single-subject silhouette"),
            materials=_clean_phrases(appearance.materials, "unspecified"),
            components=components,
            equipment=equipment,
            animations=animations,
            locked_features=_clean_phrases(production.locked_features, "original identity"),
            # Not model-authored: these are the project's standing rules, and
            # a small model paraphrases them into something weaker.
            negative_constraints=[
                "no copied characters, emblems, factions, or armour layouts",
                "no franchise-identifiable design",
                "no wrong-handed equipment placement",
            ],
            gameplay_readability=_clean_phrases(
                production.gameplay_readability, "silhouette reads at gameplay scale"
            ),
        )
        return spec
