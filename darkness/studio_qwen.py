"""Qwen description compiler and history-aware Studio gate reviewer."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import re
import secrets
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .localdeploy import LocalDeployStructuredClient
from .schemas import StrictModel
from .studio_models import (
    CORRECTION_DECISIONS,
    StudioAssetSpec,
    StudioQwenReview,
    StudioStageState,
    utc_now,
)


# Bounded prose for anything a reviewer writes.
#
# A grammar-constrained decoder generates an unbounded string until it decides
# to stop; if max_tokens runs out first, the response is cut mid-string and the
# whole object fails as invalid JSON. That is not hypothetical -- it killed D1
# (ConceptPlan ran to ~8.5k characters) and then D3 (AutomaticAssessment) in
# the same run, each time surfacing as "Invalid JSON: EOF while parsing a
# string" long after the GPU work was done.
#
# Verified on this stack that JSON Schema maxLength IS compiled into the
# grammar (a maxLength=40 field returned exactly 40 characters), so these caps
# are enforced during sampling rather than merely requested in the prompt.
# They are generous for their purpose: a gate review is read by a human in a
# card, not published.
_ReviewSummary = Annotated[str, StringConstraints(min_length=1, max_length=700)]
_ReviewPoint = Annotated[str, StringConstraints(min_length=1, max_length=240)]


class ConceptPlan(StrictModel):
    schema_version: Literal[1] = 1
    # The upper bounds are load-bearing, not decoration. A grammar-constrained
    # decoder will happily generate an unbounded string forever; if it is still
    # inside one when max_tokens runs out, the response is cut off mid-string
    # and the whole object fails as invalid JSON -- grammar guarantees shape
    # only while the model keeps producing tokens. Observed on
    # qwen3-vl:8b-instruct: positive_prompt ran to ~8.5k characters and both
    # attempts died with "EOF while parsing a string". These caps are far above
    # any useful CLIP-conditioned prompt (SDXL stops attending long before
    # this) and keep generation inside the 1600-token budget below.
    positive_prompt: str = Field(min_length=80, max_length=1400)
    negative_prompt: str = Field(min_length=20, max_length=700)
    seeds: list[int] = Field(min_length=2, max_length=2)
    rationale: str = Field(min_length=1, max_length=700)

    @model_validator(mode="after")
    def distinct_seeds(self) -> "ConceptPlan":
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("concept seeds must be distinct")
        return self


class QwenImageEditInstruction(StrictModel):
    """A model-specific instruction, never an SDXL prompt pasted into Qwen Image Edit."""

    schema_version: Literal[1] = 1
    # LocalDeploy/Ollama's JSON-grammar adapter cannot compile JSON Schema
    # ``maxLength`` constraints. The deterministic caller still keeps the
    # instruction bounded in its own prompt and evidence bundle.
    prompt: str = Field(min_length=160)
    negative_prompt: str = Field(default=" ", min_length=1)
    preserve: list[str] = Field(min_length=1, max_length=12)
    source_handling: Literal["new_text_to_image", "replace_layout_style", "revise_existing_image"]
    rationale: str = Field(min_length=1)


class ConceptCorrectionPlan(StrictModel):
    """Allowlisted image correction selected by Qwen; no arbitrary code or graph edits."""

    schema_version: Literal[1] = 1
    operation_id: Literal[
        "regenerate_complete_asset",
        "inpaint_right_hand_weapon",
        "inpaint_left_arm_shield",
        "inpaint_equipment_pair",
        "inpaint_requested_region",
    ]
    base_evidence_id: str
    edit_box_normalized: list[float] = Field(min_length=4, max_length=4)
    positive_prompt: str = Field(min_length=80)
    negative_prompt: str = Field(min_length=20)
    seeds: list[int] = Field(min_length=2, max_length=2)
    denoise: float = Field(ge=0.35, le=0.95)
    diagnosis: str = Field(min_length=1)
    preserve: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_correction(self) -> "ConceptCorrectionPlan":
        x0, y0, x1, y1 = self.edit_box_normalized
        values = [max(0.0, min(1.0, float(value))) for value in (x0, y0, x1, y1)]
        x0, y0, x1, y1 = values
        # Some vision models emit [x,y,width,height] despite the requested
        # [x0,y0,x1,y1] convention. Normalize either form into a safe box.
        if x1 <= x0:
            x1 = min(1.0, x0 + max(0.12, x1))
        if y1 <= y0:
            y1 = min(1.0, y0 + max(0.12, y1))
        if x1 <= x0:
            x0, x1 = max(0.0, x0 - 0.12), min(1.0, x0 + 0.12)
        if y1 <= y0:
            y0, y1 = max(0.0, y0 - 0.12), min(1.0, y0 + 0.12)
        self.edit_box_normalized = [x0, y0, x1, y1]
        if len(set(self.seeds)) != 2:
            raise ValueError("correction seeds must be distinct")
        return self


class GateAssessment(StrictModel):
    schema_version: Literal[1] = 1
    summary: _ReviewSummary
    strengths: list[_ReviewPoint] = Field(default_factory=list, max_length=6)
    issues: list[_ReviewPoint] = Field(default_factory=list, max_length=6)
    candidate_ranking: list[str] = Field(min_length=1, max_length=8)
    recommended_evidence_id: str
    recommended_changes: list[_ReviewPoint] = Field(default_factory=list, max_length=6)
    confidence: float = Field(ge=0, le=1)
    hard_requirements_satisfied: bool
    request_human_review: bool = True


class RevisionPlan(StrictModel):
    schema_version: Literal[1] = 1
    diagnosis: _ReviewSummary
    changes: list[_ReviewPoint] = Field(min_length=1, max_length=8)
    preserve: list[_ReviewPoint] = Field(default_factory=list, max_length=8)
    stop_reason: str | None = None


class GeometrySeedPlan(StrictModel):
    schema_version: Literal[1] = 1
    positive_prompt: str = Field(min_length=80, max_length=1400)
    negative_prompt: str = Field(min_length=20, max_length=700)
    seed: int = Field(ge=0)
    rationale: _ReviewSummary


class AutomaticAssessment(StrictModel):
    schema_version: Literal[1] = 1
    goal_satisfied: bool
    summary: _ReviewSummary
    strengths: list[_ReviewPoint] = Field(default_factory=list, max_length=6)
    issues: list[_ReviewPoint] = Field(default_factory=list, max_length=6)
    recommended_changes: list[_ReviewPoint] = Field(default_factory=list, max_length=6)
    confidence: float = Field(ge=0, le=1)


class RigidPartPlan(StrictModel):
    component_id: str
    front_box_normalized: list[float] = Field(min_length=4, max_length=4)
    pivot_normalized: list[float] = Field(min_length=2, max_length=2)
    rotation_axis: Literal["x", "y", "z"]
    minimum_degrees: float = Field(ge=-360, le=360)
    maximum_degrees: float = Field(ge=-360, le=360)
    neutral_degrees: float = Field(ge=-360, le=360)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_part(self) -> "RigidPartPlan":
        x0, y0, x1, y1 = [max(0.0, min(1.0, float(item))) for item in self.front_box_normalized]
        if x1 <= x0:
            x0, x1 = sorted((x0, x1))
        if y1 <= y0:
            y0, y1 = sorted((y0, y1))
        if x1 - x0 < 0.02 or y1 - y0 < 0.02:
            raise ValueError("rigid component front box is too small")
        self.front_box_normalized = [x0, y0, x1, y1]
        self.pivot_normalized = [max(0.0, min(1.0, float(item))) for item in self.pivot_normalized]
        if self.maximum_degrees < self.minimum_degrees:
            self.minimum_degrees, self.maximum_degrees = (
                self.maximum_degrees,
                self.minimum_degrees,
            )
        return self


class RigidStructurePlan(StrictModel):
    schema_version: Literal[1] = 1
    parts: list[RigidPartPlan] = Field(min_length=1, max_length=32)
    static_component_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


def _history(stage: StudioStageState) -> str:
    """Compact numerical/visual ledger that stays below LocalDeploy prompt limits."""

    def short(value: str, limit: int = 700) -> str:
        return value if len(value) <= limit else value[: limit - 1] + "…"

    iterations: dict[str, dict[str, object]] = {}
    for item in stage.evidence:
        iteration = int(item.metrics.get("iteration", 0) or 0)
        if iteration <= 0:
            continue
        entry = iterations.setdefault(str(iteration), {"candidates": []})
        if item.metrics.get("selectable") is True or "candidate" in item.evidence_id:
            entry["candidates"].append(
                {
                    "evidence_id": item.evidence_id,
                    "seed": item.metrics.get("seed"),
                    "operation_id": item.metrics.get("operation_id"),
                    "base_evidence_id": item.metrics.get("base_evidence_id"),
                    "width": item.metrics.get("width"),
                    "height": item.metrics.get("height"),
                    "nonblank": item.metrics.get("nonblank"),
                }
            )
    for review in stage.qwen_reviews[-6:]:
        entry = iterations.setdefault(str(review.iteration), {"candidates": []})
        entry["qwen_review"] = {
            "summary": short(review.summary),
            "issues": [short(item, 240) for item in review.issues[:6]],
            "recommended_changes": [short(item, 240) for item in review.recommended_changes[:6]],
            "ranking": review.candidate_ranking,
            "recommended_evidence_id": review.recommended_evidence_id,
            "confidence": review.confidence,
            "hard_requirements_satisfied": review.hard_requirements_satisfied,
        }
    decisions = [
        {
            "decision": item.decision,
            "comment": short(item.comment),
            "selected_evidence_id": item.selected_evidence_id,
        }
        for item in stage.human_decisions[-6:]
    ]
    return json.dumps(
        {
            "current_iteration": stage.iteration,
            "current_metrics": stage.metrics,
            "iteration_ledger": iterations,
            "ordered_human_decisions": decisions,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


# Longest edge, in pixels, of an image handed to the vision reviewer. This is
# the single highest-leverage number for local-model throughput, because a
# vision transformer's cost is driven by patch count, which is quadratic in
# image size. Qwen-VL tokenises at ~28 px patches, so D4's 1440x1392 rig
# stress board alone is roughly 2,500 vision tokens -- and Ollama serves an
# 8B model with a 4,096-token context by default. Two such images plus the
# spec and history do not fit; the server thrashes and the call runs past
# even a 900 s ceiling, which is what "ReadTimeout" at D4 actually was.
#
# At 768 px the same board is about 750 raw patches, comfortably inside the
# window, and nothing is lost for the judgement being asked: these reviews
# ask "is the deformation sane", "is the sword in the right hand" -- gross
# structure, never pixel detail. The full-resolution file remains on disk as
# evidence; only the copy handed to the model is reduced.
VISION_MAX_EDGE = 768


def _image_content(path: Path, *, max_edge: int = VISION_MAX_EDGE) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        from PIL import Image

        with Image.open(path) as image:
            if max(image.size) > max_edge:
                image = image.convert("RGB")
                image.thumbnail((max_edge, max_edge), Image.LANCZOS)
                buffer = io.BytesIO()
                # JPEG rather than PNG: these are lit renders and photographic
                # boards, where PNG is several times larger for no benefit the
                # reviewer can see, and the base64 payload is itself part of
                # the request the local server has to parse.
                image.save(buffer, format="JPEG", quality=88)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
    except Exception:
        # A reviewer running on the original bytes is far better than a stage
        # that dies because Pillow could not read one diagnostic PNG.
        pass
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def _critic_history(stage: StudioStageState) -> str:
    """Numbers and human direction only; exclude old prose that can anchor a vision critic."""
    candidates = [
        {
            "iteration": item.metrics.get("iteration"),
            "evidence_id": item.evidence_id,
            "seed": item.metrics.get("seed"),
            "operation_id": item.metrics.get("operation_id"),
            "base_evidence_id": item.metrics.get("base_evidence_id"),
        }
        for item in stage.evidence
        if "candidate" in item.evidence_id
    ][-12:]
    reviews = [
        {
            "iteration": item.iteration,
            "confidence": item.confidence,
            "hard_requirements_satisfied": item.hard_requirements_satisfied,
            "recommended_evidence_id": item.recommended_evidence_id,
        }
        for item in stage.qwen_reviews[-6:]
    ]
    decisions = [
        {
            "decision": item.decision,
            "comment": item.comment[:700],
            "selected_evidence_id": item.selected_evidence_id,
        }
        for item in stage.human_decisions[-6:]
    ]
    return json.dumps(
        {"candidate_numbers": candidates, "prior_scores": reviews, "human_decisions": decisions},
        separators=(",", ":"),
        sort_keys=True,
    )


class StudioQwen:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        spec_strategy: str = "monolithic",
        timeout_seconds: float = 120,
    ) -> None:
        # 120s suits a fast hosted endpoint. A 7-8B model generating a
        # multi-array chunk on a laptop GPU can legitimately exceed it, so
        # this is configurable rather than fixed -- a timeout here is
        # indistinguishable at the gate from a real model failure.
        self.client = LocalDeployStructuredClient(base_url=base_url, timeout_seconds=timeout_seconds)
        self.model = model
        # "monolithic" asks for the whole StudioAssetSpec in one constrained
        # call -- what the qualified 27B model does well. "chunked" composes
        # it from many small calls instead; see darkness/chunked_spec.py for
        # why a 7-8B model needs that and measurements showing it works.
        # Both paths end at the same StudioAssetSpec and the same
        # _validate_explicit_handedness check below.
        if spec_strategy not in {"monolithic", "chunked"}:
            raise ValueError("spec_strategy must be 'monolithic' or 'chunked'")
        self.spec_strategy = spec_strategy
        self.last_spec_trace: list[dict[str, object]] = []

    def compile_spec(self, description: str) -> StudioAssetSpec:
        if self.spec_strategy == "chunked":
            from .chunked_spec import ChunkedSpecCompiler

            compiler = ChunkedSpecCompiler(self.client, self.model)
            spec = compiler.compile(description)
            self.last_spec_trace = compiler.trace
            self._validate_explicit_handedness(description, spec)
            return spec
        return self._compile_spec_monolithic(description)

    def _compile_spec_monolithic(self, description: str) -> StudioAssetSpec:
        prompt = f"""
You are Qwen in ASSET ARCHITECT mode for VettedMesh Darkness. Convert the user's single description into one original,
production-ready StudioAssetSpec. First classify asset_kind and behavior. The system accepts characters, creatures,
props, architecture such as doors and walls, environments, materials, and VFX; never force a non-character into a
character schema. Describe named components, their connections, and whether each is static, rigidly articulated,
deformable, or simulated. Static assets have no animations. Rigid objects may use machine-safe actions such as open
and close. Deformable characters use canonical actions such as idle, walk, attack, hit, death, and block.

For characters and creatures, preserve explicit handedness exactly. A weapon requested in the right hand must use
side=right, socket=hand_right.grip, grip=palm_and_fingers. A left shield must use side=left,
socket=forearm_left.shield, grip=forearm_strap. Keep rigid equipment separate from the deforming body. Translate any
franchise comparison into only high-level original qualities and explicitly forbid copied characters, emblems,
factions, or armor layouts. Favor stylized, mobile-readable production art rather than photorealism unless the user
explicitly requests another original direction. Keep all ids lowercase and machine-safe. Record approximate positive
dimensions in meters and lock every explicit user requirement.

USER_DESCRIPTION={description}
""".strip()
        spec = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=StudioAssetSpec,
            max_attempts=2,
            temperature=0.15,
            max_tokens=2048,
        )
        self._validate_explicit_handedness(description, spec)
        return spec

    @staticmethod
    def _validate_explicit_handedness(description: str, spec: StudioAssetSpec) -> None:
        if spec.asset_kind not in {"character", "creature"}:
            return
        normalized = re.sub(r"\s+", " ", description.lower())
        right_sword_requested = "sword" in normalized and (
            "right hand" in normalized or "sword at the right" in normalized or "sword in the right" in normalized
        )
        left_shield_requested = "shield" in normalized and not (
            "shield right" in normalized or "shield in the right" in normalized
        )
        if right_sword_requested:
            valid = any(
                item.category == "weapon"
                and item.side == "right"
                and item.socket == "hand_right.grip"
                and item.grip == "palm_and_fingers"
                for item in spec.equipment
            )
            if not valid:
                raise ValueError("Qwen violated the explicit right-hand sword contract")
        if left_shield_requested:
            valid = any(
                item.category == "shield"
                and item.side == "left"
                and item.socket == "forearm_left.shield"
                and item.grip == "forearm_strap"
                for item in spec.equipment
            )
            if not valid:
                raise ValueError("Qwen violated the explicit left-shield contract")

    @staticmethod
    def _fresh_seeds(seeds: list[int], failed_seeds: list[int]) -> list[int]:
        result: list[int] = []
        for seed in seeds:
            candidate = int(seed)
            while candidate in failed_seeds or candidate in result:
                candidate = secrets.randbelow(2**31 - 1)
            result.append(candidate)
        return result

    def concept_plan(self, spec: StudioAssetSpec, stage: StudioStageState) -> ConceptPlan:
        rejected = [
            item.comment
            for item in stage.human_decisions
            if item.decision in CORRECTION_DECISIONS and item.comment
        ]
        failed_seeds = (
            sorted(
                {
                    int(item.metrics["seed"])
                    for item in stage.evidence
                    if item.media_type.startswith("image/") and "seed" in item.metrics
                }
            )
            if rejected
            else []
        )
        prompt = f"""
You are Qwen in DOER mode for the D1 asset concept gate. Produce one precise SDXL prompt shared by two NEW seeds.
Show the entire asset in a readable production concept, orthographic-feeling three-quarter view, centered and fully
inside frame on a FLAT PURE GREEN chroma-key background (RGB near 0,177,64) with no floor shadow and no
gradient or vignette. That backdrop is not a style choice: D2 keys this exact approved image to RGBA by
flood-filling the background inward from the border, and its test is green >= blue * 1.20, which a neutral
grey studio backdrop fails -- see make_chroma_alpha in darkness/studio_comfy.py. Apply the structured asset kind, behavior, components, dimensions,
materials, and locked features literally. For a character, show every equipment item simultaneously with correct
anatomical handedness and believable attachment/grip; the sword must be in the actual right hand and the shield on
the actual left arm when specified. For a door, wall, prop, or other static/dynamic object, show its construction and
movable pieces without inventing a body. Favor a mobile-readable silhouette and original design. Do not mention or
copy third-party IP. Avoid text, logos, cropping, missing components, duplicated components, floating attachments,
and wrong handedness.
The latest human rejection is the highest-authority diagnosis. Do not soften it, contradict it, or defer to an older
critic that called the rejected feature correct. Explicitly correct every rejection point in the positive and negative
prompts. Never reuse a seed from a rejected attempt.

STRUCTURED_SPEC={spec.model_dump_json()}
PRIOR_NUMBERS_AND_DECISIONS={_history(stage)}
LATEST_REJECTION_COMMENTS={json.dumps(rejected)}
FAILED_SEEDS_THAT_MUST_NOT_BE_REUSED={json.dumps(failed_seeds)}
""".strip()
        draft = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=ConceptPlan,
            max_attempts=2,
            temperature=0.2,
            max_tokens=1600,
        )
        referee_prompt = f"""
You are now Qwen in independent CRITIC/REFEREE mode. Audit the draft generation plan against the human's exact latest
rejection, structured spec, and failed seeds. Return a corrected complete ConceptPlan, not commentary. The final plan
must make the human feedback operational in both prompts, preserve correct features, use two distinct NEW seeds not in
FAILED_SEEDS, explicitly require all equipment simultaneously, and make missing, wrong, floating, or ungripped
equipment negative. If the user says the sword is not proper, treat the sword as failed even if an older AI review
said it was correct.

The deterministic ComfyUI worker renders the whole character in a single pass locked to one OpenPose skeleton, never
by compositing separate image regions, so there is always exactly one body. Handedness still comes entirely from the
prompt text: state which item is in the anatomical right hand and which is on the anatomical left arm in plain
language, and do not use ambiguous phrases such as "his left side" without also naming the anatomical side.

SPEC={spec.model_dump_json()}
LATEST_HUMAN_REJECTION={json.dumps(rejected[-1] if rejected else "none")}
FAILED_SEEDS={json.dumps(failed_seeds)}
DRAFT={draft.model_dump_json()}
""".strip()
        final = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": referee_prompt}],
            response_model=ConceptPlan,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1600,
        )
        safe_seeds = self._fresh_seeds(final.seeds, failed_seeds)
        return final.model_copy(update={"seeds": safe_seeds})

    def image_edit_instruction(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        plan: ConceptPlan | ConceptCorrectionPlan,
        *,
        source_handling: Literal["new_text_to_image", "replace_layout_style", "revise_existing_image"],
    ) -> QwenImageEditInstruction:
        """Rewrite a generic concept plan into one instruction appropriate for Qwen Image Edit."""
        latest_rejection = next(
            (item.comment for item in reversed(stage.human_decisions) if item.decision in CORRECTION_DECISIONS),
            "none",
        )
        source_rule = (
            "No input image is supplied. Compose the complete original asset from the structured brief rather than "
            "describing an editing operation."
            if source_handling == "new_text_to_image"
            else "The input image is only a soft composition-and-equipment layout. Replace its visual style completely; "
            "do not preserve pixels, flat vector blocks, black outlines, simple placeholder anatomy, or its background."
            if source_handling == "replace_layout_style"
            else "The input is the current candidate. Preserve only the explicitly successful locked features and "
            "replace the exact failed regions or structural defects named by the human and referee."
        )
        rewrite_prompt = f"""
You are Qwen in IMAGE-EDIT PROMPT-REWRITER mode. Return a single self-contained instruction for Qwen Image Edit,
not an SDXL prompt and not an explanation. {source_rule}

The target is an original, premium hand-painted stylized 3D fantasy game concept: smooth curved forms, believable
anatomy, physically coherent armor and equipment, controlled studio lighting, readable medium-scale details, and a
flat pure green chroma-key background. Explicitly prohibit pixel art, voxel/block forms, 8-bit rendering, retro sprites,
hard black contour lines, chibi proportions, text, logos, watermarks, duplicate people, extra limbs, equipment racks,
and floating equipment. Do not name, imitate, or refer to third-party franchises.

For character equipment, preserve exact anatomical side and physical connection. State that the complete straight
arming sword, guard, leather hilt, and pommel are held in the anatomical right hand (viewer-left in a front or
three-quarter-front view) with visibly closed fingers around the hilt. State that the single broad shield is opaque
and strapped to the anatomical left forearm (viewer-right), with no body shapes inside it. Require one connected
full-body subject with two natural separated boots on the ground, never a vehicle, hoverboard, wheel base, or pedestal.

STRUCTURED_SPEC={spec.model_dump_json()}
CURRENT_CONCEPT_PLAN={plan.model_dump_json()}
LATEST_HUMAN_REJECTION={json.dumps(latest_rejection)}
NUMERICAL_AND_DECISION_HISTORY={_history(stage)}
""".strip()
        draft = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": rewrite_prompt}],
            response_model=QwenImageEditInstruction,
            max_attempts=2,
            temperature=0.15,
            max_tokens=1700,
        )
        referee = self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Act as the image-edit instruction referee. Return a corrected complete "
                        "QwenImageEditInstruction. Check that the prompt explicitly says the input layout is not "
                        "the desired style when source_handling is replace_layout_style; that the final style is "
                        "premium hand-painted stylized 3D rather than pixel art; and that every locked component, "
                        "anatomical side, grip/strap, single-body constraint, and two separate grounded boots is "
                        "stated. Do not add trademarks or change the asset identity.\n\n"
                        f"SPEC={spec.model_dump_json()}\nSOURCE_HANDLING={source_handling}\nDRAFT={draft.model_dump_json()}"
                    ),
                }
            ],
            response_model=QwenImageEditInstruction,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1700,
        )
        # The critic is advisory.  Finish with deterministic wording for the
        # non-negotiable visual contract instead of trusting a prose model to
        # repeat every prohibition on every request.
        prompt = referee.prompt.replace("chunky heroic proportions", "heroic but natural human proportions")
        hard_contract = (
            " Do not render pixel art, retro "
            "sprites, 8-bit graphics, voxel/block forms, hard black outlines, chibi anatomy, vehicles, wheel "
            "bases, hoverboards, pedestals, duplicate people, extra limbs, extra weapons, floating equipment, "
            "text, logos, or watermarks."
        )
        if source_handling == "replace_layout_style" and "layout is not the desired rendering style" not in prompt:
            prompt += " The layout is not the desired rendering style: fully replace it."
        if "Do not render pixel art" not in prompt:
            prompt += hard_contract
        preserve = list(dict.fromkeys(
            referee.preserve
            + [
                "one connected full-body human only",
                "closed right-hand sword grip on viewer-left",
                "opaque left-forearm shield on viewer-right",
                "two separate grounded boots",
            ]
        ))
        return referee.model_copy(
            update={
                "prompt": prompt,
                "preserve": preserve[:12],
                "source_handling": source_handling,
            }
        )

    def concept_correction_plan(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        candidate_ids: list[str],
        comparison_board: Path | None = None,
    ) -> ConceptCorrectionPlan:
        latest_rejection = next(
            (item.comment for item in reversed(stage.human_decisions) if item.decision in CORRECTION_DECISIONS),
            "",
        )
        failed_seeds = sorted(
            {
                int(item.metrics["seed"])
                for item in stage.evidence
                if item.media_type.startswith("image/") and "seed" in item.metrics
            }
        )
        prompt = f"""
You are Qwen in IMAGE CORRECTION DOER mode. The human rejected the latest D1 result. Choose exactly one allowlisted
operation and one prior visual candidate as a base. Prefer a candidate that already preserves most of the asset.
Use inpainting when the complaint is localized; use regenerate_complete_asset only when no prior candidate can be
repaired. The box [x0,y0,x1,y1] uses normalized image coordinates and must tightly cover the defective object plus
its physical attachment. For a sword/grip include the entire relevant arm, hand, sword, and clearance around them.
For a shield include the entire relevant arm and shield. If both are defective, cover both sides or regenerate.
The positive prompt must restate the WHOLE approved asset and explicitly say all required components must appear
simultaneously. The negative prompt must forbid the exact rejected failure. Human text is authoritative. Use two new
seeds and preserve everything not named for change.

You choose semantic intent, not arbitrary graph edits. The deterministic worker implements your choice using one of
two qualified workflows: (1) a fresh single-pass regeneration of the whole body locked to one OpenPose skeleton, with
handedness stated directly in the prompt, or (2) a high-resolution local crop repaired with VAEEncodeForInpaint and
composited back onto the original image. If the human says both sword and shield placement are bad, choose
regenerate_complete_asset so the whole figure and its pose are solved together. If only the grip/weapon is bad and
the rest is worth preserving, choose inpaint_right_hand_weapon.

SPEC={spec.model_dump_json()}
LATEST_HUMAN_REJECTION={json.dumps(latest_rejection)}
AVAILABLE_VISUAL_EVIDENCE_IDS={json.dumps(candidate_ids)}
FAILED_SEEDS={json.dumps(failed_seeds)}
FULL_HISTORY={_history(stage)}
""".strip()
        doer_content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if comparison_board is not None:
            doer_content.extend(
                [
                    {"type": "text", "text": "LABELED VISUAL HISTORY FOR THE CORRECTION:"},
                    _image_content(comparison_board),
                ]
            )
        draft = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": doer_content}],
            response_model=ConceptCorrectionPlan,
            max_attempts=2,
            temperature=0.15,
            max_tokens=1800,
        )
        referee_prompt = f"""
You are the independent Qwen CORRECTION REFEREE. Return a complete corrected ConceptCorrectionPlan. Verify that the
chosen base_evidence_id is in AVAILABLE_IDS, the operation directly implements the human rejection, the edit box
contains the defective component and attachment, both required sword and shield are demanded simultaneously when
specified, and unrelated good features are preserved. Do not accept a vague style-only change. Do not reuse any
failed seed. If the base is missing one item but otherwise good, inpaint that missing item; if each candidate is
missing a different required item, pick the better base and add the missing item.

SPEC={spec.model_dump_json()}
HUMAN_REJECTION={json.dumps(latest_rejection)}
AVAILABLE_IDS={json.dumps(candidate_ids)}
FAILED_SEEDS={json.dumps(failed_seeds)}
DRAFT={draft.model_dump_json()}
""".strip()
        referee_content: list[dict[str, object]] = [{"type": "text", "text": referee_prompt}]
        if comparison_board is not None:
            referee_content.extend(
                [
                    {"type": "text", "text": "VERIFY AGAINST THIS SAME LABELED VISUAL HISTORY:"},
                    _image_content(comparison_board),
                ]
            )
        final = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": referee_content}],
            response_model=ConceptCorrectionPlan,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1800,
        )
        if final.base_evidence_id not in candidate_ids:
            raise ValueError("Qwen correction referee selected an unavailable base image")
        updates: dict[str, object] = {"seeds": self._fresh_seeds(final.seeds, failed_seeds)}
        rejection_lower = latest_rejection.lower()
        needs_right_weapon = any(
            item.category == "weapon" and item.side == "right" for item in spec.equipment
        ) and any(word in rejection_lower for word in ("sword", "weapon", "grip", "right hand"))
        needs_left_shield = any(
            item.category == "shield" and item.side == "left" for item in spec.equipment
        ) and "shield" in rejection_lower
        if needs_right_weapon and needs_left_shield:
            updates.update(
                {
                    "operation_id": "regenerate_complete_asset",
                    "edit_box_normalized": [0.0, 0.0, 1.0, 1.0],
                    "positive_prompt": final.positive_prompt
                    + " ONE SINGLE BODY, ONE OPENPOSE SKELETON: the anatomical right hand visibly closes around the "
                    "hilt of exactly one complete arming sword; the anatomical left forearm securely straps exactly "
                    "one broad shield. Both complete props are visible at once on the same figure.",
                    "negative_prompt": final.negative_prompt
                    + ", reversed handedness, empty right hand, sword in left hand, shield on right arm, "
                    "floating equipment, fused hilt, open weapon hand, duplicate weapon, duplicate shield, "
                    "extra limbs, duplicate body",
                }
            )
        elif final.operation_id != "regenerate_complete_asset" and needs_right_weapon:
            updates.update(
                {
                    "operation_id": "inpaint_right_hand_weapon",
                    # A front/three-quarter production view places anatomical right on viewer-left.
                    # Include hand, arm, complete weapon arc, and surrounding background.
                    "edit_box_normalized": [0.01, 0.10, 0.52, 0.94],
                    "positive_prompt": final.positive_prompt
                    + " INPAINT TASK: visibly add one complete arming sword, including blade, guard, hilt and pommel, "
                    "inside the masked viewer-left region. The closed anatomical right hand must wrap around the hilt. "
                    "Keep the existing left-arm shield visible and unchanged.",
                    "negative_prompt": final.negative_prompt
                    + ", empty right hand, hidden sword, cropped blade, sword on left, open weapon hand",
                }
            )
        elif final.operation_id != "regenerate_complete_asset" and needs_left_shield:
            updates.update(
                {
                    "operation_id": "inpaint_left_arm_shield",
                    "edit_box_normalized": [0.48, 0.10, 0.99, 0.94],
                }
            )
        return final.model_copy(update=updates)

    def review_concepts(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        images: list[tuple[str, Path, dict[str, object]]],
        *,
        comparison_board: Path | None = None,
    ) -> StudioQwenReview:
        if not 1 <= len(images) <= 2:
            raise ValueError("the qualified Qwen profile accepts one or two concept images")
        latest_rejection = next(
            (item.comment for item in reversed(stage.human_decisions) if item.decision in CORRECTION_DECISIONS),
            "none",
        )
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "You are the independent D1 visual/numerical reviewer. Rank the supplied candidates by exact "
                    "evidence_id. Check identity, full-asset readability, original design, every locked component, "
                    "behavior cues, dimensions, silhouette, cropping, and obvious "
                    "generation defects. Human rejection is authoritative: never minimize it because an older critic "
                    "disagreed. For equipped characters, check explicit equipment side, grip, and attachment; when "
                    "required, sword AND shield must exist simultaneously and a candidate missing either fails. "
                    "Use the labeled evidence IDs. Do not claim a change without visible evidence. A human must make "
                    "the final decision. In a frontal or three-quarter-front view, anatomical right normally appears "
                    "on the viewer-left; do not reverse this. The board contains both full views and enlarged hand / "
                    "equipment crops. Inspect the crop before claiming an object is absent. Ignore conclusions from "
                    "older reviews and judge only the current pixels. Set hard_requirements_satisfied=false if every "
                    "candidate misses, duplicates, "
                    "misplaces, or fails to attach/grip any locked component; style preference alone is not a hard "
                    "failure.\n\n"
                    f"SPEC={spec.model_dump_json()}\n"
                    f"CANDIDATES={json.dumps([{'evidence_id': item[0], 'metrics': item[2]} for item in images])}\n"
                    f"LATEST_HUMAN_REJECTION={json.dumps(latest_rejection)}\n"
                    f"NUMERICAL_AND_HUMAN_HISTORY_WITHOUT_OLD_PROSE={_critic_history(stage)}"
                ),
            }
        ]
        if comparison_board is not None:
            content.append({"type": "text", "text": "LABELED PREVIOUS-VS-CURRENT COMPARISON BOARD:"})
            content.append(_image_content(comparison_board))
        else:
            for evidence_id, path, metrics in images:
                content.append(
                    {
                        "type": "text",
                        "text": f"NEXT IMAGE evidence_id={evidence_id} metrics={json.dumps(metrics)}",
                    }
                )
                content.append(_image_content(path))
        first = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=GateAssessment,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1600,
        )
        referee_content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "You are a second independent vision referee. Re-check the CURRENT labeled pixels and enlarged "
                    "hand/equipment crops, then return a complete corrected GateAssessment. Do not inherit a visual "
                    "claim merely because the first critic wrote it. In particular, verify whether a sword blade, "
                    "hilt and closed grip are visibly present on viewer-left/anatomical-right, and whether the shield "
                    "is simultaneously present on viewer-right/anatomical-left. Candidate IDs and ranking must use "
                    "each supplied current ID exactly once. Separate hard missing/misplaced components from softer "
                    "style shortcomings.\n\n"
                    f"SPEC={spec.model_dump_json()}\n"
                    f"CURRENT_IDS={json.dumps([item[0] for item in images])}\n"
                    f"LATEST_HUMAN_REJECTION={json.dumps(latest_rejection)}\n"
                    f"FIRST_CRITIC_DRAFT={first.model_dump_json()}"
                ),
            }
        ]
        if comparison_board is not None:
            referee_content.append(_image_content(comparison_board))
        else:
            for evidence_id, path, metrics in images:
                referee_content.append(
                    {"type": "text", "text": f"IMAGE={evidence_id} METRICS={json.dumps(metrics)}"}
                )
                referee_content.append(_image_content(path))
        try:
            result = self.client.request(
                model=self.model,
                messages=[{"role": "user", "content": referee_content}],
                response_model=GateAssessment,
                max_attempts=2,
                temperature=0.0,
                max_tokens=1600,
            )
        except Exception:
            result = first
        ids = {item[0] for item in images}
        if set(result.candidate_ranking) != ids or result.recommended_evidence_id not in ids:
            raise ValueError("Qwen concept ranking did not contain each supplied evidence id exactly once")
        return StudioQwenReview(
            review_id=f"d1.qwen.iteration-{stage.iteration:02d}",
            created_at=utc_now(),
            stage_id="D1",
            iteration=stage.iteration,
            summary=result.summary,
            strengths=result.strengths,
            issues=result.issues,
            candidate_ranking=result.candidate_ranking,
            recommended_evidence_id=result.recommended_evidence_id,
            recommended_changes=result.recommended_changes,
            confidence=result.confidence,
            hard_requirements_satisfied=result.hard_requirements_satisfied,
            request_human_review=True,
        )

    def geometry_seed_plan(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        selected_concept: Path,
    ) -> GeometrySeedPlan:
        content = [
            {
                "type": "text",
                "text": (
                    "Prepare a single SDXL geometry-seed prompt based on this approved identity concept. The output "
                    "must preserve the same approved original asset alone, centered, fully visible, straight-on, on a "
                    "flat pure green chroma-key background with no floor shadow. For deformable characters, use an "
                    "unarmed neutral symmetrical A-pose with both open empty hands; rigid equipment is intentionally "
                    "omitted because Darkness constructs and sockets it separately. For props or architecture, show "
                    "the complete object and all structural/movable pieces in their neutral state; do not invent a "
                    "humanoid pose. Preserve identity, materials, colors and proportions. Avoid text, scenery and "
                    "perspective distortion. Include "
                    "prior numerical/Qwen/human history when choosing the seed and wording.\n\n"
                    f"SPEC={spec.model_dump_json()}\nHISTORY={_history(stage)}"
                ),
            },
            _image_content(selected_concept),
        ]
        return self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=GeometrySeedPlan,
            max_attempts=2,
            temperature=0.1,
            max_tokens=1400,
        )

    def review_geometry(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        selected_concept: Path,
        diagnostic: Path,
        metrics: dict[str, object],
    ) -> tuple[StudioQwenReview, bool]:
        content = [
            {
                "type": "text",
                "text": (
                    "Review a D2 image-to-3D candidate. Image 1 is the approved concept; image 2 is a deterministic "
                    "four-view render of the current mesh. Compare silhouette, component identity, proportions, "
                    "complete structure, a neutral production-ready state, and catastrophic geometry. For characters, "
                    "equipment absence is intentional at this stage and must not be treated as a defect. Hard numeric "
                    "gates are authoritative. goal_satisfied should be "
                    "true only if this is a credible cleanup/rigging input, not if it is production-perfect. Give "
                    "bounded changes for the next seed if false.\n\n"
                    f"SPEC={spec.model_dump_json()}\nCURRENT_METRICS={json.dumps(metrics, sort_keys=True)}\n"
                    f"FULL_HISTORY={_history(stage)}"
                ),
            },
            _image_content(selected_concept),
            _image_content(diagnostic),
        ]
        result = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=AutomaticAssessment,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1400,
        )
        review = StudioQwenReview(
            review_id=f"d2.qwen.iteration-{stage.iteration:02d}",
            stage_id="D2",
            iteration=stage.iteration,
            summary=result.summary,
            strengths=result.strengths,
            issues=result.issues,
            candidate_ranking=[f"d2-i{stage.iteration:02d}-diagnostic"],
            recommended_evidence_id=f"d2-i{stage.iteration:02d}-diagnostic",
            recommended_changes=result.recommended_changes,
            confidence=result.confidence,
            request_human_review=not result.goal_satisfied,
        )
        return review, result.goal_satisfied

    def review_cleanup(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        selected_concept: Path,
        diagnostic: Path,
        metrics: dict[str, object],
    ) -> tuple[StudioQwenReview, bool]:
        result = self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Review a D3 deterministic cleanup candidate. Image 1 is the human-approved concept; "
                                "image 2 is a labeled fixed-view mesh diagnostic after cleanup/export/re-import. "
                                "Numeric topology gates are authoritative. Judge identity preservation, missing or "
                                "floating structural pieces, silhouette damage, and whether the result is credible "
                                "for its typed behavior. Production-perfect materials are not required yet. Return "
                                "goal_satisfied=true only when cleanup can advance.\n\n"
                                f"SPEC={spec.model_dump_json()}\nMETRICS={json.dumps(metrics, sort_keys=True)}\n"
                                f"HISTORY={_history(stage)}"
                            ),
                        },
                        _image_content(selected_concept),
                        _image_content(diagnostic),
                    ],
                }
            ],
            response_model=AutomaticAssessment,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1400,
        )
        evidence_id = f"d3-i{stage.iteration:02d}-diagnostic"
        review = StudioQwenReview(
            review_id=f"d3.qwen.iteration-{stage.iteration:02d}",
            stage_id="D3",
            iteration=stage.iteration,
            summary=result.summary,
            strengths=result.strengths,
            issues=result.issues,
            candidate_ranking=[evidence_id],
            recommended_evidence_id=evidence_id,
            recommended_changes=result.recommended_changes,
            confidence=result.confidence,
            hard_requirements_satisfied=result.goal_satisfied,
            request_human_review=not result.goal_satisfied,
        )
        return review, result.goal_satisfied

    def review_deformable_rig(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        selected_concept: Path,
        stress_board: Path,
        metrics: dict[str, object],
    ) -> StudioQwenReview:
        result = self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Review a D4-D6 canonical rig/deformation proposal. Image 1 is the approved identity; "
                                "image 2 is a labeled board of neutral, shoulder/elbow stress, and hip/knee stress "
                                "renders. Use numeric landmark, bone, weight and deformation gates together with visible "
                                "evidence. Check only critical joints, silhouette preservation, collapsing shoulders, "
                                "elbows, hips and knees, and whether the body remains recognizable. This is the human "
                                "canonical-structure gate: list concrete issues and never claim perfection.\n\n"
                                f"SPEC={spec.model_dump_json()}\nMETRICS={json.dumps(metrics, sort_keys=True)}\n"
                                f"HISTORY={_history(stage)}"
                            ),
                        },
                        _image_content(selected_concept),
                        _image_content(stress_board),
                    ],
                }
            ],
            response_model=AutomaticAssessment,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1500,
        )
        evidence_id = f"d4-i{stage.iteration:02d}-rig-stress-board"
        return StudioQwenReview(
            review_id=f"d4.qwen.iteration-{stage.iteration:02d}",
            stage_id="D4",
            iteration=stage.iteration,
            summary=result.summary,
            strengths=result.strengths,
            issues=result.issues,
            candidate_ranking=[evidence_id],
            recommended_evidence_id=evidence_id,
            recommended_changes=result.recommended_changes,
            confidence=result.confidence,
            hard_requirements_satisfied=result.goal_satisfied,
            request_human_review=True,
        )

    def rigid_structure_plan(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        selected_concept: Path,
        diagnostic: Path,
    ) -> RigidStructurePlan:
        movable = [item.model_dump(mode="json") for item in spec.components if item.motion == "rigid"]
        content = [
            {
                "type": "text",
                "text": (
                    "Act as the D4 rigid-articulation DOER. From the approved concept and fixed front diagnostic, "
                    "propose one normalized front-view segmentation box and pivot for every requested rigid movable "
                    "component. Coordinates are image x0,y0,x1,y1 and pivot x,y from top-left. Use Blender world axis "
                    "x/y/z and bounded degree limits. A two-leaf doorway normally uses vertical z hinge axes at each "
                    "outer edge. Do not invent components. Static ids list every nonmoving component.\n\n"
                    f"SPEC={spec.model_dump_json()}\nMOVABLE={json.dumps(movable)}\nHISTORY={_history(stage)}"
                ),
            },
            _image_content(selected_concept),
            _image_content(diagnostic),
        ]
        plan = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=RigidStructurePlan,
            max_attempts=2,
            temperature=0.05,
            max_tokens=1800,
        )
        required = {item.component_id for item in spec.components if item.motion == "rigid"}
        supplied = {item.component_id for item in plan.parts}
        if supplied != required:
            raise ValueError(f"Qwen rigid structure plan component mismatch: {supplied} != {required}")
        return plan

    def review_rigid_motion(
        self,
        spec: StudioAssetSpec,
        stage: StudioStageState,
        selected_concept: Path,
        motion_board: Path,
        metrics: dict[str, object],
    ) -> StudioQwenReview:
        result = self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Review a D7 rigid motion board. Image 1 is the approved identity concept; image 2 "
                                "compares neutral and actuated states. Verify every requested movable part, plausible "
                                "pivot placement, correct axis/direction, no detached or badly intersecting geometry, "
                                "and readable before/after state. Numeric component/action gates are authoritative. "
                                "This result always goes to a human motion gate.\n\n"
                                f"SPEC={spec.model_dump_json()}\nMETRICS={json.dumps(metrics, sort_keys=True)}"
                            ),
                        },
                        _image_content(selected_concept),
                        _image_content(motion_board),
                    ],
                }
            ],
            response_model=AutomaticAssessment,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1400,
        )
        evidence_id = f"d7-i{stage.iteration:02d}-rigid-motion-board"
        return StudioQwenReview(
            review_id=f"d7.rigid.qwen.iteration-{stage.iteration:02d}",
            stage_id="D7",
            iteration=stage.iteration,
            summary=result.summary,
            strengths=result.strengths,
            issues=result.issues,
            candidate_ranking=[evidence_id],
            recommended_evidence_id=evidence_id,
            recommended_changes=result.recommended_changes,
            confidence=result.confidence,
            hard_requirements_satisfied=result.goal_satisfied,
            request_human_review=True,
        )

    def revision_plan(self, spec: StudioAssetSpec, stage: StudioStageState) -> RevisionPlan:
        return self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You mediate a rejected Darkness gate. Diagnose the rejection using the complete ordered "
                        "history, specify bounded changes for the same deterministic worker, and list what must be "
                        "preserved. Do not emit code, commands, or file paths. If repeated attempts show no meaningful "
                        "difference, set stop_reason and request a different production strategy.\n\n"
                        f"SPEC={spec.model_dump_json()}\nHISTORY={_history(stage)}"
                    ),
                }
            ],
            response_model=RevisionPlan,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1200,
        )
