"""Versioned persistence contracts for the local Darkness Studio control plane."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Literal

from pydantic import Field, model_validator

from .schemas import StrictModel


STAGE_DEFINITIONS = (
    ("D0", "Brief", False),
    ("D1", "Concept", True),
    ("D2", "3D generation", False),
    ("D3", "Cleanup", False),
    ("D4", "Canonical structure / skeleton", True),
    ("D5", "Rig / articulation", False),
    ("D6", "Skinning / deformation", False),
    ("D7", "Motion", True),
    ("D8", "Surface painting", True),
    ("D9", "Sprite rendering", False),
    ("D10", "Runtime validation", True),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StudioEquipment(StrictModel):
    equipment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    category: Literal["weapon", "shield", "armor", "attachment"]
    side: Literal["left", "right", "center"]
    socket: str = Field(min_length=1)
    grip: Literal["none", "palm_and_fingers", "forearm_strap"]
    rigid: bool = True
    description: str = Field(min_length=1)
    visual_requirements: list[str] = Field(default_factory=list)


class StudioComponent(StrictModel):
    """A named part whose identity, connection, and possible motion must survive production."""

    component_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    role: Literal["body", "movable_part", "attachment", "surface", "effect"]
    connection: str = Field(min_length=1)
    motion: Literal["none", "rigid", "deformable", "simulated"] = "none"
    description: str = Field(min_length=1)
    visual_requirements: list[str] = Field(default_factory=list)


class StudioAssetSpec(StrictModel):
    schema_version: Literal[1] = 1
    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    creative_direction: str = Field(min_length=1)
    asset_kind: Literal[
        "character", "creature", "prop", "architecture", "environment", "material", "vfx"
    ] = "character"
    behavior: Literal["static", "rigid_articulated", "deformable_animated", "simulated"] = (
        "deformable_animated"
    )
    anatomy_family: Literal["humanoid", "short_biped", "quadruped", "custom"] | None = "humanoid"
    height_m: float | None = Field(default=1.8, gt=0.01, lt=1000.0)
    dimensions_m: list[float] = Field(default_factory=lambda: [1.0, 1.8, 1.0], min_length=3, max_length=3)
    silhouette: list[str] = Field(min_length=1)
    materials: list[str] = Field(min_length=1)
    components: list[StudioComponent] = Field(default_factory=list)
    equipment: list[StudioEquipment] = Field(default_factory=list)
    animations: list[str] = Field(default_factory=list)
    locked_features: list[str] = Field(min_length=1)
    negative_constraints: list[str] = Field(min_length=1)
    gameplay_readability: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_asset_contract(self) -> "StudioAssetSpec":
        if any(value <= 0 for value in self.dimensions_m):
            raise ValueError("asset dimensions must be positive")
        for animation in self.animations:
            if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", animation):
                raise ValueError("animation ids must be lowercase and machine-safe")
        if self.behavior == "static" and self.animations:
            raise ValueError("static assets cannot declare animation clips")
        if self.asset_kind in {"character", "creature"} and self.behavior == "deformable_animated":
            if self.anatomy_family is None:
                raise ValueError("deformable characters and creatures require an anatomy family")
        return self


# Compatibility name retained for existing adapters and saved character runs.
StudioCharacterSpec = StudioAssetSpec


class StudioEvidence(StrictModel):
    evidence_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    label: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


class StudioQwenReview(StrictModel):
    review_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    stage_id: str = Field(pattern=r"^D(?:10|[0-9])$")
    iteration: int = Field(ge=1)
    summary: str = Field(min_length=1)
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    candidate_ranking: list[str] = Field(default_factory=list)
    recommended_evidence_id: str | None = None
    recommended_changes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    hard_requirements_satisfied: bool = True
    request_human_review: bool = True


class StudioHumanDecision(StrictModel):
    decision_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    # approve/reject are the original two verdicts. retry re-runs the same
    # stage with no implied quality judgement (a reroll). edit is a reject
    # that carries a concrete correction in `overrides` for the next attempt.
    # skip marks a stage not applicable without invalidating anything after
    # it. rollback reopens an earlier, already-decided stage and invalidates
    # everything from there forward -- see StudioStore.decide().
    decision: Literal["approve", "reject", "retry", "edit", "skip", "rollback"]
    comment: str = ""
    selected_evidence_id: str | None = None
    evidence_hashes: dict[str, str] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    target_stage_id: str | None = None


# Decisions whose comment and overrides are a correction that the next attempt
# of the same stage must consume. "reject" is the original verdict; "edit" is
# the same thing with an explicit correction attached. Anything that reads "the
# human's latest correction" must treat both identically -- filtering on
# "reject" alone silently discards an edit's comment and overrides.
CORRECTION_DECISIONS = frozenset({"reject", "edit"})


def latest_correction(stage: "StudioStageState") -> "StudioHumanDecision | None":
    """The most recent decision carrying a correction for the next attempt."""
    return next(
        (
            item
            for item in reversed(stage.human_decisions)
            if item.decision in CORRECTION_DECISIONS
        ),
        None,
    )


def awaiting_correction(stage: "StudioStageState") -> bool:
    """True when this stage's most recent decision asked for another attempt
    with a correction, rather than approving, skipping, or plain-retrying."""
    return bool(
        stage.human_decisions and stage.human_decisions[-1].decision in CORRECTION_DECISIONS
    )


def validate_stage_overrides(overrides: dict[str, Any] | None) -> None:
    """Reject malformed per-attempt override values at decision time.

    Without this the check happens deep inside the stage runner, so a typo
    like {"seed": -5} is accepted by the gate, reported as success, and only
    surfaces later as an asynchronously failed stage -- far from the input
    that caused it. Validating here means the web form (and any API caller)
    gets an immediate, actionable error instead.

    Unknown keys are deliberately allowed: a stage may define its own
    overrides, and this must not become a chokepoint that has to be updated
    before any stage can add one. Only the shape of known keys is enforced.
    """
    if not overrides:
        return
    seed = overrides.get("seed")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        raise ValueError("the 'seed' override must be a non-negative whole number")
    for key in ("concept_steps",):
        value = overrides.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 150
        ):
            raise ValueError(f"the '{key}' override must be a whole number between 1 and 150")
    for key in ("concept_cfg",):
        value = overrides.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value <= 30
        ):
            raise ValueError(f"the '{key}' override must be a number greater than 0 and at most 30")


class StudioStageState(StrictModel):
    stage_id: str = Field(pattern=r"^D(?:10|[0-9])$")
    label: str = Field(min_length=1)
    gate_required: bool
    state: Literal[
        "pending",
        "queued",
        "running",
        "awaiting_review",
        "approved",
        "rejected",
        "blocked",
        "failed",
        "skipped",
    ] = "pending"
    applicable: bool = True
    progress: float = Field(default=0, ge=0, le=1)
    message: str = "Waiting for the preceding stage."
    iteration: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    evidence: list[StudioEvidence] = Field(default_factory=list)
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    qwen_reviews: list[StudioQwenReview] = Field(default_factory=list)
    human_decisions: list[StudioHumanDecision] = Field(default_factory=list)
    error: str | None = None
    # Set by a retry or edit decision; consumed and cleared by the next run
    # of this stage. Empty for an ordinary approve/reject-driven attempt.
    pending_overrides: dict[str, Any] = Field(default_factory=dict)


class StudioRun(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    description: str = Field(min_length=1)
    title: str = "New Darkness asset"
    state: Literal[
        "created", "running", "awaiting_review", "blocked", "failed", "completed"
    ] = "created"
    current_stage: str = "D0"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    model: str = "qwen3_6_27b"
    comfy_url: str = "http://127.0.0.1:8188"
    localdeploy_url: str = "http://127.0.0.1:8000/v1"
    # "auto" prefers native Qwen Image 2512 text-to-image generation when its
    # model trio is present in ComfyUI, while keeping the existing SDXL route
    # usable on a machine that has not installed Qwen yet.
    concept_backend: Literal["auto", "qwen_image_2512", "qwen_image_edit_2511", "sdxl"] = "auto"
    checkpoint: str = "dreamshaper_xl_v2_turbo.safetensors"
    style_lora: str | None = "Warcraft style.safetensors"
    style_lora_strength: float = Field(default=0.8, ge=0.0, le=1.5)
    style_lora_trigger: str | None = "World of Warcraft cinematic style"
    prop_lora: str | None = None
    prop_lora_strength: float = Field(default=0.6, ge=0.0, le=1.5)
    # Defaults match concept_workflow()'s own long-standing steps/cfg defaults
    # in studio_comfy.py exactly, so a run created with no quality override
    # behaves identically to before these fields existed. See
    # darkness/settings.py's quality_overrides() for how [asset].quality
    # picks a [quality.<tier>] section that can change these two.
    concept_steps: int = Field(default=30, ge=1, le=150)
    concept_cfg: float = Field(default=6.0, gt=0, le=30)
    # Which profiles/<name>.toml this run resolves its per-stage settings
    # from. Stored so a resumed run keeps the configuration it started with
    # rather than silently adopting whatever the profile says today.
    profile: str = Field(default="simple", pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    spec: StudioAssetSpec | None = None
    stages: list[StudioStageState]
    event_count: int = 0

    def stage(self, stage_id: str) -> StudioStageState:
        for item in self.stages:
            if item.stage_id == stage_id:
                return item
        raise KeyError(stage_id)


def new_studio_run(
    run_id: str, description: str, overrides: dict[str, object] | None = None
) -> StudioRun:
    """Create a run. `overrides` (see darkness.settings.studio_overrides) may set
    any of StudioRun's own configuration fields; run_id/description/stages are
    always computed here and cannot be overridden this way."""
    return StudioRun(
        **(overrides or {}),
        run_id=run_id,
        description=description.strip(),
        stages=[
            StudioStageState(stage_id=stage_id, label=label, gate_required=gate)
            for stage_id, label, gate in STAGE_DEFINITIONS
        ],
    )
