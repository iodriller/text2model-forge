"""Versioned persistence contracts for the local Darkness Studio control plane."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Literal

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
    decision: Literal["approve", "reject"]
    comment: str = ""
    selected_evidence_id: str | None = None
    evidence_hashes: dict[str, str] = Field(default_factory=dict)


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
