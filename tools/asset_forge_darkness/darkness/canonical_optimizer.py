"""Typed, bounded operations for canonical short-biped rig and motion review."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    AssetStage,
    OperationDefinition,
    OperationProposal,
    OptimizerComparison,
    OptimizerDecision,
    OptimizerObservation,
    RootCauseHypothesis,
)


class StrictParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LandmarkPair(str, Enum):
    shoulders = "shoulders"
    elbows = "elbows"
    wrists = "wrists"
    hands = "hands"
    hips = "hips"
    knees = "knees"
    ankles = "ankles"
    feet = "feet"


class LandmarkAxis(str, Enum):
    lateral = "lateral"
    depth = "depth"
    height = "height"


class AdjustLandmarkPairParameters(StrictParameters):
    landmark_pair: LandmarkPair
    axis: LandmarkAxis
    delta_fraction: Annotated[float, Field(ge=-0.025, le=0.025)]

    @model_validator(mode="after")
    def require_nonzero_delta(self) -> "AdjustLandmarkPairParameters":
        if abs(self.delta_fraction) < 0.0025:
            raise ValueError("delta_fraction magnitude must be between 0.0025 and 0.025")
        return self


class AdjustMotionTimingParameters(StrictParameters):
    clip: Literal["idle", "walk", "attack", "hit", "death"]
    phase: Literal["anticipation", "contact", "impact", "recovery", "settle"]
    delta_frames: Annotated[int, Field(ge=-3, le=3)]

    @model_validator(mode="after")
    def require_nonzero_delta(self) -> "AdjustMotionTimingParameters":
        if self.delta_frames == 0:
            raise ValueError("delta_frames must not be zero")
        return self


class LockFootContactParameters(StrictParameters):
    clip: Literal["walk", "attack", "hit", "death"]
    side: Literal["left", "right"]
    start_frame: Annotated[int, Field(ge=0, le=240)]
    end_frame: Annotated[int, Field(ge=0, le=240)]

    @model_validator(mode="after")
    def bound_contact_window(self) -> "LockFootContactParameters":
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be at least start_frame")
        if self.end_frame - self.start_frame > 12:
            raise ValueError("a single bounded contact window may span at most 12 frames")
        return self


class WeightJointPair(str, Enum):
    shoulders = "shoulders"
    elbows = "elbows"
    hips = "hips"
    knees = "knees"


class WeightTransferDirection(str, Enum):
    parent_to_child = "parent_to_child"
    child_to_parent = "child_to_parent"


class RedistributeJointPairWeightsParameters(StrictParameters):
    joint_pair: WeightJointPair
    direction: WeightTransferDirection
    transfer_fraction: Annotated[float, Field(ge=0.025, le=0.15)]
    radius_fraction: Annotated[float, Field(ge=0.03, le=0.12)]


class RigQwenLandmarkParameters(StrictParameters):
    landmark_pair: LandmarkPair
    axis: LandmarkAxis
    delta_fraction: Literal[-0.025, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.025]


class RigQwenWeightParameters(StrictParameters):
    joint_pair: WeightJointPair
    direction: WeightTransferDirection
    transfer_fraction: Literal[0.05, 0.1, 0.15]
    radius_fraction: Literal[0.05, 0.08, 0.12]


class RigAdjustLandmarkProposal(OperationProposal):
    operation_id: Literal["rig.adjust_landmark_pair"]
    stage: Literal[AssetStage.rig]
    parameters: RigQwenLandmarkParameters


class RigRedistributeWeightsProposal(OperationProposal):
    operation_id: Literal["skin.redistribute_joint_pair_weights"]
    stage: Literal[AssetStage.rig]
    parameters: RigQwenWeightParameters


RigOptimizerProposal = Annotated[
    RigAdjustLandmarkProposal | RigRedistributeWeightsProposal,
    Field(discriminator="operation_id"),
]


class RigQwenComparison(StrictParameters):
    preferred: Literal["previous", "current", "tie", "uncertain"]
    reason: str = Field(min_length=1)


class RigQwenObservation(StrictParameters):
    region: str = Field(min_length=1)
    issue: str = Field(min_length=1)


class RigQwenRootCause(StrictParameters):
    cause: str = Field(min_length=1)


class CanonicalRigOptimizerDecision(StrictParameters):
    schema_version: Literal[1] = 1
    goal_satisfied: bool
    comparison: RigQwenComparison | None = None
    observations: list[RigQwenObservation] = Field(default_factory=list, max_length=4)
    root_causes: list[RigQwenRootCause] = Field(default_factory=list, max_length=3)
    proposals: list[RigOptimizerProposal] = Field(
        default_factory=list,
        max_length=1,
    )
    preserve: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    request_human_review: bool = False

    def to_optimizer_decision(self) -> OptimizerDecision:
        confidence = {"low": 0.35, "medium": 0.65, "high": 0.9}[self.confidence]
        preferred = None if self.comparison is None else self.comparison.preferred
        comparison_delta = {
            None: 0.0,
            "previous": -0.1,
            "current": 0.1,
            "tie": 0.0,
            "uncertain": 0.0,
        }[preferred]
        return OptimizerDecision(
            goal_satisfied=self.goal_satisfied,
            visual_score=None,
            technical_score=0.5,
            observations=[
                OptimizerObservation(region=item.region, issue=item.issue, severity=0.5)
                for item in self.observations
            ],
            root_causes=[
                RootCauseHypothesis(cause=item.cause, probability=confidence)
                for item in self.root_causes
            ],
            proposals=[
                OperationProposal.model_validate(item.model_dump(mode="json"))
                for item in self.proposals
            ],
            preserve=self.preserve,
            confidence=confidence,
            request_human_review=self.request_human_review,
            comparison=(
                None
                if self.comparison is None
                else OptimizerComparison(
                    preferred=self.comparison.preferred,
                    visual_delta=comparison_delta,
                    reason=self.comparison.reason,
                )
            ),
        )


PAIR_LANDMARKS = {
    LandmarkPair.shoulders: ("shoulder_l", "shoulder_r"),
    LandmarkPair.elbows: ("elbow_l", "elbow_r"),
    LandmarkPair.wrists: ("wrist_l", "wrist_r"),
    LandmarkPair.hands: ("hand_l", "hand_r"),
    LandmarkPair.hips: ("hip_l", "hip_r"),
    LandmarkPair.knees: ("knee_l", "knee_r"),
    LandmarkPair.ankles: ("ankle_l", "ankle_r"),
    LandmarkPair.feet: ("foot_l", "foot_r"),
}


def canonical_optimizer_operations() -> tuple[list[OperationDefinition], dict[str, type[BaseModel]]]:
    parameter_models: dict[str, type[BaseModel]] = {
        "rig.adjust_landmark_pair": AdjustLandmarkPairParameters,
        "skin.redistribute_joint_pair_weights": RedistributeJointPairWeightsParameters,
        "motion.adjust_timing": AdjustMotionTimingParameters,
        "motion.lock_foot_contact": LockFootContactParameters,
    }
    operations = [
        OperationDefinition(
            operation_id="rig.adjust_landmark_pair",
            worker_id="blender",
            stages=[AssetStage.anatomy, AssetStage.rig, AssetStage.skinning],
            description=(
                "Move one bilateral landmark pair in normalized body-bounds space. Positive lateral moves both "
                "landmarks outward; positive depth moves both toward Blender +Y; positive height moves both up. "
                "The deterministic worker preserves bilateral symmetry and limits each step to 2.5%."
            ),
            output_media_type="application/json",
            deterministic=True,
            parameter_schema=AdjustLandmarkPairParameters.model_json_schema(),
        ),
        OperationDefinition(
            operation_id="skin.redistribute_joint_pair_weights",
            worker_id="blender",
            stages=[AssetStage.rig, AssetStage.skinning],
            description=(
                "Symmetrically transfer a bounded fraction of existing weights between a joint pair's parent and "
                "child bones inside a spherical region. Radius is normalized by character height. The operation "
                "does not create geometry, move rest vertices, or change weights outside the declared regions."
            ),
            output_media_type="application/json",
            deterministic=True,
            parameter_schema=RedistributeJointPairWeightsParameters.model_json_schema(),
        ),
        OperationDefinition(
            operation_id="motion.adjust_timing",
            worker_id="blender",
            stages=[AssetStage.motion],
            description="Shift one semantic motion phase by at most three frames without changing joint poses.",
            output_media_type="application/json",
            deterministic=True,
            parameter_schema=AdjustMotionTimingParameters.model_json_schema(),
        ),
        OperationDefinition(
            operation_id="motion.lock_foot_contact",
            worker_id="blender",
            stages=[AssetStage.motion],
            description="Apply deterministic foot-contact IK to one declared window of at most twelve frames.",
            output_media_type="application/json",
            deterministic=True,
            parameter_schema=LockFootContactParameters.model_json_schema(),
        ),
    ]
    return operations, parameter_models


def apply_landmark_pair_adjustment(
    current: dict[str, list[float]],
    parameters: AdjustLandmarkPairParameters,
    *,
    cumulative_limit: float = 0.08,
) -> dict[str, list[float]]:
    """Return normalized landmark offsets after one symmetric, trust-region-bounded edit."""
    result = {name: [float(value) for value in values] for name, values in current.items()}
    left_name, right_name = PAIR_LANDMARKS[parameters.landmark_pair]
    left = result.setdefault(left_name, [0.0, 0.0, 0.0])
    right = result.setdefault(right_name, [0.0, 0.0, 0.0])
    delta = float(parameters.delta_fraction)
    if parameters.axis is LandmarkAxis.lateral:
        left[0] -= delta
        right[0] += delta
    else:
        axis = 1 if parameters.axis is LandmarkAxis.depth else 2
        left[axis] += delta
        right[axis] += delta
    for name in (left_name, right_name):
        if any(abs(value) > cumulative_limit + 1e-12 for value in result[name]):
            raise ValueError(f"cumulative landmark adjustment exceeds {cumulative_limit:.0%} for {name}")
    return result
