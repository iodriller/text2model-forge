from __future__ import annotations

import pytest
from pydantic import ValidationError

from darkness.canonical_optimizer import (
    AdjustLandmarkPairParameters,
    CanonicalRigOptimizerDecision,
    LandmarkAxis,
    LandmarkPair,
    RedistributeJointPairWeightsParameters,
    apply_landmark_pair_adjustment,
    canonical_optimizer_operations,
)


def test_landmark_pair_adjustment_is_symmetric_and_normalized() -> None:
    adjusted = apply_landmark_pair_adjustment(
        {},
        AdjustLandmarkPairParameters(
            landmark_pair=LandmarkPair.shoulders,
            axis=LandmarkAxis.lateral,
            delta_fraction=0.02,
        ),
    )
    assert adjusted["shoulder_l"] == [-0.02, 0.0, 0.0]
    assert adjusted["shoulder_r"] == [0.02, 0.0, 0.0]


def test_landmark_pair_adjustment_enforces_step_and_cumulative_limits() -> None:
    with pytest.raises(ValidationError):
        AdjustLandmarkPairParameters(
            landmark_pair=LandmarkPair.shoulders,
            axis=LandmarkAxis.height,
            delta_fraction=0.04,
        )
    with pytest.raises(ValueError, match="cumulative"):
        apply_landmark_pair_adjustment(
            {"shoulder_l": [0.0, 0.0, 0.075], "shoulder_r": [0.0, 0.0, 0.075]},
            AdjustLandmarkPairParameters(
                landmark_pair=LandmarkPair.shoulders,
                axis=LandmarkAxis.height,
                delta_fraction=0.01,
            ),
        )


def test_optimizer_operations_publish_the_same_typed_bounds_the_executor_uses() -> None:
    operations, models = canonical_optimizer_operations()
    operation = next(item for item in operations if item.operation_id == "rig.adjust_landmark_pair")
    assert operation.parameter_schema == models[operation.operation_id].model_json_schema()
    delta = operation.parameter_schema["properties"]["delta_fraction"]
    assert delta["minimum"] == -0.025
    assert delta["maximum"] == 0.025
    weight_operation = next(
        item for item in operations if item.operation_id == "skin.redistribute_joint_pair_weights"
    )
    assert weight_operation.parameter_schema == RedistributeJointPairWeightsParameters.model_json_schema()
    transfer = weight_operation.parameter_schema["properties"]["transfer_fraction"]
    assert transfer["minimum"] == 0.025
    assert transfer["maximum"] == 0.15


def test_rig_decision_schema_rejects_unregistered_or_out_of_bounds_nested_parameters() -> None:
    payload = {
        "schema_version": 1,
        "goal_satisfied": False,
        "technical_score": 0.5,
        "confidence": 0.7,
        "proposals": [
            {
                "schema_version": 1,
                "proposal_id": "proposal.weights.1",
                "operation_id": "skin.redistribute_joint_pair_weights",
                "stage": "D5_rig",
                "input_artifact_ids": ["rig.front"],
                "parameters": {
                    "joint_pair": "legs",
                    "direction": "parent_to_child",
                    "transfer_fraction": 0.35,
                    "radius_fraction": 0.08,
                },
                "rationale": "Invalid on purpose.",
            }
        ],
    }
    with pytest.raises(ValidationError):
        CanonicalRigOptimizerDecision.model_validate(payload)
