from __future__ import annotations

import json

from darkness.localdeploy import LocalDeployStructuredClient
from darkness.referee import (
    LocalDeployReferee,
    RefereeEvidence,
    RefereeOutcome,
    plateau_reached,
    unblind_referee,
)
from darkness.schemas import AssetStage


def test_blinded_referee_calls_a_virtual_match_a_tie_and_unblinds_it() -> None:
    evidence = RefereeEvidence(
        pair_id="motion.walk.pair.1",
        stage=AssetStage.motion,
        goal="Prefer only a materially clearer walk pose.",
        option_a_artifact_id="motion.walk.a",
        option_b_artifact_id="motion.walk.b",
        numeric_diagnostics={"changed_pixel_fraction": 0.0008},
    )

    def sender(payload):
        prompt = payload["messages"][0]["content"][0]["text"]
        assert "do not know which option is newer" in prompt
        assert "Never propose a fix" in prompt
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "schema_version": 1,
                                "preferred": "tie",
                                "material_difference": False,
                                "confidence": "high",
                                "visible_differences": [],
                                "invariant_concerns": [],
                                "reason": "The fixed-camera images are virtually identical.",
                                "request_human_review": False,
                            }
                        )
                    }
                }
            ]
        }

    referee = LocalDeployReferee(client=LocalDeployStructuredClient(sender=sender))
    decision = referee.review(evidence, image_content=[])
    assert unblind_referee(decision, candidate_option="b") is RefereeOutcome.no_meaningful_difference


def test_referee_unblinding_and_two_round_plateau_policy() -> None:
    evidence = RefereeEvidence(
        pair_id="motion.attack.pair.1",
        stage=AssetStage.motion,
        goal="Prefer the clearer attack impact pose.",
        option_a_artifact_id="motion.attack.a",
        option_b_artifact_id="motion.attack.b",
    )
    decision = {
        "schema_version": 1,
        "preferred": "a",
        "material_difference": True,
        "confidence": "medium",
        "reason": "Option A has a clearer line of action.",
    }
    from darkness.referee import RefereeDecision

    parsed = RefereeDecision.model_validate(decision)
    assert unblind_referee(parsed, candidate_option="a") is RefereeOutcome.material_improvement
    assert unblind_referee(parsed, candidate_option="b") is RefereeOutcome.regression
    assert plateau_reached(
        [RefereeOutcome.material_improvement, RefereeOutcome.no_meaningful_difference]
    ) is False
    assert plateau_reached(
        [
            RefereeOutcome.material_improvement,
            RefereeOutcome.no_meaningful_difference,
            RefereeOutcome.regression,
        ]
    ) is True
