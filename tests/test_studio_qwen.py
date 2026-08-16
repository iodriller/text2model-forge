"""Focused contract tests for Studio's structured Qwen review boundary."""
from __future__ import annotations

import pytest

from darkness.studio_qwen import GateAssessment, _validated_concept_selection


IDS = ["d1-i01-candidate-1", "d1-i01-candidate-2"]


def _assessment(**updates) -> GateAssessment:
    values = {
        "summary": "Both candidates visibly miss a locked feature.",
        "candidate_ranking": IDS,
        "recommended_evidence_id": "none",
        "confidence": 0.9,
        "hard_requirements_satisfied": False,
    }
    values.update(updates)
    return GateAssessment.model_validate(values)


def test_a_complete_reject_all_review_is_not_mislabeled_as_reviewer_unavailability():
    assert _validated_concept_selection(_assessment(), IDS) is None


def test_a_passing_review_must_recommend_one_supplied_candidate():
    with pytest.raises(ValueError, match="recommendation was not one"):
        _validated_concept_selection(
            _assessment(hard_requirements_satisfied=True),
            IDS,
        )


def test_the_ranking_must_still_contain_each_candidate_exactly_once():
    with pytest.raises(ValueError, match="ranking did not contain each"):
        _validated_concept_selection(
            _assessment(candidate_ranking=[IDS[0], IDS[0]]),
            IDS,
        )
