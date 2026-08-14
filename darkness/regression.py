"""Deterministic hard gates applied before candidate promotion."""
from __future__ import annotations

from .schemas import CandidateEvaluation, MetricResult


def evaluate_candidate(
    *,
    evaluation_id: str,
    candidate_artifact_id: str,
    baseline_artifact_id: str | None,
    metrics: list[MetricResult],
    technical_score: float,
    visual_score: float | None = None,
    locked_feature_regressions: list[str] | None = None,
    minimum_technical_score: float = 0.7,
) -> CandidateEvaluation:
    regressions = locked_feature_regressions or []
    reasons: list[str] = []
    hard_failures = [item.name for item in metrics if item.hard_failure and not item.passed]
    if hard_failures:
        reasons.append("hard failures: " + ", ".join(hard_failures))
    if regressions:
        reasons.append("locked feature regressions: " + ", ".join(regressions))
    if technical_score < minimum_technical_score:
        reasons.append(
            f"technical score {technical_score:.3f} is below {minimum_technical_score:.3f}"
        )
    return CandidateEvaluation(
        evaluation_id=evaluation_id,
        candidate_artifact_id=candidate_artifact_id,
        baseline_artifact_id=baseline_artifact_id,
        metrics=metrics,
        visual_score=visual_score,
        technical_score=technical_score,
        locked_feature_regressions=regressions,
        accepted=not reasons,
        reasons=reasons,
    )
