"""Fail-closed gate for reusing components from the underperforming legacy forge."""
from __future__ import annotations

from .schemas import LegacyQualification, LegacyQualificationDecision


def evaluate_legacy_component(
    report: LegacyQualification,
    *,
    minimum_blinded_trials: int = 3,
    minimum_win_rate: float = 0.5,
) -> LegacyQualificationDecision:
    reasons: list[str] = []
    if report.contract_tests_passed != report.contract_tests_total:
        reasons.append(
            f"contract tests passed {report.contract_tests_passed}/{report.contract_tests_total}"
        )
    if report.hard_failures:
        reasons.append(f"hard failures: {', '.join(report.hard_failures)}")
    if report.blinded_trials < minimum_blinded_trials:
        reasons.append(
            f"only {report.blinded_trials} blinded trials; need at least {minimum_blinded_trials}"
        )
    elif report.blinded_wins / report.blinded_trials <= minimum_win_rate:
        reasons.append(
            f"blinded win rate {report.blinded_wins / report.blinded_trials:.2f} must exceed {minimum_win_rate:.2f}"
        )
    if len(report.evidence_ids) < report.blinded_trials:
        reasons.append("each blinded trial must reference evidence")
    return LegacyQualificationDecision(reusable=not reasons, reasons=reasons)
