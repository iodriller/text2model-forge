from text2model_forge.qualification import evaluate_legacy_component
from text2model_forge.schemas import LegacyQualification


def report(**overrides) -> LegacyQualification:
    values = {
        "component_id": "legacy.renderer",
        "contract_tests_total": 5,
        "contract_tests_passed": 5,
        "hard_failures": [],
        "blinded_trials": 5,
        "blinded_wins": 4,
        "evidence_ids": ["e1", "e2", "e3", "e4", "e5"],
    }
    values.update(overrides)
    return LegacyQualification(**values)


def test_legacy_component_must_pass_contracts_and_win_blinded_comparison() -> None:
    assert evaluate_legacy_component(report()).reusable is True
    assert evaluate_legacy_component(report(contract_tests_passed=4)).reusable is False
    assert evaluate_legacy_component(report(blinded_wins=2)).reusable is False


def test_hard_failure_always_blocks_reuse() -> None:
    decision = evaluate_legacy_component(report(hard_failures=["silhouette regression"]))
    assert decision.reusable is False
    assert "silhouette regression" in " ".join(decision.reasons)
