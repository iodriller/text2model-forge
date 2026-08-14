from darkness.retarget_review import (
    RetargetClipReview,
    RetargetMediatorDecision,
    RetargetReviewDecision,
)


def _clip() -> RetargetClipReview:
    return RetargetClipReview(
        readability="usable_with_soft_defects",
        critical_limb_verdict="good",
        observations=["Readable at key poses."],
    )


def test_retarget_review_and_mediator_support_stop_decision() -> None:
    critic = RetargetReviewDecision(
        idle=_clip(),
        walk=_clip(),
        attack=_clip(),
        death=_clip(),
        overall="ready_for_human_gate",
        confidence="high",
        unconstrained_strategy_analysis="The four clips are sufficient for this checkpoint.",
    )
    mediator = RetargetMediatorDecision(
        critic_supported=True,
        corrected_overall="ready_for_human_gate",
        reason="No critical failure is present.",
        recommend_more_iteration=False,
    )
    assert critic.request_human_review is True
    assert mediator.recommend_more_iteration is False
