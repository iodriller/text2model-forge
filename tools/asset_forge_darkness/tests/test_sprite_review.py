from darkness.sprite_review import SpriteClipReview, SpriteMediatorDecision, SpriteReviewDecision


def _clip() -> SpriteClipReview:
    return SpriteClipReview(
        readability="good",
        scale_and_silhouette="usable_with_soft_defects",
        observations=[],
    )


def test_sprite_review_can_advance_without_more_iteration() -> None:
    critic = SpriteReviewDecision(
        idle=_clip(),
        walk=_clip(),
        attack=_clip(),
        death=_clip(),
        overall="ready_for_unity_candidate",
        confidence="high",
        unconstrained_strategy_analysis="The sprite checkpoint is readable.",
    )
    mediator = SpriteMediatorDecision(
        critic_supported=True,
        corrected_overall="ready_for_unity_candidate",
        reason="Hard gates pass and no material visual failure is present.",
        recommend_more_iteration=False,
    )
    assert critic.request_human_review is True
    assert mediator.recommend_more_iteration is False
