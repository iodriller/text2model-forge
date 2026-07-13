"""Structured Qwen critic and mediator for packaged directional sprite evidence."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .localdeploy import LocalDeployStructuredClient
from .schemas import StrictModel


class SpriteClipReview(StrictModel):
    readability: Literal["good", "usable_with_soft_defects", "needs_work", "uncertain"]
    scale_and_silhouette: Literal["good", "usable_with_soft_defects", "needs_work", "uncertain"]
    observations: list[str] = Field(default_factory=list, max_length=4)


class SpriteReviewDecision(StrictModel):
    schema_version: Literal[1] = 1
    idle: SpriteClipReview
    walk: SpriteClipReview
    attack: SpriteClipReview
    death: SpriteClipReview
    overall: Literal["ready_for_unity_candidate", "needs_one_bounded_revision", "uncertain"]
    confidence: Literal["low", "medium", "high"]
    cross_clip_issues: list[str] = Field(default_factory=list, max_length=4)
    unconstrained_strategy_analysis: str = Field(min_length=1)
    request_human_review: Literal[True] = True


class SpriteMediatorDecision(StrictModel):
    schema_version: Literal[1] = 1
    critic_supported: bool
    unsupported_or_overstated_claims: list[str] = Field(default_factory=list, max_length=5)
    corrected_overall: Literal["ready_for_unity_candidate", "needs_one_bounded_revision", "uncertain"]
    reason: str = Field(min_length=1)
    recommend_more_iteration: bool


class LocalDeploySpriteReviewer:
    def __init__(self, *, model: str = "qwen3_6_27b") -> None:
        self.client: LocalDeployStructuredClient[Any] = LocalDeployStructuredClient()
        self.model = model

    def review(
        self,
        *,
        numeric_summary: dict[str, Any],
        image_content: list[dict[str, Any]],
    ) -> SpriteReviewDecision:
        return self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are Asset Forge Darkness's broad directional-sprite critic. Review this four-clip "
                                "manual pipeline checkpoint at gameplay scale. You may freely explain important "
                                "silhouette, scale, pose, direction, or readability defects, but do not edit files or "
                                "emit code. The exact alpha/edge-clipping gates and visible-size measurements are "
                                "authoritative. A deliberately wider death camera is acceptable if the death remains "
                                "readable. Static sheets cannot prove animation timing. Recommend another render only "
                                "for a material visible problem, otherwise advance to the Unity candidate gate.\n\n"
                                f"sprite_numeric_summary={numeric_summary}"
                            ),
                        },
                        *image_content,
                    ],
                }
            ],
            response_model=SpriteReviewDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1300,
        )

    def mediate(
        self,
        *,
        numeric_summary: dict[str, Any],
        critic: SpriteReviewDecision,
        image_content: list[dict[str, Any]],
    ) -> SpriteMediatorDecision:
        return self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an independent sprite-review mediator. Check the critic against the image "
                                "and exact numeric gate. Do not invent a repair plan and do not reward more iteration. "
                                "Recommend revision only for a clearly visible material failure; otherwise advance the "
                                "candidate to Unity and human review.\n\n"
                                f"sprite_numeric_summary={numeric_summary}\ncritic={critic.model_dump_json()}"
                            ),
                        },
                        *image_content,
                    ],
                }
            ],
            response_model=SpriteMediatorDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=900,
        )
