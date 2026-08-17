"""Structured LocalDeploy review for critical short-biped motion evidence."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .localdeploy import LocalDeployStructuredClient
from .schemas import StrictModel


class MotionClipReview(StrictModel):
    readability: Literal["good", "acceptable", "needs_work", "uncertain"]
    critical_joint_verdict: Literal["good", "acceptable", "needs_work", "uncertain"]
    strengths: list[str] = Field(default_factory=list, max_length=3)
    issues: list[str] = Field(default_factory=list, max_length=4)
    recommended_strategy: str | None = None


class MotionReviewDecision(StrictModel):
    schema_version: Literal[1] = 1
    idle: MotionClipReview
    walk: MotionClipReview
    attack: MotionClipReview
    hit: MotionClipReview
    death: MotionClipReview
    overall: Literal["ready_for_human_gate", "needs_bounded_revision", "uncertain"]
    confidence: Literal["low", "medium", "high"]
    cross_clip_issues: list[str] = Field(default_factory=list, max_length=4)
    unconstrained_strategy_analysis: str = Field(min_length=1)
    requested_new_capability: str | None = None
    request_human_review: Literal[True] = True


class MotionReviewMediatorDecision(StrictModel):
    schema_version: Literal[1] = 1
    critic_supported: bool
    supported_claims: list[str] = Field(default_factory=list, max_length=5)
    unsupported_or_overstated_claims: list[str] = Field(default_factory=list, max_length=5)
    corrected_overall: Literal["ready_for_human_gate", "needs_bounded_revision", "uncertain"]
    reason: str = Field(min_length=1)
    recommend_more_iteration: bool


class LocalDeployMotionReviewer:
    def __init__(
        self,
        client: LocalDeployStructuredClient[MotionReviewDecision] | None = None,
        *,
        model: str = "qwen3_6_27b",
    ) -> None:
        self.client = client or LocalDeployStructuredClient()
        self.model = model

    def review(
        self,
        *,
        numeric_summary: dict[str, Any],
        image_content: list[dict[str, Any]],
    ) -> MotionReviewDecision:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are the broad diagnostic motion critic for Text2Model Forge Text2Model. Review the full key-pose "
                    "strips and the deterministic critical-joint/contact/deformation numbers for idle, walk, attack, "
                    "hit, and death. You have wide diagnostic freedom: explain any causal strategy you think would "
                    "help, including a new operation Text2Model does not yet implement. Do not pretend a static contact "
                    "sheet proves timing between frames, and do not override hard numerical failures. A small number of "
                    "soft severely-compressed faces is a reported defect, not automatically a failed clip. Recommend "
                    "revision only for a visible or functional issue that matters at game scale. You do not edit files "
                    "or emit code.\n\n"
                    f"motion_numeric_summary={numeric_summary}"
                ),
            }
        ]
        content.extend(image_content)
        return self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=MotionReviewDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1600,
        )

    def mediate(
        self,
        *,
        numeric_summary: dict[str, Any],
        critic: MotionReviewDecision,
        image_content: list[dict[str, Any]],
    ) -> MotionReviewMediatorDecision:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are the independent mediator for a prior motion critic. Verify the critic's factual claims "
                    "against the key-pose evidence and deterministic numbers. Do not invent a new repair plan and do "
                    "not reward extra iteration by default. Flag claims such as 'completely static', 'foot sliding', "
                    "or 'ground penetration' when the supplied measurements or images do not support that wording. "
                    "If the deterministic gates pass and remaining issues are soft or visually negligible, recommend "
                    "moving to the human gate rather than iterating.\n\n"
                    f"numeric_summary={numeric_summary}\ncritic={critic.model_dump_json()}"
                ),
            }
        ]
        content.extend(image_content)
        return self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=MotionReviewMediatorDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1000,
        )
