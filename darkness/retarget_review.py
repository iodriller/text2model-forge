"""Broad critic and independent mediator for external-library motion retargets."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .localdeploy import LocalDeployStructuredClient
from .schemas import StrictModel


class RetargetClipReview(StrictModel):
    readability: Literal["good", "usable_with_soft_defects", "needs_replacement", "uncertain"]
    critical_limb_verdict: Literal["good", "usable_with_soft_defects", "needs_work", "uncertain"]
    observations: list[str] = Field(default_factory=list, max_length=4)


class RetargetReviewDecision(StrictModel):
    schema_version: Literal[1] = 1
    idle: RetargetClipReview
    walk: RetargetClipReview
    attack: RetargetClipReview
    death: RetargetClipReview
    overall: Literal["ready_for_human_gate", "needs_one_bounded_revision", "reject_retarget", "uncertain"]
    confidence: Literal["low", "medium", "high"]
    cross_clip_issues: list[str] = Field(default_factory=list, max_length=4)
    unconstrained_strategy_analysis: str = Field(min_length=1)
    requested_new_capability: str | None = None
    request_human_review: Literal[True] = True


class RetargetMediatorDecision(StrictModel):
    schema_version: Literal[1] = 1
    critic_supported: bool
    supported_claims: list[str] = Field(default_factory=list, max_length=5)
    unsupported_or_overstated_claims: list[str] = Field(default_factory=list, max_length=5)
    corrected_overall: Literal[
        "ready_for_human_gate", "needs_one_bounded_revision", "reject_retarget", "uncertain"
    ]
    reason: str = Field(min_length=1)
    recommend_more_iteration: bool


class LocalDeployRetargetReviewer:
    def __init__(self, *, model: str = "qwen3_6_27b") -> None:
        self.client: LocalDeployStructuredClient[Any] = LocalDeployStructuredClient()
        self.model = model

    def review(
        self,
        *,
        numeric_history: dict[str, Any],
        image_content: list[dict[str, Any]],
    ) -> RetargetReviewDecision:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are VettedMesh Darkness's broad motion-retarget critic. This is a manual pipeline check, "
                    "not a request for perfection. Review only the supplied donor-to-target idle, walk, attack, "
                    "and death key poses plus exact numerical history. Pay special attention to hips, knees, ankles, "
                    "shoulders, elbows, and whether attack/death read at game scale. You have wide diagnostic freedom "
                    "to judge declared equipment semantics, weapon-side consistency, held-item silhouette, and whether the "
                    "weapon makes the attack's wind-up/contact/recovery readable. Treat source-hand motion/rest-X "
                    "metrics as authoritative for side selection and socket error as authoritative for attachment. "
                    "Those numbers do not prove a grasp: the dedicated close-up images are authoritative for whether "
                    "the fingers visibly close around the handle rather than merely touching or floating beside it. "
                    "When a previous grip image is labeled, compare it directly with the current front/side close-ups "
                    "and do not claim improvement unless shaft clearance and distinct digit wrapping visibly improve. "
                    "You may request a new system capability, but do not edit files or emit code. Static key poses do "
                    "not prove timing or foot sliding. Counts of severely compressed faces are soft warnings unless a "
                    "visible deformation is material. If the clips are usable and defects are soft, send them to the "
                    "human gate instead of asking for polish.\n\n"
                    f"retarget_numeric_history={numeric_history}"
                ),
            },
            *image_content,
        ]
        return self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=RetargetReviewDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1500,
        )

    def mediate(
        self,
        *,
        numeric_history: dict[str, Any],
        critic: RetargetReviewDecision,
        image_content: list[dict[str, Any]],
    ) -> RetargetMediatorDecision:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are the independent referee for a prior retarget critic. Verify each material claim against "
                    "the exact numbers and image sheet. Do not invent a new improvement plan and do not prefer more "
                    "iteration by default. The purpose is to stop when the retarget is usable enough for a human to "
                    "judge. Static strips cannot prove foot sliding or animation timing. Recommend one more revision "
                    "only for a clearly visible or numerically supported critical failure. Verify that the dominant "
                    "Quaternius source hand maps to the same physical rest-X side, and verify the claimed grasp against "
                    "the dedicated close-up rather than accepting socket error alone.\n\n"
                    f"retarget_numeric_history={numeric_history}\ncritic={critic.model_dump_json()}"
                ),
            },
            *image_content,
        ]
        return self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=RetargetMediatorDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1000,
        )
