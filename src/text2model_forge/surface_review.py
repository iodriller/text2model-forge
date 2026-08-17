"""Structured Qwen critic and independent mediator for a persistent surface master."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .localdeploy import LocalDeployStructuredClient
from .schemas import StrictModel


class SurfaceReviewDecision(StrictModel):
    schema_version: Literal[1] = 1
    identity_preserved: Literal["yes", "mostly", "no", "uncertain"]
    material_readability: Literal["good", "usable_with_soft_defects", "needs_work", "uncertain"]
    club_consistency: Literal["good", "usable_with_soft_defects", "needs_work", "uncertain"]
    cross_view_consistency: Literal["good", "usable_with_soft_defects", "needs_work", "uncertain"]
    visible_defects: list[str] = Field(default_factory=list, max_length=5)
    overall: Literal["ready_for_final_render", "needs_one_bounded_revision", "uncertain"]
    allowed_revision: Literal[
        "none",
        "reduce_denoise",
        "adjust_skin_palette",
        "adjust_cloth_leather_contrast",
        "adjust_club_contrast",
        "repair_projection_seam",
        "reject_candidate",
    ]
    unconstrained_strategy_analysis: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]
    request_human_review: Literal[True] = True


class SurfaceMediatorDecision(StrictModel):
    schema_version: Literal[1] = 1
    critic_supported: bool
    unsupported_or_overstated_claims: list[str] = Field(default_factory=list, max_length=5)
    corrected_overall: Literal["ready_for_final_render", "needs_one_bounded_revision", "uncertain"]
    reason: str = Field(min_length=1)
    recommend_more_iteration: bool


class LocalDeploySurfaceReviewer:
    def __init__(self, *, model: str = "qwen3_6_27b") -> None:
        self.client: LocalDeployStructuredClient[Any] = LocalDeployStructuredClient()
        self.model = model

    def review(
        self,
        *,
        numeric_history: dict[str, Any],
        image_content: list[dict[str, Any]],
    ) -> SurfaceReviewDecision:
        return self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are Text2Model Forge Text2Model's broad surface-art critic. The left column is the "
                                "previous deterministic 3D master; the right column is the current ComfyUI-painted "
                                "views projected once into one persistent master. The report declares whether the "
                                "accepted storage is a baked atlas or stable-UV corner colors and lists any rejected "
                                "intermediate atlas. Review identity, "
                                "olive skin readability, dark cloth versus leather separation, wooden club clarity, "
                                "projection seams, baked lighting, and consistency across views. You may reason "
                                "freely, but choose at most one revision from the allowlist and never edit files or "
                                "emit code. Exact alpha, atlas, source-hash, and workflow metrics are authoritative. "
                                "Soft painterly differences do not justify endless iteration; if usable, advance to "
                                "final motion rendering and human review.\n\n"
                                f"previous_current_numeric_history={numeric_history}"
                            ),
                        },
                        *image_content,
                    ],
                }
            ],
            response_model=SurfaceReviewDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1300,
        )

    def mediate(
        self,
        *,
        numeric_history: dict[str, Any],
        critic: SurfaceReviewDecision,
        image_content: list[dict[str, Any]],
    ) -> SurfaceMediatorDecision:
        return self.client.request(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are the independent Text2Model surface referee. Check every material critic claim "
                                "against the same blinded before/current board and exact metrics. Do not reward more "
                                "iteration merely because diffusion could add detail. Recommend one bounded revision "
                                "only for a clear production-visible failure; otherwise advance to final render and "
                                "human review.\n\n"
                                f"numeric_history={numeric_history}\ncritic={critic.model_dump_json()}"
                            ),
                        },
                        *image_content,
                    ],
                }
            ],
            response_model=SurfaceMediatorDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=900,
        )
