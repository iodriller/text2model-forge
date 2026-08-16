"""Independent blinded LocalDeploy referee and plateau policy."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from .localdeploy import LocalDeployStructuredClient
from .schemas import AssetStage, StrictModel


class RefereeEvidence(StrictModel):
    schema_version: Literal[1] = 1
    pair_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    stage: AssetStage
    goal: str = Field(min_length=1)
    option_a_artifact_id: str = Field(min_length=1)
    option_b_artifact_id: str = Field(min_length=1)
    numeric_diagnostics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    locked_invariants: list[str] = Field(default_factory=list)


class RefereeDecision(StrictModel):
    schema_version: Literal[1] = 1
    preferred: Literal["a", "b", "tie", "uncertain"]
    material_difference: bool
    confidence: Literal["low", "medium", "high"]
    visible_differences: list[str] = Field(default_factory=list, max_length=4)
    invariant_concerns: list[str] = Field(default_factory=list, max_length=4)
    reason: str = Field(min_length=1)
    request_human_review: bool = False


class RefereeOutcome(str, Enum):
    material_improvement = "material_improvement"
    no_meaningful_difference = "no_meaningful_difference"
    regression = "regression"
    uncertain = "uncertain"


class LocalDeployReferee:
    def __init__(
        self,
        client: LocalDeployStructuredClient[RefereeDecision] | None = None,
        *,
        model: str = "qwen3_6_27b",
    ) -> None:
        self.client = client or LocalDeployStructuredClient()
        self.model = model

    def review(
        self,
        evidence: RefereeEvidence,
        *,
        image_content: list[dict[str, Any]],
    ) -> RefereeDecision:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are VettedMesh Darkness's independent blinded visual referee. You do not know which option "
                    "is newer, which operation created it, or what the proposer expected. Compare only labeled option "
                    "A with labeled option B and the supplied numerical evidence. Report a preference only for a "
                    "visible, material quality difference. If the images are virtually identical, choose tie and set "
                    "material_difference=false. Never propose a fix, code, parameter, or operation. Deterministic hard "
                    "invariant failures outrank visual preference. Use uncertain when the evidence is insufficient.\n\n"
                    f"referee_evidence={evidence.model_dump_json()}"
                ),
            }
        ]
        content.extend(image_content)
        return self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=RefereeDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=900,
        )


def unblind_referee(
    decision: RefereeDecision,
    *,
    candidate_option: Literal["a", "b"],
) -> RefereeOutcome:
    if decision.preferred == "uncertain":
        return RefereeOutcome.uncertain
    if not decision.material_difference or decision.preferred == "tie":
        return RefereeOutcome.no_meaningful_difference
    if decision.preferred == candidate_option:
        return RefereeOutcome.material_improvement
    return RefereeOutcome.regression


def plateau_reached(
    outcomes: list[RefereeOutcome],
    *,
    non_improvement_limit: int = 2,
) -> bool:
    if non_improvement_limit < 1:
        raise ValueError("non_improvement_limit must be positive")
    consecutive = 0
    for outcome in reversed(outcomes):
        if outcome is RefereeOutcome.material_improvement:
            break
        if outcome is RefereeOutcome.uncertain:
            break
        consecutive += 1
    return consecutive >= non_improvement_limit
