"""LocalDeploy visual/numerical optimizer constrained to registered operations."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from .localdeploy import LocalDeployStructuredClient
from .schemas import EvidenceBundle, OptimizerDecision, OperationDefinition


class LocalDeployOptimizer:
    def __init__(
        self,
        client: LocalDeployStructuredClient[OptimizerDecision] | None = None,
        *,
        model: str = "qwen3_6_27b",
    ) -> None:
        self.client = client or LocalDeployStructuredClient()
        self.model = model

    def diagnose(
        self,
        evidence: EvidenceBundle,
        operations: list[OperationDefinition],
        *,
        image_content: list[dict[str, Any]] | None = None,
        parameter_models: dict[str, type[BaseModel]] | None = None,
        decision_model: type[BaseModel] = OptimizerDecision,
        decision_validator: Callable[[OptimizerDecision], str | None] | None = None,
    ) -> OptimizerDecision:
        allowed = [item.model_dump(mode="json") for item in operations if evidence.stage in item.stages]
        if decision_model is not OptimizerDecision:
            for item in allowed:
                item.pop("parameter_schema", None)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are Asset Forge Darkness's diagnostic optimizer. Diagnose from the supplied evidence. "
                    "You may propose only operation_id values present in allowed_operations; never emit code or paths. "
                    "Prefer deterministic evidence, preserve locked features, and request human review when ambiguous. "
                    "When the evidence contains previous accepted and current attempted results, compare them directly: "
                    "populate the comparison fields available in the response schema; if it includes visual_delta, set "
                    "visual_delta positive only when current is visually better, and "
                    "explain the visible difference. The supplied hard numeric verdict is authoritative: never approve a "
                    "numeric regression. Treat rejected operation signatures and their measured outcomes as hard history; "
                    "do not repeat them or merely reduce their magnitude. Select at most one smallest causal intervention "
                    "for the next attempt.\n\n"
                    f"evidence={evidence.model_dump_json()}\nallowed_operations={allowed}"
                ),
            }
        ]
        content.extend(image_content or [])
        allowed_ids = {item["operation_id"] for item in allowed}
        validators = parameter_models or {}
        for semantic_attempt in range(2):
            decision = self.client.request(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                response_model=decision_model,
                max_attempts=2,
                temperature=0.0,
                max_tokens=1600,
            )
            try:
                disallowed = [
                    item.operation_id
                    for item in decision.proposals
                    if item.operation_id not in allowed_ids
                ]
                if disallowed:
                    raise ValueError(
                        "optimizer proposed unregistered operations: " + ", ".join(disallowed)
                    )
                if any(item.stage is not evidence.stage for item in decision.proposals):
                    raise ValueError("optimizer proposal stage does not match the evidence stage")
                adapter = getattr(decision, "to_optimizer_decision", None)
                normalized_decision = (
                    adapter()
                    if callable(adapter)
                    else OptimizerDecision.model_validate(decision.model_dump(mode="json"))
                )
                for proposal_index, proposal in enumerate(normalized_decision.proposals):
                    parameter_model = validators.get(proposal.operation_id)
                    if parameter_model is None:
                        continue
                    raw_parameters = (
                        proposal.parameters.model_dump(mode="json")
                        if isinstance(proposal.parameters, BaseModel)
                        else proposal.parameters
                    )
                    try:
                        validated = parameter_model.model_validate(raw_parameters)
                    except ValidationError as exc:
                        raise ValueError(
                            f"optimizer proposed invalid parameters for {proposal.operation_id}: {exc}"
                        ) from exc
                    normalized_decision.proposals[proposal_index].parameters = validated.model_dump(mode="json")
                if decision_validator is not None:
                    inconsistency = decision_validator(normalized_decision)
                    if inconsistency:
                        raise ValueError(
                            "optimizer decision contradicts deterministic evidence: " + inconsistency
                        )
                return normalized_decision
            except ValueError as exc:
                if semantic_attempt == 1:
                    raise
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Your preceding decision was rejected by the deterministic operation validator. "
                            f"validator_error={exc}. preceding_decision={decision.model_dump_json()}. "
                            "Return a corrected complete decision. Use the exact enum values and numeric bounds in "
                            "allowed_operations. Do not repeat the invalid parameter value."
                        ),
                    }
                )
        raise AssertionError("semantic optimizer validation loop exited unexpectedly")
