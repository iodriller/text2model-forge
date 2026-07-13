"""LocalDeploy visual/numerical optimizer constrained to registered operations."""
from __future__ import annotations

from typing import Any

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
    ) -> OptimizerDecision:
        allowed = [item.model_dump(mode="json") for item in operations if evidence.stage in item.stages]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are Asset Forge Darkness's diagnostic optimizer. Diagnose from the supplied evidence. "
                    "You may propose only operation_id values present in allowed_operations; never emit code or paths. "
                    "Prefer deterministic evidence, preserve locked features, and request human review when ambiguous.\n\n"
                    f"evidence={evidence.model_dump_json()}\nallowed_operations={allowed}"
                ),
            }
        ]
        content.extend(image_content or [])
        decision = self.client.request(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_model=OptimizerDecision,
            max_attempts=2,
            temperature=0.0,
            max_tokens=1600,
        )
        allowed_ids = {item["operation_id"] for item in allowed}
        disallowed = [item.operation_id for item in decision.proposals if item.operation_id not in allowed_ids]
        if disallowed:
            raise ValueError("optimizer proposed unregistered operations: " + ", ".join(disallowed))
        if any(item.stage is not evidence.stage for item in decision.proposals):
            raise ValueError("optimizer proposal stage does not match the evidence stage")
        return decision
