"""Deterministic worker used to prove orchestration without a heavyweight model."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .operations import OperationContext, OperationOutput, OperationRegistry
from .schemas import AssetStage, OperationDefinition


class FakeAdvanceParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_stage: AssetStage
    remove_defect: str | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)


def _advance(context: OperationContext, parameters: BaseModel) -> OperationOutput:
    parsed = FakeAdvanceParameters.model_validate(parameters)
    source = json.loads(context.input_bytes[0].decode("utf-8"))
    defects = list(source.get("defects", []))
    if parsed.remove_defect in defects:
        defects.remove(parsed.remove_defect)
    result = {
        "schema_version": 1,
        "run_id": context.run_id,
        "stage": parsed.target_stage.value,
        "source_artifact_id": context.input_records[0].artifact_id,
        "defects": defects,
        "locked_features": source.get("locked_features", []),
        "annotations": parsed.annotations,
    }
    return OperationOutput(
        data=json.dumps(result, sort_keys=True, indent=2).encode("utf-8"),
        media_type="application/json",
        metadata={"worker_id": "fake.deterministic", "target_stage": parsed.target_stage.value},
    )


def build_fake_registry() -> OperationRegistry:
    registry = OperationRegistry()
    registry.register(
        OperationDefinition(
            operation_id="fake.advance",
            worker_id="fake.deterministic",
            stages=list(AssetStage),
            description="Copy a fixture forward while optionally repairing one seeded defect.",
            output_media_type="application/json",
            deterministic=True,
        ),
        FakeAdvanceParameters,
        _advance,
    )
    return registry
