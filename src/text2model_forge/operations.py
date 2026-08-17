"""Allowlisted operation registry; models can propose IDs and parameters, never code."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from .schemas import ArtifactRecord, AssetStage, OperationDefinition


class UnknownOperationError(KeyError):
    pass


class OperationContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class OperationContext:
    run_id: str
    stage: AssetStage
    input_records: list[ArtifactRecord]
    input_bytes: list[bytes]


@dataclass(frozen=True)
class OperationOutput:
    data: bytes
    media_type: str
    producer_candidate_ids: list[str] = field(default_factory=list)
    source_license_ids: list[str] = field(default_factory=list)
    source_license_status: str = "cleared"
    metadata: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[OperationContext, BaseModel], OperationOutput]


@dataclass(frozen=True)
class RegisteredOperation:
    definition: OperationDefinition
    parameter_model: type[BaseModel]
    handler: Handler


class OperationRegistry:
    def __init__(self) -> None:
        self._operations: dict[str, RegisteredOperation] = {}

    def register(
        self,
        definition: OperationDefinition,
        parameter_model: type[BaseModel],
        handler: Handler,
    ) -> None:
        if definition.input_count_max < definition.input_count_min:
            raise OperationContractError("input_count_max must be at least input_count_min")
        if definition.operation_id in self._operations:
            raise OperationContractError(f"duplicate operation '{definition.operation_id}'")
        self._operations[definition.operation_id] = RegisteredOperation(definition, parameter_model, handler)

    def definitions(self) -> list[OperationDefinition]:
        return [item.definition for item in self._operations.values()]

    def execute(
        self,
        operation_id: str,
        context: OperationContext,
        parameters: dict[str, Any],
    ) -> OperationOutput:
        registered = self._operations.get(operation_id)
        if registered is None:
            raise UnknownOperationError(operation_id)
        definition = registered.definition
        if context.stage not in definition.stages:
            raise OperationContractError(
                f"operation '{operation_id}' is not allowed at stage {context.stage.value}"
            )
        count = len(context.input_records)
        if count < definition.input_count_min or count > definition.input_count_max:
            raise OperationContractError(
                f"operation '{operation_id}' requires {definition.input_count_min}..{definition.input_count_max} inputs"
            )
        validated = registered.parameter_model.model_validate(parameters)
        output = registered.handler(context, validated)
        if output.media_type != definition.output_media_type:
            raise OperationContractError("handler output media type violates its registered contract")
        return output
