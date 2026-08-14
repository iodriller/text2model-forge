"""Registered Darkness mesh operations backed by the independent OBJ kernel."""
from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from .mesh import MeshHealth, MeshRepairDecision, TriangleMesh
from .operations import OperationContext, OperationOutput, OperationRegistry
from .regression import evaluate_candidate
from .schemas import (
    AssetStage,
    CandidateEvaluation,
    MetricResult,
    OperationDefinition,
)


class MeshRepairParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weld_tolerance: float = Field(default=1e-9, gt=0, le=0.01)
    minimum_component_faces: int = Field(default=2, ge=1, le=10000)


class MeshAnalyzeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tolerance: float = Field(default=1e-9, gt=0, le=0.01)


def evaluate_mesh_repair(
    *,
    evaluation_id: str,
    candidate_artifact_id: str,
    baseline_artifact_id: str,
    decision: MeshRepairDecision,
) -> CandidateEvaluation:
    before = decision.before
    after = decision.candidate if decision.accepted else before
    metrics = [
        MetricResult(
            name="repair_branch_accepted",
            value=decision.accepted,
            passed=decision.accepted,
            hard_failure=True,
        ),
        MetricResult(
            name="usable_geometry",
            value=after.faces > 0 and after.vertices > 0,
            passed=after.faces > 0 and after.vertices > 0,
            hard_failure=True,
        ),
        MetricResult(
            name="finite_coordinates",
            value=after.finite_coordinates,
            passed=after.finite_coordinates,
            hard_failure=True,
        ),
        MetricResult(
            name="degenerate_faces",
            value=after.degenerate_faces,
            threshold=0,
            passed=after.degenerate_faces == 0,
            hard_failure=True,
        ),
        MetricResult(
            name="non_manifold_edges_not_worse",
            value=after.non_manifold_edges,
            threshold=before.non_manifold_edges,
            passed=after.non_manifold_edges <= before.non_manifold_edges,
            hard_failure=True,
        ),
        MetricResult(
            name="closed_surface",
            value=after.boundary_edges,
            threshold=0,
            passed=after.boundary_edges == 0,
            hard_failure=True,
        ),
        MetricResult(
            name="component_count_not_worse",
            value=after.connected_components,
            threshold=before.connected_components,
            passed=after.connected_components <= before.connected_components,
        ),
        MetricResult(
            name="winding_consistent",
            value=after.inconsistent_winding_edges,
            threshold=0,
            passed=after.inconsistent_winding_edges == 0,
            hard_failure=True,
        ),
    ]
    technical_score = sum(metric.passed for metric in metrics) / len(metrics)
    return evaluate_candidate(
        evaluation_id=evaluation_id,
        candidate_artifact_id=candidate_artifact_id,
        baseline_artifact_id=baseline_artifact_id,
        metrics=metrics,
        technical_score=technical_score,
    )


def _repair(context: OperationContext, raw: BaseModel) -> OperationOutput:
    parameters = MeshRepairParameters.model_validate(raw)
    mesh = TriangleMesh.from_obj(context.input_bytes[0].decode("utf-8"))
    before = mesh.health(parameters.weld_tolerance)
    repaired, decision = mesh.guarded_repair(
        tolerance=parameters.weld_tolerance,
        minimum_component_faces=parameters.minimum_component_faces,
    )
    after = repaired.health(parameters.weld_tolerance)
    return OperationOutput(
        data=repaired.to_obj().encode("utf-8"),
        media_type="model/obj",
        metadata={
            "before": before.model_dump(),
            "after": after.model_dump(),
            "repair_decision": decision.model_dump(),
            "worker_id": "mesh.deterministic",
        },
    )


def _analyze(context: OperationContext, raw: BaseModel) -> OperationOutput:
    parameters = MeshAnalyzeParameters.model_validate(raw)
    mesh = TriangleMesh.from_obj(context.input_bytes[0].decode("utf-8"))
    report = mesh.health(parameters.tolerance)
    return OperationOutput(
        data=json.dumps(report.model_dump(), sort_keys=True, indent=2).encode("utf-8"),
        media_type="application/json",
        metadata={"worker_id": "mesh.deterministic"},
    )


def register_mesh_operations(registry: OperationRegistry) -> None:
    registry.register(
        OperationDefinition(
            operation_id="mesh.repair_structural",
            worker_id="mesh.deterministic",
            stages=[AssetStage.geometry, AssetStage.topology],
            description="Weld duplicate vertices, remove degenerate faces, and remove tiny disconnected components.",
            output_media_type="model/obj",
            deterministic=True,
        ),
        MeshRepairParameters,
        _repair,
    )
    registry.register(
        OperationDefinition(
            operation_id="mesh.analyze_health",
            worker_id="mesh.deterministic",
            stages=[AssetStage.geometry, AssetStage.topology],
            description="Produce deterministic topology health metrics for a triangular OBJ.",
            output_media_type="application/json",
            deterministic=True,
        ),
        MeshAnalyzeParameters,
        _analyze,
    )
