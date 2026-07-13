"""Stateful asset compiler that stages every operation before promotion."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .events import EventLog
from .hashing import sha256_bytes
from .operations import OperationContext, OperationRegistry
from .schemas import (
    ApprovalRecord,
    ArtifactLineage,
    ArtifactRecord,
    AssetBrief,
    AssetStage,
    CandidateEvaluation,
    ExternalWorkerResponse,
)


class DarknessCompiler:
    def __init__(self, workspace: str | Path, operations: OperationRegistry) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(self.workspace / "artifacts")
        self.operations = operations

    def log(self, run_id: str) -> EventLog:
        return EventLog(self.workspace, run_id)

    def create_run(self, run_id: str, brief: AssetBrief) -> ArtifactRecord:
        log = self.log(run_id)
        if log.read():
            raise RuntimeError(f"run '{run_id}' already exists")
        data = brief.model_dump_json(indent=2).encode("utf-8")
        artifact_id = f"{brief.asset_id}.brief.v1"
        digest = sha256_bytes(data)
        lineage = ArtifactLineage(
            artifact_id=artifact_id,
            artifact_sha256=digest,
            stage=AssetStage.brief.value,
            source_license_ids=["user-authored"],
        )
        record = self.artifacts.record_bytes(
            data,
            artifact_id=artifact_id,
            media_type="application/json",
            stage=AssetStage.brief,
            lineage=lineage,
            metadata={"asset_id": brief.asset_id, "kind": "asset_brief"},
        )
        log.append("run_created", AssetStage.brief, {"brief_artifact_id": artifact_id})
        log.append("artifact_recorded", AssetStage.brief, {"artifact_id": artifact_id, "sha256": digest})
        log.append("candidate_staged", AssetStage.brief, {"artifact_id": artifact_id})
        return record

    def execute_candidate(
        self,
        run_id: str,
        *,
        operation_id: str,
        artifact_id: str,
        input_artifact_ids: list[str],
        parameters: dict[str, Any],
    ) -> ArtifactRecord:
        log = self.log(run_id)
        state = log.state()
        if state.status != "running":
            raise RuntimeError("candidate execution requires a running run")
        records = [self.artifacts.load_record(item) for item in input_artifact_ids]
        context = OperationContext(
            run_id=run_id,
            stage=state.stage,
            input_records=records,
            input_bytes=[self.artifacts.read_bytes(item) for item in input_artifact_ids],
        )
        output = self.operations.execute(operation_id, context, parameters)
        digest = sha256_bytes(output.data)
        lineage = ArtifactLineage(
            artifact_id=artifact_id,
            artifact_sha256=digest,
            stage=state.stage.value,
            producer_candidate_ids=output.producer_candidate_ids,
            parent_artifact_ids=input_artifact_ids,
            source_license_ids=output.source_license_ids,
            source_license_status=output.source_license_status,
        )
        record = self.artifacts.record_bytes(
            output.data,
            artifact_id=artifact_id,
            media_type=output.media_type,
            stage=state.stage,
            lineage=lineage,
            metadata={"operation_id": operation_id, **output.metadata},
        )
        log.append("artifact_recorded", state.stage, {"artifact_id": artifact_id, "sha256": digest})
        log.append("candidate_staged", state.stage, {"artifact_id": artifact_id})
        return record

    def record_evaluation(self, run_id: str, evaluation: CandidateEvaluation) -> None:
        log = self.log(run_id)
        state = log.state()
        if evaluation.candidate_artifact_id not in state.staged_candidates:
            raise RuntimeError("evaluation target is not a staged candidate")
        log.append(
            "candidate_evaluated",
            state.stage,
            {"evaluation": evaluation.model_dump(mode="json")},
        )
        event_type = "candidate_promoted" if evaluation.accepted else "candidate_rejected"
        log.append(event_type, state.stage, {"artifact_id": evaluation.candidate_artifact_id})

    def import_external_response(
        self,
        run_id: str,
        *,
        response: ExternalWorkerResponse,
        candidate_id: str,
        parent_artifact_ids: list[str],
        artifact_ids_by_role: dict[str, str],
        primary_role: str,
        source_license_ids: list[str],
        source_license_status: str = "cleared",
    ) -> dict[str, ArtifactRecord]:
        if response.status != "succeeded":
            raise RuntimeError(response.error or f"external worker status is {response.status}")
        log = self.log(run_id)
        state = log.state()
        if state.status != "running":
            raise RuntimeError("external import requires a running run")
        if primary_role not in artifact_ids_by_role:
            raise ValueError("primary_role requires an artifact ID mapping")
        imported: dict[str, ArtifactRecord] = {}
        for output in response.outputs:
            artifact_id = artifact_ids_by_role.get(output.role)
            if artifact_id is None:
                raise ValueError(f"missing artifact ID mapping for output role '{output.role}'")
            path = Path(output.path).resolve()
            data = path.read_bytes()
            digest = sha256_bytes(data)
            lineage = ArtifactLineage(
                artifact_id=artifact_id,
                artifact_sha256=digest,
                stage=state.stage.value,
                producer_candidate_ids=[candidate_id],
                parent_artifact_ids=parent_artifact_ids,
                source_license_ids=source_license_ids,
                source_license_status=source_license_status,
            )
            record = self.artifacts.record_bytes(
                data,
                artifact_id=artifact_id,
                media_type=output.media_type,
                stage=state.stage,
                lineage=lineage,
                metadata={"job_id": response.job_id, "role": output.role, **output.metadata},
            )
            imported[output.role] = record
            log.append("artifact_recorded", state.stage, {"artifact_id": artifact_id, "sha256": digest})
        primary = imported.get(primary_role)
        if primary is None:
            raise ValueError(f"worker response did not include primary role '{primary_role}'")
        log.append("candidate_staged", state.stage, {"artifact_id": primary.artifact_id})
        return imported

    def approve(
        self,
        run_id: str,
        *,
        artifact_id: str,
        approved_by: str,
        locked_features: list[str] | None = None,
        notes: str = "",
    ) -> ApprovalRecord:
        log = self.log(run_id)
        state = log.state()
        if state.promoted_by_stage.get(state.stage.value) != artifact_id:
            raise RuntimeError("only the promoted artifact for the current stage can be approved")
        record = self.artifacts.load_record(artifact_id)
        approval = ApprovalRecord(
            approval_id=f"approval.{run_id}.{state.stage.name}.{state.sequence + 1}",
            artifact_id=artifact_id,
            stage=state.stage,
            artifact_sha256=record.sha256,
            approved_by=approved_by,
            approved_at=datetime.now(timezone.utc),
            locked_features=locked_features or [],
            notes=notes,
        )
        log.append("approval_recorded", state.stage, {"approval": approval.model_dump(mode="json")})
        return approval

    def advance(self, run_id: str) -> AssetStage:
        log = self.log(run_id)
        state = log.state()
        promoted = state.promoted_by_stage.get(state.stage.value)
        approval = state.approvals.get(state.stage.value)
        if not promoted or not approval or approval.artifact_id != promoted:
            raise RuntimeError("current promoted artifact requires a current approval before stage advance")
        stages = list(AssetStage)
        index = stages.index(state.stage)
        if index == len(stages) - 1:
            raise RuntimeError("D10_export is the final stage")
        target = stages[index + 1]
        log.append("stage_advanced", state.stage, {"to_stage": target.value})
        return target

    def complete(self, run_id: str) -> None:
        log = self.log(run_id)
        state = log.state()
        if state.stage is not AssetStage.export:
            raise RuntimeError("run can only complete at D10_export")
        promoted = state.promoted_by_stage.get(state.stage.value)
        approval = state.approvals.get(state.stage.value)
        if not promoted or not approval or approval.artifact_id != promoted:
            raise RuntimeError("final promoted export requires current approval")
        log.append("run_completed", state.stage, {"artifact_id": promoted})
