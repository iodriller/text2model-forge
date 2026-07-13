from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from darkness.artifacts import ArtifactConflictError, ArtifactIntegrityError, ArtifactStore
from darkness.events import EventLog, EventLogIntegrityError
from darkness.export_policy import decide_export
from darkness.hashing import sha256_bytes
from darkness.registry import load_registry
from darkness.schemas import (
    ApprovalRecord,
    ArtifactLineage,
    AssetStage,
    DestinationContract,
)


def record(store: ArtifactStore, artifact_id: str, data: bytes, producer: str, stage: AssetStage = AssetStage.brief):
    digest = sha256_bytes(data)
    lineage = ArtifactLineage(
        artifact_id=artifact_id,
        artifact_sha256=digest,
        stage=stage.value,
        producer_candidate_ids=[producer],
    )
    return store.record_bytes(
        data,
        artifact_id=artifact_id,
        media_type="application/json",
        stage=stage,
        lineage=lineage,
    )


def test_artifact_store_is_content_addressed_idempotent_and_immutable(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    created = record(store, "brief.v1", b'{"asset":"goblin"}', "qwen3.6-27b")
    repeated = record(store, "brief.v1", b'{"asset":"goblin"}', "qwen3.6-27b")

    assert repeated == created
    assert store.read_bytes("brief.v1") == b'{"asset":"goblin"}'
    assert created.blob_path.endswith(created.sha256)

    with pytest.raises(ArtifactConflictError):
        record(store, "brief.v1", b'{"asset":"orc"}', "qwen3.6-27b")


def test_artifact_store_rejects_false_lineage_and_tampering(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    bad = ArtifactLineage(
        artifact_id="brief.v1",
        artifact_sha256="0" * 64,
        stage=AssetStage.brief.value,
    )
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        store.record_bytes(
            b"real bytes",
            artifact_id="brief.v1",
            media_type="text/plain",
            stage=AssetStage.brief,
            lineage=bad,
        )

    created = record(store, "brief.v1", b"real bytes", "qwen3.6-27b")
    (store.root / created.blob_path).write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        store.read_bytes("brief.v1")


def test_event_log_reduces_branch_promote_approve_advance_fail_and_resume(tmp_path) -> None:
    log = EventLog(tmp_path, "goblin.run1")
    digest = "a" * 64
    log.append("run_created", AssetStage.brief, {"brief_id": "goblin.brief"})
    log.append("artifact_recorded", AssetStage.brief, {"artifact_id": "brief.v1", "sha256": digest})
    log.append("candidate_staged", AssetStage.brief, {"artifact_id": "brief.v1"})
    log.append("candidate_promoted", AssetStage.brief, {"artifact_id": "brief.v1"})
    approval = ApprovalRecord(
        approval_id="approval.brief.v1",
        artifact_id="brief.v1",
        stage=AssetStage.brief,
        artifact_sha256=digest,
        approved_by="human",
        approved_at=datetime.now(timezone.utc),
    )
    log.append("approval_recorded", AssetStage.brief, {"approval": approval.model_dump(mode="json")})
    log.append("stage_advanced", AssetStage.brief, {"to_stage": AssetStage.concept.value})
    log.append("run_failed", AssetStage.concept, {"reason": "simulated worker crash"})
    log.append("run_resumed", AssetStage.concept, {})

    state = log.state()
    assert state.stage is AssetStage.concept
    assert state.status == "running"
    assert state.promoted_by_stage[AssetStage.brief.value] == "brief.v1"
    assert state.approvals[AssetStage.brief.value].artifact_sha256 == digest
    assert state.failure is None


def test_event_log_detects_tampering(tmp_path) -> None:
    log = EventLog(tmp_path, "goblin.run1")
    log.append("run_created", AssetStage.brief, {})
    line = json.loads(log.path.read_text(encoding="utf-8"))
    line["payload"] = {"tampered": True}
    log.path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(EventLogIntegrityError):
        log.read()


def test_export_requires_permitted_lineage_current_approval_and_contract_tests(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = record(store, "export.v1", b"game asset", "qwen3.6-27b", AssetStage.export)
    approval = ApprovalRecord(
        approval_id="approval.export.v1",
        artifact_id=artifact.artifact_id,
        stage=artifact.stage,
        artifact_sha256=artifact.sha256,
        approved_by="human",
        approved_at=datetime.now(timezone.utc),
    )
    destination = DestinationContract(delivery=["glb"])
    decision = decide_export(
        artifact,
        destination=destination,
        all_artifacts=store.all_records(),
        registry=load_registry(),
        approval=approval,
        contract_tests_passed=True,
    )
    assert decision.allowed is True

    restricted = record(store, "export.hunyuan", b"research asset", "hunyuan3d-2.1", AssetStage.export)
    restricted_approval = approval.model_copy(
        update={
            "approval_id": "approval.export.hunyuan",
            "artifact_id": restricted.artifact_id,
            "artifact_sha256": restricted.sha256,
        }
    )
    blocked = decide_export(
        restricted,
        destination=destination,
        all_artifacts=store.all_records(),
        registry=load_registry(),
        approval=restricted_approval,
        contract_tests_passed=True,
    )
    assert blocked.allowed is False
    assert "review_required" in " ".join(blocked.reasons)
