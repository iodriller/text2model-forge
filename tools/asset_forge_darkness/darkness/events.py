"""Append-only, hash-chained run events and deterministic state reduction."""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .hashing import sha256_json
from .schemas import ApprovalRecord, AssetStage, RunEvent, RunState


class EventLogIntegrityError(RuntimeError):
    pass


class InvalidRunTransition(RuntimeError):
    pass


STAGES = list(AssetStage)


def _event_hash(event_data: dict[str, Any]) -> str:
    material = dict(event_data)
    material.pop("event_hash", None)
    return sha256_json(material)


def reduce_events(events: list[RunEvent]) -> RunState:
    if not events:
        raise InvalidRunTransition("a run requires at least one event")
    state = RunState(run_id=events[0].run_id)
    for event in events:
        if event.run_id != state.run_id:
            raise InvalidRunTransition("event run_id does not match the event log")
        if event.stage is not state.stage:
            raise InvalidRunTransition(
                f"event stage {event.stage.value} does not match current stage {state.stage.value}"
            )
        state.sequence = event.sequence
        payload = event.payload
        if event.event_type == "run_created":
            if event.sequence != 1:
                raise InvalidRunTransition("run_created must be the first event")
            state.status = "running"
        elif event.event_type == "artifact_recorded":
            state.artifacts[str(payload["artifact_id"])] = str(payload["sha256"])
        elif event.event_type == "candidate_staged":
            candidate_id = str(payload["artifact_id"])
            if candidate_id not in state.staged_candidates:
                state.staged_candidates.append(candidate_id)
        elif event.event_type == "candidate_promoted":
            candidate_id = str(payload["artifact_id"])
            if candidate_id not in state.staged_candidates:
                raise InvalidRunTransition("only staged candidates can be promoted")
            state.promoted_by_stage[event.stage.value] = candidate_id
            state.staged_candidates.remove(candidate_id)
        elif event.event_type == "candidate_rejected":
            candidate_id = str(payload["artifact_id"])
            if candidate_id in state.staged_candidates:
                state.staged_candidates.remove(candidate_id)
            if candidate_id not in state.rejected_candidates:
                state.rejected_candidates.append(candidate_id)
        elif event.event_type == "approval_recorded":
            approval = ApprovalRecord.model_validate(payload["approval"])
            current_sha = state.artifacts.get(approval.artifact_id)
            if current_sha != approval.artifact_sha256:
                raise InvalidRunTransition("approval does not match the current artifact digest")
            state.approvals[approval.stage.value] = approval
        elif event.event_type == "approval_invalidated":
            state.approvals.pop(event.stage.value, None)
        elif event.event_type == "stage_advanced":
            target = AssetStage(str(payload["to_stage"]))
            current_index = STAGES.index(state.stage)
            if STAGES.index(target) != current_index + 1:
                raise InvalidRunTransition("stages must advance exactly one step")
            state.stage = target
        elif event.event_type == "run_failed":
            state.status = "failed"
            state.failure = str(payload.get("reason", "unknown failure"))
        elif event.event_type == "run_resumed":
            if state.status != "failed":
                raise InvalidRunTransition("only a failed run can be resumed")
            state.status = "running"
            state.failure = None
        elif event.event_type == "run_completed":
            if state.stage is not AssetStage.export:
                raise InvalidRunTransition("run can only complete at D10_export")
            state.status = "completed"
    return state


class EventLog:
    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.run_dir = self.root / "runs" / run_id
        self.path = self.run_dir / "events.jsonl"
        self.lock_path = self.run_dir / ".events.lock"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self, timeout_seconds: float = 5.0) -> Iterator[None]:
        deadline = time.monotonic() + timeout_seconds
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, str(os.getpid()).encode("ascii"))
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"event log is locked: {self.lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)

    def read(self, *, verify: bool = True) -> list[RunEvent]:
        if not self.path.exists():
            return []
        events = [
            RunEvent.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if verify:
            previous: str | None = None
            for expected_sequence, event in enumerate(events, start=1):
                if event.sequence != expected_sequence:
                    raise EventLogIntegrityError("event sequence is not contiguous")
                if event.previous_event_hash != previous:
                    raise EventLogIntegrityError("event hash chain is broken")
                if _event_hash(event.model_dump(mode="json")) != event.event_hash:
                    raise EventLogIntegrityError(f"event {event.sequence} hash is invalid")
                previous = event.event_hash
        return events

    def append(self, event_type: str, stage: AssetStage, payload: dict[str, Any] | None = None) -> RunEvent:
        with self._lock():
            events = self.read()
            previous = events[-1].event_hash if events else None
            event_data: dict[str, Any] = {
                "run_id": self.run_id,
                "sequence": len(events) + 1,
                "event_id": uuid.uuid4().hex,
                "event_type": event_type,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "stage": stage.value,
                "payload": payload or {},
                "previous_event_hash": previous,
                "event_hash": "0" * 64,
            }
            provisional = RunEvent.model_validate(event_data)
            event_data = provisional.model_dump(mode="json")
            event_data["event_hash"] = _event_hash(event_data)
            event = RunEvent.model_validate(event_data)
            proposed = events + [event]
            reduce_events(proposed)  # validate transition before persisting
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(event.model_dump_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return event

    def state(self) -> RunState:
        return reduce_events(self.read())
