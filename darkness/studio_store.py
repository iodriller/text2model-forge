"""Atomic run persistence, evidence hashing, and review decision semantics."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from .studio_models import (
    StudioEvidence,
    StudioHumanDecision,
    StudioQwenReview,
    StudioRun,
    new_studio_run,
)


class StudioStore:
    def __init__(self, workspace: str | Path) -> None:
        self.root = Path(workspace).resolve() / "studio"
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def run_root(self, run_id: str) -> Path:
        if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789.-_" for ch in run_id):
            raise ValueError("invalid run id")
        return self.runs_root / run_id

    def create(
        self, run_id: str, description: str, overrides: dict[str, Any] | None = None
    ) -> StudioRun:
        run = new_studio_run(run_id, description, overrides)
        root = self.run_root(run_id)
        with self._lock:
            if root.exists():
                raise FileExistsError(f"studio run already exists: {run_id}")
            root.mkdir(parents=True)
            self.save(run)
            self.event(run, "run_created", {"description": description})
        return run

    def load(self, run_id: str) -> StudioRun:
        path = self.run_root(run_id) / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"unknown studio run: {run_id}")
        with self._lock:
            return StudioRun.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, run: StudioRun) -> None:
        with self._lock:
            run.updated_at = datetime.now(timezone.utc)
            path = self.run_root(run.run_id) / "run.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
            # Path.replace() is atomic on POSIX but can still raise a transient
            # PermissionError on Windows if another process or thread (real-time
            # antivirus, an unlocked reader) briefly has run.json open at the
            # exact moment of the rename. This is not a real conflict -- retry
            # rather than fail the whole stage over a race that clears in
            # milliseconds; only propagate if it is still happening after that.
            attempts = 8
            for attempt in range(attempts):
                try:
                    temporary.replace(path)
                    return
                except PermissionError:
                    if attempt == attempts - 1:
                        raise
                    time.sleep(0.05 * (attempt + 1))

    def list(self) -> list[StudioRun]:
        with self._lock:
            runs = []
            for path in self.runs_root.glob("*/run.json"):
                try:
                    runs.append(StudioRun.model_validate_json(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
            return sorted(runs, key=lambda item: item.updated_at, reverse=True)

    def recover_interrupted_runs(self) -> list[str]:
        """Release runs left in `running` by a stopped local Studio process."""
        recovered: list[str] = []
        for run in self.list():
            stage = run.stage(run.current_stage)
            interrupted_failure = (
                run.state == "failed"
                and stage.error is not None
                and "local Studio process stopped" in stage.error
            )
            if run.state != "running" and not interrupted_failure:
                continue
            if stage.state not in {"running", "failed"}:
                run.state = "failed"
                stage.error = "Studio stopped between stages; resume is safe."
                stage.message = "The prior local process stopped. Use Resume to continue from saved state."
            else:
                current_candidates = [
                    item
                    for item in stage.evidence
                    if item.metrics.get("iteration") == stage.iteration
                    and item.metrics.get("selectable") is not False
                    and item.media_type.startswith("image/")
                    and "candidate" in item.evidence_id
                ]
                if stage.gate_required and current_candidates:
                    for item in current_candidates:
                        item.metrics["selectable"] = True
                    stage.qwen_reviews.append(
                        StudioQwenReview(
                            review_id=f"{stage.stage_id.lower()}.interrupted-{stage.iteration:02d}",
                            stage_id=stage.stage_id,
                            iteration=stage.iteration,
                            summary=(
                                "The deterministic images finished, but the prior Qwen critic process was "
                                "interrupted. The images are released for human review instead of leaving the "
                                "pipeline stuck."
                            ),
                            issues=["No completed Qwen visual ranking is available for this attempt."],
                            candidate_ranking=[item.evidence_id for item in current_candidates],
                            recommended_evidence_id=None,
                            recommended_changes=[
                                "Approve only if one result satisfies the brief; otherwise reject with the exact defect."
                            ],
                            confidence=0,
                            request_human_review=True,
                        )
                    )
                    stage.state = "awaiting_review"
                    stage.progress = 1
                    stage.error = None
                    stage.message = "Generated evidence recovered after interruption and ready for your decision."
                    run.state = "awaiting_review"
                else:
                    stage.state = "failed"
                    stage.error = "The local Studio process stopped before this automatic stage completed."
                    stage.message = "Use Resume to retry this stage from its persisted inputs and history."
                    run.state = "failed"
            stage.finished_at = datetime.now(timezone.utc)
            self.event(
                run,
                "stage_recovered",
                {"stage_id": stage.stage_id, "new_state": stage.state, "iteration": stage.iteration},
            )
            recovered.append(run.run_id)
        return recovered

    def event(self, run: StudioRun, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            run.event_count += 1
            record = {
                "sequence": run.event_count,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "stage_id": run.current_stage,
                "payload": payload,
            }
            with (self.run_root(run.run_id) / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            self.save(run)

    def read_events(self, run_id: str) -> list[dict[str, Any]]:
        path = self.run_root(run_id) / "events.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def evidence(
        self,
        run: StudioRun,
        stage_id: str,
        path: Path,
        *,
        evidence_id: str,
        label: str,
        media_type: str,
        metrics: dict[str, float | int | bool | str | None] | None = None,
    ) -> StudioEvidence:
        resolved = path.resolve()
        root = self.run_root(run.run_id).resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise ValueError("evidence must be a file inside the studio run")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        item = StudioEvidence(
            evidence_id=evidence_id,
            label=label,
            relative_path=resolved.relative_to(root).as_posix(),
            media_type=media_type,
            sha256=digest,
            metrics=metrics or {},
        )
        stage = run.stage(stage_id)
        stage.evidence = [existing for existing in stage.evidence if existing.evidence_id != evidence_id]
        stage.evidence.append(item)
        self.save(run)
        return item

    _DECISIONS = {"approve", "reject", "retry", "edit", "skip", "rollback"}

    @staticmethod
    def _invalidate_from(run: StudioRun, index: int, reason: str) -> None:
        """Reset every stage from `index` onward to pending, preserving each
        stage's own human_decisions (an append-only audit trail) but
        discarding evidence, reviews, and any stale pending_overrides built
        on what is now invalidated work."""
        for downstream in run.stages[index:]:
            downstream.state = "pending"
            downstream.progress = 0
            downstream.message = reason
            downstream.evidence = []
            downstream.qwen_reviews = []
            downstream.pending_overrides = {}
            downstream.error = None

    def decide(
        self,
        run_id: str,
        stage_id: str,
        decision: str,
        comment: str,
        selected_evidence_id: str | None,
        *,
        overrides: dict[str, Any] | None = None,
        target_stage_id: str | None = None,
    ) -> StudioRun:
        """Record one human decision at a stage gate.

        approve: pick a candidate and unlock the next stage.
        reject: comment required; Qwen's next attempt sees it.
        retry: reroll the same stage, no comment required, no quality
            judgement implied. `overrides` (if given) become the stage's
            pending_overrides for its next attempt.
        edit: like reject, but the correction is concrete: comment,
            `overrides`, or both are required.
        skip: mark this stage not applicable; comment required as the
            reason; does not invalidate anything downstream.
        rollback: reopen an earlier stage named by `target_stage_id` and
            invalidate everything from there forward, including this stage.
        """
        if decision not in self._DECISIONS:
            raise ValueError(f"decision must be one of {sorted(self._DECISIONS)}")
        if decision == "reject" and not comment.strip():
            raise ValueError("a rejection comment is required so Qwen knows what to improve")
        if decision == "edit" and not comment.strip() and not overrides:
            raise ValueError("an edit needs a comment, override values, or both")
        if decision == "skip" and not comment.strip():
            raise ValueError("a skip reason is required so the record explains why the stage was bypassed")
        if decision == "rollback" and not target_stage_id:
            raise ValueError("rollback requires target_stage_id")
        with self._lock:
            run = self.load(run_id)
            stage = run.stage(stage_id)
            if not stage.gate_required or stage.state != "awaiting_review":
                raise ValueError(f"{stage_id} is not waiting for a human decision")
            evidence_ids = {item.evidence_id for item in stage.evidence}
            if selected_evidence_id and selected_evidence_id not in evidence_ids:
                raise ValueError("selected evidence does not belong to this gate")
            if decision == "approve" and selected_evidence_id:
                selected = next(item for item in stage.evidence if item.evidence_id == selected_evidence_id)
                if selected.metrics.get("selectable") is not True:
                    raise ValueError("select a production candidate, not a comparison or report file")
            if decision == "approve" and len(stage.evidence) > 1 and not selected_evidence_id:
                recommended = stage.qwen_reviews[-1].recommended_evidence_id if stage.qwen_reviews else None
                recommended_item = next(
                    (item for item in stage.evidence if item.evidence_id == recommended), None
                )
                if recommended_item is None or recommended_item.metrics.get("selectable") is not True:
                    raise ValueError("select one candidate before approving")
                selected_evidence_id = recommended
            target_index: int | None = None
            if decision == "rollback":
                stage_ids = [item.stage_id for item in run.stages]
                if target_stage_id not in stage_ids:
                    raise ValueError(f"unknown stage: {target_stage_id}")
                target_index = stage_ids.index(target_stage_id)
                current_index = stage_ids.index(stage_id)
                if target_index >= current_index:
                    raise ValueError(
                        "rollback target must be an earlier stage than the stage the decision is recorded against"
                    )
                if run.stages[target_index].state not in {"approved", "skipped", "rejected", "failed"}:
                    raise ValueError(f"{target_stage_id} has no prior decision to roll back to")
            record = StudioHumanDecision(
                decision_id=f"{stage_id.lower()}.{uuid.uuid4().hex[:12]}",
                decision=decision,
                comment=comment.strip(),
                selected_evidence_id=selected_evidence_id,
                evidence_hashes={item.evidence_id: item.sha256 for item in stage.evidence},
                overrides=overrides or {},
                target_stage_id=target_stage_id if decision == "rollback" else None,
            )
            stage.human_decisions.append(record)
            if decision != "rollback":
                # rollback's state/message for every affected stage, including
                # this one when it falls in range, is set by _invalidate_from below.
                state_by_decision = {
                    "approve": "approved",
                    "reject": "rejected",
                    "retry": "pending",
                    "edit": "rejected",
                    "skip": "skipped",
                }
                message_by_decision = {
                    "approve": "Approved. The next deterministic stage may run.",
                    "reject": "Rejected. Qwen will use the comment and complete history for the next attempt.",
                    "retry": "Retrying with a fresh attempt.",
                    "edit": "Correction recorded. The next attempt will apply it.",
                    "skip": "Skipped: " + comment.strip(),
                }
                stage.state = state_by_decision[decision]
                stage.message = message_by_decision[decision]
            if decision in {"retry", "edit"}:
                stage.pending_overrides = overrides or {}
            run.state = "running"
            event_type_by_decision = {
                "approve": "gate_approved",
                "reject": "gate_rejected",
                "retry": "gate_retried",
                "edit": "gate_edited",
                "skip": "gate_skipped",
                "rollback": "gate_rolled_back",
            }
            self.event(run, event_type_by_decision[decision], record.model_dump(mode="json"))
            if decision in {"reject", "retry", "edit"}:
                index = next(i for i, item in enumerate(run.stages) if item.stage_id == stage_id)
                self._invalidate_from(run, index + 1, "Invalidated by an upstream " + decision + ".")
                run.current_stage = stage_id
            elif decision == "rollback":
                assert target_index is not None
                self._invalidate_from(run, target_index, f"Reopened by a rollback from {stage_id}.")
                run.current_stage = target_stage_id
            self.save(run)
            return run

    def artifact_path(self, run_id: str, relative_path: str) -> Path:
        root = self.run_root(run_id).resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents or not target.is_file():
            raise FileNotFoundError("artifact not found")
        return target
