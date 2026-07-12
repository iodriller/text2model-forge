"""Final export gate combining lineage, approvals, and contract tests."""
from __future__ import annotations

from .lineage import evaluate_release
from .schemas import ApprovalRecord, ArtifactRecord, CandidateRegistry, DestinationContract, ExportDecision


def decide_export(
    artifact: ArtifactRecord,
    *,
    destination: DestinationContract,
    all_artifacts: dict[str, ArtifactRecord],
    registry: CandidateRegistry,
    approval: ApprovalRecord | None,
    contract_tests_passed: bool,
) -> ExportDecision:
    lineage = {artifact_id: item.lineage for artifact_id, item in all_artifacts.items()}
    release = evaluate_release(artifact.artifact_id, lineage, registry)
    approval_current = bool(
        approval
        and approval.artifact_id == artifact.artifact_id
        and approval.artifact_sha256 == artifact.sha256
        and approval.stage == artifact.stage
    )
    reasons = list(release.reasons)
    if not approval_current:
        reasons.append("missing or stale human approval")
    if not contract_tests_passed:
        reasons.append("destination contract tests did not pass")
    return ExportDecision(
        artifact_id=artifact.artifact_id,
        destination=destination,
        release_policy=release,
        approvals_current=approval_current,
        contract_tests_passed=contract_tests_passed,
        allowed=not reasons,
        reasons=reasons,
    )
