"""Research and release policy over recursive artifact lineage."""
from __future__ import annotations

from .schemas import ArtifactLineage, CandidateRegistry, PolicyDecision, ReleaseEligibility


def _walk(root_id: str, artifacts: dict[str, ArtifactLineage]) -> tuple[list[ArtifactLineage], list[str]]:
    ordered: list[ArtifactLineage] = []
    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in visited:
            return
        if artifact_id in visiting:
            errors.append(f"artifact lineage cycle detected at '{artifact_id}'")
            return
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            errors.append(f"missing parent artifact lineage '{artifact_id}'")
            return
        visiting.add(artifact_id)
        for parent_id in artifact.parent_artifact_ids:
            visit(parent_id)
        visiting.remove(artifact_id)
        visited.add(artifact_id)
        ordered.append(artifact)

    visit(root_id)
    return ordered, errors


def evaluate_research(
    root_id: str, artifacts: dict[str, ArtifactLineage], registry: CandidateRegistry
) -> PolicyDecision:
    lineage, reasons = _walk(root_id, artifacts)
    candidates = {item.candidate_id: item for item in registry.candidates}
    for artifact in lineage:
        if artifact.source_license_status == "blocked":
            reasons.append(f"artifact '{artifact.artifact_id}' has blocked source licensing")
        for candidate_id in artifact.producer_candidate_ids:
            candidate = candidates.get(candidate_id)
            if candidate is None:
                reasons.append(f"artifact '{artifact.artifact_id}' references unknown candidate '{candidate_id}'")
            elif not candidate.research_allowed:
                reasons.append(f"candidate '{candidate_id}' is not approved even for research")
    return PolicyDecision(
        allowed=not reasons,
        reasons=reasons,
        traversed_artifact_ids=[item.artifact_id for item in lineage],
    )


def evaluate_release(
    root_id: str, artifacts: dict[str, ArtifactLineage], registry: CandidateRegistry
) -> PolicyDecision:
    lineage, reasons = _walk(root_id, artifacts)
    candidates = {item.candidate_id: item for item in registry.candidates}
    for artifact in lineage:
        if artifact.source_license_status != "cleared":
            reasons.append(
                f"artifact '{artifact.artifact_id}' source licensing is {artifact.source_license_status}"
            )
        for candidate_id in artifact.producer_candidate_ids:
            candidate = candidates.get(candidate_id)
            if candidate is None:
                reasons.append(f"artifact '{artifact.artifact_id}' references unknown candidate '{candidate_id}'")
            elif candidate.release_eligibility is not ReleaseEligibility.permitted:
                reasons.append(
                    f"candidate '{candidate_id}' release eligibility is {candidate.release_eligibility.value}"
                )
    return PolicyDecision(
        allowed=not reasons,
        reasons=reasons,
        traversed_artifact_ids=[item.artifact_id for item in lineage],
    )
