"""Load and validate the audited worker/model candidate registry."""
from __future__ import annotations

import json
from pathlib import Path
from vettedmesh_paths import resource_root

from .schemas import CandidatePriority, CandidateRegistry


def default_registry_path() -> Path:
    return resource_root() / "registry" / "candidates.json"


def load_registry(path: Path | None = None) -> CandidateRegistry:
    target = path or default_registry_path()
    registry = CandidateRegistry.model_validate_json(target.read_text(encoding="utf-8"))
    ids = [item.candidate_id for item in registry.candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate registry contains duplicate candidate_id values")
    return registry


def candidates_in_priority_order(registry: CandidateRegistry) -> list[str]:
    rank = {
        CandidatePriority.open_source_first: 0,
        CandidatePriority.research_primary: 1,
        CandidatePriority.later_candidate: 2,
    }
    return [
        item.candidate_id
        for item in sorted(registry.candidates, key=lambda item: (rank[item.priority], item.candidate_id))
    ]


def registry_as_json(registry: CandidateRegistry) -> str:
    return json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=True)
