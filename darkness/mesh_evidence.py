"""Human-readable evidence output for the deterministic station-3 mesh cleanup slice."""
from __future__ import annotations

import json
from pathlib import Path

from .hashing import sha256_bytes
from .mesh import MeshHealth, MeshRepairDecision, TriangleMesh


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _health_rows(before: MeshHealth, after: MeshHealth) -> str:
    labels = {
        "vertices": "Vertices",
        "faces": "Faces",
        "duplicate_vertices": "Duplicate vertices",
        "isolated_vertices": "Isolated vertices",
        "degenerate_faces": "Degenerate faces",
        "connected_components": "Connected components",
        "boundary_edges": "Boundary edges",
        "non_manifold_edges": "Non-manifold edges",
        "inconsistent_winding_edges": "Inconsistent winding edges",
        "finite_coordinates": "Finite coordinates",
    }
    before_values = before.model_dump()
    after_values = after.model_dump()
    return "\n".join(
        f"| {label} | {before_values[key]} | {after_values[key]} |"
        for key, label in labels.items()
    )


def build_mesh_evidence(
    source_path: str | Path,
    output_directory: str | Path,
    *,
    tolerance: float = 1e-9,
    minimum_component_faces: int = 2,
) -> dict[str, object]:
    source_path = Path(source_path).resolve()
    output_directory = Path(output_directory).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_directory.mkdir(parents=True, exist_ok=False)

    source_bytes = source_path.read_bytes()
    mesh = TriangleMesh.from_obj(source_bytes.decode("utf-8"))
    selected, decision = mesh.guarded_repair(
        tolerance=tolerance,
        minimum_component_faces=minimum_component_faces,
    )
    selected_bytes = selected.to_obj().encode("utf-8")
    after = selected.health(tolerance)

    before_path = output_directory / "before_health.json"
    after_path = output_directory / "after_health.json"
    decision_path = output_directory / "repair_decision.json"
    repaired_path = output_directory / "repaired.obj"
    report_path = output_directory / "report.md"

    _write_json(before_path, decision.before.model_dump())
    _write_json(after_path, after.model_dump())
    _write_json(decision_path, decision.model_dump())
    repaired_path.write_bytes(selected_bytes)

    before_diagnoses = ", ".join(decision.before.diagnoses) or "none"
    after_diagnoses = ", ".join(after.diagnoses) or "none"
    reasons = "\n".join(f"- {reason}" for reason in decision.reasons) or "- none"
    report = f"""# Darkness station-3 mesh cleanup evidence

- Source: `{source_path.name}`
- Source SHA-256: `{sha256_bytes(source_bytes)}`
- Selected output SHA-256: `{sha256_bytes(selected_bytes)}`
- Repair branch accepted: **{str(decision.accepted).lower()}**
- Before diagnoses: {before_diagnoses}
- After diagnoses: {after_diagnoses}

| Metric | Before | Selected output |
|---|---:|---:|
{_health_rows(decision.before, after)}

## Repair decision reasons

{reasons}

This report is deterministic evidence only. It does not replace Blender renders or human approval of silhouette.
"""
    report_path.write_text(report, encoding="utf-8")

    return {
        "accepted": decision.accepted,
        "before_diagnoses": decision.before.diagnoses,
        "after_diagnoses": after.diagnoses,
        "source_sha256": sha256_bytes(source_bytes),
        "selected_sha256": sha256_bytes(selected_bytes),
        "files": {
            "report": str(report_path),
            "before_health": str(before_path),
            "after_health": str(after_path),
            "repair_decision": str(decision_path),
            "repaired_mesh": str(repaired_path),
        },
    }
