"""Validation for canonical topology/deformation family contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import ForgeError
from .genesis import ALLOWED_BODY_PLANS, sha256_file


def inspect_family_contract(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForgeError(f"Family contract could not be read: {error}") from error

    failures: list[str] = []
    if value.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    if value.get("stage") != "family_contract":
        failures.append("stage must be family_contract")
    family_id = str(value.get("id", "")).strip()
    if not family_id:
        failures.append("id is required")
    if value.get("body_plan") not in ALLOWED_BODY_PLANS:
        failures.append("body_plan is invalid")

    graph = value.get("topology_graph") or {}
    points = graph.get("points")
    edges = graph.get("edges")
    if not isinstance(points, list) or len(points) < 2 or any(not isinstance(item, str) for item in points):
        failures.append("topology_graph.points must contain at least two landmark names")
        points = []
    if len(set(points)) != len(points):
        failures.append("topology_graph.points must be unique")
    if not isinstance(edges, list) or not edges:
        failures.append("topology_graph.edges must be non-empty")
        edges = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2 or any(point not in points for point in edge):
            failures.append(f"topology_graph.edges[{index}] must reference two declared points")
    radii = graph.get("radii") or {}
    for point in points:
        radius = radii.get(point)
        if not isinstance(radius, (int, float)) or float(radius) <= 0:
            failures.append(f"topology_graph.radii.{point} must be positive")

    bones = value.get("skeleton") or []
    if not isinstance(bones, list) or not bones:
        failures.append("skeleton must be a non-empty array")
        bones = []
    names: set[str] = set()
    for index, bone in enumerate(bones):
        label = f"skeleton[{index}]"
        if not isinstance(bone, dict):
            failures.append(f"{label} must be an object")
            continue
        name = str(bone.get("name", "")).strip()
        if not name:
            failures.append(f"{label}.name is required")
        elif name in names:
            failures.append(f"duplicate bone name: {name}")
        names.add(name)
        if bone.get("head") not in points or bone.get("tail") not in points:
            failures.append(f"{label} head/tail must reference topology points")
    for bone in bones:
        if isinstance(bone, dict) and bone.get("parent") and bone["parent"] not in names:
            failures.append(f"bone {bone.get('name')} references missing parent {bone['parent']}")

    required_actions = value.get("required_actions")
    if not isinstance(required_actions, list) or not required_actions:
        failures.append("required_actions must be non-empty")
        required_actions = []
    if len(set(required_actions)) != len(required_actions):
        failures.append("required_actions must be unique")

    return {
        "schema_version": 1,
        "stage": "family_contract_validation",
        "family": family_id,
        "body_plan": value.get("body_plan"),
        "contract": str(source),
        "contract_sha256": sha256_file(source),
        "point_count": len(points),
        "edge_count": len(edges),
        "bone_count": len(bones),
        "required_actions": required_actions,
        "passed": not failures,
        "failures": failures,
    }


def validate_family_contract(path: Path) -> dict[str, Any]:
    report = inspect_family_contract(path)
    if not report["passed"]:
        raise ForgeError("Invalid family contract: " + "; ".join(report["failures"][:16]))
    return report


def inspect_canonical_master(contract_path: Path, master_path: Path, report_path: Path) -> dict[str, Any]:
    contract = inspect_family_contract(contract_path)
    master = Path(master_path).resolve()
    build_report = Path(report_path).resolve()
    failures = list(contract["failures"])
    try:
        build = json.loads(build_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForgeError(f"Canonical-master report could not be read: {error}") from error

    if build.get("stage") != "canonical_master":
        failures.append("report stage must be canonical_master")
    if build.get("production_eligible") is not False:
        failures.append("canonical master must remain production_eligible=false until surface/sprite gates pass")
    if build.get("family") != contract["family"]:
        failures.append("report family does not match the contract")
    if build.get("contract_sha256") != contract["contract_sha256"]:
        failures.append("report is not hash-bound to the current family contract")
    if not master.is_file():
        failures.append(f"canonical master is missing: {master}")
    elif build.get("master_sha256") != sha256_file(master):
        failures.append("master_sha256 does not match the canonical master")

    mesh = build.get("mesh") or {}
    if mesh.get("connected_components") != 1:
        failures.append("canonical body must be one connected component")
    if mesh.get("boundary_edges") != 0 or mesh.get("non_manifold_edges") != 0:
        failures.append("canonical body must be watertight and manifold")
    if not mesh.get("has_uv"):
        failures.append("canonical body has no UV map")
    if int(mesh.get("vertices", 0) or 0) < 100:
        failures.append("canonical body is too coarse")

    weights = build.get("weights") or {}
    if weights.get("unweighted_vertices") != 0:
        failures.append("canonical body has unweighted vertices")
    max_sum_error = weights.get("max_sum_error")
    if max_sum_error is None or float(max_sum_error) > 0.001:
        failures.append("vertex weights do not normalize to one")
    max_influences = weights.get("max_influences")
    if max_influences is None or int(max_influences) > 4:
        failures.append("canonical body exceeds four bone influences per vertex")

    missing_bones = sorted(set(build.get("missing_bones") or []))
    missing_actions = sorted(set(build.get("missing_actions") or []))
    if missing_bones:
        failures.append("missing bones: " + ", ".join(missing_bones))
    if missing_actions:
        failures.append("missing actions: " + ", ".join(missing_actions))
    motion = build.get("motion_audit") or {}
    if not motion.get("passed"):
        failures.append("motion/deformation audit did not pass")

    return {
        "schema_version": 1,
        "stage": "canonical_master_validation",
        "family": contract["family"],
        "master": str(master),
        "report": str(build_report),
        "passed": not failures,
        "canonical_stage_eligible": not failures,
        "production_eligible": False,
        "next_required_stage": "surface_bake_and_sprite_qa",
        "failures": failures,
        "mesh": mesh,
        "weights": weights,
        "motion_audit": motion,
    }


def validate_canonical_master(contract_path: Path, master_path: Path, report_path: Path) -> dict[str, Any]:
    result = inspect_canonical_master(contract_path, master_path, report_path)
    if not result["passed"]:
        raise ForgeError("Canonical master failed: " + "; ".join(result["failures"][:16]))
    return result
