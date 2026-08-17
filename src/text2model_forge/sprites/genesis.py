"""Contracts for the from-scratch Text2Model Forge "Genesis" branch.

Genesis deliberately separates a disposable anatomy target from the mesh that may
eventually ship.  The target may be voxel-remeshed because its only job is to
describe volume and landmarks.  A later canonical-topology fitting stage must
preserve the owned template's vertex order, UVs, weights, and deformation loops.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .core import ForgeError


ALLOWED_BODY_PLANS = {
    "biped",
    "quadruped",
    "winged_quadruped",
    "serpentine",
    "arthropod",
    "prop",
}
ALLOWED_VOLUME_TYPES = {"ellipsoid", "capsule", "hull"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point(value: object, label: str, failures: list[str]) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        failures.append(f"{label} must be a three-number array")
        return False
    if any(not isinstance(axis, (int, float)) for axis in value):
        failures.append(f"{label} must contain only numbers")
        return False
    return True


def inspect_genesis_spec(spec_path: Path) -> dict[str, Any]:
    path = Path(spec_path).resolve()
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForgeError(f"Genesis spec could not be read: {error}") from error

    failures: list[str] = []
    if spec.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    if spec.get("stage") != "genesis_spec":
        failures.append("stage must be genesis_spec")
    if not str(spec.get("id", "")).strip():
        failures.append("id is required")
    if spec.get("body_plan") not in ALLOWED_BODY_PLANS:
        failures.append("body_plan must be one of: " + ", ".join(sorted(ALLOWED_BODY_PLANS)))
    if not str(spec.get("family_template", "")).strip():
        failures.append("family_template is required; Genesis never promotes a raw voxel target")

    landmarks = spec.get("landmarks")
    if not isinstance(landmarks, dict) or not landmarks:
        failures.append("landmarks must be a non-empty object")
        landmarks = {}
    else:
        for name, point in landmarks.items():
            _point(point, f"landmarks.{name}", failures)

    volumes = spec.get("volumes")
    if not isinstance(volumes, list) or not volumes:
        failures.append("volumes must be a non-empty array")
        volumes = []
    seen: set[str] = set()
    for index, volume in enumerate(volumes):
        label = f"volumes[{index}]"
        if not isinstance(volume, dict):
            failures.append(f"{label} must be an object")
            continue
        volume_id = str(volume.get("id", "")).strip()
        if not volume_id:
            failures.append(f"{label}.id is required")
        elif volume_id in seen:
            failures.append(f"duplicate volume id: {volume_id}")
        seen.add(volume_id)
        kind = volume.get("type")
        if kind not in ALLOWED_VOLUME_TYPES:
            failures.append(f"{label}.type must be ellipsoid, capsule, or hull")
            continue
        if not str(volume.get("region", "")).strip():
            failures.append(f"{label}.region is required")
        if kind == "ellipsoid":
            center = volume.get("center")
            if isinstance(center, str):
                if center not in landmarks:
                    failures.append(f"{label}.center references missing landmark {center!r}")
            else:
                _point(center, f"{label}.center", failures)
            radii = volume.get("radii")
            if _point(radii, f"{label}.radii", failures) and any(float(v) <= 0 for v in radii):
                failures.append(f"{label}.radii must be positive")
            if "rotation_degrees" in volume:
                _point(volume["rotation_degrees"], f"{label}.rotation_degrees", failures)
        if kind == "capsule":
            for endpoint in ("start", "end"):
                value = volume.get(endpoint)
                if isinstance(value, str):
                    if value not in landmarks:
                        failures.append(f"{label}.{endpoint} references missing landmark {value!r}")
                else:
                    _point(value, f"{label}.{endpoint}", failures)
            radius = volume.get("radius")
            if not isinstance(radius, (int, float)) or float(radius) <= 0:
                failures.append(f"{label}.radius must be positive")
        if kind == "hull":
            points = volume.get("points")
            if not isinstance(points, list) or len(points) < 3:
                failures.append(f"{label}.points must contain at least three landmarks/points")
            else:
                for point_index, value in enumerate(points):
                    if isinstance(value, str):
                        if value not in landmarks:
                            failures.append(f"{label}.points[{point_index}] references missing landmark {value!r}")
                    else:
                        _point(value, f"{label}.points[{point_index}]", failures)
            thickness = volume.get("thickness")
            if not isinstance(thickness, (int, float)) or float(thickness) <= 0:
                failures.append(f"{label}.thickness must be positive")

    target = spec.get("anatomy_target") or {}
    voxel_size = target.get("voxel_size")
    if not isinstance(voxel_size, (int, float)) or not 0.005 <= float(voxel_size) <= 0.25:
        failures.append("anatomy_target.voxel_size must be between 0.005 and 0.25 meters")

    details = spec.get("details") or []
    if not isinstance(details, list):
        failures.append("details must be an array")
        details = []
    for index, detail in enumerate(details):
        label = f"details[{index}]"
        if not isinstance(detail, dict):
            failures.append(f"{label} must be an object")
            continue
        if detail.get("type") not in {"sphere", "ellipsoid", "cone", "box"}:
            failures.append(f"{label}.type must be sphere, ellipsoid, cone, or box")
        if not str(detail.get("bone", "")).strip():
            failures.append(f"{label}.bone is required")
        for key in ("center", "start", "end"):
            if key not in detail:
                continue
            point = detail[key]
            if isinstance(point, str):
                if point not in landmarks:
                    failures.append(f"{label}.{key} references missing landmark {point!r}")
            else:
                _point(point, f"{label}.{key}", failures)
        if detail.get("type") == "box":
            scale = detail.get("scale")
            if _point(scale, f"{label}.scale", failures) and any(float(axis) <= 0 for axis in scale):
                failures.append(f"{label}.scale must be positive")
            if "rotation_degrees" in detail:
                _point(detail["rotation_degrees"], f"{label}.rotation_degrees", failures)

    return {
        "schema_version": 1,
        "stage": "genesis_spec_validation",
        "spec": str(path),
        "spec_sha256": sha256_file(path),
        "asset_id": spec.get("id"),
        "body_plan": spec.get("body_plan"),
        "family_template": spec.get("family_template"),
        "landmark_count": len(landmarks),
        "volume_count": len(volumes),
        "detail_count": len(details),
        "passed": not failures,
        "failures": failures,
    }


def validate_genesis_spec(spec_path: Path) -> dict[str, Any]:
    report = inspect_genesis_spec(spec_path)
    if not report["passed"]:
        raise ForgeError("Invalid Genesis spec: " + "; ".join(report["failures"][:12]))
    return report


def inspect_anatomy_target(spec_path: Path, target_path: Path, build_report_path: Path) -> dict[str, Any]:
    spec_validation = inspect_genesis_spec(spec_path)
    target = Path(target_path).resolve()
    report_path = Path(build_report_path).resolve()
    failures = list(spec_validation["failures"])
    try:
        build = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForgeError(f"Genesis anatomy-target report could not be read: {error}") from error

    if build.get("stage") != "anatomy_target":
        failures.append("build report stage must be anatomy_target")
    if build.get("production_eligible") is not False:
        failures.append("an anatomy target must explicitly be production_eligible=false")
    if build.get("next_required_stage") != "canonical_topology_fit":
        failures.append("next_required_stage must be canonical_topology_fit")
    if build.get("spec_sha256") != spec_validation["spec_sha256"]:
        failures.append("build report is not hash-bound to the current Genesis spec")
    if not target.is_file():
        failures.append(f"anatomy target is missing: {target}")
    elif build.get("target_sha256") != sha256_file(target):
        failures.append("build report target_sha256 does not match the anatomy target")

    mesh = build.get("mesh") or {}
    if mesh.get("connected_components") != 1:
        failures.append("anatomy target must contain exactly one connected mesh component")
    if mesh.get("boundary_edges") != 0:
        failures.append("anatomy target must be watertight (zero boundary edges)")
    if mesh.get("non_manifold_edges") != 0:
        failures.append("anatomy target must have zero non-manifold edges")
    if int(mesh.get("vertices", 0) or 0) < 100:
        failures.append("anatomy target is too coarse (fewer than 100 vertices)")

    return {
        "schema_version": 1,
        "stage": "anatomy_target_validation",
        "asset_id": spec_validation.get("asset_id"),
        "target": str(target),
        "build_report": str(report_path),
        "passed": not failures,
        "production_eligible": False,
        "next_required_stage": "canonical_topology_fit",
        "failures": failures,
        "mesh": mesh,
    }


def validate_anatomy_target(spec_path: Path, target_path: Path, build_report_path: Path) -> dict[str, Any]:
    report = inspect_anatomy_target(spec_path, target_path, build_report_path)
    if not report["passed"]:
        raise ForgeError("Genesis anatomy target failed: " + "; ".join(report["failures"][:12]))
    return report
