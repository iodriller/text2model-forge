"""Strict deterministic Instant Meshes D2b wrapper with immutable-source topology gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from text2model_forge.polygon_obj import PolygonObj


OPERATION_ID = "retopology.instant_meshes"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(parameters: dict[str, object], name: str, default: int, minimum: int, maximum: int) -> int:
    value = parameters.get(name, default)
    if isinstance(value, bool) or int(value) != float(value):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _boolean(parameters: dict[str, object], name: str, default: bool) -> bool:
    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(request: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    if request.get("operation_id") != OPERATION_ID:
        raise ValueError(f"unsupported Instant Meshes operation: {request.get('operation_id')}")
    input_paths = dict(request.get("input_paths") or {})
    if len(input_paths) != 1:
        raise ValueError("Instant Meshes requires exactly one OBJ input")
    source = Path(next(iter(input_paths.values()))).resolve()
    if source.suffix.lower() != ".obj" or not source.is_file():
        raise ValueError("Instant Meshes input must be an existing OBJ file")
    executable_value = os.environ.get("TEXT2MODEL_FORGE_INSTANT_MESHES_EXE")
    if not executable_value:
        raise ValueError("TEXT2MODEL_FORGE_INSTANT_MESHES_EXE is required")
    executable = Path(os.path.expandvars(executable_value)).resolve()
    if not executable.is_file():
        raise ValueError(f"Instant Meshes executable does not exist: {executable}")

    output_root = Path(str(request["output_directory"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_obj = output_root / "retopology.obj"
    analysis_path = output_root / "retopology_analysis.json"
    stdout_path = output_root / "instant_meshes_stdout.log"
    stderr_path = output_root / "instant_meshes_stderr.log"
    for path in (output_obj, analysis_path, stdout_path, stderr_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    parameters = dict(request.get("parameters") or {})
    field_faces = _integer(parameters, "field_faces", 12500, 100, 100000)
    max_output_faces = _integer(parameters, "maximum_output_faces", 75000, 100, 400000)
    smooth_iterations = _integer(parameters, "smooth_iterations", 2, 0, 10)
    intrinsic = _boolean(parameters, "intrinsic", False)
    align_boundaries = _boolean(parameters, "align_boundaries", False)
    crease_value = parameters.get("crease_degrees")
    crease_degrees: float | None = None
    if crease_value is not None:
        crease_degrees = float(crease_value)
        if not 0.0 <= crease_degrees <= 180.0:
            raise ValueError("crease_degrees must be between zero and 180")

    source_hash_before = _sha256(source)
    command = [
        str(executable),
        "--deterministic",
        "--threads", "1",
        "--rosy", "4",
        "--posy", "4",
        "--faces", str(field_faces),
        "--smooth", str(smooth_iterations),
        "--output", str(output_obj),
    ]
    if crease_degrees is not None:
        command.extend(["--crease", str(crease_degrees)])
    if intrinsic:
        command.append("--intrinsic")
    if align_boundaries:
        command.append("--boundaries")
    command.append(str(source))
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    completed = subprocess.run(
        command,
        cwd=executable.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
        creationflags=creation_flags,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Instant Meshes exited with code {completed.returncode}")
    if not output_obj.is_file() or output_obj.stat().st_size == 0:
        raise RuntimeError("Instant Meshes exited successfully without a non-empty OBJ output")
    source_hash_after = _sha256(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Instant Meshes modified its source OBJ")

    topology = PolygonObj.parse(output_obj.read_text(encoding="utf-8-sig")).analyze()
    structural_gate_passed = (
        bool(topology["finite_coordinates"])
        and int(topology["connected_components"]) == 1
        and int(topology["boundary_edges"]) == 0
        and int(topology["non_manifold_edges"]) == 0
        and int(topology["degenerate_faces"]) == 0
        and int(topology["isolated_vertices"]) == 0
    )
    all_quad_gate_passed = int(topology["non_quad_faces"]) == 0
    face_budget_gate_passed = int(topology["faces"]) <= max_output_faces
    promotion_gate = structural_gate_passed and all_quad_gate_passed and face_budget_gate_passed
    report = {
        "schema_version": 1,
        "worker": "Instant Meshes master 7b3160864a2e1025af498c84cfed91cbfb613698",
        "source_path": str(source),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_overwritten": False,
        "parameters": {
            "deterministic": True,
            "threads": 1,
            "rosy": 4,
            "posy": 4,
            "field_faces": field_faces,
            "maximum_output_faces": max_output_faces,
            "smooth_iterations": smooth_iterations,
            "crease_degrees": crease_degrees,
            "intrinsic": intrinsic,
            "align_boundaries": align_boundaries,
        },
        "topology": topology,
        "structural_gate_passed": structural_gate_passed,
        "all_quad_gate_passed": all_quad_gate_passed,
        "face_budget_gate_passed": face_budget_gate_passed,
        "automatic_promotion_gate_passed": promotion_gate,
        "human_approval_required": True,
        "approved": False,
    }
    _write_json(analysis_path, report)
    outputs = [
        {"path": str(output_obj), "media_type": "model/obj", "role": "retopology_candidate", "metadata": {}},
        {"path": str(analysis_path), "media_type": "application/json", "role": "retopology_analysis", "metadata": {}},
        {"path": str(stdout_path), "media_type": "text/plain", "role": "worker_stdout", "metadata": {}},
        {"path": str(stderr_path), "media_type": "text/plain", "role": "worker_stderr", "metadata": {}},
    ]
    diagnostics = {
        "vertices": int(topology["vertices"]),
        "faces": int(topology["faces"]),
        "quads": int(topology["quads"]),
        "non_quad_faces": int(topology["non_quad_faces"]),
        "connected_components": int(topology["connected_components"]),
        "boundary_edges": int(topology["boundary_edges"]),
        "structural_gate_passed": structural_gate_passed,
        "all_quad_gate_passed": all_quad_gate_passed,
        "face_budget_gate_passed": face_budget_gate_passed,
        "automatic_promotion_gate_passed": promotion_gate,
    }
    warnings = ["Retopology is a D4 topology target, not an approved visual master, rig, skin, or animation body."]
    if not structural_gate_passed:
        warnings.append("The candidate failed structural topology gates and must not be promoted.")
    if not all_quad_gate_passed:
        warnings.append("The candidate failed the all-quad gate.")
    if not face_budget_gate_passed:
        warnings.append("The candidate exceeded the configured face budget.")
    return outputs, diagnostics, warnings


def main() -> int:
    args = _arguments()
    started = time.monotonic()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8-sig"))
        outputs, diagnostics, warnings = execute(request)
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 3)
        response = {
            "schema_version": 1,
            "job_id": request["job_id"],
            "status": "succeeded",
            "outputs": outputs,
            "diagnostics": diagnostics,
            "warnings": warnings,
            "error": None,
        }
    except Exception as exc:
        request = locals().get("request", {})
        response = {
            "schema_version": 1,
            "job_id": request.get("job_id", "unknown"),
            "status": "failed",
            "outputs": [],
            "diagnostics": {"elapsed_seconds": round(time.monotonic() - started, 3)},
            "warnings": [],
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if response["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
