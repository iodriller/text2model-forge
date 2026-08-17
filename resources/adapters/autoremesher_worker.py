"""Strict AutoRemesher 1.0 D2b wrapper with immutable-source and topology evidence gates."""
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


OPERATION_ID = "retopology.autoremesher"


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


def _number(parameters: dict[str, object], name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(parameters.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(request: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    if request.get("operation_id") != OPERATION_ID:
        raise ValueError(f"unsupported AutoRemesher operation: {request.get('operation_id')}")
    input_paths = dict(request.get("input_paths") or {})
    if len(input_paths) != 1:
        raise ValueError("AutoRemesher requires exactly one OBJ input")
    source = Path(next(iter(input_paths.values()))).resolve()
    if source.suffix.lower() != ".obj" or not source.is_file():
        raise ValueError("AutoRemesher input must be an existing OBJ file")
    executable_value = os.environ.get("TEXT2MODEL_FORGE_AUTOREMESHER_EXE")
    if not executable_value:
        raise ValueError("TEXT2MODEL_FORGE_AUTOREMESHER_EXE is required")
    executable = Path(os.path.expandvars(executable_value)).resolve()
    if not executable.is_file():
        raise ValueError(f"AutoRemesher executable does not exist: {executable}")

    output_root = Path(str(request["output_directory"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_obj = output_root / "retopology.obj"
    report_path = output_root / "autoremesher_report.txt"
    analysis_path = output_root / "retopology_analysis.json"
    stdout_path = output_root / "autoremesher_stdout.log"
    stderr_path = output_root / "autoremesher_stderr.log"
    for path in (output_obj, report_path, analysis_path, stdout_path, stderr_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    parameters = dict(request.get("parameters") or {})
    target_quads = int(_number(parameters, "target_quads", 20000, 100, 250000))
    edge_scaling = _number(parameters, "edge_scaling", 1.0, 0.1, 10.0)
    sharp_edge = _number(parameters, "sharp_edge_degrees", 60.0, 0.0, 180.0)
    smooth_normal = _number(parameters, "smooth_normal_degrees", 0.0, 0.0, 180.0)
    adaptivity = _number(parameters, "adaptivity", 0.5, 0.0, 1.0)
    source_hash_before = _sha256(source)
    command = [
        str(executable),
        "--input", str(source),
        "--output", str(output_obj),
        "--report", str(report_path),
        "--target-quads", str(target_quads),
        "--edge-scaling", str(edge_scaling),
        "--sharp-edge", str(sharp_edge),
        "--smooth-normal", str(smooth_normal),
        "--adaptivity", str(adaptivity),
    ]
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
        raise RuntimeError(f"AutoRemesher exited with code {completed.returncode}")
    if not output_obj.is_file() or output_obj.stat().st_size == 0:
        raise RuntimeError("AutoRemesher exited successfully without a non-empty OBJ output")
    if not report_path.is_file():
        raise RuntimeError("AutoRemesher exited successfully without its report")
    source_hash_after = _sha256(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("AutoRemesher modified its source OBJ")

    mesh = PolygonObj.parse(output_obj.read_text(encoding="utf-8-sig"))
    topology = mesh.analyze()
    structural_gate_passed = (
        bool(topology["finite_coordinates"])
        and int(topology["connected_components"]) == 1
        and int(topology["boundary_edges"]) == 0
        and int(topology["non_manifold_edges"]) == 0
        and int(topology["degenerate_faces"]) == 0
        and int(topology["isolated_vertices"]) == 0
    )
    all_quad_gate_passed = int(topology["non_quad_faces"]) == 0
    report = {
        "schema_version": 1,
        "worker": "AutoRemesher 1.0.0",
        "source_path": str(source),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_overwritten": False,
        "parameters": {
            "target_quads": target_quads,
            "edge_scaling": edge_scaling,
            "sharp_edge_degrees": sharp_edge,
            "smooth_normal_degrees": smooth_normal,
            "adaptivity": adaptivity,
        },
        "topology": topology,
        "structural_gate_passed": structural_gate_passed,
        "all_quad_gate_passed": all_quad_gate_passed,
        "automatic_promotion_gate_passed": structural_gate_passed and all_quad_gate_passed,
        "human_approval_required": True,
        "approved": False,
    }
    _write_json(analysis_path, report)
    outputs = [
        {"path": str(output_obj), "media_type": "model/obj", "role": "retopology_candidate", "metadata": {}},
        {"path": str(report_path), "media_type": "text/plain", "role": "autoremesher_report", "metadata": {}},
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
        "automatic_promotion_gate_passed": structural_gate_passed and all_quad_gate_passed,
    }
    warnings = ["Retopology is a D4 topology target, not an approved visual master, rig, skin, or animation body."]
    if not structural_gate_passed:
        warnings.append("The candidate failed structural topology gates and must not be promoted.")
    if not all_quad_gate_passed:
        warnings.append("The candidate contains non-quad faces and failed the plan's all-quad gate.")
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
