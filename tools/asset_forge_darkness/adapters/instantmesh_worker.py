"""Darkness external-worker wrapper for Apache-2.0 InstantMesh.

Runs on the Linux side of the WSL2 boundary (see adapters/wsl_launch.py).
Request/response file contract matches triposg_worker.py and
trellis2_worker.py so SubprocessWorkerAdapter does not need to know which
side of the WSL2 boundary a worker runs on, nor which generator it wraps.

Added 2026-07-12 as a fully hands-off comparison candidate: unlike
TRELLIS.2 (blocked pending Meta's manual review of the gated
facebook/dinov3-vitl16-pretrain-lvd1689m repo), InstantMesh's two
checkpoints (sudo-ai/zero123plus-v1.2, TencentARC/InstantMesh) are both
publicly downloadable with no Hugging Face gate.

InstantMesh's own run.py defaults to rembg (u2net) for background removal.
Darkness always passes --no_rembg and a caller-owned pre-masked RGBA
composite instead, matching the policy already applied to TripoSG and
TRELLIS.2 -- this worker never downloads or runs rembg's model. run.py is
invoked as a subprocess (its own CLI, not an importable pipeline object) and
its OBJ output is converted to GLB via trimesh for a consistent response
contract across all three geometry workers.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SOURCE_REVISION_ENV = "DARKNESS_INSTANTMESH_SOURCE_REVISION"
CONFIG_NAME = "instant-mesh-large"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser


def _valid_alpha(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as image:
        if image.mode != "RGBA":
            return False
        alpha = image.getchannel("A")
        low, high = alpha.getextrema()
        if low == high:
            return False
        histogram = alpha.histogram()
        pixels = image.width * image.height
        return histogram[0] >= pixels * 0.01 and histogram[255] >= pixels * 0.01


def _composite_on_white(path: Path):
    from PIL import Image

    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background


def _run(request: dict) -> tuple[Path, dict]:
    import trimesh

    source = Path(os.environ["DARKNESS_INSTANTMESH_SOURCE"]).resolve()
    if not source.is_dir():
        raise FileNotFoundError("InstantMesh source checkout is missing")

    inputs = request.get("inputs", [])
    input_paths = request.get("input_paths", {})
    if len(inputs) != 1:
        raise ValueError("InstantMesh requires exactly one input artifact")
    input_id = inputs[0]["artifact_id"]
    image_path = Path(input_paths[input_id]).resolve()
    if not _valid_alpha(image_path):
        raise ValueError(
            "InstantMesh input must be an RGBA PNG with at least 1% transparent and 1% opaque pixels; "
            "Darkness composites its own background and passes --no_rembg rather than trusting rembg/u2net."
        )

    parameters = request.get("parameters", {})
    seed = int(parameters.get("seed", 42))
    diffusion_steps = int(parameters.get("diffusion_steps", 75))
    export_texmap = bool(parameters.get("export_texmap", False))
    if not 1 <= diffusion_steps <= 200:
        raise ValueError("diffusion_steps must be between 1 and 200")

    output_root = Path(request["output_directory"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    composite = _composite_on_white(image_path)
    prepared_name = "instantmesh_input"
    prepared_path = output_root / f"{prepared_name}.png"
    composite.save(prepared_path)

    run_output_dir = output_root / "run_py_output"
    command = [
        sys.executable,
        "run.py",
        f"configs/{CONFIG_NAME}.yaml",
        str(prepared_path),
        "--output_path", str(run_output_dir),
        "--seed", str(seed),
        "--diffusion_steps", str(diffusion_steps),
        "--no_rembg",
    ]
    if export_texmap:
        command.append("--export_texmap")

    completed = subprocess.run(command, cwd=str(source))
    if completed.returncode != 0:
        raise RuntimeError(f"InstantMesh run.py exited with code {completed.returncode}")

    obj_path = run_output_dir / CONFIG_NAME / "meshes" / f"{prepared_name}.obj"
    if not obj_path.is_file():
        raise RuntimeError(f"InstantMesh did not produce the expected mesh at {obj_path}")

    mesh = trimesh.load(obj_path, force="mesh")
    output_path = output_root / "instantmesh_candidate.glb"
    mesh.export(output_path)

    diagnostics = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "connected_components": int(len(mesh.split(only_watertight=False))),
        "seed": seed,
        "diffusion_steps": diffusion_steps,
        "source_revision": os.environ.get(SOURCE_REVISION_ENV, "unpinned"),
        "background_composite": "white_rgb_over_owned_rgba_no_rembg",
        "checkpoints": "sudo-ai/zero123plus-v1.2 + TencentARC/InstantMesh (both ungated)",
    }
    return output_path, diagnostics


def main() -> int:
    args = _parser().parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    response: dict
    try:
        output_path, diagnostics = _run(request)
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 3)
        response = {
            "schema_version": 1,
            "job_id": request["job_id"],
            "status": "succeeded",
            "outputs": [
                {
                    "path": str(output_path),
                    "media_type": "model/gltf-binary",
                    "role": "geometry_candidate",
                    "metadata": {
                        "candidate_id": "instantmesh",
                        "source_revision": diagnostics["source_revision"],
                    },
                }
            ],
            "diagnostics": diagnostics,
            "warnings": [
                "Generated bodies remain topology targets until the D2b retopology sub-stage "
                "and deformation-ready fitting pass.",
                "InstantMesh is an April 2024 architecture; treat as a fast hands-off baseline, "
                "not a substitute for TRELLIS.2 once its gated dependency clears review.",
            ],
        }
    except Exception as exc:
        response = {
            "schema_version": 1,
            "job_id": request.get("job_id", "unknown"),
            "status": "failed",
            "outputs": [],
            "diagnostics": {"elapsed_seconds": round(time.monotonic() - started, 3)},
            "warnings": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
