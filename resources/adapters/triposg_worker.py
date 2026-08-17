"""Text2Model external-worker wrapper for MIT TripoSG without BRIA RMBG or DiSo."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


SOURCE_REVISION = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
WEIGHTS_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"


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


def _run(request: dict) -> tuple[Path, dict]:
    import numpy as np
    import torch
    import trimesh

    source = Path(os.environ["TEXT2MODEL_FORGE_TRIPOSG_SOURCE"]).resolve()
    weights = Path(os.environ["TEXT2MODEL_FORGE_TRIPOSG_WEIGHTS"]).resolve()
    compat = Path(os.environ["TEXT2MODEL_FORGE_TRIPOSG_COMPAT"]).resolve()
    if not source.is_dir() or not weights.is_dir() or not compat.is_dir():
        raise FileNotFoundError("TripoSG source, weights, or compatibility directory is missing")
    sys.path[:0] = [str(compat), str(source), str(source / "scripts")]

    from image_process import prepare_image
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    inputs = request.get("inputs", [])
    input_paths = request.get("input_paths", {})
    if len(inputs) != 1:
        raise ValueError("TripoSG requires exactly one input artifact")
    input_id = inputs[0]["artifact_id"]
    image_path = Path(input_paths[input_id]).resolve()
    if not _valid_alpha(image_path):
        raise ValueError(
            "TripoSG input must be an RGBA PNG with at least 1% transparent and 1% opaque pixels; "
            "Text2Model will not fall back to non-commercial BRIA RMBG."
        )

    parameters = request.get("parameters", {})
    seed = int(parameters.get("seed", 42))
    steps = int(parameters.get("num_inference_steps", 30))
    guidance = float(parameters.get("guidance_scale", 7.0))
    dense_depth = int(parameters.get("dense_octree_depth", 7))
    hierarchical_depth = int(parameters.get("hierarchical_octree_depth", 8))
    if not 1 <= steps <= 100:
        raise ValueError("num_inference_steps must be between 1 and 100")
    if not 6 <= dense_depth <= 8 or not dense_depth <= hierarchical_depth <= 9:
        raise ValueError("octree depths are outside the qualified safety range")

    torch.cuda.reset_peak_memory_stats()
    prepared = prepare_image(str(image_path), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)
    pipeline = TripoSGPipeline.from_pretrained(str(weights), local_files_only=True).to("cuda", torch.float16)
    output = pipeline(
        image=prepared,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        num_inference_steps=steps,
        guidance_scale=guidance,
        use_flash_decoder=False,
        dense_octree_depth=dense_depth,
        hierarchical_octree_depth=hierarchical_depth,
    ).samples[0]
    if output[0] is None or output[1] is None:
        raise RuntimeError("TripoSG returned no mesh")
    mesh = trimesh.Trimesh(output[0].astype(np.float32), np.ascontiguousarray(output[1]))
    output_root = Path(request["output_directory"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "triposg_candidate.glb"
    mesh.export(output_path)
    diagnostics = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "seed": seed,
        "source_revision": SOURCE_REVISION,
        "weights_revision": WEIGHTS_REVISION,
        "background_removal_model": "none_rgba_required",
        "extractor": "skimage_cpu_marching_cubes",
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
                        "candidate_id": "triposg-1.5b",
                        "source_revision": SOURCE_REVISION,
                        "weights_revision": WEIGHTS_REVISION,
                    },
                }
            ],
            "diagnostics": diagnostics,
            "warnings": [
                "Generated bodies remain topology targets until deformation-ready fitting passes."
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
