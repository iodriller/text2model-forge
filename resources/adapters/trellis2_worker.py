"""Text2Model external-worker wrapper for MIT TRELLIS.2-4B.

Runs on the Linux side of the WSL2 boundary (see resources/adapters/wsl_launch.py,
which bridges the Windows-side --request/--response paths into this
process). The request/response file contract exactly matches
triposg_worker.py so SubprocessWorkerAdapter does not need to know which
side of the WSL2 boundary a given worker runs on.

Grounded against the official TRELLIS.2 README example
(https://github.com/microsoft/TRELLIS.2) as of 2026-07-12: pipeline import
path, `Trellis2ImageTo3DPipeline.from_pretrained(...).cuda()`, `pipeline.run(image)`,
and the `o_voxel.postprocess.to_glb(...)` export call are all taken verbatim
from that example. The README does not document a per-call seed/generator
parameter, so this worker approximates reproducibility with a global torch
seed pending confirmation of a real per-call parameter.

Background removal (found by reading trellis2/pipelines/trellis2_image_to_3d.py
directly after a live smoke test failed): `preprocess_image()` uses the input
image's own alpha channel unchanged whenever it already carries real
transparency, and only invokes the pipeline's internal `rembg_model` otherwise.
That internal model is a `BiRefNet` instance whose TRELLIS.2-4B pipeline
config points at the gated, non-commercial `briaai/RMBG-2.0` -- the exact
model Text2Model's RGBA-in policy (already applied to TripoSG and InstantMesh)
exists to avoid. Worse, `from_pretrained()` unconditionally *constructs* that
`BiRefNet` object regardless of whether `run()` will ever call it, which
downloads RMBG-2.0's config eagerly at load time. This worker therefore does
two things: (1) passes the caller-owned RGBA image through with its alpha
channel intact (not composited to opaque RGB) so `preprocess_image()` takes
the has-alpha path, and (2) monkeypatches `trellis2.pipelines.rembg.BiRefNet`
to a stub that never touches the network, so RMBG-2.0 is never downloaded or
constructed even during pipeline load, and fails loudly if ever actually
invoked (which would mean the alpha channel was insufficient for a given
input).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


SOURCE_REVISION_ENV = "TEXT2MODEL_FORGE_TRELLIS2_SOURCE_REVISION"


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


class _NoNetworkRembg:
    """Stand-in for trellis2.pipelines.rembg.BiRefNet that never touches the network.

    preprocess_image() only calls self.rembg_model when the input lacks real
    alpha; this worker always supplies real alpha, so __call__ should never
    actually run. It raises instead of silently falling back to a
    non-commercial gated model if that assumption is ever wrong.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def to(self, device) -> None:
        pass

    def cuda(self) -> None:
        pass

    def cpu(self) -> None:
        pass

    def __call__(self, image):
        raise RuntimeError(
            "TRELLIS.2 tried to run its internal background-removal model (briaai/RMBG-2.0, "
            "gated and non-commercial). Text2Model never invokes it; this means the supplied "
            "RGBA image did not carry the meaningful alpha channel preprocess_image() expects."
        )


def _dinov3_extract_features_compat(self, image):
    """Compatibility shim for trellis2.modules.image_feature_extractor.DinoV3FeatureExtractor.

    TRELLIS.2's own code (found live 2026-07-12, source_revision
    75fbf018...) reads `self.model.layer` directly, assuming the
    transformers version at the time it was written kept DINOv3ViTModel's
    transformer blocks at the top level. transformers==5.13.1 -- what
    setup.sh's unpinned `pip install transformers` actually resolves to --
    moved them one level deeper, under `model.model.layer`
    (DINOv3ViTModel.model is now a DINOv3ViTEncoder). This is TRELLIS.2's own
    unpinned-dependency drift, not a Text2Model policy decision; the fix
    below is the smallest change that makes both layouts work.
    """
    import torch.nn.functional as F

    image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
    hidden_states = self.model.embeddings(image, bool_masked_pos=None)
    position_embeddings = self.model.rope_embeddings(image)
    encoder = self.model.model if hasattr(self.model, "model") and hasattr(self.model.model, "layer") else self.model
    for layer_module in encoder.layer:
        hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
    return F.layer_norm(hidden_states, hidden_states.shape[-1:])


def _run(request: dict) -> tuple[Path, dict]:
    import torch

    source = Path(os.environ["TEXT2MODEL_FORGE_TRELLIS2_SOURCE"]).resolve()
    if not source.is_dir():
        raise FileNotFoundError("TRELLIS.2 source checkout is missing")
    # trellis2 is a repo-relative package, not pip-installed: its own README
    # examples only work run from the repo root. cwd is changed (not just
    # sys.path) because verified live behavior showed the package resolving
    # its sparse-conv/attention backends relative to the working directory.
    os.chdir(source)
    sys.path.insert(0, str(source))

    import o_voxel
    import trellis2.modules.image_feature_extractor as ife_module
    import trellis2.pipelines.rembg as rembg_module
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Both must happen before from_pretrained(): it unconditionally
    # constructs rembg_model and image_cond_model. See the module docstring
    # and _dinov3_extract_features_compat's docstring for why each is needed.
    rembg_module.BiRefNet = _NoNetworkRembg
    ife_module.DinoV3FeatureExtractor.extract_features = _dinov3_extract_features_compat

    weights = os.environ.get("TEXT2MODEL_FORGE_TRELLIS2_WEIGHTS", "microsoft/TRELLIS.2-4B")

    inputs = request.get("inputs", [])
    input_paths = request.get("input_paths", {})
    if len(inputs) != 1:
        raise ValueError("TRELLIS.2 requires exactly one input artifact")
    input_id = inputs[0]["artifact_id"]
    image_path = Path(input_paths[input_id]).resolve()
    if not _valid_alpha(image_path):
        raise ValueError(
            "TRELLIS.2 input must be an RGBA PNG with at least 1% transparent and 1% opaque pixels; "
            "the alpha channel is passed through as-is so the pipeline's has-alpha path is used and "
            "its internal (gated, non-commercial) background-removal model is never invoked."
        )
    with Image.open(image_path) as source:
        image = source.convert("RGBA")

    parameters = request.get("parameters", {})
    seed = int(parameters.get("seed", 42))
    decimation_target = int(parameters.get("decimation_target", 1_000_000))
    texture_size = int(parameters.get("texture_size", 4096))
    remesh = bool(parameters.get("remesh", True))
    if decimation_target <= 0:
        raise ValueError("decimation_target must be positive")
    if texture_size <= 0:
        raise ValueError("texture_size must be positive")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(weights)
    pipeline.cuda()
    mesh = pipeline.run(image)[0]

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
    )

    output_root = Path(request["output_directory"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "trellis2_candidate.glb"
    # extension_webp=True (the README's default) failed live 2026-07-12 with
    # "module 'PIL._webp' has no attribute 'HAVE_WEBPANIM'" -- a libwebp/
    # Pillow build mismatch in this environment. PNG textures are the safe,
    # universally-supported fallback and cost only file size, not quality.
    glb.export(str(output_path), extension_webp=False)

    diagnostics = {
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "seed": seed,
        "decimation_target": decimation_target,
        "texture_size": texture_size,
        "remesh": remesh,
        "source_revision": os.environ.get(SOURCE_REVISION_ENV, "unpinned"),
        "weights": weights,
        "background_removal": "owned_rgba_alpha_passthrough_no_rmbg",
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
                        "candidate_id": "trellis2-4b",
                        "source_revision": diagnostics["source_revision"],
                    },
                }
            ],
            "diagnostics": diagnostics,
            "warnings": [
                "Generated bodies remain topology targets until the D2b retopology sub-stage "
                "and deformation-ready fitting pass.",
                "pipeline.run() reproducibility surface is approximated with a global torch seed "
                "pending live confirmation of a per-call generator parameter.",
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
