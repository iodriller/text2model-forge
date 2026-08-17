"""Install official native ComfyUI Qwen Image model sets.

This installer downloads only the files required for native ComfyUI generation/editing and
hard-links them from the Hugging Face cache when possible, avoiding duplicate
30 GB copies on disk.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download


IMAGE_EDIT_2511_FILES = (
    (
        "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        "diffusion_models",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "text_encoders",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
        "vae",
    ),
)

IMAGE_2512_FILES = (
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors",
        "diffusion_models",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "text_encoders",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
        "vae",
    ),
)

Z_IMAGE_TURBO_FILES = (
    (
        "Comfy-Org/z_image_turbo",
        "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
        "diffusion_models",
    ),
    (
        "Comfy-Org/z_image_turbo",
        "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
        "text_encoders",
    ),
    (
        "Comfy-Org/z_image_turbo",
        "split_files/vae/ae.safetensors",
        "vae",
    ),
)

STYLIZED_SDXL_FILES = (
    # destination filename pinned to StudioRun.checkpoint's default
    # ("dreamshaper_xl_v2_turbo.safetensors") -- the upstream file is
    # differently cased/spelled and ComfyUI's checkpoint dropdown, D0/D1
    # config, and check_comfy_checkpoints all key off the literal filename.
    (
        "Lykon/dreamshaper-xl-v2-turbo",
        "DreamShaperXL_Turbo_v2.safetensors",
        "checkpoints",
        "dreamshaper_xl_v2_turbo.safetensors",
    ),
)

MODEL_SETS = {
    "image-edit-2511": IMAGE_EDIT_2511_FILES,
    "image-2512": IMAGE_2512_FILES,
    "z-image-turbo": Z_IMAGE_TURBO_FILES,
    "stylized-sdxl": STYLIZED_SDXL_FILES,
}


def install(models_root: Path, profile: str) -> list[dict[str, object]]:
    if profile not in MODEL_SETS:
        raise ValueError(f"unknown Qwen Image profile: {profile}")
    results: list[dict[str, object]] = []
    for entry in MODEL_SETS[profile]:
        repo_id, filename, destination_folder = entry[0], entry[1], entry[2]
        destination_name = entry[3] if len(entry) > 3 else None
        cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
        destination = models_root / destination_folder / (destination_name or cached.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(cached, destination)
                materialization = "hardlink"
            except OSError:
                shutil.copy2(cached, destination)
                materialization = "copy"
        else:
            materialization = "existing"
        results.append(
            {
                "repo_id": repo_id,
                "source": filename,
                "destination": str(destination),
                "size_bytes": destination.stat().st_size,
                "materialization": materialization,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(MODEL_SETS), default="image-edit-2511")
    args = parser.parse_args()
    print(json.dumps({"profile": args.profile, "installed": install(args.models_root, args.profile)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
