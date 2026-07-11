"""Texture-master bake driver: paint canonical views once, bake them into the mesh.

Sequence (one command, resumable by signature):
  1. Blender renders N orbit views of the character master in rest pose, with true depth.
  2. All views ride ONE SDXL img2img call as a grid alongside the unit's persistent style
     anchor, constrained by the true-depth ControlNet.  One diffusion call means one
     style decision for the entire character, forever.
  3. Blender projects the painted views back onto the meshes (facing- and
     visibility-weighted) and bakes a UV atlas per mesh, saving <master>_baked.blend.

After this, production renders use the baked master directly: no per-frame diffusion,
no drift, no equipment protection needed, and a full 4-direction rebuild costs minutes.
The bake is skipped when an existing baked master matches the recorded run signature.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from .comfy import ComfyClient
from .core import ForgeError, sha256_file, sha256_json, utc_now, write_json
from .overpaint import (
    BACKGROUND,
    DEPTH_CONTROLNET,
    NEGATIVE,
    STYLE_SUFFIX,
    _flatten,
    _registered_model_hash,
    build_overpaint_workflow,
)

BAKE_SCRIPT = Path(__file__).resolve().parent.parent / "blender" / "bake_texture_master.py"


def _texture_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("texture_master")
    if not isinstance(block, dict) or not block.get("enabled"):
        raise ForgeError("Character config has no enabled 'texture_master' block")
    return block


def _style_block(config: dict[str, Any]) -> dict[str, Any]:
    """Prompt/checkpoint/seed come from the overpaint block even when its per-frame
    stage is disabled; one style declaration serves both mechanisms."""
    block = config.get("overpaint")
    if not isinstance(block, dict):
        raise ForgeError("Character config needs an 'overpaint' block for style settings")
    return block


def grow_edges(image: Image.Image, steps: int = 8) -> Image.Image:
    """Push silhouette colors outward into transparent texels so bilinear taps just
    outside the alpha edge blend character color instead of grid background."""
    result = image
    original_alpha = image.getchannel("A")
    solid = original_alpha.point(lambda a: 255 if a > 0 else 0)
    for _ in range(steps):
        grown = result.filter(ImageFilter.MaxFilter(3))
        result = Image.composite(result, grown, solid)
        result.putalpha(original_alpha)
    return result


def _run_blender(blender: str, blend_path: Path, mode: str, config_path: Path, work: Path, repo_root: Path, output: Path | None = None) -> None:
    command = [
        blender, "--background", str(blend_path), "--python-exit-code", "1",
        "--python", str(BAKE_SCRIPT), "--",
        "--mode", mode,
        "--config", str(config_path),
        "--work", str(work),
        "--repo-root", str(repo_root),
    ]
    if output is not None:
        command.extend(["--output", str(output)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = (result.stdout or "")[-3000:] + "\n" + (result.stderr or "")[-2000:]
        raise ForgeError(f"Blender bake step '{mode}' failed:\n{tail}")


def bake_texture_master(
    config_path: Path,
    repo_root: Path,
    blender: str,
    base_url: str = "http://127.0.0.1:8188",
    timeout_seconds: float = 600.0,
    force: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    repo_root = repo_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    block = _texture_block(config)
    style = _style_block(config)
    unit = str(config["id"])

    master = repo_root / str(config["source"])
    if not master.is_file():
        raise ForgeError(f"Character master was not found (build it first): {master}")
    baked_value = str(block.get("baked_source") or "").strip()
    if not baked_value:
        raise ForgeError("texture_master.baked_source must name the baked .blend path")
    baked = repo_root / baked_value

    work = repo_root / "artifacts" / "asset-forge-work" / unit / "bake"
    views_folder = work / "views"
    provenance_path = work / "bake-provenance.json"

    checkpoint = str(style.get("checkpoint", "dreamshaper_xl_v2_turbo.safetensors"))
    seed = int(style.get("seed", 77))
    steps = int(style.get("steps", 8))
    cfg = float(style.get("cfg", 3.0))
    denoise = float(block.get("denoise", 0.5))
    control_strength = float(block.get("control_strength", style.get("control_strength", 0.65)))
    view_count = int(block.get("views", 8))
    positive = str(style.get("prompt", "")).strip()
    positive = f"{positive}. {STYLE_SUFFIX}" if positive else STYLE_SUFFIX
    negative = str(style.get("negative", "")).strip()
    negative = f"{NEGATIVE}, {negative}" if negative else NEGATIVE

    anchors_folder = config_path.parent / "anchors"
    anchor_path = anchors_folder / f"{unit}.anchor.png"
    anchor_depth_path = anchors_folder / f"{unit}.anchor.depth.png"

    signature = sha256_json({
        "unit": unit,
        "master_sha256": sha256_file(master),
        "bake_script_sha256": sha256_file(BAKE_SCRIPT),
        "views": view_count,
        "extra_views": block.get("extra_views"),
        "view_size": int(block.get("view_size", 768)),
        "view_scale": float(block.get("view_scale", 4.2)),
        "view_elevation": float(block.get("view_elevation", 18.0)),
        "bake_resolution": int(block.get("bake_resolution", 2048)),
        "checkpoint": checkpoint,
        "checkpoint_sha256": _registered_model_hash(config_path, checkpoint),
        "controlnet": DEPTH_CONTROLNET,
        "controlnet_sha256": _registered_model_hash(config_path, DEPTH_CONTROLNET),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "control_strength": control_strength,
        "positive": positive,
        "negative": negative,
        "anchor_sha256": sha256_file(anchor_path) if anchor_path.is_file() else None,
    })

    if not force and baked.is_file() and provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") == signature and previous.get("baked_sha256") == sha256_file(baked):
            previous["cached"] = True
            return previous

    work.mkdir(parents=True, exist_ok=True)
    _run_blender(blender, master, "render-views", config_path, work, repo_root)

    metadata = json.loads((work / "views.json").read_text(encoding="utf-8"))
    resolution = int(metadata["resolution"])
    views: list[Image.Image] = []
    depths: list[Image.Image] = []
    for record in metadata["views"]:
        index = int(record["index"])
        with Image.open(views_folder / f"view_{index:02d}.png") as source:
            views.append(source.convert("RGBA"))
        with Image.open(views_folder / f"depth_{index:02d}.png") as source:
            depths.append(source.convert("RGB"))

    client = ComfyClient(base_url)
    available = client.checkpoints()
    if checkpoint not in available:
        raise ForgeError(f"Bake checkpoint {checkpoint!r} is not installed. Available: {available}")

    cell = int(block.get("grid_cell", 512))
    columns = int(block.get("grid_columns", 3))

    def upload(image: Image.Image, name: str) -> str:
        stream = io.BytesIO()
        image.save(stream, "PNG")
        return client.upload_image(name, stream.getvalue())

    def paint(beauty: Image.Image, control: Image.Image | None, tag: str) -> Image.Image:
        uploaded = upload(beauty, f"{unit}_bake_{tag}.png")
        control_name = upload(control, f"{unit}_bake_{tag}_ctl.png") if control is not None else None
        workflow = build_overpaint_workflow(
            checkpoint, positive, negative, uploaded, seed, steps, cfg, denoise,
            f"AssetForge/bake/{unit}/{tag}",
            control_image=control_name, control_strength=control_strength,
        )
        prompt_id = client.submit(workflow)
        history = client.wait(prompt_id, timeout_seconds=timeout_seconds)
        scratch = work / "comfy-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        outputs = client.download_outputs(history, scratch)
        with Image.open(outputs[0]) as painted_source:
            painted = painted_source.convert("RGB")
        for item in scratch.iterdir():
            item.unlink()
        if painted.size != beauty.size:
            painted = painted.resize(beauty.size, Image.Resampling.LANCZOS)
        return painted

    # The persistent anchor keeps the bake in the same visual family as everything the
    # unit has ever shown the owner.  Create it from the front view when absent.
    anchor_created = False
    if anchor_path.is_file():
        anchor_image = Image.open(anchor_path).convert("RGB")
        anchor_depth = (
            Image.open(anchor_depth_path).convert("RGB")
            if anchor_depth_path.is_file()
            else Image.new("RGB", anchor_image.size, (0, 0, 0))
        )
    else:
        front = _flatten(views[0]).resize((cell, cell), Image.Resampling.LANCZOS)
        front_depth = depths[0].resize((cell, cell), Image.Resampling.LANCZOS)
        painted_front = paint(front, front_depth, "anchor")
        anchors_folder.mkdir(parents=True, exist_ok=True)
        painted_front.save(anchor_path, "PNG")
        front_depth.save(anchor_depth_path, "PNG")
        anchor_image, anchor_depth = painted_front, front_depth
        anchor_created = True

    slots = 1 + len(views)
    rows = math.ceil(slots / columns)

    def compose(cells: list[Image.Image], background: tuple[int, int, int]) -> Image.Image:
        canvas = Image.new("RGB", (columns * cell, rows * cell), background)
        for index, item in enumerate(cells):
            canvas.paste(item.resize((cell, cell), Image.Resampling.LANCZOS), ((index % columns) * cell, (index // columns) * cell))
        return canvas

    beauty = compose([anchor_image] + [_flatten(view) for view in views], BACKGROUND)
    control = compose([anchor_depth] + depths, (0, 0, 0))
    painted_grid = paint(beauty, control, "views")

    painted_hashes: dict[str, str] = {}
    for index, view in enumerate(views):
        slot = index + 1
        x, y = (slot % columns) * cell, (slot // columns) * cell
        piece = painted_grid.crop((x, y, x + cell, y + cell)).resize((resolution, resolution), Image.Resampling.LANCZOS)
        painted = piece.convert("RGBA")
        painted.putalpha(view.getchannel("A"))
        painted = grow_edges(painted)
        painted_path = views_folder / f"painted_{index:02d}.png"
        painted.save(painted_path, "PNG")
        painted_hashes[painted_path.name] = sha256_file(painted_path)

    _run_blender(blender, master, "project-bake", config_path, work, repo_root, output=baked)
    if not baked.is_file():
        raise ForgeError(f"Projection bake reported success but no baked master exists: {baked}")

    icons_folder = repo_root / "asset_sources" / "ember-defense" / "assets" / unit / "icons"
    _run_blender(blender, baked, "icons", config_path, work, repo_root, output=icons_folder)

    manifest = json.loads((work / "bake-manifest.json").read_text(encoding="utf-8"))
    provenance = {
        "schema_version": 1,
        "stage": "texture-master-bake",
        "unit": unit,
        "run_signature": signature,
        "master": str(master),
        "master_sha256": sha256_file(master),
        "baked_master": str(baked),
        "baked_sha256": sha256_file(baked),
        "views": view_count,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _registered_model_hash(config_path, checkpoint),
        "controlnet": DEPTH_CONTROLNET,
        "controlnet_sha256": _registered_model_hash(config_path, DEPTH_CONTROLNET),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "control_strength": control_strength,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "anchor": str(anchor_path),
        "anchor_sha256": sha256_file(anchor_path),
        "anchor_created": anchor_created,
        "painted_views_sha256": painted_hashes,
        "atlases": manifest.get("objects", []),
        "icons": str(icons_folder),
        "completed_utc": utc_now(),
    }
    write_json(provenance_path, provenance)
    return provenance
