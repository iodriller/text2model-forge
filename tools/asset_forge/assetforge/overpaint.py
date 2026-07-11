"""Structure-locked overpaint v3: depth-controlled, anchor-consistent, equipment-gated.

The Blender render supplies structure; SDXL repaints surfaces. Three mechanisms keep the
result consistent and complete:

1. Depth ControlNet — every frame's TRUE 3D depth (rendered by Blender) constrains
   generation, so thin equipment (sword, club) cannot be painted away and forms cannot
   drift off the geometry.
2. Persistent style anchor — one painted reference cell per unit, created once and stored
   next to the character config, is embedded into EVERY grid of EVERY run. Diffusion
   copies materials/emblems from it, making frame N, frame N+1, and next month's rebuild
   agree.
3. Equipment integrity gate — Blender also renders an equipment-only mask per frame; after
   painting, QA verifies the equipment region still reads distinctly (color/contrast vs
   its surroundings). Failing chunks retry once at lower denoise, and remaining failures
   are reported in provenance.

The ORIGINAL render alpha is re-applied to every painted frame, so silhouettes and
baselines never drift. Downstream packing/QA/Unity are unchanged.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from .comfy import ComfyClient
from .core import ForgeError, sha256_json, utc_now, write_json

WORK_SIZE = 768
BACKGROUND = (40, 44, 52)
DEPTH_CONTROLNET = "controlnet_depth_sdxl_xinsir.safetensors"
STYLE_SUFFIX = (
    "dark high fantasy hand-painted game art, grim muted desaturated palette, battle-worn "
    "blackened metal with sharp cold highlights, gritty painted armor and cloth texture, "
    "deep shadows, painterly brushwork, readable stylized forms, dramatic moody lighting"
)
NEGATIVE = (
    "photo, photorealistic, plastic, toy, lego, minecraft, voxel, blocky, cartoon sticker, "
    "flat untextured surfaces, blurry, soft focus, changed pose, extra limbs, extra weapons, "
    "text, watermark, frame, border"
)
INTEGRITY_MIN_PIXELS = 30
INTEGRITY_MIN_DISTANCE = 16.0


def _config_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("overpaint")
    if not isinstance(block, dict) or not block.get("enabled"):
        raise ForgeError("Character config has no enabled 'overpaint' block")
    return block


def build_overpaint_workflow(
    checkpoint: str,
    positive: str,
    negative: str,
    image_name: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    prefix: str,
    control_image: str | None = None,
    controlnet: str = DEPTH_CONTROLNET,
    control_strength: float = 0.65,
) -> dict[str, Any]:
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": positive}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0],
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": "dpmpp_sde",
                "scheduler": "karras",
                "denoise": float(denoise),
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["7", 0]}},
    }
    if control_image is not None:
        workflow["9"] = {"class_type": "LoadImage", "inputs": {"image": control_image}}
        workflow["10"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet}}
        workflow["11"] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "control_net": ["10", 0],
                "image": ["9", 0],
                "strength": float(control_strength),
                "start_percent": 0.0,
                "end_percent": 0.9,
            },
        }
        workflow["6"]["inputs"]["positive"] = ["11", 0]
        workflow["6"]["inputs"]["negative"] = ["11", 1]
    return workflow


def _flatten(original: Image.Image) -> Image.Image:
    flat = Image.new("RGB", original.size, BACKGROUND)
    flat.paste(original, (0, 0), original)
    return flat


def _sibling(frame_path: Path, unit_frames: Path, suffix: str) -> Path:
    relative = frame_path.relative_to(unit_frames)
    return unit_frames.parent / (unit_frames.name + suffix) / relative


def equipment_integrity(painted: Image.Image, alpha: Image.Image, equip_mask_path: Path) -> tuple[bool, float]:
    """True when the painted equipment region still reads against its surroundings."""
    if not equip_mask_path.is_file():
        return True, -1.0
    with Image.open(equip_mask_path) as source:
        mask = source.convert("RGBA").getchannel("A").resize(painted.size, Image.Resampling.NEAREST)
    mask = mask.point(lambda a: 255 if a >= 100 else 0)
    coverage = ImageStat.Stat(mask).sum[0] / 255
    if coverage < INTEGRITY_MIN_PIXELS:
        return True, -1.0
    ring = mask.filter(ImageFilter.MaxFilter(25))
    ring = Image.composite(Image.new("L", mask.size, 255), Image.new("L", mask.size, 0), ring)
    ring.paste(0, (0, 0), mask)
    character = alpha.point(lambda a: 255 if a >= 100 else 0)
    ring.paste(0, (0, 0), Image.composite(Image.new("L", mask.size, 0), Image.new("L", mask.size, 255), character))
    if ImageStat.Stat(ring).sum[0] / 255 < INTEGRITY_MIN_PIXELS:
        return True, -1.0
    inside = ImageStat.Stat(painted.convert("RGB"), mask).mean
    around = ImageStat.Stat(painted.convert("RGB"), ring).mean
    distance = sum(abs(a - b) for a, b in zip(inside, around)) / 3.0
    return distance >= INTEGRITY_MIN_DISTANCE, distance


def overpaint_frames(
    config_path: Path,
    frames_root: Path,
    limit: int | None = None,
    base_url: str = "http://127.0.0.1:8188",
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    block = _config_block(config)
    unit_id = config["id"]
    checkpoint = block.get("checkpoint", "dreamshaper_xl_v2_turbo.safetensors")
    seed = int(block.get("seed", 77))
    steps = int(block.get("steps", 8))
    cfg = float(block.get("cfg", 3.0))
    denoise = float(block.get("denoise", 0.42))
    control_strength = float(block.get("control_strength", 0.65))
    use_anchor = bool(block.get("anchor", True))
    positive = str(block.get("prompt", "")).strip()
    positive = f"{positive}. {STYLE_SUFFIX}" if positive else STYLE_SUFFIX
    negative = str(block.get("negative", "")).strip()
    negative = f"{NEGATIVE}, {negative}" if negative else NEGATIVE

    unit_frames = frames_root.resolve() / unit_id
    frames = sorted(unit_frames.glob("*/*/[0-9][0-9].png"))
    if not frames:
        raise ForgeError(f"No rendered frames found under {unit_frames}")
    if limit is not None:
        frames = frames[: int(limit)]

    depth_available = (frames_root.resolve() / f"{unit_id}-depth").is_dir()
    use_depth = bool(block.get("control", "depth") == "depth" and depth_available)

    client = ComfyClient(base_url)
    available = client.checkpoints()
    if checkpoint not in available:
        raise ForgeError(f"Overpaint checkpoint {checkpoint!r} is not installed. Available: {available}")

    def upload(image: Image.Image, name: str) -> str:
        stream = io.BytesIO()
        image.save(stream, "PNG")
        return client.upload_image(name, stream.getvalue())

    def run_workflow(image: Image.Image, tag: str, run_denoise: float, control: Image.Image | None) -> Image.Image:
        uploaded = upload(image, f"{unit_id}_{tag}.png")
        control_name = upload(control, f"{unit_id}_{tag}_ctl.png") if control is not None else None
        workflow = build_overpaint_workflow(
            checkpoint, positive, negative, uploaded, seed, steps, cfg, run_denoise,
            f"AssetForge/overpaint/{unit_id}/{tag}",
            control_image=control_name, control_strength=control_strength,
        )
        prompt_id = client.submit(workflow)
        history = client.wait(prompt_id, timeout_seconds=timeout_seconds)
        scratch = frames_root.resolve() / f"{unit_id}-overpaint-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        outputs = client.download_outputs(history, scratch)
        with Image.open(outputs[0]) as painted_source:
            painted = painted_source.convert("RGB")
        for item in scratch.iterdir():
            item.unlink()
        return painted

    def load_depth(frame_path: Path, size: tuple[int, int]) -> Image.Image:
        depth_path = _sibling(frame_path, unit_frames, "-depth")
        if depth_path.is_file():
            with Image.open(depth_path) as source:
                return source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return Image.new("RGB", size, (0, 0, 0))

    # Persistent style anchor: painted once per unit, embedded into every grid afterwards.
    anchors_folder = config_path.resolve().parent / "anchors"
    anchor_path = anchors_folder / f"{unit_id}.anchor.png"
    anchor_depth_path = anchors_folder / f"{unit_id}.anchor.depth.png"
    anchor_image: Image.Image | None = None
    anchor_depth: Image.Image | None = None
    anchor_created = False
    if use_anchor:
        if anchor_path.is_file():
            anchor_image = Image.open(anchor_path).convert("RGB")
            anchor_depth = (
                Image.open(anchor_depth_path).convert("RGB")
                if anchor_depth_path.is_file()
                else Image.new("RGB", anchor_image.size, (0, 0, 0))
            )
        else:
            preferred = [p for p in frames if p.match("*/idle/south/00.png")] or frames
            source_frame = preferred[0]
            with Image.open(source_frame) as source:
                original = source.convert("RGBA")
            work = _flatten(original).resize((WORK_SIZE, WORK_SIZE), Image.Resampling.LANCZOS)
            depth = load_depth(source_frame, (WORK_SIZE, WORK_SIZE)) if use_depth else None
            painted = run_workflow(work, "anchor", denoise, depth)
            if painted.size != (WORK_SIZE, WORK_SIZE):
                painted = painted.resize((WORK_SIZE, WORK_SIZE), Image.Resampling.LANCZOS)
            anchors_folder.mkdir(parents=True, exist_ok=True)
            painted.save(anchor_path, "PNG")
            (depth or Image.new("RGB", (WORK_SIZE, WORK_SIZE), (0, 0, 0))).save(anchor_depth_path, "PNG")
            anchor_image, anchor_depth = painted, depth or Image.new("RGB", (WORK_SIZE, WORK_SIZE), (0, 0, 0))
            anchor_created = True

    raw_backup = frames_root.resolve() / f"{unit_id}-raw"

    def backup(original: Image.Image, frame_path: Path) -> None:
        backup_path = raw_backup / frame_path.relative_to(unit_frames)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        original.save(backup_path, "PNG")

    cell = int(block.get("grid_cell", 512))
    columns = int(block.get("grid_columns", 3))
    rows = int(block.get("grid_rows", 2))
    slots = columns * rows
    payload_slots = slots - 1 if anchor_image is not None else slots

    groups: dict[Path, list[Path]] = {}
    for frame_path in frames:
        groups.setdefault(frame_path.parent, []).append(frame_path)

    processed = 0
    integrity_failures: list[dict[str, Any]] = []
    for group_index, (folder, members) in enumerate(sorted(groups.items())):
        for chunk_index in range(0, len(members), payload_slots):
            chunk = members[chunk_index : chunk_index + payload_slots]
            originals = []
            for frame_path in chunk:
                with Image.open(frame_path) as source:
                    originals.append(source.convert("RGBA"))

            def compose(cells: list[Image.Image], anchor: Image.Image | None) -> Image.Image:
                canvas = Image.new("RGB", (columns * cell, rows * cell), BACKGROUND)
                offset = 0
                if anchor is not None:
                    canvas.paste(anchor.resize((cell, cell), Image.Resampling.LANCZOS), (0, 0))
                    offset = 1
                for index, item in enumerate(cells):
                    slot = index + offset
                    canvas.paste(item.resize((cell, cell), Image.Resampling.LANCZOS), ((slot % columns) * cell, (slot // columns) * cell))
                return canvas

            beauty = compose([_flatten(original) for original in originals], anchor_image)
            control = None
            if use_depth:
                control = compose([load_depth(path, (cell, cell)) for path in chunk], anchor_depth)

            def attempt(run_denoise: float) -> tuple[list[Image.Image], list[dict[str, Any]]]:
                painted = run_workflow(beauty, f"{group_index:03d}_{chunk_index:02d}_{int(run_denoise * 100)}", run_denoise, control)
                if painted.size != beauty.size:
                    painted = painted.resize(beauty.size, Image.Resampling.LANCZOS)
                pieces: list[Image.Image] = []
                failures: list[dict[str, Any]] = []
                offset = 1 if anchor_image is not None else 0
                for index, (frame_path, original) in enumerate(zip(chunk, originals)):
                    slot = index + offset
                    x, y = (slot % columns) * cell, (slot // columns) * cell
                    piece = painted.crop((x, y, x + cell, y + cell)).resize(original.size, Image.Resampling.LANCZOS)
                    ok, distance = equipment_integrity(piece, original.getchannel("A"), _sibling(frame_path, unit_frames, "-equip"))
                    if not ok:
                        failures.append({"frame": str(frame_path), "distance": round(distance, 2), "denoise": run_denoise})
                    pieces.append(piece)
                return pieces, failures

            pieces, failures = attempt(denoise)
            if failures:
                retry_pieces, retry_failures = attempt(max(0.3, denoise - 0.1))
                if len(retry_failures) < len(failures):
                    pieces, failures = retry_pieces, retry_failures
            integrity_failures.extend(failures)

            for frame_path, original, piece in zip(chunk, originals, pieces):
                backup(original, frame_path)
                result = piece.convert("RGBA")
                result.putalpha(original.getchannel("A"))
                result.save(frame_path, "PNG")
                processed += 1

    provenance = {
        "schema_version": 2,
        "stage": "overpaint",
        "unit": unit_id,
        "batch": "grid",
        "checkpoint": checkpoint,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "depth_control": use_depth,
        "control_strength": control_strength if use_depth else None,
        "anchor": str(anchor_path) if anchor_image is not None else None,
        "anchor_created": anchor_created,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "frames_processed": processed,
        "equipment_integrity_failures": integrity_failures,
        "alpha_source": "blender-render",
        "workflow_sha256": sha256_json(
            build_overpaint_workflow(checkpoint, positive, negative, "x", seed, steps, cfg, denoise, "x")
        ),
        "completed_utc": utc_now(),
    }
    write_json(frames_root.resolve() / f"{unit_id}-overpaint.json", provenance)
    if integrity_failures:
        print(f"OVERPAINT_INTEGRITY_WARNINGS={len(integrity_failures)}")
    return provenance
