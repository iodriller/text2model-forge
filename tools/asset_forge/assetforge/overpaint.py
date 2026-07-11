"""Structure-locked overpaint v3: depth-controlled, anchor-consistent, equipment-gated.

The Blender render supplies structure; SDXL repaints surfaces. Three mechanisms keep the
result consistent and complete:

1. Depth ControlNet — every frame's TRUE 3D depth (rendered by Blender) constrains
   generation. This improves structure but is not treated as an equipment guarantee.
2. Persistent style anchor — one painted reference cell per unit, created once and stored
   next to the character config, is embedded into EVERY grid of EVERY run. Diffusion
   copies materials/emblems from it, making frame N, frame N+1, and next month's rebuild
   agree.
3. Protected equipment — Blender also renders an equipment-only mask per frame. After
   painting, authored pixels are restored under that mask. Missing/empty masks or any
   protected-pixel mismatch fail the job without promoting partial outputs.

The ORIGINAL render alpha is re-applied to every painted frame, so silhouettes and
baselines never drift. Chunks are resumable and promoted only as a complete passing set.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageStat

from .comfy import ComfyClient
from .core import ForgeError, sha256_file, sha256_json, utc_now, write_json

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


def _mask_channel(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    rgba = source.convert("RGBA").resize(size, Image.Resampling.NEAREST)
    red, green, blue, alpha = rgba.split()
    visible_color = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    return ImageChops.multiply(visible_color, alpha)


def equipment_integrity(painted: Image.Image, alpha: Image.Image, equip_mask_path: Path) -> tuple[bool, float]:
    """True when the painted equipment region still reads against its surroundings."""
    if not equip_mask_path.is_file():
        return False, -1.0
    with Image.open(equip_mask_path) as source:
        mask = _mask_channel(source, painted.size)
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


def restore_protected_parts(
    painted: Image.Image,
    original: Image.Image,
    equip_mask_path: Path,
    feather_pixels: float = 0.0,
) -> tuple[Image.Image, int, float]:
    """Restore authored equipment pixels and report coverage and reconstruction error.

    Diffusion is useful for surface treatment but cannot be trusted with a sword edge,
    shield silhouette, or emblem.  The Blender equipment pass is therefore a protection
    mask, not merely a warning hint.  Pixels under it come from the deterministic master.
    """
    if not equip_mask_path.is_file():
        raise ForgeError(f"Required equipment mask is missing: {equip_mask_path}")
    with Image.open(equip_mask_path) as source:
        mask = _mask_channel(source, painted.size)
    mask = mask.point(lambda value: 255 if value >= 100 else 0)
    coverage = round(ImageStat.Stat(mask).sum[0] / 255)
    if coverage < INTEGRITY_MIN_PIXELS:
        raise ForgeError(f"Equipment mask is empty or too small ({coverage} pixels): {equip_mask_path}")
    if feather_pixels > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather_pixels)))

    authored = original.convert("RGB").resize(painted.size, Image.Resampling.LANCZOS)
    restored = Image.composite(authored, painted.convert("RGB"), mask)
    _, error = protected_part_metrics(restored, authored, equip_mask_path)
    return restored, coverage, error


def protected_part_metrics(
    image: Image.Image,
    original: Image.Image,
    equip_mask_path: Path,
) -> tuple[int, float]:
    if not equip_mask_path.is_file():
        raise ForgeError(f"Required equipment mask is missing: {equip_mask_path}")
    with Image.open(equip_mask_path) as source:
        mask = _mask_channel(source, image.size)
    hard_mask = mask.point(lambda value: 255 if value >= 100 else 0)
    coverage = round(ImageStat.Stat(hard_mask).sum[0] / 255)
    authored = original.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
    difference = ImageStat.Stat(ImageChops.difference(image.convert("RGB"), authored), hard_mask).mean
    return coverage, sum(difference) / 3.0


def masks_equivalent(first_path: Path, second_path: Path) -> bool:
    if first_path.resolve() == second_path.resolve():
        return True
    if not first_path.is_file() or not second_path.is_file():
        return False
    with Image.open(first_path) as first_source, Image.open(second_path) as second_source:
        first = _mask_channel(first_source, first_source.size)
        second = _mask_channel(second_source, first.size)
    return ImageChops.difference(first, second).getbbox() is None


def _registered_model_hash(config_path: Path, checkpoint: str) -> str | None:
    for parent in config_path.resolve().parents:
        registry_path = parent / "asset_sources" / "ember-defense" / "models" / "model-registry.json"
        if not registry_path.is_file():
            continue
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for model in registry.get("models", []):
            if model.get("filename") == checkpoint:
                value = model.get("installed_sha256")
                return str(value) if value else None
    return None


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
    protect_equipment = bool(block.get("protect_equipment", True))
    fail_closed = bool(block.get("fail_closed", True))
    protection_feather = float(block.get("protection_feather", 0.0))
    equipment_required = bool(config.get("equipment_objects"))
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
    input_hashes = {
        str(path.relative_to(unit_frames)).replace("\\", "/"): sha256_file(path)
        for path in frames
    }
    input_manifest_sha256 = sha256_json(input_hashes)

    depth_available = (frames_root.resolve() / f"{unit_id}-depth").is_dir()
    use_depth = bool(block.get("control", "depth") == "depth" and depth_available)
    structural_hashes: dict[str, str] = {}
    required_passes = ["-depth"] if use_depth else []
    if equipment_required:
        required_passes.extend(["-equip", "-protect"])
    for frame_path in frames:
        relative = str(frame_path.relative_to(unit_frames)).replace("\\", "/")
        for suffix in required_passes:
            pass_path = _sibling(frame_path, unit_frames, suffix)
            if not pass_path.is_file():
                raise ForgeError(f"Required structural pass is missing: {pass_path}")
            structural_hashes[f"{suffix}/{relative}"] = sha256_file(pass_path)
    structural_manifest_sha256 = sha256_json(structural_hashes)

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
    staging_root = frames_root.resolve() / f"{unit_id}-overpaint-staging"
    state_path = staging_root / "state.json"
    run_signature = sha256_json({
        "unit": unit_id,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _registered_model_hash(config_path, checkpoint),
        "controlnet": DEPTH_CONTROLNET if use_depth else None,
        "controlnet_sha256": _registered_model_hash(config_path, DEPTH_CONTROLNET) if use_depth else None,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "control_strength": control_strength,
        "positive": positive,
        "negative": negative,
        "input_manifest_sha256": input_manifest_sha256,
        "structural_manifest_sha256": structural_manifest_sha256,
        "anchor_sha256": sha256_file(anchor_path) if anchor_path.is_file() else None,
        "protect_equipment": protect_equipment,
        "protection_feather": protection_feather,
        "grid_cell": int(block.get("grid_cell", 512)),
        "grid_columns": int(block.get("grid_columns", 3)),
        "grid_rows": int(block.get("grid_rows", 2)),
    })
    state: dict[str, Any] = {"run_signature": run_signature, "completed_chunks": []}
    if state_path.is_file():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_state.get("run_signature") == run_signature:
            state = saved_state
        else:
            shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    completed_chunks = set(str(value) for value in state.get("completed_chunks", []))

    def backup(original: Image.Image, frame_path: Path) -> None:
        backup_path = raw_backup / frame_path.relative_to(unit_frames)
        if backup_path.is_file():
            return
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
    contrast_warnings: list[dict[str, Any]] = []
    repaired_generation_failures: list[dict[str, Any]] = []
    protection_records: list[dict[str, Any]] = []
    for group_index, (folder, members) in enumerate(sorted(groups.items())):
        for chunk_index in range(0, len(members), payload_slots):
            chunk = members[chunk_index : chunk_index + payload_slots]
            chunk_key = f"{group_index:03d}:{chunk_index:03d}"
            staged_paths = [staging_root / frame_path.relative_to(unit_frames) for frame_path in chunk]
            if chunk_key in completed_chunks and all(path.is_file() for path in staged_paths):
                processed += len(chunk)
                continue
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
                    equipment_mask = _sibling(frame_path, unit_frames, "-equip")
                    protection_mask = _sibling(frame_path, unit_frames, "-protect")
                    fully_protected = masks_equivalent(protection_mask, equipment_mask)
                    ok, distance = (
                        (True, -1.0)
                        if fully_protected
                        else equipment_integrity(piece, original.getchannel("A"), equipment_mask)
                    )
                    if not ok:
                        failures.append({"frame": str(frame_path), "distance": round(distance, 2), "denoise": run_denoise})
                    pieces.append(piece)
                return pieces, failures

            pieces, failures = attempt(denoise)
            if failures:
                retry_pieces, retry_failures = attempt(max(0.3, denoise - 0.1))
                if len(retry_failures) < len(failures):
                    pieces, failures = retry_pieces, retry_failures
            repaired_generation_failures.extend(failures)

            for frame_path, original, piece in zip(chunk, originals, pieces):
                mask_path = _sibling(frame_path, unit_frames, "-equip")
                protection_mask_path = _sibling(frame_path, unit_frames, "-protect")
                if not protection_mask_path.is_file():
                    protection_mask_path = mask_path
                if equipment_required and protect_equipment:
                    try:
                        piece, coverage, reconstruction_error = restore_protected_parts(
                            piece, original, protection_mask_path, protection_feather
                        )
                        protection_records.append({
                            "frame": str(frame_path),
                            "mask_pixels": coverage,
                            "protected_pixel_mae": round(reconstruction_error, 6),
                        })
                    except ForgeError as error:
                        integrity_failures.append({"frame": str(frame_path), "error": str(error)})
                full_equipment_protected = masks_equivalent(protection_mask_path, mask_path)
                ok, distance = (
                    (True, -1.0)
                    if full_equipment_protected
                    else equipment_integrity(piece, original.getchannel("A"), mask_path)
                )
                if equipment_required and not full_equipment_protected and not ok:
                    contrast_warnings.append({
                        "frame": str(frame_path),
                        "distance": round(distance, 2),
                        "warning": "equipment palette is close to adjacent character pixels",
                    })
                result = piece.convert("RGBA")
                result.putalpha(original.getchannel("A"))
                staged_path = staging_root / frame_path.relative_to(unit_frames)
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(staged_path, "PNG")
                processed += 1
            completed_chunks.add(chunk_key)
            state["completed_chunks"] = sorted(completed_chunks)
            state["updated_utc"] = utc_now()
            write_json(state_path, state)

    passed = not integrity_failures
    protection_records = []
    if passed:
        for frame_path in frames:
            staged_path = staging_root / frame_path.relative_to(unit_frames)
            if not staged_path.is_file():
                integrity_failures.append({"frame": str(frame_path), "error": "staged output is missing"})
                continue
            if equipment_required and protect_equipment:
                try:
                    with Image.open(staged_path) as staged_source, Image.open(frame_path) as original_source:
                        coverage, reconstruction_error = protected_part_metrics(
                            staged_source.convert("RGB"),
                            original_source.convert("RGBA"),
                            (
                                _sibling(frame_path, unit_frames, "-protect")
                                if _sibling(frame_path, unit_frames, "-protect").is_file()
                                else _sibling(frame_path, unit_frames, "-equip")
                            ),
                        )
                    if coverage < INTEGRITY_MIN_PIXELS or reconstruction_error > 0.01:
                        integrity_failures.append({
                            "frame": str(frame_path),
                            "error": "protected equipment differs from the authored master",
                            "protected_pixel_mae": round(reconstruction_error, 6),
                        })
                    protection_records.append({
                        "frame": str(frame_path),
                        "mask_pixels": coverage,
                        "protected_pixel_mae": round(reconstruction_error, 6),
                    })
                except ForgeError as error:
                    integrity_failures.append({"frame": str(frame_path), "error": str(error)})
        passed = not integrity_failures

    provenance = {
        "schema_version": 3,
        "stage": "overpaint",
        "unit": unit_id,
        "passed": passed,
        "fail_closed": fail_closed,
        "transactional_promotion": True,
        "resumable_chunks": True,
        "run_signature": run_signature,
        "batch": "grid",
        "grid_cell": cell,
        "grid_columns": columns,
        "grid_rows": rows,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _registered_model_hash(config_path, checkpoint),
        "controlnet": DEPTH_CONTROLNET if use_depth else None,
        "controlnet_sha256": _registered_model_hash(config_path, DEPTH_CONTROLNET) if use_depth else None,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "depth_control": use_depth,
        "control_strength": control_strength if use_depth else None,
        "anchor": str(anchor_path) if anchor_image is not None else None,
        "anchor_sha256": sha256_file(anchor_path) if anchor_path.is_file() else None,
        "anchor_created": anchor_created,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "frames_processed": processed,
        "input_manifest_sha256": input_manifest_sha256,
        "structural_manifest_sha256": structural_manifest_sha256,
        "protected_equipment": equipment_required and protect_equipment,
        "protected_frames": len(protection_records),
        "protection_records": protection_records,
        "generation_equipment_warnings_repaired": repaired_generation_failures,
        "equipment_contrast_warnings": contrast_warnings,
        "equipment_integrity_failures": integrity_failures,
        "alpha_source": "blender-render",
        "workflow_sha256": sha256_json(
            build_overpaint_workflow(checkpoint, positive, negative, "x", seed, steps, cfg, denoise, "x")
        ),
        "completed_utc": utc_now(),
    }
    write_json(frames_root.resolve() / f"{unit_id}-overpaint.json", provenance)
    if integrity_failures:
        message = f"Overpaint rejected {len(integrity_failures)} equipment-integrity failure(s)"
        if fail_closed:
            raise ForgeError(message + "; original Blender frames were left unchanged")
        print(f"OVERPAINT_INTEGRITY_WARNINGS={len(integrity_failures)}")
        return provenance

    for frame_path in frames:
        with Image.open(frame_path) as source:
            backup(source.convert("RGBA"), frame_path)
        staged_path = staging_root / frame_path.relative_to(unit_frames)
        staged_path.replace(frame_path)
    shutil.rmtree(staging_root)
    return provenance


def repair_protected_frames(config_path: Path, frames_root: Path) -> dict[str, Any]:
    """Salvage an older/partial overpaint by restoring protected master pixels.

    The result remains review-only until a complete schema-3 overpaint run succeeds, but
    this makes interrupted and historical work useful for visual comparison without
    spending more GPU time.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unit_id = str(config["id"])
    unit_frames = frames_root.resolve() / unit_id
    raw_root = frames_root.resolve() / f"{unit_id}-raw"
    staging_root = frames_root.resolve() / f"{unit_id}-repair-staging"
    frames = sorted(unit_frames.glob("*/*/[0-9][0-9].png"))
    if not frames:
        raise ForgeError(f"No painted frames found under {unit_frames}")
    if not raw_root.is_dir():
        raise ForgeError(f"No deterministic raw backup found under {raw_root}")
    if staging_root.exists():
        shutil.rmtree(staging_root)

    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for frame_path in frames:
        relative = frame_path.relative_to(unit_frames)
        raw_path = raw_root / relative
        protection_mask = _sibling(frame_path, unit_frames, "-protect")
        if not protection_mask.is_file():
            protection_mask = _sibling(frame_path, unit_frames, "-equip")
        try:
            with Image.open(frame_path) as painted_source, Image.open(raw_path) as raw_source:
                raw = raw_source.convert("RGBA")
                restored, coverage, error = restore_protected_parts(
                    painted_source.convert("RGB"), raw, protection_mask
                )
                equipment_mask = _sibling(frame_path, unit_frames, "-equip")
                full_equipment_protected = masks_equivalent(protection_mask, equipment_mask)
                ok, distance = (
                    (True, -1.0)
                    if full_equipment_protected
                    else equipment_integrity(restored, raw.getchannel("A"), equipment_mask)
                )
                result = restored.convert("RGBA")
                result.putalpha(raw.getchannel("A"))
            if not full_equipment_protected and not ok:
                failures.append({"frame": str(frame_path), "error": "equipment remains unreadable", "distance": distance})
            target = staging_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            result.save(target, "PNG")
            records.append({"frame": str(frame_path), "mask_pixels": coverage, "protected_pixel_mae": error})
        except (ForgeError, OSError) as error:
            failures.append({"frame": str(frame_path), "error": str(error)})

    report = {
        "schema_version": 1,
        "stage": "protected-part-repair",
        "unit": unit_id,
        "passed": not failures,
        "production_eligible": False,
        "production_blocker": "complete_schema_3_overpaint_run_required",
        "frames_processed": len(records),
        "failures": failures,
        "records": records,
        "completed_utc": utc_now(),
    }
    report_path = frames_root.resolve() / f"{unit_id}-repair.json"
    write_json(report_path, report)
    if failures:
        raise ForgeError(
            f"Protected-part repair rejected {len(failures)} frame(s); existing painted frames were unchanged"
        )
    for frame_path in frames:
        (staging_root / frame_path.relative_to(unit_frames)).replace(frame_path)
    shutil.rmtree(staging_root)
    return report
