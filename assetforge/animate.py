"""Motion storyboard prototype: pose-controlled diffusion frame generation.

Every frame is img2img from the SAME reference image with a fixed seed; motion comes
from an authored pose pack applied through an OpenPose ControlNet. Frames land in the
same frames/{id}/{action}/{direction}/NN.png contract used by review tooling. Diffusion
does not guarantee identity or transition continuity, so these outputs are never eligible
for production packaging or Unity import.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .comfy import ComfyClient
from .core import (
    ForgeError,
    asset_root,
    load_asset,
    load_project,
    package_root,
    read_json,
    record_job,
    sha256_file,
    sha256_json,
    utc_now,
    write_json,
)
from .pose import load_pack, pack_sha256, render_action_frames, standing_extent

STUDIO_BACKGROUND = (24, 28, 32)
ACTION_PHRASES = {
    "idle": "standing guard at ease, weapon held ready at the side",
    "walk": "marching forward mid-stride, purposeful walk cycle",
    "attack": "swinging the weapon in a powerful forward strike",
    "shoot": "drawing and loosing a projectile at a distant target",
    "defend": "bracing behind the raised shield, guarding against a blow",
    "cast": "channeling and thrusting both hands forward to release a spell",
    "hit": "flinching backward, staggered by an impact",
    "death": "collapsing to the ground, defeated",
}
DEFAULT_CHECKPOINT = "dreamshaper_xl_v2_turbo.safetensors"
DEFAULT_CONTROLNET = "controlnet_openpose_sdxl_xinsir.safetensors"


def model_record(workspace: Path, filename: str) -> dict[str, Any] | None:
    project = load_project(workspace)
    registry = read_json(workspace.resolve() / project["model_registry"])
    return next((item for item in registry.get("models", []) if item.get("filename") == filename), None)


def sampler_settings(workspace: Path, checkpoint: str) -> dict[str, Any]:
    project = load_project(workspace)
    profile = read_json(workspace.resolve() / "hardware" / f"{project['hardware_profile']}.profile.json")
    settings = dict(profile["comfyui"])
    record = model_record(workspace, checkpoint)
    if record and isinstance(record.get("sampler_settings"), dict):
        settings.update(record["sampler_settings"])
    return settings


def build_animation_prompt(workspace: Path, asset_id: str, action_name: str) -> tuple[str, str]:
    project = load_project(workspace)
    asset = load_asset(workspace, asset_id)
    style = read_json(workspace.resolve() / project["style_profile"])
    brief = asset.get("brief", {})
    equipment = ", ".join(brief.get("equipment", [])) or "equipment defined by the role"
    materials = ", ".join(brief.get("materials", [])) or "weathered practical fantasy materials"
    phrase = ACTION_PHRASES.get(action_name, action_name)
    positive = (
        f"{asset['display_name']}, {asset['role']}; equipment: {equipment}; materials: {materials}. "
        f"{phrase}. Side view profile facing right, one single character, full body with feet visible, "
        f"flat dark studio background. {style['positive']}"
    )
    must_not = ", ".join(brief.get("must_not_have", []))
    negative = (
        f"{style['negative']}, multiple characters, duplicated figure, turnaround sheet, "
        f"front view, back view, cropped limbs" + (f", {must_not}" if must_not else "")
    )
    return positive, negative


def prepare_reference(
    reference: Path,
    canvas: tuple[int, int],
    pack: dict[str, Any],
    cutout: Callable[[list[tuple[Path, Path]]], None],
    work_folder: Path,
) -> bytes:
    """Normalize the reference figure to the authored pose extent.

    The img2img latent and the ControlNet skeleton must agree on where the character is:
    the reference is cut out, scaled so head-to-ankle matches the pose pack's standing
    extent, and placed feet-on-baseline. Otherwise generated figures inherit the
    reference framing and overflow the authored ground line.
    """
    work_folder.mkdir(parents=True, exist_ok=True)
    cut_path = work_folder / "reference_cutout.png"
    cutout([(reference, cut_path)])
    with Image.open(cut_path) as source:
        image = source.convert("RGBA")
    box = image.getchannel("A").point(lambda a: 255 if a >= 40 else 0).getbbox()
    if box is None:
        raise ForgeError(f"Reference cutout is fully transparent: {reference}")
    figure = image.crop(box)

    width, height = canvas
    head_y, ankle_y = standing_extent(pack)
    target_height = max(1.0, (ankle_y - (head_y - 0.05)) * height)
    scale = min(target_height / figure.height, (width * 0.7) / figure.width)
    size = (max(1, round(figure.width * scale)), max(1, round(figure.height * scale)))
    resized = figure.resize(size, Image.Resampling.LANCZOS)
    destination = Image.new("RGB", (width, height), STUDIO_BACKGROUND)
    baseline = round(ankle_y * height)
    destination.paste(resized, ((width - size[0]) // 2, baseline - size[1]), resized)
    prepared_path = work_folder / "reference_prepared.png"
    destination.save(prepared_path, "PNG")
    stream = io.BytesIO()
    destination.save(stream, "PNG")
    return stream.getvalue()


def build_frame_workflow(
    positive: str,
    negative: str,
    checkpoint: str,
    controlnet: str,
    reference_name: str,
    pose_name: str,
    seed: int,
    settings: dict[str, Any],
    denoise: float,
    control_strength: float,
    prefix: str,
) -> dict[str, Any]:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": positive}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative}},
        "4": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": pose_name}},
        "7": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet}},
        "8": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "control_net": ["7", 0],
                "image": ["6", 0],
                "strength": float(control_strength),
                "start_percent": 0.0,
                "end_percent": 0.8,
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["5", 0],
                "seed": int(seed),
                "steps": int(settings["steps"]),
                "cfg": float(settings["cfg"]),
                "sampler_name": settings["sampler"],
                "scheduler": settings["scheduler"],
                "denoise": float(denoise),
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["10", 0]}},
    }


def default_cutout(pairs: list[tuple[Path, Path]]) -> None:
    python = package_root() / "runtime" / "triposr" / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise ForgeError(
            f"Background-removal worker needs the TripoSR venv (rembg) at {python}. "
            "Run .\\scripts\\setup_local_ai.ps1 install-triposr -AcceptTripoSrLicense"
        )
    worker = Path(__file__).resolve().parent / "cutout_worker.py"
    payload = "\n".join(f"{source}|{destination}" for source, destination in pairs) + "\n"
    result = subprocess.run(
        [str(python), str(worker)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=120 + 60 * len(pairs),
    )
    if result.returncode != 0:
        raise ForgeError(f"Background removal failed: {result.stderr.strip()[:2000]}")


def generate_animation(
    workspace: Path,
    asset_id: str,
    actions: list[str],
    checkpoint: str = DEFAULT_CHECKPOINT,
    controlnet: str = DEFAULT_CONTROLNET,
    reference: Path | None = None,
    pack_id: str = "humanoid_side_east",
    seed: int = 1001,
    denoise: float = 0.6,
    control_strength: float = 0.85,
    base_url: str = "http://127.0.0.1:8188",
    timeout_seconds: float = 600.0,
    cutout: Callable[[list[tuple[Path, Path]]], None] | None = None,
) -> dict[str, Any]:
    project = load_project(workspace)
    asset = load_asset(workspace, asset_id)
    pack = load_pack(workspace, pack_id)
    settings = sampler_settings(workspace, checkpoint)
    canvas = tuple(int(value) for value in pack.get("canvas", [768, 768]))
    view = pack.get("view", "east")

    if reference is None:
        raise ForgeError("An approved reference image is required (--reference); it locks the unit identity")
    reference = reference.resolve()
    if not reference.is_file():
        raise ForgeError(f"Reference image was not found: {reference}")

    for action_name in actions:
        if action_name not in pack.get("actions", {}):
            raise ForgeError(f"Pose pack has no action {action_name!r}; available: {sorted(pack.get('actions', {}))}")

    client = ComfyClient(base_url)
    available = client.checkpoints()
    if checkpoint not in available:
        raise ForgeError(f"Checkpoint {checkpoint!r} is not installed. Available: {available}")

    reference_upload = client.upload_image(
        f"{asset_id}_reference.png",
        prepare_reference(
            reference, canvas, pack, cutout or default_cutout,
            asset_root(workspace, asset_id) / "renders",
        ),
    )
    record = model_record(workspace, checkpoint)
    controlnet_record = model_record(workspace, controlnet)
    frames_root = asset_root(workspace, asset_id) / "renders" / "frames"
    raw_root = asset_root(workspace, asset_id) / "renders" / "raw"
    generated: dict[str, Any] = {"asset_id": asset_id, "actions": {}, "frames_root": str(frames_root)}

    for action_name in actions:
        positive, negative = build_animation_prompt(workspace, asset_id, action_name)
        pose_images = render_action_frames(pack, action_name)
        cutout_pairs: list[tuple[Path, Path]] = []
        frame_records = []
        for index, pose_image in enumerate(pose_images):
            stream = io.BytesIO()
            pose_image.save(stream, "PNG")
            pose_upload = client.upload_image(f"{asset_id}_{action_name}_{index:02d}_pose.png", stream.getvalue())
            prefix = f"AssetForge/{project['project_id']}/{asset_id}/anim/{action_name}/{index:02d}"
            workflow = build_frame_workflow(
                positive, negative, checkpoint, controlnet, reference_upload, pose_upload,
                seed, settings, denoise, control_strength, prefix,
            )
            prompt_id = client.submit(workflow)
            history = client.wait(prompt_id, timeout_seconds=timeout_seconds)
            raw_folder = raw_root / asset_id / action_name / view
            outputs = client.download_outputs(history, raw_folder)
            raw_frame = raw_folder / f"{index:02d}.png"
            outputs[0].replace(raw_frame)
            final_folder = frames_root / asset_id / action_name / view
            final_folder.mkdir(parents=True, exist_ok=True)
            final_frame = final_folder / f"{index:02d}.png"
            cutout_pairs.append((raw_frame, final_frame))
            frame_records.append({
                "index": index,
                "workflow_sha256": sha256_json(workflow),
                "prompt_id": prompt_id,
                "raw": str(raw_frame),
                "final": str(final_frame),
            })

        (cutout or default_cutout)(cutout_pairs)
        for item in frame_records:
            final_path = Path(item["final"])
            if not final_path.is_file():
                raise ForgeError(f"Cutout did not produce {final_path}")
            item["final_sha256"] = sha256_file(final_path)

        job = {
            "engine": "AssetForge Motion Storyboard Prototype",
            "production_eligible": False,
            "production_blocker": "diffusion_frame_sequence_has_no_single_master_identity_lock",
            "workflow_type": "pose-controlled-img2img",
            "asset_id": asset_id,
            "action": action_name,
            "view": view,
            "checkpoint": checkpoint,
            "controlnet": controlnet,
            "model": {
                "id": record.get("id") if record else None,
                "license": record.get("license") if record else None,
                "installed_sha256": record.get("installed_sha256") if record else None,
                "registry_status": "hash-registered" if record and record.get("installed_sha256") else "unregistered",
            },
            "controlnet_model": {
                "id": controlnet_record.get("id") if controlnet_record else None,
                "license": controlnet_record.get("license") if controlnet_record else None,
                "installed_sha256": controlnet_record.get("installed_sha256") if controlnet_record else None,
            },
            "pose_pack": {"id": pack.get("id"), "sha256": pack_sha256(pack)},
            "reference": {"path": str(reference), "sha256": sha256_file(reference)},
            "seed": int(seed),
            "denoise": float(denoise),
            "control_strength": float(control_strength),
            "sampler": {key: settings[key] for key in ("steps", "cfg", "sampler", "scheduler")},
            "frames": frame_records,
            "completed_utc": utc_now(),
        }
        record_job(workspace, asset_id, job)
        generated["actions"][action_name] = {
            "frames": len(frame_records),
            "folder": str(frames_root / asset_id / action_name / view),
        }

    del asset
    return generated
