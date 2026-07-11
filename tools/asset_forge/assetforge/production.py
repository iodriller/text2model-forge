"""Production packaging and human-readable review media for owned Blender units.

Diffusion frame sequences are useful motion boards, but they cannot prove identity
continuity.  A production package therefore requires one hash-locked Blender master,
four rendered directions, a passing sheet report, and explicit action timing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .core import ForgeError, sha256_file, sha256_json, utc_now, write_json


REQUIRED_DIRECTIONS = ("north", "south", "east", "west")
REQUIRED_ACTIONS = ("idle", "walk", "attack", "hit", "death")
PRODUCTION_SOURCE_KIND = "owned_blender_master"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForgeError(f"Could not load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ForgeError(f"{label.capitalize()} must contain a JSON object: {path}")
    return value


def _absolute(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def validate_production_config(config: dict[str, Any], repo_root: Path) -> list[str]:
    failures: list[str] = []
    if config.get("source_kind") != PRODUCTION_SOURCE_KIND:
        failures.append(f"source_kind must be {PRODUCTION_SOURCE_KIND!r}")
    source_value = config.get("source")
    source = _absolute(repo_root, source_value) if isinstance(source_value, str) else None
    if source is None or not source.is_file() or source.suffix.lower() != ".blend":
        failures.append("source must be an existing owned .blend master")
    directions = tuple(config.get("directions", []))
    if any(direction not in directions for direction in REQUIRED_DIRECTIONS):
        failures.append("production characters require genuine north, south, east, and west renders")
    animations = config.get("animations", {})
    if not isinstance(animations, dict):
        failures.append("animations must be an object")
        return failures
    for action_name in REQUIRED_ACTIONS:
        if action_name not in animations:
            failures.append(f"missing required action {action_name!r}")
    for action_name, action in animations.items():
        if not isinstance(action, dict):
            failures.append(f"action {action_name!r} must be an object")
            continue
        frames = int(action.get("frames", 0))
        fps = float(action.get("fps", 0))
        if frames < 2:
            failures.append(f"action {action_name!r} needs at least two frames")
        if fps <= 0:
            failures.append(f"action {action_name!r} needs fps > 0")
        samples = action.get("sample_frames")
        if samples is not None and len(samples) != frames:
            failures.append(f"action {action_name!r} sample_frames must contain exactly {frames} values")
        phases = action.get("phases", {})
        if action_name in ("attack", "heavy_attack", "shoot") and "contact_frame" not in phases:
            failures.append(f"action {action_name!r} needs phases.contact_frame")
        for phase_name, frame in phases.items():
            if not isinstance(frame, int) or frame < 0 or frame >= frames:
                failures.append(f"action {action_name!r} phase {phase_name!r} is outside 0..{max(0, frames - 1)}")
    return failures


def _load_cells(sheet_path: Path, cell_width: int, cell_height: int, frame_count: int) -> list[Image.Image]:
    with Image.open(sheet_path) as source:
        sheet = source.convert("RGBA")
    expected = (cell_width * frame_count, cell_height)
    if sheet.size != expected:
        raise ForgeError(f"Wrong dimensions for {sheet_path}: {sheet.size}, expected {expected}")
    return [
        sheet.crop((index * cell_width, 0, (index + 1) * cell_width, cell_height))
        for index in range(frame_count)
    ]


def _fit_rgba(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    output = Image.new("RGBA", size, (0, 0, 0, 0))
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    output.alpha_composite(copy, ((size[0] - copy.width) // 2, size[1] - copy.height))
    return output


def create_review_media(config: dict[str, Any], repo_root: Path, review_root: Path) -> dict[str, Any]:
    """Create action/direction and gameplay-transition GIFs that expose bad motion."""
    review_root.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height = (int(value) for value in config["cell_size"])
    directions = list(config["directions"])
    cache: dict[tuple[str, str], list[Image.Image]] = {}
    for action_name, action in config["animations"].items():
        for direction in directions:
            relative = config["output_path_pattern"].format(animation=action_name, direction=direction)
            cache[(action_name, direction)] = _load_cells(
                _absolute(repo_root, relative), cell_width, cell_height, int(action["frames"])
            )

    grid_frames: list[Image.Image] = []
    grid_durations: list[int] = []
    tile = 144
    for action_name, action in config["animations"].items():
        count = int(action["frames"])
        duration = max(45, round(1000 / float(action["fps"])))
        for frame_index in range(count):
            frame = Image.new("RGB", (tile * 2, tile * 2 + 28), (14, 20, 24))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, frame.width, 27), fill=(30, 38, 44))
            draw.text((8, 8), f"{config['display_name']}  |  {action_name}", fill=(239, 219, 180))
            for direction_index, direction in enumerate(REQUIRED_DIRECTIONS):
                source_frames = cache[(action_name, direction)]
                source = source_frames[min(frame_index, len(source_frames) - 1)]
                fitted = _fit_rgba(source, (tile, tile))
                x = (direction_index % 2) * tile
                y = 28 + (direction_index // 2) * tile
                frame.paste(fitted, (x, y), fitted)
                draw.text((x + 5, y + 5), direction, fill=(198, 207, 211))
            grid_frames.append(frame)
            grid_durations.append(duration)
        grid_frames.extend([grid_frames[-1]] * 2)
        grid_durations.extend([180, 180])

    all_actions = review_root / f"{config['id']}_all_actions.gif"
    grid_frames[0].save(
        all_actions,
        save_all=True,
        append_images=grid_frames[1:],
        duration=grid_durations,
        loop=0,
        optimize=False,
    )

    default_direction = str(config.get("default_facing", "south"))
    if default_direction not in directions:
        default_direction = "south"
    sequence = [name for name in ("idle", "walk", "attack", "idle", "hit", "idle", "death") if name in config["animations"]]
    stage_frames: list[Image.Image] = []
    stage_durations: list[int] = []
    for action_name in sequence:
        action = config["animations"][action_name]
        cells = cache[(action_name, default_direction)]
        duration = max(45, round(1000 / float(action["fps"])))
        for cell in cells:
            stage = Image.new("RGB", (360, 640), (17, 25, 29))
            draw = ImageDraw.Draw(stage)
            for y in range(92, 560, 58):
                shade = 31 + ((y // 58) % 2) * 4
                draw.polygon(((46, y), (314, y), (328, y + 48), (32, y + 48)), fill=(shade, shade + 6, shade + 4))
            draw.rectangle((22, 24, 338, 67), fill=(30, 38, 43))
            draw.text((34, 39), f"{config['display_name']}  {action_name} / {default_direction}", fill=(239, 219, 180))
            draw.rectangle((45, 560, 315, 624), fill=(43, 37, 34), outline=(104, 83, 57), width=3)
            draw.text((123, 584), "CASTLE GATE", fill=(211, 166, 90))
            sprite = _fit_rgba(cell, (154, 154))
            stage.paste(sprite, (103, 360), sprite)
            stage_frames.append(stage)
            stage_durations.append(duration)
        if action_name == "death":
            stage_frames.extend([stage_frames[-1]] * 4)
            stage_durations.extend([220] * 4)

    transition_reel = review_root / f"{config['id']}_gameplay_transition.gif"
    stage_frames[0].save(
        transition_reel,
        save_all=True,
        append_images=stage_frames[1:],
        duration=stage_durations,
        loop=0,
        optimize=False,
    )

    result = {
        "all_actions": str(all_actions.resolve()),
        "gameplay_transition": str(transition_reel.resolve()),
        "default_direction": default_direction,
        "sequence": sequence,
    }
    write_json(review_root / "review_manifest.json", result)
    return result


def package_production_unit(
    config_path: Path,
    repo_root: Path,
    qa_report_path: Path,
    destination: Path,
    review_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_json(config_path.resolve(), "character config")
    failures = validate_production_config(config, repo_root)
    qa_report = _load_json(qa_report_path.resolve(), "QA report")
    if not qa_report.get("passed"):
        failures.append(f"sheet QA report is not passing: {qa_report_path}")
    if qa_report.get("asset_id") not in (None, config.get("id")) and qa_report.get("character") != config.get("id"):
        failures.append("sheet QA report belongs to a different character")
    if failures:
        raise ForgeError("Production package rejected:\n- " + "\n- ".join(failures))

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = _absolute(repo_root, config["source"])
    actions: list[dict[str, Any]] = []
    copied: list[str] = []
    for action_name, action in config["animations"].items():
        for direction in config["directions"]:
            source_sheet = _absolute(
                repo_root,
                config["output_path_pattern"].format(animation=action_name, direction=direction),
            )
            if not source_sheet.is_file():
                raise ForgeError(f"Missing validated sheet during packaging: {source_sheet}")
            filename = f"{config['id']}_{action_name}_{direction}_sheet.png"
            target = destination / filename
            shutil.copy2(source_sheet, target)
            copied.append(str(target))
            actions.append({
                "name": action_name,
                "direction": direction,
                "frames": int(action["frames"]),
                "fps": float(action["fps"]),
                "loop": bool(action.get("loop", False)),
                "sheet": filename,
                "phases": dict(action.get("phases", {})),
            })

    manifest = {
        "schema_version": 2,
        "asset_id": config["id"],
        "display_name": config.get("display_name", config["id"]),
        "default_facing": config.get("default_facing", "south"),
        "cell_width": int(config["cell_size"][0]),
        "cell_height": int(config["cell_size"][1]),
        "pixels_per_unit": float(config.get("pixels_per_unit", config["cell_size"][1])),
        "pivot_x": float(config.get("pivot_x", 0.5)),
        "pivot_y": float(config.get("pivot_y", 0.035)),
        "prefab_scale": float(config.get("prefab_scale", 1.0)),
        "health_bar_y": float(config.get("health_bar_y", 0.72)),
        "production": {
            "eligible": True,
            "source_kind": PRODUCTION_SOURCE_KIND,
            "identity_lock": "single_master_sha256",
            "master": _relative(repo_root, source),
            "master_sha256": sha256_file(source),
            "master_license": config.get("master_license", "project-original"),
            "config_sha256": sha256_json(config),
            "qa_report": _relative(repo_root, qa_report_path),
            "qa_report_sha256": sha256_file(qa_report_path),
            "packaged_utc": utc_now(),
        },
        "actions": actions,
    }
    manifest_path = destination / "unit_manifest.json"
    write_json(manifest_path, manifest)
    review = create_review_media(config, repo_root, review_root.resolve())
    return {
        "manifest": str(manifest_path),
        "sheets": copied,
        "review": review,
        "production_eligible": True,
    }
