"""Frames -> engine sheets for the Animation Engine path.

Packing uses ONE global crop window and ONE transform for every frame of every action:
motion stays authentic inside the window and there is zero per-frame anchoring jitter.
The vertical anchor is the authored ankle baseline from the pose pack (not the alpha
bounding box), so feet stay locked to the same ground line in every frame. A Unity-ready
unit manifest (frames, fps, loop, pivot, pixels-per-unit) is emitted next to the sheets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .core import SCHEMA_VERSION, ForgeError, asset_root, load_asset, write_json
from .pose import load_pack, standing_extent
from .qa import validate_sheets

GAMEPLAY_CELL = 96
WINDOW_PADDING = 6
ACTION_FPS = {
    "idle": 6.0, "walk": 10.0, "attack": 12.0, "shoot": 12.0,
    "defend": 10.0, "cast": 10.0, "hit": 10.0, "death": 8.0,
}
LOOPING_ACTIONS = {"idle", "walk"}


ALPHA_THRESHOLD = 40


def _alpha_box(image: Image.Image) -> tuple[int, int, int, int]:
    # Threshold first: rembg leaves faint semi-transparent ghosts (motion smears, cape
    # halos) that would otherwise inflate the shared window and shrink the character.
    solid = image.getchannel("A").point(lambda a: 255 if a >= ALPHA_THRESHOLD else 0)
    box = solid.getbbox()
    if box is None:
        raise ForgeError("Frame is fully transparent")
    return box


def _pose_window(pack: dict[str, Any], actions: list[str], canvas: tuple[int, int]) -> tuple[int, int, int, int]:
    """Where the character can possibly be, from authored keypoints plus generous margins."""
    width, height = canvas
    xs: list[float] = []
    ys: list[float] = []
    for name in actions:
        action = pack["actions"].get(name)
        if not action:
            continue
        for keyframe in action["keyframes"]:
            for value in keyframe["pose"].values():
                if value:
                    xs.append(value[0])
                    ys.append(value[1])
    if not xs:
        return (0, 0, width, height)
    pad_x = 0.16
    pad_top = 0.13
    pad_bottom = 0.05
    return (
        max(0, round((min(xs) - pad_x) * width)),
        max(0, round((min(ys) - pad_top) * height)),
        min(width, round((max(xs) + pad_x) * width)),
        min(height, round((max(ys) + pad_bottom) * height)),
    )


def _load_frames(frames_root: Path, asset_id: str, action: str, view: str) -> list[Path]:
    folder = frames_root / asset_id / action / view
    frames = sorted(folder.glob("[0-9][0-9].png"))
    if not frames:
        raise ForgeError(f"No frames found in {folder}; run assetforge animate first")
    return frames


def authored_baseline(pack: dict[str, Any]) -> float:
    """Ankle ground line (0..1 canvas fraction) from the standing action's keyframes."""
    return standing_extent(pack)[1]


def pack_sheets(
    workspace: Path,
    asset_id: str,
    actions: list[str] | None = None,
    pack_id: str = "humanoid_side_east",
    report_path: Path | None = None,
    contact_sheet_path: Path | None = None,
) -> dict[str, Any]:
    asset = load_asset(workspace, asset_id)
    pack = load_pack(workspace, pack_id)
    view = pack.get("view", "east")
    canvas_width, canvas_height = [int(value) for value in pack.get("canvas", [768, 768])]
    production = asset.get("production", {})
    cell_width, cell_height = [int(value) for value in production.get("cell_size", [256, 256])]
    height_fraction = float(production.get("target_height_fraction", 0.68))
    root = asset_root(workspace, asset_id)
    frames_root = root / "renders" / "frames"

    if actions is None:
        actions = [
            name for name in pack.get("actions", {})
            if (frames_root / asset_id / name / view).is_dir()
        ]
    if not actions:
        raise ForgeError(f"No rendered actions found under {frames_root / asset_id}")

    loaded: dict[str, list[Image.Image]] = {}
    for action in actions:
        loaded[action] = [Image.open(path).convert("RGBA") for path in _load_frames(frames_root, asset_id, action, view)]

    # One window over every frame of every action; one transform for all of them.
    # The alpha union is clamped to the authored pose extents so stray artifacts
    # cannot inflate the window.
    boxes = [_alpha_box(frame) for frames in loaded.values() for frame in frames]
    pose_left, pose_top, pose_right, pose_bottom = _pose_window(pack, list(loaded), (canvas_width, canvas_height))
    window_left = max(pose_left, min(min(box[0] for box in boxes) - WINDOW_PADDING, pose_right - 1))
    window_top = max(pose_top, min(min(box[1] for box in boxes) - WINDOW_PADDING, pose_bottom - 1))
    window_right = min(pose_right, max(max(box[2] for box in boxes) + WINDOW_PADDING, window_left + 1))
    window_bottom = min(pose_bottom, max(max(box[3] for box in boxes) + WINDOW_PADDING, window_top + 1))
    window_width = window_right - window_left
    baseline_canvas = authored_baseline(pack) * canvas_height

    baseline_cell = cell_height - max(3, round(cell_height * 0.035))
    # The height budget applies to the STANDING figure (idle), not to the whole motion
    # window — a raised weapon must not shrink the character. Guards below keep raised
    # or lying extremes from clipping the cell.
    standing_frames = loaded.get("idle") or next(iter(loaded.values()))
    standing_boxes = [_alpha_box(frame) for frame in standing_frames]
    standing_height = max(1.0, baseline_canvas - min(box[1] for box in standing_boxes))
    reach_above = max(1.0, baseline_canvas - window_top)
    below_baseline = max(0.0, window_bottom - baseline_canvas)
    scale = min(
        (cell_width - 2) / window_width,
        (cell_height * height_fraction) / standing_height,
        (baseline_cell - 2) / reach_above,
        ((cell_height - baseline_cell - 1) / below_baseline) if below_baseline > 0 else float("inf"),
    )
    scaled_width = max(1, round(window_width * scale))
    offset_x = (cell_width - scaled_width) // 2
    offset_y = baseline_cell - round((baseline_canvas - window_top) * scale)

    sheets_folder = root / "sheets"
    sheets_folder.mkdir(parents=True, exist_ok=True)
    pattern = "assets/{asset}/sheets/{asset}_{{animation}}_{{direction}}_sheet.png".format(asset=asset_id)

    for action, frames in loaded.items():
        sheet = Image.new("RGBA", (cell_width * len(frames), cell_height), (0, 0, 0, 0))
        for index, frame in enumerate(frames):
            cropped = frame.crop((window_left, window_top, window_right, window_bottom))
            scaled_height = max(1, round(cropped.height * scale))
            resized = cropped.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            sheet.alpha_composite(resized, (index * cell_width + offset_x, offset_y))
        output = workspace.resolve() / pattern.format(animation=action, direction=view)
        sheet.save(output, "PNG", optimize=True)

    reports_folder = root / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)
    config = {
        "id": asset_id,
        "cell_size": [cell_width, cell_height],
        "directions": [view],
        "animations": {action: {"frames": len(loaded[action])} for action in loaded},
        "output_path_pattern": pattern,
        # Generated animation frames move inside a fixed window; feet vary a few pixels.
        "baseline_tolerance": max(2, round(cell_height * 0.03)),
        "min_occupancy": 0.015,
    }
    config_path = reports_folder / f"{asset_id}_sheet_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    report_path = report_path or (reports_folder / f"{asset_id}_sheet_qa.json")
    contact_sheet_path = contact_sheet_path or (reports_folder / f"{asset_id}_contact_sheet.png")
    report = validate_sheets(config_path, workspace.resolve(), report_path, contact_sheet_path)

    manifest = {
        "schema_version": 2,
        "asset_id": asset_id,
        "display_name": asset.get("display_name", asset_id),
        "default_facing": view,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "pixels_per_unit": cell_height,
        "pivot_x": 0.5,
        "pivot_y": round((cell_height - baseline_cell) / cell_height, 4),
        "prefab_scale": 1.0,
        "health_bar_y": 0.58,
        "production": {
            "eligible": False,
            "source_kind": "diffusion_frame_sequence",
            "identity_lock": "reference_conditioning_only",
            "blockers": [
                "no_single_master_identity_lock",
                "missing_genuine_north_south_west_views",
                "requires_human_motion_review",
            ],
        },
        "actions": [
            {
                "name": action,
                "direction": view,
                "frames": len(loaded[action]),
                "fps": ACTION_FPS.get(action, 10.0),
                "loop": action in LOOPING_ACTIONS,
                "sheet": f"{asset_id}_{action}_{view}_sheet.png",
            }
            for action in sorted(loaded)
        ],
    }
    write_json(sheets_folder / "unit_manifest.json", manifest)
    report["unit_manifest"] = str(sheets_folder / "unit_manifest.json")

    preview_rows = len(loaded)
    preview_columns = max(len(frames) for frames in loaded.values())
    preview = Image.new(
        "RGBA", (preview_columns * GAMEPLAY_CELL, preview_rows * GAMEPLAY_CELL), (18, 24, 28, 255)
    )
    for row, (action, frames) in enumerate(sorted(loaded.items())):
        relative = pattern.format(animation=action, direction=view)
        with Image.open(workspace.resolve() / relative) as sheet_image:
            sheet = sheet_image.convert("RGBA")
        for column in range(len(frames)):
            cell = sheet.crop((column * cell_width, 0, (column + 1) * cell_width, cell_height))
            cell = cell.resize((GAMEPLAY_CELL, GAMEPLAY_CELL), Image.Resampling.LANCZOS)
            preview.alpha_composite(cell, (column * GAMEPLAY_CELL, row * GAMEPLAY_CELL))
    preview_path = reports_folder / f"{asset_id}_gameplay_preview.png"
    preview.save(preview_path, "PNG", optimize=True)
    report["gameplay_preview"] = str(preview_path)
    return report
