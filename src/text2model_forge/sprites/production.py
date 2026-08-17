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
    texture_master = config.get("texture_master")
    if isinstance(texture_master, dict) and texture_master.get("enabled"):
        if not str(texture_master.get("baked_source", "")).strip():
            failures.append("texture_master requires baked_source")
    morphology_value = str(config.get("morphology_profile", "")).strip()
    if morphology_value:
        morphology_path = _absolute(repo_root, morphology_value)
        if not morphology_path.is_file() or morphology_path.suffix.lower() != ".json":
            failures.append("morphology_profile must name an existing tracked JSON file")
    overpaint = config.get("overpaint")
    if isinstance(overpaint, dict) and overpaint.get("enabled"):
        if not config.get("depth_pass") or overpaint.get("control") != "depth":
            failures.append("production overpaint requires the Blender depth pass and depth control")
        if not overpaint.get("anchor"):
            failures.append("production overpaint requires a persistent style anchor")
        if not overpaint.get("protect_equipment", True):
            failures.append("production overpaint must protect authored equipment pixels")
        if not overpaint.get("fail_closed", True):
            failures.append("production overpaint must fail closed")
        if not config.get("equipment_objects"):
            failures.append("production overpaint requires explicit equipment_objects")
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


def _head_crop(image: Image.Image) -> Image.Image:
    """Crop the upper anatomy at review scale; diffusion cannot hide a human skull here."""
    bounds = image.getchannel("A").getbbox()
    if not bounds:
        return image
    left, top, right, bottom = bounds
    subject_height = max(1, bottom - top)
    head_bottom = min(bottom, top + max(12, round(subject_height * 0.38)))
    head_height = head_bottom - top
    center = (left + right) // 2
    half_width = max(head_height, (right - left) // 2)
    padding = 5
    return image.crop((
        max(0, center - half_width - padding),
        max(0, top - padding),
        min(image.width, center + half_width + padding),
        min(image.height, head_bottom + padding),
    ))


def _silhouette(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    value = Image.new("RGBA", image.size, (39, 55, 48, 0))
    value.putalpha(alpha)
    return value


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

    def review_frame(action_name: str, direction: str, frame_index: int) -> Image.Image:
        """Prefer the high-resolution Blender render for close review, with packed
        cells as the portable/test fallback.  Approval should expose real face detail
        instead of enlarging a 256 px shipping cell after the fact."""
        raw = (
            repo_root / "artifacts" / "text2model-forge-work" / str(config["id"]) /
            "frames" / str(config["id"]) / action_name / direction / f"{frame_index:02d}.png"
        )
        if raw.is_file():
            with Image.open(raw) as source:
                return source.convert("RGBA")
        frames = cache[(action_name, direction)]
        return frames[min(max(frame_index, 0), len(frames) - 1)]

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

    morphology_path = None
    morphology = None
    if str(config.get("morphology_profile", "")).strip():
        morphology_path = _absolute(repo_root, str(config["morphology_profile"]))
        morphology = _load_json(morphology_path, "morphology profile")

    acceptance_board = review_root / f"{config['id']}_acceptance_board.png"
    board_height = 1040 if morphology else 760
    board = Image.new("RGB", (1280, board_height), (13, 19, 23))
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, board.width, 70), fill=(29, 37, 43))
    draw.text((28, 20), f"{config['display_name']}  |  visual acceptance board", fill=(241, 220, 180))
    draw.text((28, 45), "identity directions  /  authored attack phases  /  gameplay scale", fill=(167, 181, 186))

    def board_cell(source: Image.Image, x: int, y: int, label: str, size: int = 210) -> None:
        draw.rounded_rectangle((x, y, x + size, y + size + 28), radius=8, fill=(24, 32, 37), outline=(62, 76, 80))
        subject = source
        bounds = source.getchannel("A").getbbox()
        if bounds:
            padding = 6
            subject = source.crop((
                max(0, bounds[0] - padding),
                max(0, bounds[1] - padding),
                min(source.width, bounds[2] + padding),
                min(source.height, bounds[3] + padding),
            ))
        fitted = _fit_rgba(subject, (size - 18, size - 18))
        board.paste(fitted, (x + 9, y + 9), fitted)
        draw.text((x + 9, y + size + 7), label, fill=(205, 211, 210))

    idle_name = "idle" if "idle" in config["animations"] else next(iter(config["animations"]))
    for direction_index, direction in enumerate(REQUIRED_DIRECTIONS):
        board_cell(review_frame(idle_name, direction, 0), 28 + direction_index * 244, 94, f"{idle_name} / {direction}")

    attack_name = "attack" if "attack" in config["animations"] else idle_name
    attack_action = config["animations"][attack_name]
    phases = attack_action.get("phases", {})
    phase_items = [
        ("anticipation", int(phases.get("anticipation_frame", 0))),
        ("contact", int(phases.get("contact_frame", max(0, len(cache[(attack_name, default_direction)]) // 2)))),
        ("recovery", int(phases.get("recovery_frame", len(cache[(attack_name, default_direction)]) - 1))),
    ]
    attack_cells = cache[(attack_name, default_direction)]
    attack_y = 658 if morphology else 376
    if morphology:
        south = review_frame(idle_name, "south", 0)
        east = review_frame(idle_name, "east", 0)
        board_cell(_head_crop(south), 28, 376, "face proof / south")
        board_cell(_head_crop(east), 272, 376, "snout + jaw proof / east")
        board_cell(_silhouette(east), 516, 376, "species silhouette / east")
        board_cell(east, 760, 376, "posture proof / east")
        draw.rounded_rectangle((1004, 376, 1252, 614), radius=8, fill=(20, 28, 32), outline=(62, 76, 80))
        silhouette = morphology.get("silhouette_contract", {})
        draw.text((1020, 394), "Creature DNA contract", fill=(224, 207, 176))
        draw.text((1020, 426), str(morphology.get("species", "creature")), fill=(198, 207, 211))
        draw.text((1020, 452), str(morphology.get("family", "unknown family")), fill=(166, 180, 184))
        draw.text((1020, 490), "Required at a glance:", fill=(198, 207, 211))
        draw.text((1020, 516), "ears / brow / snout", fill=(166, 180, 184))
        draw.text((1020, 540), "jaw / tusks / hunch", fill=(166, 180, 184))
        draw.text((1020, 574), f"landmarks: {len(silhouette.get('required_landmarks', []))}", fill=(218, 154, 112))
    for phase_index, (phase_name, frame_index) in enumerate(phase_items):
        frame_index = min(max(frame_index, 0), len(attack_cells) - 1)
        board_cell(review_frame(attack_name, default_direction, frame_index), 28 + phase_index * 244, attack_y, f"{attack_name} {phase_name} / f{frame_index}")

    preview_x = 784
    preview_y = attack_y
    draw.rounded_rectangle((preview_x, preview_y, 1252, preview_y + 340), radius=8, fill=(20, 28, 32), outline=(62, 76, 80))
    draw.text((preview_x + 18, preview_y + 16), "96 px gameplay readability", fill=(224, 207, 176))
    preview_frames = [
        ("idle", review_frame(idle_name, default_direction, 0)),
        ("contact", review_frame(attack_name, default_direction, min(max(int(phases.get("contact_frame", 0)), 0), len(attack_cells) - 1))),
    ]
    for preview_index, (label, source) in enumerate(preview_frames):
        sprite = _fit_rgba(source, (96, 96))
        x = preview_x + 38 + preview_index * 205
        y = preview_y + 70
        board.paste(sprite, (x, y), sprite)
        draw.text((x + 22, y + 108), label, fill=(194, 202, 203))
    approval_label = "anatomy, role, weapon, and contact pose" if morphology else "role, weapon, shield/emblem, and contact pose"
    draw.text((preview_x + 18, preview_y + 218), f"Approve only if {approval_label}", fill=(166, 180, 184))
    draw.text((preview_x + 18, preview_y + 243), "and four-view identity all read clearly.", fill=(166, 180, 184))
    draw.text((preview_x + 18, preview_y + 289), "This board is review evidence, not automatic approval.", fill=(218, 154, 112))
    board.save(acceptance_board, "PNG")

    result = {
        "all_actions": str(all_actions.resolve()),
        "gameplay_transition": str(transition_reel.resolve()),
        "acceptance_board": str(acceptance_board.resolve()),
        "default_direction": default_direction,
        "sequence": sequence,
        "morphology_profile": str(morphology_path.resolve()) if morphology_path else None,
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
    qa_sheets = qa_report.get("sheets")
    if not isinstance(qa_sheets, list):
        failures.append("sheet QA report has no hash-bound sheet inventory")
    else:
        inventory = {
            str(entry.get("path", "")).replace("\\", "/"): str(entry.get("sha256", ""))
            for entry in qa_sheets if isinstance(entry, dict)
        }
        for action_name in config.get("animations", {}):
            for direction in config.get("directions", []):
                sheet = _absolute(
                    repo_root,
                    config["output_path_pattern"].format(animation=action_name, direction=direction),
                )
                relative = _relative(repo_root, sheet)
                recorded = inventory.get(relative)
                if not recorded:
                    failures.append(f"sheet QA report does not cover current sheet: {relative}")
                elif not sheet.is_file() or recorded != sha256_file(sheet):
                    failures.append(f"sheet changed after QA: {relative}")
    overpaint_report_path: Path | None = None
    overpaint_report: dict[str, Any] | None = None
    bake_report_path: Path | None = None
    bake_report: dict[str, Any] | None = None
    texture_master = config.get("texture_master")
    texture_baked = isinstance(texture_master, dict) and texture_master.get("enabled")
    overpaint = config.get("overpaint")
    if texture_baked:
        # Baked lane: paint lives in the master's textures, frames are plain renders,
        # so per-frame overpaint provenance does not exist.  The gate instead proves
        # that the baked master the sheets were rendered from is the one the recorded
        # bake produced, with registered models and the persistent anchor.
        bake_report_path = (
            repo_root / "artifacts" / "text2model-forge-work" / str(config.get("id")) /
            "bake" / "bake-provenance.json"
        )
        if not bake_report_path.is_file():
            failures.append("texture-master unit has no bake provenance report")
        else:
            bake_report = _load_json(bake_report_path, "bake provenance")
            baked_source = _absolute(repo_root, str(texture_master.get("baked_source", "")))
            if bake_report.get("unit") != config.get("id"):
                failures.append("bake provenance belongs to a different character")
            if not baked_source.is_file():
                failures.append(f"baked master is missing: {baked_source}")
            elif bake_report.get("baked_sha256") != sha256_file(baked_source):
                failures.append("baked master does not match the recorded bake provenance hash")
            if not bake_report.get("checkpoint_sha256") or not bake_report.get("controlnet_sha256"):
                failures.append("bake provenance is missing registered model hashes")
            if not bake_report.get("anchor_sha256"):
                failures.append("bake provenance is missing the persistent anchor hash")
    elif isinstance(overpaint, dict) and overpaint.get("enabled"):
        overpaint_report_path = (
            repo_root / "artifacts" / "text2model-forge-work" / str(config.get("id")) /
            "frames" / f"{config.get('id')}-overpaint.json"
        )
        if not overpaint_report_path.is_file():
            failures.append("enabled production overpaint has no provenance report")
        else:
            overpaint_report = _load_json(overpaint_report_path, "overpaint provenance")
            if overpaint_report.get("unit") != config.get("id"):
                failures.append("overpaint provenance belongs to a different character")
            if int(overpaint_report.get("schema_version", 0)) < 3:
                failures.append("overpaint provenance predates protected-part production gates")
            if not overpaint_report.get("passed"):
                failures.append("overpaint provenance is not passing")
            if not overpaint_report.get("depth_control"):
                failures.append("overpaint provenance does not prove depth control")
            if not overpaint_report.get("protected_equipment"):
                failures.append("overpaint provenance does not prove protected equipment")
            if not overpaint_report.get("transactional_promotion"):
                failures.append("overpaint provenance does not prove transactional promotion")
            if not overpaint_report.get("checkpoint_sha256") or not overpaint_report.get("controlnet_sha256"):
                failures.append("overpaint provenance is missing registered model hashes")
            if not overpaint_report.get("anchor_sha256"):
                failures.append("overpaint provenance is missing the persistent anchor hash")
            expected_protected_frames = sum(
                int(action.get("frames", 0)) for action in config.get("animations", {}).values()
            ) * len(config.get("directions", []))
            if int(overpaint_report.get("protected_frames", -1)) != expected_protected_frames:
                failures.append(
                    "overpaint provenance does not cover every configured frame "
                    f"({overpaint_report.get('protected_frames')} != {expected_protected_frames})"
                )
            if overpaint_report.get("equipment_integrity_failures"):
                failures.append("overpaint provenance contains equipment-integrity failures")
    if failures:
        raise ForgeError("Production package rejected:\n- " + "\n- ".join(failures))

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = _absolute(repo_root, config["source"])
    morphology_validation: dict[str, Any] | None = None
    morphology_profile_value = str(config.get("morphology_profile", "")).strip()
    if morphology_profile_value:
        from .morphology import validate_morphology_build
        morphology_profile_path = _absolute(repo_root, morphology_profile_value)
        morphology_validation = validate_morphology_build(
            morphology_profile_path,
            Path(str(source) + ".morphology-report.json"),
            source,
            repo_root,
        )
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
            "identity_lock": (
                "single_baked_master_sha256+structural_paint"
                if bake_report is not None
                else "single_master_sha256+protected_equipment"
                if overpaint_report is not None else "single_master_sha256"
            ),
            "master": _relative(repo_root, source),
            "master_sha256": sha256_file(source),
            "master_license": config.get("master_license", "project-original"),
            "config_sha256": sha256_json(config),
            "qa_report": _relative(repo_root, qa_report_path),
            "qa_report_sha256": sha256_file(qa_report_path),
            "overpaint_report": (
                _relative(repo_root, overpaint_report_path)
                if overpaint_report_path is not None else None
            ),
            "overpaint_report_sha256": (
                sha256_file(overpaint_report_path)
                if overpaint_report_path is not None and overpaint_report_path.is_file() else None
            ),
            "bake_report": (
                _relative(repo_root, bake_report_path)
                if bake_report_path is not None else None
            ),
            "bake_report_sha256": (
                sha256_file(bake_report_path)
                if bake_report_path is not None and bake_report_path.is_file() else None
            ),
            "baked_master_sha256": (bake_report or {}).get("baked_sha256"),
            "packaged_utc": utc_now(),
            "morphology_profile": (
                _relative(repo_root, Path(morphology_validation["profile"]))
                if morphology_validation else None
            ),
            "morphology_profile_sha256": (
                morphology_validation.get("profile_sha256") if morphology_validation else None
            ),
            "morphology_family": (
                morphology_validation.get("family") if morphology_validation else None
            ),
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
