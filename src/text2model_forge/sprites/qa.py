from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from .core import ForgeError, sha256_file, utc_now, write_json


def _absolute(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _alpha_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    return image.getchannel("A").getbbox()


def _difference_ratio(first: Image.Image, second: Image.Image) -> float:
    difference = ImageChops.difference(first, second)
    histogram = difference.histogram()
    changed = sum(value for index, value in enumerate(histogram) if index % 256 != 0)
    maximum = first.width * first.height * 4
    return changed / maximum if maximum else 0.0


def validate_sheets(config_path: Path, repo_root: Path, report_path: Path, contact_sheet_path: Path | None = None) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForgeError(f"Could not load sheet config {config_path}: {error}") from error
    cell_width, cell_height = [int(value) for value in config["cell_size"]]
    baseline_tolerance = int(config.get("baseline_tolerance", 1))
    min_occupancy = float(config.get("min_occupancy", 0.035))
    min_motion = float(config.get("min_motion", 0.01))
    failures: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    previews: list[tuple[str, Image.Image]] = []
    directional_first_frames: dict[tuple[str, str], Image.Image] = {}

    for animation, animation_config in config["animations"].items():
        frame_count = int(animation_config["frames"])
        action_baseline_tolerance = int(animation_config.get("baseline_tolerance", baseline_tolerance))
        for direction in config["directions"]:
            relative = config["output_path_pattern"].format(animation=animation, direction=direction)
            path = _absolute(repo_root, relative)
            if not path.is_file():
                failures.append(f"Missing sheet: {relative}")
                continue
            with Image.open(path) as source:
                image = source.convert("RGBA")
            expected = (cell_width * frame_count, cell_height)
            if image.size != expected:
                failures.append(f"Wrong dimensions for {relative}: {image.size}, expected {expected}")
                continue
            baselines = []
            occupancy = []
            alpha_bounds = []
            cells = []
            for index in range(frame_count):
                cell = image.crop((index * cell_width, 0, (index + 1) * cell_width, cell_height))
                cells.append(cell)
                box = _alpha_box(cell)
                if box is None:
                    failures.append(f"Transparent frame {index} in {relative}")
                    continue
                if box[0] <= 1 or box[2] >= cell_width - 1 or box[1] <= 1 or box[3] >= cell_height - 1:
                    failures.append(f"Clipped frame {index} in {relative}: alpha bounds {box}")
                baselines.append(box[3])
                alpha_bounds.append(box)
                occupancy.append(((box[2] - box[0]) * (box[3] - box[1])) / (cell_width * cell_height))
            if baselines and max(baselines) - min(baselines) > action_baseline_tolerance:
                failures.append(f"Baseline drift in {relative}: {baselines}")
            if occupancy and (min(occupancy) < min_occupancy or max(occupancy) > 0.82):
                failures.append(f"Unsafe frame occupancy in {relative}: {[round(value, 3) for value in occupancy]}")
            differences = [_difference_ratio(cells[index - 1], cells[index]) for index in range(1, len(cells))]
            if differences and min(differences) == 0:
                failures.append(f"Exact duplicate adjacent frames in {relative}")
            elif differences and max(differences) < min_motion:
                failures.append(f"Action is visually static in {relative}: {[round(value, 6) for value in differences]}")
            elif differences and min(differences) < min_motion * 0.12:
                warnings.append(f"Nearly static adjacent frames in {relative}: {[round(value, 6) for value in differences]}")
            if animation in ("attack", "heavy_attack", "shoot") and cells:
                contact_index = int(animation_config.get("phases", {}).get("contact_frame", len(cells) // 2))
                contact_index = max(0, min(contact_index, len(cells) - 1))
                contact_change = _difference_ratio(cells[0], cells[contact_index])
                if contact_change < min_motion * 2:
                    failures.append(f"Unreadable contact pose in {relative}: difference={contact_change:.6f}")
            if animation == "hit" and len(cells) >= 3:
                hit_change = _difference_ratio(cells[0], cells[len(cells) // 2])
                if hit_change < min_motion * 2:
                    failures.append(f"Unreadable hit reaction in {relative}: difference={hit_change:.6f}")
            if animation == "death" and len(alpha_bounds) >= 2:
                first = alpha_bounds[0]
                final = alpha_bounds[-1]
                first_height = max(1, first[3] - first[1])
                final_height = final[3] - final[1]
                first_width = max(1, first[2] - first[0])
                final_width = final[2] - final[0]
                if final_height / first_height > 0.86 and final_width / first_width < 1.12:
                    failures.append(
                        f"Death does not finish collapsed in {relative}: "
                        f"height ratio={final_height / first_height:.3f}, width ratio={final_width / first_width:.3f}"
                    )
            if bool(animation_config.get("loop", False)) and len(cells) >= 3:
                seam = _difference_ratio(cells[-1], cells[0])
                typical = max(differences) if differences else 0.0
                if typical > 0 and seam > typical * 1.8:
                    warnings.append(f"Loop seam may pop in {relative}: seam={seam:.4f}, max adjacent={typical:.4f}")
            if cells:
                directional_first_frames[(animation, direction)] = cells[0]
                previews.append((f"{animation} / {direction}", cells[0].copy()))
            records.append({
                "path": relative,
                "sha256": sha256_file(path),
                "animation": animation,
                "direction": direction,
                "frames": frame_count,
                "baselines": baselines,
                "occupancy": occupancy,
                "alpha_bounds": alpha_bounds,
                "adjacent_difference": differences,
            })

    for animation in config["animations"]:
        north = directional_first_frames.get((animation, "north"))
        south = directional_first_frames.get((animation, "south"))
        if north is not None and south is not None and _difference_ratio(north, south) == 0:
            failures.append(f"North and south are identical for {animation}; genuine front/back views are required")
        east = directional_first_frames.get((animation, "east"))
        west = directional_first_frames.get((animation, "west"))
        if east is not None and west is not None and _difference_ratio(east, west) == 0:
            failures.append(f"East and west are identical for {animation}")

    if contact_sheet_path and previews:
        columns = min(4, len(previews))
        label_height = 24
        rows = (len(previews) + columns - 1) // columns
        contact = Image.new("RGBA", (columns * cell_width, rows * (cell_height + label_height)), (18, 24, 28, 255))
        draw = ImageDraw.Draw(contact)
        for index, (label, cell) in enumerate(previews):
            x = (index % columns) * cell_width
            y = (index // columns) * (cell_height + label_height)
            contact.alpha_composite(cell, (x, y))
            draw.text((x + 6, y + cell_height + 4), label, fill=(235, 226, 207, 255))
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        contact.save(contact_sheet_path, "PNG", optimize=True)

    report = {
        "schema_version": 1,
        "asset_id": config["id"],
        "checked_utc": utc_now(),
        "passed": not failures,
        "sheets": records,
        "contact_sheet": str(contact_sheet_path.resolve()) if contact_sheet_path else None,
        "failures": failures,
        "warnings": warnings,
    }
    write_json(report_path.resolve(), report)
    return report
