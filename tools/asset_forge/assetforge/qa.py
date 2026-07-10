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
    failures: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    previews: list[tuple[str, Image.Image]] = []
    directional_first_frames: dict[tuple[str, str], Image.Image] = {}

    for animation, animation_config in config["animations"].items():
        frame_count = int(animation_config["frames"])
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
            cells = []
            for index in range(frame_count):
                cell = image.crop((index * cell_width, 0, (index + 1) * cell_width, cell_height))
                cells.append(cell)
                box = _alpha_box(cell)
                if box is None:
                    failures.append(f"Transparent frame {index} in {relative}")
                    continue
                if box[0] <= 1 or box[2] >= cell_width - 1 or box[1] <= 1:
                    failures.append(f"Clipped frame {index} in {relative}: alpha bounds {box}")
                baselines.append(box[3])
                occupancy.append(((box[2] - box[0]) * (box[3] - box[1])) / (cell_width * cell_height))
            if baselines and max(baselines) - min(baselines) > 1:
                failures.append(f"Baseline drift in {relative}: {baselines}")
            if occupancy and (min(occupancy) < 0.035 or max(occupancy) > 0.82):
                failures.append(f"Unsafe frame occupancy in {relative}: {[round(value, 3) for value in occupancy]}")
            differences = [_difference_ratio(cells[index - 1], cells[index]) for index in range(1, len(cells))]
            if differences and min(differences) == 0:
                failures.append(f"Exact duplicate adjacent frames in {relative}")
            elif differences and min(differences) < 0.0005:
                warnings.append(f"Nearly static adjacent frames in {relative}: {[round(value, 6) for value in differences]}")
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
