"""Pack directional Blender frames into candidate sprite sheets and a review manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat


ACTIONS = {"idle": (8, True), "walk": (8, True), "attack": (10, False), "death": (10, False)}
DIRECTIONS = ("north", "south", "east", "west")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--source-id", default="text2model_short_biped_candidate")
    parser.add_argument("--asset-id", default="short_biped_motion_candidate")
    parser.add_argument("--display-name", default="Short-biped motion candidate")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visible_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    return image.getchannel("A").getbbox()


def _review_sheet(output: Path) -> Path:
    width = 1440
    label_width = 130
    title_height = 52
    rows: list[tuple[str, Image.Image]] = []
    for action in ACTIONS:
        for direction in ("south", "east"):
            with Image.open(output / f"{action}_{direction}.png").convert("RGBA") as source:
                background = Image.new("RGBA", source.size, (14, 17, 21, 255))
                background.alpha_composite(source)
                ratio = width / background.width
                rows.append(
                    (
                        f"{action} {direction}",
                        background.resize((width, max(1, int(background.height * ratio)))),
                    )
                )
    canvas = Image.new(
        "RGB",
        (width + label_width, title_height + sum(image.height for _, image in rows)),
        "#0d1014",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 17), "Text2Model Quaternius retarget - gameplay sprite proof", fill="white")
    y = title_height
    for label, image in rows:
        draw.text((14, y + 16), label.upper(), fill="white")
        canvas.paste(image.convert("RGB"), (label_width, y))
        y += image.height
    path = output / "sprite_review.png"
    canvas.save(path)
    return path


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    frames_root = args.frames_root.resolve() / args.source_id
    output = args.output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    failures: list[str] = []
    for action, (expected_frames, loop) in ACTIONS.items():
        for direction in DIRECTIONS:
            folder = frames_root / action / direction
            paths = sorted(folder.glob("*.png"))
            if len(paths) != expected_frames:
                failures.append(f"{action}/{direction}:expected_{expected_frames}_frames_got_{len(paths)}")
                continue
            with Image.open(paths[0]).convert("RGBA") as first:
                cell = first.size
            sheet = Image.new("RGBA", (cell[0] * len(paths), cell[1]), (0, 0, 0, 0))
            visible_bounds: list[tuple[int, int, int, int]] = []
            visible_luminance: list[float] = []
            for index, path in enumerate(paths):
                with Image.open(path).convert("RGBA") as frame:
                    if frame.size != cell:
                        failures.append(f"{action}/{direction}:inconsistent_frame_size")
                        continue
                    bounds = _visible_bbox(frame)
                    if bounds is None:
                        failures.append(f"{action}/{direction}:{path.name}:empty_alpha")
                    else:
                        visible_bounds.append(bounds)
                        channel_means = ImageStat.Stat(frame.convert("RGB"), frame.getchannel("A")).mean
                        visible_luminance.append(sum(channel_means) / len(channel_means))
                        if (
                            bounds[0] <= 1
                            or bounds[1] <= 1
                            or bounds[2] >= frame.width - 1
                            or bounds[3] >= frame.height - 1
                        ):
                            failures.append(f"{action}/{direction}:{path.name}:edge_clipping")
                    sheet.alpha_composite(frame, (index * cell[0], 0))
            name = f"{action}_{direction}.png"
            path = output / name
            sheet.save(path)
            width_fractions = [
                (bounds[2] - bounds[0]) / cell[0] for bounds in visible_bounds
            ]
            height_fractions = [
                (bounds[3] - bounds[1]) / cell[1] for bounds in visible_bounds
            ]
            if visible_bounds and min(height_fractions) < 0.08:
                failures.append(f"{action}/{direction}:silhouette_too_small")
            if visible_luminance and min(visible_luminance) < 28.0:
                failures.append(
                    f"{action}/{direction}:visible_surface_too_dark:"
                    f"{min(visible_luminance):.2f}_lt_28.00"
                )
            records.append(
                {
                    "name": action,
                    "direction": direction,
                    "frames": len(paths),
                    "fps": 10.0,
                    "loop": loop,
                    "sheet": name,
                    "sha256": _sha256(path),
                    "minimum_visible_width_fraction": min(width_fractions, default=0.0),
                    "minimum_visible_height_fraction": min(height_fractions, default=0.0),
                    "maximum_visible_width_fraction": max(width_fractions, default=0.0),
                    "maximum_visible_height_fraction": max(height_fractions, default=0.0),
                    "minimum_visible_luminance_8bit": min(visible_luminance, default=0.0),
                    "mean_visible_luminance_8bit": (
                        sum(visible_luminance) / len(visible_luminance)
                        if visible_luminance
                        else 0.0
                    ),
                }
            )
    review_sheet = _review_sheet(output)
    manifest = {
        "schema_version": 1,
        "quality_gate_version": 2,
        "asset_id": args.asset_id,
        "display_name": args.display_name,
        "status": "human_review_candidate",
        "default_facing": "south",
        "cell_width": 256,
        "cell_height": 256,
        "pixels_per_unit": 256.0,
        "pivot_x": 0.5,
        "pivot_y": 0.18,
        "source_master": str(args.master.resolve()),
        "source_master_sha256": _sha256(args.master.resolve()),
        "actions": records,
        "automatic_gate_passed": not failures and len(records) == len(ACTIONS) * len(DIRECTIONS),
        "hard_failures": failures,
        "review_sheet": review_sheet.name,
        "review_sheet_sha256": _sha256(review_sheet),
        "human_approval_required": True,
        "human_approved": False,
    }
    (output / "candidate_unit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("sprite packaging failed: " + ", ".join(failures))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
