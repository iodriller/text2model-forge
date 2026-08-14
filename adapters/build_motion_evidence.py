"""Build human-readable motion key-pose sheets from Blender evidence renders."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from PIL import Image, ImageDraw


PATTERN = re.compile(r"^motion_([a-z0-9_]+)_(\d{3})_(front|left|right|back)\.png$")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=("idle", "walk", "attack", "death"))
    parser.add_argument("--title", default="Darkness motion evidence")
    return parser.parse_args(argv)


def _discover(root: Path, clips: list[str]) -> dict[tuple[str, str], list[tuple[int, Path]]]:
    result: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for path in root.glob("motion_*.png"):
        match = PATTERN.match(path.name)
        if match is None:
            continue
        clip, frame, view = match.groups()
        if clip in clips and view in {"front", "left"}:
            result.setdefault((clip, view), []).append((int(frame), path))
    for frames in result.values():
        frames.sort()
    missing = [f"{clip}/{view}" for clip in clips for view in ("front", "left") if not result.get((clip, view))]
    if missing:
        raise FileNotFoundError("missing key-pose evidence: " + ", ".join(missing))
    return result


def _strip(frames: list[tuple[int, Path]], output: Path, *, clip: str, view: str) -> Path:
    cell = 320
    header = 46
    canvas = Image.new("RGB", (cell * len(frames), cell + header), "#14171c")
    draw = ImageDraw.Draw(canvas)
    for index, (frame, path) in enumerate(frames):
        with Image.open(path).convert("RGB") as image:
            image.thumbnail((cell, cell))
            canvas.paste(image, (index * cell + (cell - image.width) // 2, header))
        draw.text((index * cell + 10, 15), f"{clip.upper()}  frame {frame}  {view}", fill="white")
    canvas.save(output)
    return output


def _master(rows: list[tuple[str, Path]], output: Path, *, title: str) -> Path:
    width = 1500
    label = 118
    title_height = 54
    images: list[tuple[str, Image.Image]] = []
    for name, path in rows:
        with Image.open(path).convert("RGB") as source:
            ratio = width / source.width
            images.append((name, source.resize((width, int(source.height * ratio)))))
    canvas = Image.new(
        "RGB",
        (width + label, title_height + sum(image.height for _, image in images)),
        "#0d1014",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 18), title, fill="white")
    y = title_height
    for name, image in images:
        draw.text((14, y + 18), name.upper(), fill="white")
        canvas.paste(image, (label, y))
        y += image.height
    canvas.save(output)
    return output


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    source = args.input_directory.resolve()
    output = args.output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    discovered = _discover(source, args.clips)
    front_rows: list[tuple[str, Path]] = []
    for clip in args.clips:
        for view in ("front", "left"):
            path = _strip(
                discovered[(clip, view)],
                output / f"{clip}_{view}_keyposes.png",
                clip=clip,
                view=view,
            )
            if view == "front":
                front_rows.append((clip, path))
    master = _master(front_rows, output / "all_motion_front_keyposes.png", title=args.title)
    print(master)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
