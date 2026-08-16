"""Compose canonical-master renders and mechanical evidence into one review board."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .core import ForgeError


def compose_canonical_review(frames_root: Path, report_path: Path, output_path: Path) -> dict[str, Any]:
    frames = sorted(Path(frames_root).glob("*.png"))
    if not frames:
        raise ForgeError(f"No canonical review frames found under {frames_root}")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    cell = 320
    margin = 24
    header = 118
    columns = min(4, len(frames))
    rows = (len(frames) + columns - 1) // columns
    board = Image.new("RGB", (margin * 2 + columns * cell, header + margin + rows * (cell + 34)), (18, 21, 27))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    draw.text((margin, 18), f"VettedMesh Genesis — {report.get('family')} canonical master", fill=(235, 225, 205), font=font)
    mesh = report.get("mesh") or {}
    motion = report.get("motion_audit") or {}
    weights = report.get("weights") or {}
    evidence = (
        f"manifold components={mesh.get('connected_components')}  boundary={mesh.get('boundary_edges')}  "
        f"UV={mesh.get('has_uv')}  unweighted={weights.get('unweighted_vertices')}\n"
        f"actions={len(report.get('actions') or [])}  foot slip max={max((motion.get('foot_slip_m') or {'none': 0}).values()):.4f}m  "
        f"attack travel={float(motion.get('attack_tip_travel_m', 0)):.3f}m  intersections={motion.get('worst_non_adjacent_triangle_overlaps')}"
    )
    draw.multiline_text((margin, 44), evidence, fill=(170, 190, 180), font=font, spacing=5)
    for index, path in enumerate(frames):
        row, column = divmod(index, columns)
        x = margin + column * cell
        y = header + row * (cell + 34)
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((cell - 8, cell - 8), Image.Resampling.LANCZOS)
        board.paste(image, (x + (cell - image.width) // 2, y + (cell - image.height) // 2))
        draw.text((x + 6, y + cell + 8), path.stem.replace("-", "  frame ", 1), fill=(225, 220, 205), font=font)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, "PNG")
    return {"stage": "canonical_review", "family": report.get("family"), "frames": len(frames), "output": str(output.resolve())}
