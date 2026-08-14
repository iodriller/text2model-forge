"""Render deterministic directional sprite sheets from embedded glTF animations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from validate_glb_skinning import accessor, load_glb, pose_vertices


DIRECTIONS = {"south": 0, "west": 90, "north": 180, "east": 270}


def rotate_y(vertices, degrees):
    angle = math.radians(degrees)
    matrix = np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]],
        dtype=np.float32,
    )
    return vertices @ matrix.T


def render_frame(vertices, faces, colors, *, degrees, size=192):
    vertices = rotate_y(vertices, degrees)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices = vertices - center
    span = max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]), 1e-6)
    scale = size * 0.82 / span
    screen = np.column_stack([size / 2 + vertices[:, 0] * scale, size / 2 - vertices[:, 1] * scale, vertices[:, 2]])
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    depth = screen[faces, 2].mean(axis=1)
    for face_index in np.argsort(depth):
        indices = faces[face_index]
        triangle = vertices[indices]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        # Small triangles dominate low-resolution meshes. Per-triangle outlines
        # turn the surface into a dark stipple at sprite resolution, so use a
        # bright, outline-free clay render and let the silhouette carry the edge.
        light = 0.72 + 0.48 * abs(float(normal[2]))
        base = colors[indices, :3].mean(axis=0)
        color = tuple(int(np.clip(value * light * 255, 0, 255)) for value in base) + (255,)
        points = [(float(screen[index, 0]), float(screen[index, 1])) for index in indices]
        draw.polygon(points, fill=color)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-size", type=int, default=192)
    args = parser.parse_args()
    doc, binary = load_glb(args.input)
    primitive = doc["meshes"][0]["primitives"][0]
    faces = accessor(doc, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
    colors = accessor(doc, binary, primitive["attributes"]["COLOR_0"]).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for animation in doc["animations"]:
        for direction, degrees in DIRECTIONS.items():
            frames = []
            for keyframe in range(3):
                _, posed = pose_vertices(doc, binary, animation, keyframe_index=keyframe)
                frames.append(render_frame(posed, faces, colors, degrees=degrees, size=args.frame_size))
            sheet = Image.new("RGBA", (args.frame_size * 3, args.frame_size), (0, 0, 0, 0))
            for index, frame in enumerate(frames):
                sheet.paste(frame, (index * args.frame_size, 0), frame)
            path = args.output_dir / f"{animation['name']}_{direction}.png"
            sheet.save(path)
            # Keep the report portable when the validated output tree is copied
            # into a package or moved to another machine.
            outputs.append(path.name)
    report = {"animations": len(doc["animations"]), "directions": list(DIRECTIONS), "frames_per_sheet": 3, "sheets": outputs}
    (args.output_dir / "sprite_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
