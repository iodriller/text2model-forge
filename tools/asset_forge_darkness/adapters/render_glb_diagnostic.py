"""Dependency-light orthographic GLB diagnostic renderer for research candidates."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]],
        dtype=np.float64,
    )


def render_view(vertices: np.ndarray, faces: np.ndarray, rotation_degrees: float, size: int = 512) -> Image.Image:
    rotated = vertices @ _rotation_y(rotation_degrees).T
    center = (rotated.min(axis=0) + rotated.max(axis=0)) / 2
    rotated = rotated - center
    span = max(float(np.ptp(rotated[:, 0])), float(np.ptp(rotated[:, 1])), 1e-9)
    scale = size * 0.84 / span
    screen = np.column_stack(
        [size / 2 + rotated[:, 0] * scale, size / 2 - rotated[:, 1] * scale, rotated[:, 2]]
    )
    image = Image.new("RGB", (size, size), (24, 27, 32))
    draw = ImageDraw.Draw(image)
    face_depth = screen[faces, 2].mean(axis=1)
    for face_index in np.argsort(face_depth):
        indices = faces[face_index]
        triangle = rotated[indices]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length <= 1e-12:
            continue
        normal /= length
        light = float(np.clip(0.25 + 0.75 * abs(normal[2]), 0, 1))
        color = (
            int(55 + 80 * light),
            int(75 + 105 * light),
            int(55 + 65 * light),
        )
        points = [(float(screen[index, 0]), float(screen[index, 1])) for index in indices]
        draw.polygon(points, fill=color, outline=(18, 20, 22))
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = trimesh.load(args.input, force="scene")
    meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError("GLB contains no triangle mesh")
    mesh = trimesh.util.concatenate(meshes)
    views = [render_view(mesh.vertices, mesh.faces, angle) for angle in (0, 45, 90, 180)]
    sheet = Image.new("RGB", (1024, 1024), (15, 17, 20))
    for image, position in zip(views, ((0, 0), (512, 0), (0, 512), (512, 512))):
        sheet.paste(image, position)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(
        {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "components": int(len(mesh.split(only_watertight=False))),
            "watertight": bool(mesh.is_watertight),
            "bounds": mesh.bounds.tolist(),
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
