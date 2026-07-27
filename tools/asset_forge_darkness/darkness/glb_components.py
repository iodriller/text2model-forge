"""Non-destructive GLB component audit and bounded cleanup candidates."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pydantic import Field

from .hashing import sha256_bytes
from .schemas import StrictModel


class GlbComponent(StrictModel):
    index: int = Field(ge=0)
    vertices: int = Field(ge=0)
    faces: int = Field(ge=0)
    face_fraction: float = Field(ge=0, le=1)
    absolute_volume: float = Field(ge=0)
    centroid: list[float]
    bounds: list[list[float]]


class GlbComponentAudit(StrictModel):
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vertices: int = Field(ge=0)
    faces: int = Field(ge=0)
    connected_components: int = Field(ge=0)
    watertight: bool
    finite_coordinates: bool
    components: list[GlbComponent]
    structural_component_indices: list[int]
    primary_component_index: int


def _load_combined(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    meshes = [item for item in loaded.dump() if isinstance(item, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("GLB contains no triangle mesh")
    return trimesh.util.concatenate(meshes)


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]],
        dtype=np.float64,
    )


def _render_view(mesh: trimesh.Trimesh, degrees: float, size: int = 512) -> Image.Image:
    vertices = np.asarray(mesh.vertices) @ _rotation_y(degrees).T
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices = vertices - center
    span = max(float(np.ptp(vertices[:, 0])), float(np.ptp(vertices[:, 1])), 1e-9)
    scale = size * 0.84 / span
    screen = np.column_stack(
        [size / 2 + vertices[:, 0] * scale, size / 2 - vertices[:, 1] * scale, vertices[:, 2]]
    )
    image = Image.new("RGB", (size, size), (24, 27, 32))
    draw = ImageDraw.Draw(image)
    faces = np.asarray(mesh.faces)
    for face_index in np.argsort(screen[faces, 2].mean(axis=1)):
        indices = faces[face_index]
        triangle = vertices[indices]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length <= 1e-12:
            continue
        normal /= length
        light = float(np.clip(0.25 + 0.75 * abs(normal[2]), 0, 1))
        color = (int(55 + 80 * light), int(75 + 105 * light), int(55 + 65 * light))
        points = [(float(screen[index, 0]), float(screen[index, 1])) for index in indices]
        draw.polygon(points, fill=color, outline=(18, 20, 22))
    return image


def _render_turntable(mesh: trimesh.Trimesh, path: Path) -> None:
    sheet = Image.new("RGB", (1024, 1024), (15, 17, 20))
    for degrees, position in zip((0, 45, 90, 180), ((0, 0), (512, 0), (0, 512), (512, 512))):
        sheet.paste(_render_view(mesh, degrees), position)
    sheet.save(path)


def _export_glb(mesh: trimesh.Trimesh, path: Path) -> str:
    data = trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh))
    path.write_bytes(data)
    return sha256_bytes(data)


def build_component_audit(
    source_path: str | Path,
    output_directory: str | Path,
    *,
    minimum_structural_faces: int = 100,
    minimum_structural_fraction: float = 0.001,
) -> dict[str, object]:
    if minimum_structural_faces < 1:
        raise ValueError("minimum_structural_faces must be positive")
    if not 0 <= minimum_structural_fraction <= 1:
        raise ValueError("minimum_structural_fraction must be between zero and one")
    source_path = Path(source_path).resolve()
    output_directory = Path(output_directory).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_directory.mkdir(parents=True, exist_ok=False)

    source_bytes = source_path.read_bytes()
    combined = _load_combined(source_path)
    components = sorted(combined.split(only_watertight=False), key=lambda item: len(item.faces), reverse=True)
    if not components:
        raise ValueError("GLB contains no connected triangle components")
    total_faces = max(len(combined.faces), 1)
    records = [
        GlbComponent(
            index=index,
            vertices=len(component.vertices),
            faces=len(component.faces),
            face_fraction=len(component.faces) / total_faces,
            absolute_volume=float(abs(component.volume)),
            centroid=np.asarray(component.centroid, dtype=float).tolist(),
            bounds=np.asarray(component.bounds, dtype=float).tolist(),
        )
        for index, component in enumerate(components)
    ]
    structural_indices = [
        item.index
        for item in records
        if item.faces >= minimum_structural_faces and item.face_fraction >= minimum_structural_fraction
    ]
    if 0 not in structural_indices:
        structural_indices.insert(0, 0)
    audit = GlbComponentAudit(
        source_sha256=sha256_bytes(source_bytes),
        vertices=len(combined.vertices),
        faces=len(combined.faces),
        connected_components=len(components),
        watertight=bool(combined.is_watertight),
        finite_coordinates=bool(np.isfinite(combined.vertices).all()),
        components=records,
        structural_component_indices=structural_indices,
        primary_component_index=0,
    )

    structural = trimesh.util.concatenate([components[index] for index in structural_indices])
    primary = components[0]
    candidates = {
        "source": combined,
        "structural": structural,
        "primary": primary,
    }
    candidate_files: dict[str, dict[str, object]] = {}
    for name, mesh in candidates.items():
        image_path = output_directory / f"{name}_turntable.png"
        _render_turntable(mesh, image_path)
        entry: dict[str, object] = {
            "components": len(mesh.split(only_watertight=False)),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "turntable": str(image_path),
        }
        if name != "source":
            glb_path = output_directory / f"{name}_candidate.glb"
            entry["glb"] = str(glb_path)
            entry["sha256"] = _export_glb(mesh, glb_path)
        candidate_files[name] = entry

    audit_path = output_directory / "component_audit.json"
    audit_path.write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(
        f"| {item.index} | {item.vertices} | {item.faces} | {item.face_fraction:.6f} | "
        f"{item.centroid[0]:.4f}, {item.centroid[1]:.4f}, {item.centroid[2]:.4f} |"
        for item in records
    )
    report_path = output_directory / "report.md"
    report_path.write_text(
        f"""# GLB component audit

- Source: `{source_path.name}`
- Source SHA-256: `{audit.source_sha256}`
- Components: {audit.connected_components}
- Structural branch keeps: {audit.structural_component_indices}
- Primary-only branch keeps: [0]
- Human approval: **required; no branch is auto-promoted**

| Component | Vertices | Faces | Face fraction | Centroid |
|---:|---:|---:|---:|---|
{rows}

## Review branches

- `source_turntable.png`: unchanged geometry rendered for comparison.
- `structural_turntable.png`: keeps components meeting both configured structural thresholds.
- `primary_turntable.png`: keeps only the largest connected surface.

The source GLB is never overwritten. Derived GLBs are research candidates and may lose scene/material metadata.
""",
        encoding="utf-8",
    )
    return {
        "audit": audit.model_dump(),
        "candidates": candidate_files,
        "files": {"audit": str(audit_path), "report": str(report_path)},
    }
