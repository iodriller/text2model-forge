"""Project painted orbit views into persistent mesh-corner colors without per-frame AI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Vector


ATTRIBUTE = "Text2ModelPaint"


def _arguments() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pose-action", default="idle")
    parser.add_argument("--pose-frame", type=int, default=1)
    return parser.parse_args(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paint_material(name: str):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 0.78
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = ATTRIBUTE
    links.new(color.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _ensure_uv(obj) -> None:
    mesh = obj.data
    existing = mesh.uv_layers.get("Text2ModelSurfaceUV")
    if existing is not None:
        mesh.uv_layers.remove(existing)
    mesh.uv_layers.new(name="Text2ModelSurfaceUV")
    mesh.uv_layers.active = mesh.uv_layers["Text2ModelSurfaceUV"]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66.0), island_margin=0.02, correct_aspect=True
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def _load_views(metadata: dict[str, object], folder: Path):
    loaded = []
    for record in metadata["views"]:
        index = int(record["index"])
        image = bpy.data.images.load(str(folder / f"painted_{index:02d}.png"), check_existing=False)
        pixels = list(image.pixels[:])
        loaded.append(
            {
                "right": Vector(record["right"]),
                "up": Vector(record["up"]),
                "forward": Vector(record["forward"]),
                "camera": Vector(record["camera_location"]),
                "scale": float(record["ortho_scale"]),
                "width": int(image.size[0]),
                "height": int(image.size[1]),
                "pixels": pixels,
            }
        )
    return loaded


def _sample(view, position: Vector, normal: Vector):
    facing = max(float(normal.dot(-view["forward"])), 0.0)
    if facing <= 0.01:
        return None
    relative = position - view["camera"]
    u = float(relative.dot(view["right"]) / view["scale"] + 0.5)
    v = float(relative.dot(view["up"]) / view["scale"] + 0.5)
    if not 0.002 <= u <= 0.998 or not 0.002 <= v <= 0.998:
        return None
    x = min(view["width"] - 1, max(0, round(u * (view["width"] - 1))))
    y = min(view["height"] - 1, max(0, round(v * (view["height"] - 1))))
    offset = (y * view["width"] + x) * 4
    rgba = tuple(float(value) for value in view["pixels"][offset : offset + 4])
    if rgba[3] < 0.2:
        return None
    return facing**4, rgba


def _project_object(obj, views) -> dict[str, float | int]:
    _ensure_uv(obj)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    if evaluated_mesh is None:
        raise RuntimeError(f"could not evaluate mesh for {obj.name}")
    if len(evaluated_mesh.vertices) != len(obj.data.vertices):
        evaluated.to_mesh_clear()
        raise RuntimeError(f"evaluated topology changed for {obj.name}")
    normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
    colors = []
    confidence = []
    try:
        for vertex in evaluated_mesh.vertices:
            position = evaluated.matrix_world @ vertex.co
            normal = (normal_matrix @ vertex.normal).normalized()
            samples = [result for view in views if (result := _sample(view, position, normal)) is not None]
            if samples:
                # Choose the most face-on view. This avoids averaging opposite-side
                # texture details and keeps the result deterministic.
                weight, rgba = max(samples, key=lambda item: item[0])
                colors.append(rgba)
                confidence.append(weight)
            else:
                colors.append((0.18, 0.25, 0.08, 1.0))
                confidence.append(0.0)
    finally:
        evaluated.to_mesh_clear()

    existing = obj.data.color_attributes.get(ATTRIBUTE)
    if existing is not None:
        obj.data.color_attributes.remove(existing)
    attribute = obj.data.color_attributes.new(name=ATTRIBUTE, type="BYTE_COLOR", domain="CORNER")
    luminance = []
    for loop in obj.data.loops:
        rgba = colors[loop.vertex_index]
        attribute.data[loop.index].color = rgba
        luminance.append(0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2])
    obj.data.materials.clear()
    obj.data.materials.append(_paint_material(f"Text2Model Projected {obj.name}"))
    obj["text2model_surface_projection"] = "comfyui_multiview_vertex_color_v1"
    return {
        "vertices": len(obj.data.vertices),
        "loops": len(obj.data.loops),
        "mean_confidence": sum(confidence) / max(len(confidence), 1),
        "uncovered_vertices": sum(value == 0.0 for value in confidence),
        "minimum_luminance": min(luminance, default=0.0),
        "mean_luminance": sum(luminance) / max(len(luminance), 1),
        "maximum_luminance": max(luminance, default=0.0),
    }


def main() -> int:
    args = _arguments()
    error_path = args.report.resolve().parent / "painted_vertex_projection_error.txt"
    if error_path.is_file():
        error_path.unlink()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    armature = bpy.data.objects.get("Text2ModelShortBipedRig")
    if armature is not None:
        action = next(
            (item for item in bpy.data.actions if item.name.lower() == args.pose_action.lower()), None
        )
        if action is None:
            raise RuntimeError(f"pose action not found: {args.pose_action}")
        armature.animation_data_create()
        armature.animation_data.action = action
    bpy.context.scene.frame_set(args.pose_frame)
    bpy.context.view_layer.update()
    views = _load_views(metadata, args.views.resolve())
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise RuntimeError("paint projection found no visible meshes")
    records = {obj.name: _project_object(obj, views) for obj in meshes}
    if armature is not None:
        armature.animation_data.action = None
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    failures = []
    if any(record["mean_luminance"] < 0.04 for record in records.values()):
        failures.append("projected_surface_too_dark")
    if any(record["uncovered_vertices"] / max(record["vertices"], 1) > 0.08 for record in records.values()):
        failures.append("too_many_uncovered_vertices")
    report = {
        "schema_version": 1,
        "stage": "painted_vertex_projection",
        "method": "comfyui_multiview_vertex_color_v1",
        "metadata": str(args.metadata.resolve()),
        "painted_views": str(args.views.resolve()),
        "output": str(output),
        "output_sha256": _sha256(output),
        "objects": records,
        "hard_failures": failures,
        "automatic_gate_passed": not failures,
    }
    args.report.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("painted vertex projection failed: " + ", ".join(failures))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        if "--report" in raw:
            report = Path(raw[raw.index("--report") + 1]).resolve()
            (report.parent / "painted_vertex_projection_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
