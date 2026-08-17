import argparse
import json
import os
import sys

import bpy
from mathutils import Vector


def parse_args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(raw)


def action_match(aliases):
    available = {action.name.lower(): action.name for action in bpy.data.actions}
    return next((available[name.lower()] for name in aliases if name.lower() in available), None)


def world_bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        return None
    minimum = [min(point[index] for point in points) for index in range(3)]
    maximum = [max(point[index] for point in points) for index in range(3)]
    return {"min": minimum, "max": maximum, "size": [maximum[index] - minimum[index] for index in range(3)]}


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)

    failures = []
    warnings = []
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    polygon_count = sum(len(obj.data.polygons) for obj in meshes)
    if not meshes:
        failures.append("No renderable mesh objects")
    max_polygons = int(config.get("max_polygons", 75000))
    if polygon_count > max_polygons:
        failures.append(f"Polygon count {polygon_count} exceeds configured maximum {max_polygons}")

    missing_materials = [obj.name for obj in meshes if not obj.material_slots]
    if missing_materials:
        failures.append("Mesh objects without materials: " + ", ".join(missing_materials))
    unapplied_scale = [obj.name for obj in meshes if any(abs(value - 1.0) > 0.001 for value in obj.scale)]
    if unapplied_scale:
        warnings.append("Objects with unapplied scale: " + ", ".join(unapplied_scale))

    missing_textures = []
    for image in bpy.data.images:
        if image.source != "FILE" or not image.filepath:
            continue
        path = bpy.path.abspath(image.filepath)
        if not os.path.isfile(path):
            missing_textures.append(f"{image.name}: {path}")
    if missing_textures:
        failures.append("Missing external textures: " + "; ".join(missing_textures))

    animation_object_name = config.get("animation_object")
    animation_object = bpy.data.objects.get(animation_object_name) if animation_object_name else None
    if config.get("animations") and animation_object is None:
        failures.append(f"Animation object not found: {animation_object_name}")

    actions = {}
    for name, settings in config.get("animations", {}).items():
        aliases = settings.get("actions", [])
        matched = action_match(aliases) if aliases else None
        actions[name] = matched
        if aliases and matched is None:
            failures.append(f"Required action {name!r} missing; aliases: {aliases}")

    bounds = world_bounds(meshes)
    if bounds and bounds["size"][2] <= 0.01:
        failures.append("Master has near-zero world height")

    report = {
        "asset_id": config.get("id"),
        "passed": not failures,
        "blender_version": bpy.app.version_string,
        "source": bpy.data.filepath,
        "mesh_objects": len(meshes),
        "polygon_count": polygon_count,
        "max_polygons": max_polygons,
        "materials": len(bpy.data.materials),
        "images": len(bpy.data.images),
        "bounds": bounds,
        "actions": actions,
        "failures": failures,
        "warnings": warnings,
    }
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("TEXT2MODEL_FORGE_MASTER_AUDIT=" + report_path)
    if failures:
        raise RuntimeError("Master audit failed: " + " | ".join(failures))


main()
