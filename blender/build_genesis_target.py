"""Build a disposable, watertight anatomy target from semantic volumes.

Run with Blender:
  blender --background --python build_genesis_target.py -- \
    --spec creature.genesis.json --output target.blend --report target.report.json

The output is intentionally NOT a production mesh.  Voxel remeshing gives the next
stage one continuous shape, but destroys useful animation topology.  Production is
blocked until an owned family template is fitted to this target.
"""

from __future__ import annotations

import argparse
import bmesh
import hashlib
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def arguments():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--preview")
    return parser.parse_args(values)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_point(value, landmarks):
    return Vector(landmarks[value] if isinstance(value, str) else value)


def apply_scale(value):
    bpy.context.view_layer.objects.active = value
    value.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    value.select_set(False)


def add_ellipsoid(volume, landmarks):
    center = resolve_point(volume["center"], landmarks)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=center)
    value = bpy.context.object
    value.name = "VOL_" + volume["id"]
    value.scale = Vector(volume["radii"])
    value.rotation_euler = tuple(math.radians(axis) for axis in volume.get("rotation_degrees", [0, 0, 0]))
    apply_scale(value)
    return value


def add_sphere(name, center, radius):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=center)
    value = bpy.context.object
    value.name = name
    return value


def add_capsule(volume, landmarks):
    start = resolve_point(volume["start"], landmarks)
    end = resolve_point(volume["end"], landmarks)
    radius = float(volume["radius"])
    direction = end - start
    length = direction.length
    if length <= 1e-6:
        raise RuntimeError(f"capsule {volume['id']} has coincident endpoints")
    midpoint = (start + end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=radius, depth=length, location=midpoint)
    cylinder = bpy.context.object
    cylinder.name = "VOL_" + volume["id"] + "_shaft"
    cylinder.rotation_mode = "QUATERNION"
    cylinder.rotation_quaternion = direction.to_track_quat("Z", "Y")
    apply_scale(cylinder)
    return [
        cylinder,
        add_sphere("VOL_" + volume["id"] + "_start", start, radius),
        add_sphere("VOL_" + volume["id"] + "_end", end, radius),
    ]


def add_hull(volume, landmarks):
    points = [resolve_point(value, landmarks) for value in volume["points"]]
    normal = (points[1] - points[0]).cross(points[2] - points[0])
    if normal.length <= 1e-6:
        raise RuntimeError(f"hull {volume['id']} has collinear first three points")
    offset = normal.normalized() * (float(volume["thickness"]) * 0.5)
    vertices = [point + offset for point in points] + [point - offset for point in points]
    mesh = bpy.data.meshes.new("VOL_" + volume["id"] + "_mesh")
    value = bpy.data.objects.new("VOL_" + volume["id"], mesh)
    bpy.context.collection.objects.link(value)
    bm = bmesh.new()
    for point in vertices:
        bm.verts.new(point)
    bm.verts.ensure_lookup_table()
    bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
    bm.to_mesh(mesh)
    bm.free()
    return value


def join_objects(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for value in objects:
        value.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    result = bpy.context.object
    result.name = "AF_AnatomyTarget"
    return result


def add_landmarks(landmarks):
    collection = bpy.data.collections.new("AF_SemanticLandmarks")
    bpy.context.scene.collection.children.link(collection)
    for name, point in landmarks.items():
        marker = bpy.data.objects.new("LM_" + name, None)
        marker.empty_display_type = "SPHERE"
        marker.empty_display_size = 0.035
        marker.location = point
        marker["asset_forge_landmark"] = name
        collection.objects.link(marker)


def mesh_metrics(value):
    mesh = value.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    unseen = set(bm.verts)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            vertex = stack.pop()
            for edge in vertex.link_edges:
                neighbor = edge.other_vert(vertex)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    metrics = {
        "vertices": len(bm.verts),
        "edges": len(bm.edges),
        "polygons": len(bm.faces),
        "connected_components": components,
        "boundary_edges": sum(1 for edge in bm.edges if edge.is_boundary),
        "non_manifold_edges": sum(1 for edge in bm.edges if not edge.is_manifold),
    }
    bm.free()
    return metrics


def render_preview(target, path):
    scene = bpy.context.scene
    try:
        # Blender 4.2 called this BLENDER_EEVEE_NEXT; 5.x renamed it back to
        # BLENDER_EEVEE. Pick whichever this build offers.
        scene.render.engine = next(
            (name for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
             if name in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()),
            "BLENDER_EEVEE",
        )
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("AF_PreviewWorld")
    scene.world.color = (0.025, 0.03, 0.04)

    camera_data = bpy.data.cameras.new("AF_PreviewCamera")
    camera = bpy.data.objects.new("AF_PreviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (3.0, 5.6, 2.55)
    camera.rotation_euler = (Vector((0.0, 0.08, 1.05)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.75
    scene.camera = camera

    for name, location, energy, size in (
        ("AF_Key", (3.5, 4.0, 5.0), 900.0, 4.0),
        ("AF_Rim", (-3.0, -2.0, 3.5), 650.0, 3.0),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (target.location - light.location).to_track_quat("-Z", "Y").to_euler()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    args = arguments()
    spec_path = os.path.abspath(args.spec)
    output_path = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    with open(spec_path, "r", encoding="utf-8") as stream:
        spec = json.load(stream)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    landmarks = spec["landmarks"]
    objects = []
    for volume in spec["volumes"]:
        if volume["type"] == "ellipsoid":
            objects.append(add_ellipsoid(volume, landmarks))
        elif volume["type"] == "capsule":
            objects.extend(add_capsule(volume, landmarks))
        elif volume["type"] == "hull":
            objects.append(add_hull(volume, landmarks))
        else:
            raise RuntimeError("unsupported volume type: " + str(volume["type"]))

    target = join_objects(objects)
    modifier = target.modifiers.new("AF_VolumetricUnion", "REMESH")
    modifier.mode = "VOXEL"
    modifier.voxel_size = float(spec["anatomy_target"]["voxel_size"])
    modifier.use_smooth_shade = True
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    target["asset_forge_stage"] = "anatomy_target"
    target["asset_forge_production_eligible"] = False
    target["asset_forge_family_template"] = spec["family_template"]
    add_landmarks(landmarks)

    material = bpy.data.materials.new("AF Anatomy Target (Non-Production)")
    material.diffuse_color = (0.16, 0.34, 0.22, 1.0)
    material.roughness = 0.82
    target.data.materials.append(material)

    metrics = mesh_metrics(target)
    if args.preview:
        render_preview(target, os.path.abspath(args.preview))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    report = {
        "schema_version": 1,
        "stage": "anatomy_target",
        "asset_id": spec["id"],
        "body_plan": spec["body_plan"],
        "family_template": spec["family_template"],
        "production_eligible": False,
        "next_required_stage": "canonical_topology_fit",
        "spec": spec_path,
        "spec_sha256": sha256(spec_path),
        "target": output_path,
        "target_sha256": sha256(output_path),
        "semantic_volume_count": len(spec["volumes"]),
        "landmark_count": len(landmarks),
        "voxel_size": float(spec["anatomy_target"]["voxel_size"]),
        "mesh": metrics,
        "preview": os.path.abspath(args.preview) if args.preview else None,
    }
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if metrics["connected_components"] != 1 or metrics["non_manifold_edges"] != 0:
        raise RuntimeError("voxel anatomy target is not one watertight component: " + json.dumps(metrics))
    print("ASSET_FORGE_GENESIS_TARGET=" + output_path)
    print("ASSET_FORGE_GENESIS_REPORT=" + report_path)


main()
