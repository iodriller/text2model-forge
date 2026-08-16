"""Create a fitted, skinned canonical master from a family contract and anatomy target.

This is a deterministic reference-template authoring backend.  It uses Blender's Skin
modifier once to establish family topology, fits that topology to the disposable
anatomy target, then adds an owned semantic armature, normalized weights, family
actions, equipment sockets, and motion/deformation metrics.
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
from mathutils.bvhtree import BVHTree


def args_value():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--target", required=True)
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


def topology_hash(mesh):
    digest = hashlib.sha256()
    digest.update(str(len(mesh.vertices)).encode())
    for polygon in mesh.polygons:
        digest.update(",".join(str(index) for index in polygon.vertices).encode())
        digest.update(b";")
    return digest.hexdigest()


def mesh_metrics(value):
    bm = bmesh.new()
    bm.from_mesh(value.data)
    unseen = set(bm.verts)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            vertex = stack.pop()
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
    result = {
        "vertices": len(bm.verts),
        "edges": len(bm.edges),
        "polygons": len(bm.faces),
        "connected_components": components,
        "boundary_edges": sum(1 for edge in bm.edges if edge.is_boundary),
        "non_manifold_edges": sum(1 for edge in bm.edges if not edge.is_manifold),
        "has_uv": bool(value.data.uv_layers),
        "topology_sha256": topology_hash(value.data),
    }
    bm.free()
    return result


def build_skin_graph(contract, landmarks):
    graph = contract["topology_graph"]
    points = graph["points"]
    vertices = [landmarks[name] for name in points]
    index = {name: number for number, name in enumerate(points)}
    edges = [(index[a], index[b]) for a, b in graph["edges"]]
    mesh = bpy.data.meshes.new("AF_CanonicalBodyMesh")
    mesh.from_pydata(vertices, edges, [])
    mesh.update()
    body = bpy.data.objects.new("AF_CanonicalBody", mesh)
    bpy.context.collection.objects.link(body)
    bpy.context.view_layer.objects.active = body
    body.select_set(True)

    skin = body.modifiers.new("AF_FamilyTopology", "SKIN")
    bpy.context.view_layer.objects.active = body
    # Force creation of the skin-vertex custom-data layer before assigning radii.
    bpy.context.view_layer.objects.active = body
    skin_data = body.data.skin_vertices[0].data
    for number, name in enumerate(points):
        radius = float(graph["radii"][name])
        skin_data[number].radius = (radius, radius)
        skin_data[number].use_root = name == graph.get("root_point")
    bpy.ops.object.modifier_apply(modifier=skin.name)

    # This is the one-time topology-establishment step for a new family. Skin-graph
    # branches can emit touching shells at high-valence joints; fuse them before the
    # result becomes the immutable canonical template. Variant fitting never remeshes.
    union = body.modifiers.new("AF_EstablishCanonicalSurface", "REMESH")
    union.mode = "VOXEL"
    union.voxel_size = 0.045
    union.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=union.name)

    subdivision = body.modifiers.new("AF_JointLoops", "SUBSURF")
    subdivision.subdivision_type = "CATMULL_CLARK"
    subdivision.levels = min(1, int(graph.get("subdivision_levels", 1)))
    subdivision.render_levels = subdivision.levels
    bpy.ops.object.modifier_apply(modifier=subdivision.name)
    return body


def establish_reference_topology(target):
    """Create a family's first immutable topology from its accepted watertight target.

    Remeshing is permitted exactly here, while the topology is being authored. Once
    this master is accepted, variants must preserve its topology hash.
    """
    body = target.copy()
    body.data = target.data.copy()
    body.name = "AF_CanonicalBody"
    body.data.name = "AF_CanonicalBodyMesh"
    bpy.context.collection.objects.link(body)
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    union = body.modifiers.new("AF_EstablishCanonicalSurface", "REMESH")
    union.mode = "VOXEL"
    union.voxel_size = 0.045
    union.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=union.name)
    smooth = body.modifiers.new("AF_CanonicalRelax", "SMOOTH")
    smooth.factor = 0.12
    smooth.iterations = 2
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    return body


def fit_to_target(body, target):
    modifier = body.modifiers.new("AF_CanonicalSurfaceFit", "SHRINKWRAP")
    modifier.target = target
    modifier.wrap_method = "NEAREST_SURFACEPOINT"
    modifier.wrap_mode = "ON_SURFACE"
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    smooth = body.modifiers.new("AF_FitRelax", "CORRECTIVE_SMOOTH")
    smooth.factor = 0.18
    smooth.iterations = 3
    bpy.ops.object.modifier_apply(modifier=smooth.name)


def smart_uv(body):
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(55), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")


def build_armature(contract, landmarks):
    data = bpy.data.armatures.new("AF_DeformationRig")
    rig = bpy.data.objects.new("AF_Rig", data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    created = {}
    for item in contract["skeleton"]:
        bone = data.edit_bones.new(item["name"])
        bone.head = landmarks[item["head"]]
        bone.tail = landmarks[item["tail"]]
        if (bone.tail - bone.head).length < 0.01:
            bone.tail.z += 0.02
        created[item["name"]] = bone
    for item in contract["skeleton"]:
        if item.get("parent"):
            created[item["name"]].parent = created[item["parent"]]
    bpy.ops.object.mode_set(mode="POSE")
    for bone in rig.pose.bones:
        bone.rotation_mode = "XYZ"
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.show_in_front = True
    return rig


def segment_distance(point, start, end):
    axis = end - start
    length_squared = axis.length_squared
    if length_squared <= 1e-12:
        return (point - start).length
    amount = max(0.0, min(1.0, (point - start).dot(axis) / length_squared))
    return (point - (start + axis * amount)).length


def bind_weights(body, rig, contract, landmarks):
    bones = contract["skeleton"]
    groups = {item["name"]: body.vertex_groups.new(name=item["name"]) for item in bones}
    max_influences = 0
    max_sum_error = 0.0
    unweighted = 0
    for vertex in body.data.vertices:
        point = body.matrix_world @ vertex.co
        candidates = []
        for item in bones:
            distance = segment_distance(point, Vector(landmarks[item["head"]]), Vector(landmarks[item["tail"]]))
            score = 1.0 / max(distance * distance, 0.0004)
            candidates.append((score, item["name"]))
        selected = sorted(candidates, reverse=True)[:4]
        total = sum(score for score, _ in selected)
        if total <= 0:
            unweighted += 1
            continue
        normalized = [(score / total, name) for score, name in selected]
        max_influences = max(max_influences, len(normalized))
        max_sum_error = max(max_sum_error, abs(sum(weight for weight, _ in normalized) - 1.0))
        for weight, name in normalized:
            groups[name].add([vertex.index], weight, "REPLACE")
    armature = body.modifiers.new("AF_Armature", "ARMATURE")
    armature.object = rig
    body.parent = rig
    return {
        "unweighted_vertices": unweighted,
        "max_sum_error": max_sum_error,
        "max_influences": max_influences,
    }


DETAIL_COLORS = {
    "eye_ivory": (0.58, 0.52, 0.35, 1.0),
    "pupil_ember": (0.65, 0.08, 0.015, 1.0),
    "bone": (0.62, 0.52, 0.34, 1.0),
    "horn": (0.12, 0.08, 0.055, 1.0),
    "claw": (0.2, 0.15, 0.09, 1.0),
    "leather": (0.11, 0.045, 0.018, 1.0),
    "dark_cloth": (0.035, 0.045, 0.055, 1.0),
}


def detail_material(name):
    material_name = "AF_Detail_" + name
    existing = bpy.data.materials.get(material_name)
    if existing:
        return existing
    value = bpy.data.materials.new(material_name)
    color = DETAIL_COLORS.get(name, (0.3, 0.3, 0.3, 1.0))
    value.diffuse_color = color
    value.roughness = 0.72
    value.use_nodes = True
    shader = value.node_tree.nodes.get("Principled BSDF")
    if shader:
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = 0.72
        if name == "pupil_ember":
            shader.inputs["Metallic"].default_value = 0.15
    return value


def attach_rigid_mesh(value, rig, bone_name):
    group = value.vertex_groups.new(name=bone_name)
    group.add([vertex.index for vertex in value.data.vertices], 1.0, "REPLACE")
    modifier = value.modifiers.new("AF_DetailArmature", "ARMATURE")
    modifier.object = rig


def create_details(spec, rig, landmarks):
    created = []
    for detail in spec.get("details") or []:
        kind = detail["type"]
        if kind in {"sphere", "ellipsoid"}:
            center = Vector(landmarks[detail["center"]] if isinstance(detail["center"], str) else detail["center"])
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=float(detail.get("radius", 1.0)), location=center)
            value = bpy.context.object
            if kind == "ellipsoid":
                value.scale = detail["radii"]
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        elif kind == "cone":
            start = Vector(landmarks[detail["start"]] if isinstance(detail["start"], str) else detail["start"])
            end = Vector(landmarks[detail["end"]] if isinstance(detail["end"], str) else detail["end"])
            axis = end - start
            bpy.ops.mesh.primitive_cone_add(
                vertices=16,
                radius1=float(detail["radius"]),
                radius2=0.006,
                depth=axis.length,
                location=(start + end) * 0.5,
            )
            value = bpy.context.object
            value.rotation_mode = "QUATERNION"
            value.rotation_quaternion = axis.to_track_quat("Z", "Y")
        else:
            center = Vector(landmarks[detail["center"]] if isinstance(detail["center"], str) else detail["center"])
            bpy.ops.mesh.primitive_cube_add(location=center)
            value = bpy.context.object
            value.scale = detail["scale"]
            value.rotation_euler = tuple(math.radians(axis) for axis in detail.get("rotation_degrees", [0, 0, 0]))
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        value.name = "AF_Detail_" + detail["id"]
        value.data.materials.append(detail_material(detail.get("material", "bone")))
        attach_rigid_mesh(value, rig, detail["bone"])
        created.append(value.name)
    return created


def set_pose(rig, frame, rotations=None, locations=None):
    rotations = rotations or {}
    locations = locations or {}
    for bone in rig.pose.bones:
        bone.rotation_euler = tuple(math.radians(axis) for axis in rotations.get(bone.name, (0, 0, 0)))
        bone.location = locations.get(bone.name, (0, 0, 0))
        bone.keyframe_insert(data_path="rotation_euler", frame=frame)
        bone.keyframe_insert(data_path="location", frame=frame)


def action(rig, name, keys):
    value = bpy.data.actions.new(name=name)
    value.use_fake_user = True
    rig.animation_data_create()
    rig.animation_data.action = value
    for frame, rotations, locations in keys:
        set_pose(rig, frame, rotations, locations)
    value.frame_start = keys[0][0]
    value.frame_end = keys[-1][0]
    rig.animation_data.action = None
    return value


def create_short_biped_actions(rig):
    neutral = {}
    actions = {}
    actions["idle"] = action(rig, "idle", [
        (1, neutral, {}), (12, {"spine": (2, 0, 0), "head": (-1, 0, 0)}, {}), (24, neutral, {}),
    ])
    pose_a = {"thigh.L": (-24, 0, 0), "shin.L": (18, 0, 0), "thigh.R": (20, 0, 0), "upper_arm.L": (18, 0, 0), "upper_arm.R": (-18, 0, 0)}
    pose_b = {"thigh.L": (20, 0, 0), "thigh.R": (-24, 0, 0), "shin.R": (18, 0, 0), "upper_arm.L": (-18, 0, 0), "upper_arm.R": (18, 0, 0)}
    actions["walk"] = action(rig, "walk", [
        (1, pose_a, {}), (4, pose_a, {}), (7, neutral, {}),
        (13, pose_b, {}), (16, pose_b, {}), (19, neutral, {}), (25, pose_a, {}),
    ])
    actions["attack"] = action(rig, "attack", [
        (1, neutral, {}),
        (6, {"spine": (-5, 0, -7), "upper_arm.R": (-50, 0, -28), "forearm.R": (-28, 0, 0)}, {}),
        (11, {"spine": (7, 0, 9), "upper_arm.R": (55, 0, 30), "forearm.R": (26, 0, 0)}, {}),
        (17, {"spine": (3, 0, 3), "upper_arm.R": (18, 0, 8)}, {}), (24, neutral, {}),
    ])
    actions["hit"] = action(rig, "hit", [
        (1, neutral, {}), (6, {"spine": (-18, 0, 10), "head": (-10, 0, -8)}, {}), (12, neutral, {}),
    ])
    actions["death"] = action(rig, "death", [
        (1, neutral, {}), (10, {"hips": (0, 0, -18), "head": (0, 0, 6)}, {}),
        (20, {"hips": (0, 0, -58), "head": (0, 0, 10)}, {}),
        (28, {"hips": (0, 0, -84)}, {}),
    ])
    return actions


def create_winged_quadruped_actions(rig):
    neutral = {}
    actions = {}
    actions["idle"] = action(rig, "idle", [
        (1, neutral, {}),
        (12, {"neck": (2, 0, 0), "wing_upper.L": (0, -3, 0), "wing_upper.R": (0, 3, 0), "tail.2": (0, 0, 3)}, {}),
        (24, neutral, {}),
    ])
    pose_a = {
        "front_upper.L": (-16, 0, 0), "front_lower.L": (10, 0, 0), "rear_upper.R": (14, 0, 0),
        "front_upper.R": (12, 0, 0), "rear_upper.L": (-14, 0, 0), "tail.1": (0, 0, 4),
    }
    pose_b = {
        "front_upper.R": (-16, 0, 0), "front_lower.R": (10, 0, 0), "rear_upper.L": (14, 0, 0),
        "front_upper.L": (12, 0, 0), "rear_upper.R": (-14, 0, 0), "tail.1": (0, 0, -4),
    }
    actions["walk"] = action(rig, "walk", [
        (1, pose_a, {}), (4, pose_a, {}), (7, neutral, {}),
        (13, pose_b, {}), (16, pose_b, {}), (19, neutral, {}), (25, pose_a, {}),
    ])
    actions["attack"] = action(rig, "attack", [
        (1, neutral, {}), (6, {"neck": (-12, 0, 0), "head": (-10, 0, 0), "jaw": (12, 0, 0)}, {}),
        (11, {"neck": (16, 0, 0), "head": (14, 0, 0), "jaw": (-8, 0, 0), "wing_upper.L": (0, -5, 0), "wing_upper.R": (0, 5, 0)}, {}),
        (18, {"neck": (4, 0, 0), "head": (3, 0, 0)}, {}), (24, neutral, {}),
    ])
    actions["hit"] = action(rig, "hit", [
        (1, neutral, {}), (6, {"spine": (0, 0, 8), "neck": (-8, 0, 0), "wing_upper.L": (0, 4, 0)}, {}), (12, neutral, {}),
    ])
    actions["death"] = action(rig, "death", [
        (1, neutral, {}), (12, {"hips": (0, 0, -15), "wing_upper.L": (0, -8, 0), "wing_upper.R": (0, 8, 0)}, {}),
        (24, {"hips": (0, 0, -55)}, {}), (32, {"hips": (0, 0, -82)}, {}),
    ])
    actions["fly"] = action(rig, "fly", [
        (1, {"wing_upper.L": (0, -18, 0), "wing_upper.R": (0, 18, 0)}, {}),
        (8, {"wing_upper.L": (0, 22, 0), "wing_upper.R": (0, -22, 0), "wing_lower.L": (0, 10, 0), "wing_lower.R": (0, -10, 0)}, {}),
        (16, {"wing_upper.L": (0, -18, 0), "wing_upper.R": (0, 18, 0)}, {}),
    ])
    return actions


def create_weapon(rig, landmarks):
    wrist = Vector(landmarks["wrist_r"])
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.04, depth=0.62, location=wrist + Vector((0, 0, 0.31)))
    weapon = bpy.context.object
    weapon.name = "AF_Weapon"
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.14, location=wrist + Vector((0, 0, 0.58)))
    head = bpy.context.object
    head.name = "AF_WeaponHead"
    bpy.context.view_layer.objects.active = weapon
    weapon.select_set(True)
    head.select_set(True)
    bpy.ops.object.join()
    group = weapon.vertex_groups.new(name="forearm.R")
    group.add([vertex.index for vertex in weapon.data.vertices], 1.0, "REPLACE")
    modifier = weapon.modifiers.new("AF_WeaponArmature", "ARMATURE")
    modifier.object = rig
    tip_vertex = max(
        weapon.data.vertices,
        key=lambda vertex: (weapon.matrix_world @ vertex.co).z,
    ).index
    distances = [((weapon.matrix_world @ vertex.co) - wrist).length for vertex in weapon.data.vertices]
    closest = min(distances)
    grip_vertices = [index for index, distance in enumerate(distances) if distance <= closest + 0.002]
    return weapon, (weapon, tip_vertex), (weapon, grip_vertices)


def create_attack_tip(rig, bone_name, world_location):
    tip = bpy.data.objects.new("AF_AttackTip", None)
    tip.empty_display_type = "SPHERE"
    tip.empty_display_size = 0.035
    bpy.context.collection.objects.link(tip)
    tip.location = world_location
    world = tip.matrix_world.copy()
    tip.parent = rig
    tip.parent_type = "BONE"
    tip.parent_bone = bone_name
    tip.matrix_world = world
    return tip


def world_position(value):
    bpy.context.view_layer.update()
    return value.matrix_world.translation.copy()


def attack_tip_position(value):
    if isinstance(value, tuple):
        object_value, vertex_index = value
        evaluated = object_value.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        position = evaluated.matrix_world @ mesh.vertices[vertex_index].co
        evaluated.to_mesh_clear()
        return position
    return world_position(value)


def evaluated_vertices_center(descriptor):
    object_value, indices = descriptor
    evaluated = object_value.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    total = Vector((0, 0, 0))
    for index in indices:
        total += evaluated.matrix_world @ mesh.vertices[index].co
    evaluated.to_mesh_clear()
    return total / max(1, len(indices))


def non_adjacent_self_intersections(body, rig, actions):
    worst = 0
    checked = 0
    samples = []
    for value in actions.values():
        rig.animation_data.action = value
        start, end = map(int, value.frame_range)
        for frame in sorted({start, (start + end) // 2, end}):
            bpy.context.scene.frame_set(frame)
            evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            vertices = [vertex.co.copy() for vertex in mesh.vertices]
            polygons = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
            tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True)
            overlaps = 0
            for first, second in tree.overlap(tree):
                if first >= second:
                    continue
                if set(polygons[first]).isdisjoint(polygons[second]):
                    overlaps += 1
            worst = max(worst, overlaps)
            checked += 1
            samples.append({"action": value.name, "frame": frame, "overlaps": overlaps})
            evaluated.to_mesh_clear()
    rig.animation_data.action = None
    return checked, worst, samples


def audit_motion(body, rig, actions, attack_tip, contract, grip=None):
    thresholds = contract.get("motion_thresholds") or {}
    metrics = {"foot_slip_m": {}, "attack_tip_travel_m": 0.0, "max_grip_error_m": 0.0}
    walk = actions.get("walk")
    if walk:
        rig.animation_data.action = walk
        provider = contract.get("motion_provider")
        contacts = (
            (("foot.L", (1, 4)), ("foot.R", (13, 16)))
            if provider == "procedural_short_biped_v1"
            else (
                ("front_foot.L", (1, 4)), ("rear_foot.R", (1, 4)),
                ("front_foot.R", (13, 16)), ("rear_foot.L", (13, 16)),
            )
        )
        for bone_name, frames in contacts:
            positions = []
            for frame in frames:
                bpy.context.scene.frame_set(frame)
                bone = rig.pose.bones.get(bone_name)
                positions.append((rig.matrix_world @ bone.tail).copy())
            metrics["foot_slip_m"][bone_name] = (positions[1] - positions[0]).length
    attack_value = actions.get("attack")
    if attack_value and attack_tip:
        rig.animation_data.action = attack_value
        bpy.context.scene.frame_set(6)
        start = attack_tip_position(attack_tip)
        bpy.context.scene.frame_set(11)
        contact = attack_tip_position(attack_tip)
        metrics["attack_tip_travel_m"] = (contact - start).length
    checked, intersections, intersection_samples = non_adjacent_self_intersections(body, rig, actions)
    metrics["self_intersection_frames_checked"] = checked
    metrics["worst_non_adjacent_triangle_overlaps"] = intersections
    metrics["intersection_samples"] = intersection_samples
    if grip:
        for value in actions.values():
            rig.animation_data.action = value
            start, end = map(int, value.frame_range)
            for frame in sorted({start, (start + end) // 2, end}):
                bpy.context.scene.frame_set(frame)
                expected = rig.matrix_world @ rig.pose.bones["forearm.R"].tail
                actual = evaluated_vertices_center(grip)
                metrics["max_grip_error_m"] = max(metrics["max_grip_error_m"], (actual - expected).length)
    max_slip = max(metrics["foot_slip_m"].values(), default=0.0)
    failures = []
    if max_slip > float(thresholds.get("max_foot_slip_m", 0.03)):
        failures.append(f"foot slip {max_slip:.4f}m exceeds threshold")
    if metrics["attack_tip_travel_m"] < float(thresholds.get("min_attack_tip_travel_m", 0.25)):
        failures.append("attack weapon-tip travel is too small")
    if metrics["max_grip_error_m"] > float(thresholds.get("max_grip_error_m", 0.01)):
        failures.append(f"weapon grip error {metrics['max_grip_error_m']:.4f}m exceeds threshold")
    max_overlaps = int(thresholds.get("max_non_adjacent_triangle_overlaps", 0))
    if intersections > max_overlaps:
        failures.append(
            f"non-adjacent triangle overlaps {intersections} exceed threshold {max_overlaps}"
        )
    rig.animation_data.action = None
    return {"passed": not failures, "failures": failures, "thresholds": thresholds, **metrics}


def apply_display_materials(body, contract):
    material = bpy.data.materials.new("AF_GoblinSurface")
    base_color = (0.055, 0.12, 0.032, 1.0) if contract["body_plan"] == "biped" else (0.18, 0.035, 0.022, 1.0)
    material.diffuse_color = base_color
    material.roughness = 0.82
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader:
        shader.inputs["Base Color"].default_value = base_color
        shader.inputs["Roughness"].default_value = 0.82
    body.data.materials.clear()
    body.data.materials.append(material)
    weapon = bpy.data.objects.get("AF_Weapon")
    if weapon:
        wood = bpy.data.materials.new("AF_WornWood")
        wood.diffuse_color = (0.16, 0.07, 0.025, 1.0)
        wood.roughness = 0.9
        wood.use_nodes = True
        wood_shader = wood.node_tree.nodes.get("Principled BSDF")
        if wood_shader:
            wood_shader.inputs["Base Color"].default_value = wood.diffuse_color
            wood_shader.inputs["Roughness"].default_value = 0.9
        weapon.data.materials.clear()
        weapon.data.materials.append(wood)


def render_preview(body, rig, path, contract):
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
    if scene.world is None:
        scene.world = bpy.data.worlds.new("AF_World")
    scene.world.color = (0.018, 0.022, 0.03)
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    camera_data = bpy.data.cameras.new("AF_ReviewCamera")
    camera = bpy.data.objects.new("AF_ReviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    preview = contract.get("preview") or {}
    camera.location = preview.get("camera", (3.0, 5.8, 2.55))
    camera_target = Vector(preview.get("target", (0, 0.08, 1.05)))
    camera.rotation_euler = (camera_target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(preview.get("ortho_scale", 2.8))
    scene.camera = camera
    for name, location, energy, size in (
        ("AF_Key", (3.5, 4.5, 5.0), 1050, 4.0),
        ("AF_Rim", (-3.0, -2.5, 3.5), 750, 3.0),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (camera_target - light.location).to_track_quat("-Z", "Y").to_euler()
    if "idle" in bpy.data.actions:
        rig.animation_data.action = bpy.data.actions["idle"]
        scene.frame_set(1)
    scene.render.filepath = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.render.render(write_still=True)


def main():
    args = args_value()
    contract_path = os.path.abspath(args.contract)
    spec_path = os.path.abspath(args.spec)
    target_path = os.path.abspath(args.target)
    output_path = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    with open(contract_path, encoding="utf-8") as stream:
        contract = json.load(stream)
    with open(spec_path, encoding="utf-8") as stream:
        spec = json.load(stream)
    landmarks = spec["landmarks"]

    bpy.ops.wm.open_mainfile(filepath=target_path)
    target = bpy.data.objects.get("AF_AnatomyTarget")
    if target is None:
        raise RuntimeError("target blend has no AF_AnatomyTarget")
    body = establish_reference_topology(target)
    smart_uv(body)
    rig = build_armature(contract, landmarks)
    weights = bind_weights(body, rig, contract, landmarks)
    details = create_details(spec, rig, landmarks)

    provider = contract.get("motion_provider")
    if provider == "procedural_short_biped_v1":
        actions = create_short_biped_actions(rig)
        _, attack_tip, grip = create_weapon(rig, landmarks)
    elif provider == "procedural_winged_quadruped_v1":
        actions = create_winged_quadruped_actions(rig)
        attack_tip = create_attack_tip(rig, "head", Vector(landmarks["snout"]))
        grip = None
    else:
        raise RuntimeError("unsupported motion provider in this builder: " + str(provider))
    motion_audit = audit_motion(body, rig, actions, attack_tip, contract, grip)
    apply_display_materials(body, contract)

    bpy.data.objects.remove(target, do_unlink=True)
    metrics = mesh_metrics(body)
    if args.preview:
        render_preview(body, rig, os.path.abspath(args.preview), contract)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    required_bones = [item["name"] for item in contract["skeleton"]]
    report = {
        "schema_version": 1,
        "stage": "canonical_master",
        "production_eligible": False,
        "next_required_stage": "surface_bake_and_sprite_qa",
        "family": contract["id"],
        "body_plan": contract["body_plan"],
        "contract_sha256": sha256(contract_path),
        "spec_sha256": sha256(spec_path),
        "target_sha256": sha256(target_path),
        "master_sha256": sha256(output_path),
        "mesh": metrics,
        "weights": weights,
        "details": details,
        "bones": required_bones,
        "actions": sorted(actions),
        "missing_bones": sorted(set(required_bones) - set(rig.data.bones.keys())),
        "missing_actions": sorted(set(contract["required_actions"]) - set(actions)),
        "motion_audit": motion_audit,
        "preview": os.path.abspath(args.preview) if args.preview else None,
    }
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("ASSET_FORGE_CANONICAL_MASTER=" + output_path)
    print("ASSET_FORGE_CANONICAL_REPORT=" + report_path)


main()
