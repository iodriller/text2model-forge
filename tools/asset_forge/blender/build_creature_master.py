"""Build a deterministic creature master from a Creature DNA morphology profile.

This first production adapter targets the biped_humanoid_v1 family.  The profile,
not this script, owns species choices: posture, head proportions, face landmarks,
ears, tusks, removed donor parts, and equipment.  Future ogres therefore become a
new profile; dragons use the separate winged-quadruped family adapter rather than a
pile of Goblin-specific conditionals.

Usage:
  blender --background <base.blend> --python build_creature_master.py -- \
    --profile <profile.json> --output <master.blend> [--report <report.json>]
"""

import argparse
import hashlib
import json
import math
import os
import sys

import bpy
from mathutils import Euler, Quaternion, Vector


def parse_args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    return parser.parse_args(raw)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_material(name, spec):
    color = tuple(float(v) for v in spec.get("color", (0.5, 0.5, 0.5)))
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Metallic"].default_value = float(spec.get("metallic", 0.0))
    node.inputs["Roughness"].default_value = float(spec.get("roughness", 0.8))
    emission = float(spec.get("emission", 0.0))
    if emission > 0.0:
        emission_input = node.inputs.get("Emission Color") or node.inputs.get("Emission")
        strength_input = node.inputs.get("Emission Strength")
        if emission_input:
            emission_input.default_value = (*color, 1.0)
        if strength_input:
            strength_input.default_value = emission
    material.diffuse_color = (*color, 1.0)
    return material


def bone_parent(value, armature, bone_name):
    world = value.matrix_world.copy()
    value.parent = armature
    value.parent_type = "BONE"
    value.parent_bone = bone_name
    value.matrix_world = world


def smooth(value):
    if value.type != "MESH":
        return
    for polygon in value.data.polygons:
        polygon.use_smooth = True


def add_ellipsoid(name, location, scale, rotation, material, armature, bone, subdivisions=3):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions, location=location, rotation=rotation)
    value = bpy.context.object
    value.name = name
    value.scale = tuple(float(v) for v in scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    smooth(value)
    value.data.materials.append(material)
    bone_parent(value, armature, bone)
    return value


def add_cone(name, location, radius, length, rotation, material, armature, bone, vertices=12):
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=float(radius),
        radius2=0.0,
        depth=float(length),
        location=location,
        rotation=rotation,
    )
    value = bpy.context.object
    value.name = name
    modifier = value.modifiers.new("Soft forged edge", "BEVEL")
    modifier.width = min(float(radius) * 0.25, 0.008)
    modifier.segments = 2
    smooth(value)
    value.data.materials.append(material)
    bone_parent(value, armature, bone)
    return value


def add_cylinder(name, location, radius, length, rotation, material, armature, bone, vertices=12):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=float(radius),
        depth=float(length),
        location=location,
        rotation=rotation,
    )
    value = bpy.context.object
    value.name = name
    modifier = value.modifiers.new("Forged edge", "BEVEL")
    modifier.width = min(float(radius) * 0.2, 0.008)
    modifier.segments = 2
    value.data.materials.append(material)
    bone_parent(value, armature, bone)
    return value


def head_frame(armature, bone_name):
    bone = armature.pose.bones[bone_name]
    matrix = armature.matrix_world @ bone.matrix
    rotation = matrix.to_3x3()
    origin = armature.matrix_world @ ((bone.head + bone.tail) / 2)
    forward = (rotation @ Vector((0.0, 0.0, 1.0))).normalized()
    up = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
    right = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
    return origin, forward, up, right


def bone_frame(armature, bone_name, at_tail=False):
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        raise RuntimeError(f"Body module bone {bone_name!r} is missing")
    matrix = armature.matrix_world @ bone.matrix
    rotation = matrix.to_3x3()
    head = armature.matrix_world @ bone.head
    tail = armature.matrix_world @ bone.tail
    direction = (tail - head).normalized()
    center = tail if at_tail else (head + tail) * 0.5
    right = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
    forward = (rotation @ Vector((0.0, 0.0, 1.0))).normalized()
    return center, direction, forward, right, max(0.02, (tail - head).length)


def point(origin, forward, up, right, forward_value=0.0, up_value=0.0, right_value=0.0):
    return origin + forward * float(forward_value) + up * float(up_value) + right * float(right_value)


def face_rotation(forward):
    return tuple(forward.to_track_quat("Z", "Y").to_euler())


def add_ear(name, sign, spec, frame, material, armature, bone):
    origin, forward, up, right = frame
    root_x = sign * float(spec["root_spacing"])
    length = float(spec["length"])
    root_height = float(spec["root_height"])
    thickness = float(spec["thickness"])
    angle = math.radians(float(spec.get("down_angle_degrees", 0.0)))
    tip_x = root_x + sign * length * math.cos(angle)
    tip_y = float(spec.get("up", 0.0)) - length * math.sin(angle)
    tip_z = float(spec.get("forward", 0.0)) - float(spec.get("back_sweep", 0.0))
    root_y = float(spec.get("up", 0.0))
    root_z = float(spec.get("forward", 0.0))

    local = [
        (root_x, root_y - root_height / 2, root_z),
        (root_x, root_y + root_height / 2, root_z),
        (tip_x, tip_y, tip_z),
        (root_x, root_y - root_height / 2, root_z - thickness),
        (root_x, root_y + root_height / 2, root_z - thickness),
        (tip_x, tip_y, tip_z - thickness * 0.4),
    ]
    vertices = [point(origin, forward, up, right, z, y, x) for x, y, z in local]
    faces = [
        (0, 1, 2), (5, 4, 3),
        (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0),
    ]
    mesh = bpy.data.meshes.new(name + ".Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    value = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(value)
    modifier = value.modifiers.new("Organic ear edge", "BEVEL")
    modifier.width = thickness * 0.28
    modifier.segments = 3
    smooth(value)
    value.data.materials.append(material)
    bone_parent(value, armature, bone)
    return value


def bake_posture(armature, actions, corrections):
    quaternions = {
        bone: Euler(tuple(math.radians(float(v)) for v in angles), "XYZ").to_quaternion()
        for bone, angles in corrections.items()
    }
    applied = []
    for action_name in actions:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Cannot bake creature posture: action {action_name!r} not found")
        armature.animation_data.action = action
        start, end = action.frame_range
        for frame_number in range(int(math.floor(start)), int(math.ceil(end)) + 1):
            bpy.context.scene.frame_set(frame_number)
            for bone_name, correction in quaternions.items():
                bone = armature.pose.bones.get(bone_name)
                if bone is None:
                    raise RuntimeError(f"Posture bone {bone_name!r} is missing")
                if bone.rotation_mode != "QUATERNION":
                    bone.rotation_mode = "QUATERNION"
                bone.rotation_quaternion = correction @ bone.rotation_quaternion
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)
        applied.append(action_name)
    return applied


def add_spiked_club(grip_matrix, armature, bone, created):
    grip_origin = grip_matrix.translation.copy()
    grip_up = (grip_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    rotation = tuple(grip_up.to_track_quat("Z", "Y").to_euler())
    wood = make_material("Creature.Club.Wood", {"color": [0.12, 0.07, 0.035], "roughness": 0.92})
    iron = make_material("Creature.Club.Iron", {"color": [0.055, 0.065, 0.07], "roughness": 0.55, "metallic": 0.28})
    for name, distance, radius, length, material in (
        ("Club.Haft", 0.20, 0.032, 0.54, wood),
        ("Club.Band.Lower", 0.20, 0.041, 0.035, iron),
        ("Club.Band.Upper", 0.45, 0.043, 0.035, iron),
        ("Club.Head", 0.52, 0.105, 0.28, wood),
    ):
        value = add_cylinder(name, grip_origin + grip_up * distance, radius, length, rotation, material, armature, bone)
        created.append(value.name)
    # Spikes form a radial crown in the club's local grip frame.
    right = (grip_matrix.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    forward = grip_up.cross(right).normalized()
    for index, radial in enumerate((right, -right, forward, -forward)):
        location = grip_origin + grip_up * 0.54 + radial * 0.105
        rotation = tuple(radial.to_track_quat("Z", "Y").to_euler())
        value = add_cone(f"Club.Spike.{index:02d}", location, 0.027, 0.13, rotation, iron, armature, bone)
        created.append(value.name)


def build_head(profile, armature, materials, created):
    head = profile["head"]
    bone = head.get("bone", "Head")
    modules = head["modules"]
    frame = head_frame(armature, bone)
    origin, forward, up, right = frame
    base_rotation = forward.to_track_quat("Z", "Y")

    def ellipsoid(name, spec, scale_key="scale", rotation=None):
        location = point(
            origin, forward, up, right,
            spec.get("forward", 0.0), spec.get("up", 0.0), spec.get("right", 0.0),
        )
        value = add_ellipsoid(
            name, location, spec[scale_key], tuple((rotation or base_rotation).to_euler()),
            materials[spec["material"]], armature, bone,
        )
        created.append(value.name)
        return value

    if "cranium" in modules:
        ellipsoid("Morph.Cranium", modules["cranium"])
    if "eyes" in modules:
        spec = modules["eyes"]
        for side, sign in (("L", -1.0), ("R", 1.0)):
            eye_spec = dict(spec)
            eye_spec["right"] = sign * float(spec["spacing"])
            eye_spec["scale"] = [spec["radius"], spec["radius"], spec["radius"] * 0.72]
            ellipsoid(f"Morph.Eye.{side}", eye_spec)
    if "brow" in modules:
        spec = modules["brow"]
        for side, sign in (("L", -1.0), ("R", 1.0)):
            brow_spec = dict(spec)
            brow_spec["right"] = sign * float(spec["width"]) * 0.48
            brow_spec["scale"] = [float(spec["width"]) * 0.55, spec["height"], spec["depth"]]
            scowl = Quaternion((0.0, 0.0, 1.0), math.radians(sign * float(spec["scowl_degrees"])))
            ellipsoid(f"Morph.Brow.{side}", brow_spec, rotation=base_rotation @ scowl)
    if "snout" in modules:
        spec = modules["snout"]
        upturn = Quaternion((1.0, 0.0, 0.0), math.radians(float(spec.get("upturn_degrees", 0.0))))
        ellipsoid("Morph.Snout", spec, rotation=base_rotation @ upturn)
    if "jaw" in modules:
        ellipsoid("Morph.Jaw", modules["jaw"])
    if "mouth" in modules:
        ellipsoid("Morph.Mouth", modules["mouth"])
    if "nostrils" in modules:
        spec = modules["nostrils"]
        for side, sign in (("L", -1.0), ("R", 1.0)):
            nostril_spec = dict(spec)
            nostril_spec["right"] = sign * float(spec["spacing"])
            nostril_spec["scale"] = [spec["radius"], spec["radius"] * 0.65, spec["radius"] * 0.42]
            ellipsoid(f"Morph.Nostril.{side}", nostril_spec)
    if "tusks" in modules:
        spec = modules["tusks"]
        rotation = tuple(up.to_track_quat("Z", "Y").to_euler())
        for side, sign in (("L", -1.0), ("R", 1.0)):
            location = point(origin, forward, up, right, spec["forward"], spec["up"], sign * spec["spacing"])
            value = add_cone(
                f"Morph.Tusk.{side}", location, spec["radius"], spec["length"], rotation,
                materials[spec["material"]], armature, bone,
            )
            created.append(value.name)
    if "pointed_ears" in modules:
        spec = modules["pointed_ears"]
        for side, sign in (("L", -1.0), ("R", 1.0)):
            value = add_ear(f"Morph.Ear.{side}", sign, spec, frame, materials[spec["material"]], armature, bone)
            created.append(value.name)
    if modules.get("skullcap", {}).get("enabled"):
        cap = {
            "scale": [0.25, 0.20, 0.22], "forward": -0.12, "up": 0.08,
            "material": modules["skullcap"]["material"],
        }
        ellipsoid("Morph.Skullcap", cap)


def build_body(profile, armature, materials, created):
    body = profile.get("body")
    if not isinstance(body, dict):
        return
    for spec in body.get("segments", []):
        center, direction, _forward, _right, length = bone_frame(armature, spec["bone"])
        rotation = tuple(direction.to_track_quat("Z", "Y").to_euler())
        scale = (
            float(spec["radius_right"]),
            float(spec["radius_depth"]),
            length * 0.5 * float(spec.get("length_scale", 1.0)),
        )
        value = add_ellipsoid(
            spec["name"], center, scale, rotation, materials[spec["material"]],
            armature, spec["bone"], subdivisions=2,
        )
        created.append(value.name)
    for spec in body.get("attachments", []):
        center, direction, forward, right, _length = bone_frame(
            armature, spec["bone"], bool(spec.get("at_tail")),
        )
        offset = spec.get("offset", [0.0, 0.0, 0.0])
        location = center + right * float(offset[0]) + direction * float(offset[1]) + forward * float(offset[2])
        rotation = tuple(direction.to_track_quat("Z", "Y").to_euler())
        value = add_ellipsoid(
            spec["name"], location, spec["scale"], rotation, materials[spec["material"]],
            armature, spec["bone"], subdivisions=2,
        )
        created.append(value.name)


def main():
    args = parse_args()
    profile_path = os.path.abspath(args.profile)
    output = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report or (output + ".morphology-report.json"))
    profile = load_json(profile_path)
    if profile.get("family") != "biped_humanoid_v1":
        raise RuntimeError("This adapter only implements biped_humanoid_v1")

    contract_value = profile["family_contract"]
    contract_path = contract_value if os.path.isabs(contract_value) else os.path.abspath(contract_value)
    contract = load_json(contract_path)
    armature = bpy.data.objects.get("CharacterArmature")
    if armature is None:
        raise RuntimeError("Base file has no CharacterArmature")
    missing_bones = [name for name in contract.get("required_bones", []) if armature.pose.bones.get(name) is None]
    if missing_bones:
        raise RuntimeError("Base rig is missing family bones: " + ", ".join(missing_bones))

    armature.rotation_euler = (0.0, 0.0, math.radians(180.0))
    scale = float(profile.get("scale", 1.0))
    armature.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    equipment = profile.get("equipment", {}).get("weapon", {})
    grip_source = bpy.data.objects.get(equipment.get("source_grip_object", ""))
    if grip_source is None:
        raise RuntimeError("Morphology weapon source grip object is missing")
    grip_matrix = grip_source.matrix_world.copy()

    armature.animation_data_create()
    posture_actions = bake_posture(
        armature,
        profile.get("animation_actions", []),
        profile["posture"]["bones_degrees_xyz"],
    )
    armature.animation_data.action = bpy.data.actions[posture_actions[0]]
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    removed = []
    for name in profile.get("remove_objects", []):
        value = bpy.data.objects.get(name)
        if value is not None:
            removed.append(name)
            bpy.data.objects.remove(value, do_unlink=True)

    materials = {
        name: make_material("Creature." + name.title(), spec)
        for name, spec in profile.get("materials", {}).items()
    }
    created = []
    build_body(profile, armature, materials, created)
    build_head(profile, armature, materials, created)
    if equipment.get("kind") == "spiked_club":
        add_spiked_club(grip_matrix, armature, equipment.get("bone", "Weapon.R"), created)

    required_landmarks = set(profile.get("silhouette_contract", {}).get("required_landmarks", []))
    failures = []
    missing_landmarks = sorted(required_landmarks - set(created))
    if missing_landmarks:
        failures.append("missing landmarks: " + ", ".join(missing_landmarks))
    if set(profile.get("animation_actions", [])) != set(posture_actions):
        failures.append("posture action coverage is incomplete")

    armature.animation_data.action = None
    armature["asset_forge_asset_id"] = profile.get("species", profile["id"])
    armature["asset_forge_source_kind"] = "owned_blender_master"
    armature["asset_forge_generator"] = "creature_dna_biped_v1"
    armature["asset_forge_morphology_profile"] = profile["id"]
    armature["asset_forge_morphology_sha256"] = sha256_file(profile_path)
    armature["asset_forge_family"] = profile["family"]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    bpy.data.use_autopack = False
    bpy.ops.wm.save_as_mainfile(filepath=output)

    report = {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "profile_id": profile["id"],
        "profile_sha256": sha256_file(profile_path),
        "family": profile["family"],
        "family_contract": contract_path,
        "created_parts": created,
        "removed_objects": removed,
        "posture_actions": posture_actions,
        "output": output,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if failures:
        raise RuntimeError("Creature DNA build failed: " + "; ".join(failures))
    print("ASSET_FORGE_CREATURE=" + output)
    print("ASSET_FORGE_MORPHOLOGY_REPORT=" + report_path)


main()
