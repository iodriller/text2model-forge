import math
import os
import sys

import bpy
from mathutils import Matrix


def script_args():
    args = sys.argv
    return args[args.index("--") + 1 :] if "--" in args else []


def output_path():
    args = script_args()
    if len(args) != 2 or args[0] != "--output":
        raise RuntimeError(
            "Usage: blender --background --python create_humanoid_template.py -- --output <template.blend>"
        )
    return os.path.abspath(args[1])


def material(name, color, metallic=0.0, roughness=0.75):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1.0)
    value.metallic = metallic
    value.roughness = roughness
    return value


def add_bone(armature, name, head, tail, parent=None):
    bone = armature.edit_bones.new(name)
    bone.head = head
    bone.tail = tail
    bone.parent = parent
    return bone


def create_armature():
    data = bpy.data.armatures.new("EmberHumanoidRig")
    armature = bpy.data.objects.new("Armature", data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    root = add_bone(data, "root", (0, 0, 0.08), (0, 0, 0.42))
    hips = add_bone(data, "hips", (0, 0, 0.42), (0, 0, 1.00), root)
    spine = add_bone(data, "spine", (0, 0, 1.00), (0, 0, 1.78), hips)
    neck = add_bone(data, "neck", (0, 0, 1.78), (0, 0, 2.06), spine)
    add_bone(data, "head", (0, 0, 2.06), (0, 0, 2.48), neck)

    upper_arm_l = add_bone(data, "upper_arm.L", (-0.30, 0, 1.70), (-0.68, 0, 1.42), spine)
    add_bone(data, "forearm.L", (-0.68, 0, 1.42), (-0.88, 0, 1.08), upper_arm_l)
    upper_arm_r = add_bone(data, "upper_arm.R", (0.30, 0, 1.70), (0.68, 0, 1.42), spine)
    add_bone(data, "forearm.R", (0.68, 0, 1.42), (0.88, 0, 1.08), upper_arm_r)

    thigh_l = add_bone(data, "thigh.L", (-0.18, 0, 0.96), (-0.20, 0, 0.52), hips)
    add_bone(data, "shin.L", (-0.20, 0, 0.52), (-0.20, 0, 0.08), thigh_l)
    thigh_r = add_bone(data, "thigh.R", (0.18, 0, 0.96), (0.20, 0, 0.52), hips)
    add_bone(data, "shin.R", (0.20, 0, 0.52), (0.20, 0, 0.08), thigh_r)

    bpy.ops.object.mode_set(mode="POSE")
    for bone in armature.pose.bones:
        bone.rotation_mode = "XYZ"
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    return armature


def bevel(object_value, width=0.05):
    modifier = object_value.modifiers.new("Soft silhouette", "BEVEL")
    modifier.width = width
    modifier.segments = 2


def parent_to_bone(object_value, armature, bone_name):
    world = object_value.matrix_world.copy()
    object_value.parent = armature
    object_value.parent_type = "BONE"
    object_value.parent_bone = bone_name
    object_value.matrix_world = world


def add_cube(name, location, scale, material_value, armature, bone, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    value = bpy.context.object
    value.name = name
    value.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel(value)
    value.data.materials.append(material_value)
    parent_to_bone(value, armature, bone)
    return value


def add_sphere(name, location, scale, material_value, armature, bone):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, location=location)
    value = bpy.context.object
    value.name = name
    value.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    value.data.materials.append(material_value)
    parent_to_bone(value, armature, bone)
    return value


def build_character(armature):
    steel = material("Tarnished Steel", (0.13, 0.18, 0.21), metallic=0.62, roughness=0.44)
    steel_light = material("Steel Edge", (0.28, 0.34, 0.37), metallic=0.7, roughness=0.38)
    cloth = material("Muted Defender Blue", (0.035, 0.11, 0.17), roughness=0.88)
    leather = material("Worn Leather", (0.14, 0.065, 0.028), roughness=0.94)
    ember = material("Ember Accent", (0.9, 0.16, 0.025), metallic=0.08, roughness=0.5)
    skin = material("Neutral Skin", (0.43, 0.27, 0.18), roughness=0.86)

    add_cube("Torso", (0, 0, 1.43), (0.40, 0.25, 0.48), cloth, armature, "spine")
    add_cube("Belt", (0, 0, 1.02), (0.42, 0.27, 0.09), leather, armature, "hips")
    add_cube("Tabard", (0, 0.27, 1.18), (0.27, 0.035, 0.47), cloth, armature, "spine")
    add_sphere("Head", (0, 0.01, 2.18), (0.29, 0.25, 0.31), skin, armature, "head")
    add_sphere("Helmet", (0, -0.015, 2.28), (0.34, 0.30, 0.30), steel, armature, "head")
    add_cube("Helmet Brow", (0, 0.275, 2.22), (0.34, 0.055, 0.055), steel_light, armature, "head")
    add_cube("Nose Guard", (0, 0.33, 2.10), (0.045, 0.045, 0.18), steel_light, armature, "head")

    add_cube("Upper Arm L", (-0.48, 0, 1.54), (0.18, 0.18, 0.31), steel, armature, "upper_arm.L", rotation=(0, math.radians(-34), 0))
    add_cube("Forearm L", (-0.78, 0, 1.25), (0.15, 0.16, 0.27), leather, armature, "forearm.L", rotation=(0, math.radians(-25), 0))
    add_cube("Upper Arm R", (0.48, 0, 1.54), (0.18, 0.18, 0.31), steel, armature, "upper_arm.R", rotation=(0, math.radians(34), 0))
    add_cube("Forearm R", (0.78, 0, 1.25), (0.15, 0.16, 0.27), leather, armature, "forearm.R", rotation=(0, math.radians(25), 0))

    add_cube("Thigh L", (-0.19, 0, 0.76), (0.16, 0.19, 0.29), cloth, armature, "thigh.L")
    add_cube("Shin L", (-0.20, 0, 0.29), (0.15, 0.18, 0.27), steel, armature, "shin.L")
    add_cube("Boot L", (-0.20, 0.09, 0.08), (0.17, 0.27, 0.10), leather, armature, "shin.L")
    add_cube("Thigh R", (0.19, 0, 0.76), (0.16, 0.19, 0.29), cloth, armature, "thigh.R")
    add_cube("Shin R", (0.20, 0, 0.29), (0.15, 0.18, 0.27), steel, armature, "shin.R")
    add_cube("Boot R", (0.20, 0.09, 0.08), (0.17, 0.27, 0.10), leather, armature, "shin.R")

    add_cube("Shield", (-0.90, 0.12, 1.25), (0.08, 0.38, 0.52), steel, armature, "forearm.L")
    add_cube("Shield Ember", (-0.99, 0.14, 1.25), (0.025, 0.10, 0.10), ember, armature, "forearm.L")
    add_cube("Sword Blade", (0.98, 0, 1.37), (0.055, 0.07, 0.60), steel_light, armature, "forearm.R", rotation=(0, math.radians(-12), 0))
    add_cube("Sword Guard", (0.86, 0, 0.91), (0.20, 0.08, 0.055), steel, armature, "forearm.R")
    add_cube("Ember Badge", (0, 0.31, 1.46), (0.10, 0.035, 0.10), ember, armature, "spine", rotation=(0, 0, math.radians(45)))


def pose_frame(armature, frame, rotations=None, locations=None, scales=None):
    rotations = rotations or {}
    locations = locations or {}
    scales = scales or {}
    for bone in armature.pose.bones:
        bone.rotation_euler = rotations.get(bone.name, (0, 0, 0))
        bone.location = locations.get(bone.name, (0, 0, 0))
        bone.scale = scales.get(bone.name, (1, 1, 1))
        bone.keyframe_insert("rotation_euler", frame=frame)
        bone.keyframe_insert("location", frame=frame)
        bone.keyframe_insert("scale", frame=frame)


def create_action(armature, name, frames):
    action = bpy.data.actions.new(name=name)
    action.use_fake_user = True
    armature.animation_data_create()
    armature.animation_data.action = action
    for frame, rotations, locations, scales in frames:
        pose_frame(armature, frame, rotations, locations, scales)
    action.frame_start = min(frame[0] for frame in frames)
    action.frame_end = max(frame[0] for frame in frames)
    armature.animation_data.action = None


def radians_map(values):
    return {name: tuple(math.radians(axis) for axis in rotation) for name, rotation in values.items()}


def build_actions(armature):
    neutral = {}
    create_action(armature, "idle", [
        (1, neutral, {}, {}),
        (12, radians_map({"spine": (2, 0, 0), "upper_arm.L": (-2, 0, 0), "upper_arm.R": (-2, 0, 0)}), {"root": (0, 0, 0.025)}, {}),
        (24, neutral, {}, {}),
    ])

    walk_a = radians_map({"thigh.L": (-28, 0, 0), "shin.L": (16, 0, 0), "thigh.R": (28, 0, 0), "upper_arm.L": (20, 0, 0), "upper_arm.R": (-20, 0, 0)})
    walk_b = radians_map({"thigh.L": (28, 0, 0), "thigh.R": (-28, 0, 0), "shin.R": (16, 0, 0), "upper_arm.L": (-20, 0, 0), "upper_arm.R": (20, 0, 0)})
    create_action(armature, "walk", [
        (1, walk_a, {}, {}),
        (7, neutral, {"root": (0, 0, 0.055)}, {}),
        (13, walk_b, {}, {}),
        (19, neutral, {"root": (0, 0, 0.055)}, {}),
        (25, walk_a, {}, {}),
    ])

    create_action(armature, "attack", [
        (1, neutral, {}, {}),
        (6, radians_map({"hips": (0, 0, -8), "spine": (-8, 0, -14), "upper_arm.R": (-46, -8, -38), "forearm.R": (-28, 0, 0)}), {"root": (0, -0.06, 0)}, {}),
        (11, radians_map({"hips": (0, 0, 8), "spine": (12, 0, 18), "upper_arm.R": (72, 5, 48), "forearm.R": (38, 0, 0), "upper_arm.L": (-16, 0, -8)}), {"root": (0, 0.20, 0.025)}, {}),
        (17, radians_map({"spine": (4, 0, 7), "upper_arm.R": (28, 0, 18)}), {"root": (0, 0.10, 0)}, {}),
        (24, neutral, {}, {}),
    ])

    create_action(armature, "shoot", [
        (1, neutral, {}, {}),
        (6, radians_map({"upper_arm.L": (0, -15, -58), "forearm.L": (0, 0, 36), "upper_arm.R": (0, 15, 58), "forearm.R": (0, 0, -70), "spine": (0, 0, -5)}), {}, {}),
        (12, radians_map({"upper_arm.L": (0, -15, -58), "forearm.L": (0, 0, 36), "upper_arm.R": (0, 8, 25), "forearm.R": (0, 0, -12)}), {"root": (0, -0.025, 0)}, {}),
        (20, neutral, {}, {}),
    ])

    create_action(armature, "hit", [
        (1, neutral, {}, {}),
        (7, radians_map({"hips": (-8, 0, 0), "spine": (-18, 0, 7), "upper_arm.L": (12, 0, -10), "upper_arm.R": (12, 0, 10)}), {"root": (0, -0.14, -0.025)}, {"root": (0.97, 1.0, 0.96)}),
        (12, neutral, {}, {}),
    ])

    create_action(armature, "death", [
        (1, neutral, {}, {}),
        (9, radians_map({"hips": (-12, 0, -10), "spine": (-18, 0, -18), "upper_arm.L": (24, 0, -18), "upper_arm.R": (20, 0, 22)}), {"root": (0.08, -0.06, -0.08)}, {}),
        (17, radians_map({"hips": (-54, 12, -28), "spine": (-24, 0, -12), "thigh.L": (30, 0, 0), "thigh.R": (-18, 0, 0)}), {"root": (0.22, -0.05, -0.62)}, {}),
        (25, radians_map({"hips": (-82, 18, -45), "spine": (-18, 0, -10), "upper_arm.L": (35, 0, -30), "upper_arm.R": (28, 0, 35)}), {"root": (0.34, -0.02, -0.98)}, {}),
    ])


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armature = create_armature()
    build_character(armature)
    build_actions(armature)

    destination = output_path()
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=destination)
    print(f"TEXT2MODEL_FORGE_HUMANOID_TEMPLATE={destination}")


main()
