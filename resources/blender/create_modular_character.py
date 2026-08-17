"""Build an original, rigged, four-direction Text2Model Forge character from JSON.

The factory intentionally uses only Blender primitives and project-authored data.  It
is not a replacement for final art direction, but it provides a coherent owned master,
stable topology, reusable actions, and a fast place for an artist to refine the result.
"""

import argparse
import json
import math
import os
import sys

import bpy


def parse_args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(raw)


def rgb(value):
    if isinstance(value, str):
        text = value.lstrip("#")
        return tuple(int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
    return tuple(float(channel) for channel in value)


def make_material(name, color, metallic=0.0, roughness=0.78, emission=0.0):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    node = value.node_tree.nodes.get("Principled BSDF")
    base = rgb(color)
    node.inputs["Base Color"].default_value = (*base, 1.0)
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    if emission > 0:
        emission_input = node.inputs.get("Emission Color") or node.inputs.get("Emission")
        if emission_input is not None:
            emission_input.default_value = (*base, 1.0)
        strength = node.inputs.get("Emission Strength")
        if strength is not None:
            strength.default_value = emission
    value.diffuse_color = (*base, 1.0)
    return value


def add_bone(data, name, head, tail, parent=None):
    bone = data.edit_bones.new(name)
    bone.head = head
    bone.tail = tail
    bone.parent = parent
    return bone


def create_armature(scale):
    data = bpy.data.armatures.new("EmberCharacterRig")
    armature = bpy.data.objects.new("Armature", data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    def p(x, y, z):
        return (x * scale, y * scale, z * scale)

    root = add_bone(data, "root", p(0, 0, 0.08), p(0, 0, 0.42))
    hips = add_bone(data, "hips", p(0, 0, 0.42), p(0, 0, 1.00), root)
    spine = add_bone(data, "spine", p(0, 0, 1.00), p(0, 0, 1.78), hips)
    neck = add_bone(data, "neck", p(0, 0, 1.78), p(0, 0, 2.06), spine)
    add_bone(data, "head", p(0, 0, 2.06), p(0, 0, 2.48), neck)
    upper_l = add_bone(data, "upper_arm.L", p(-0.30, 0, 1.70), p(-0.68, 0, 1.42), spine)
    add_bone(data, "forearm.L", p(-0.68, 0, 1.42), p(-0.88, 0, 1.08), upper_l)
    upper_r = add_bone(data, "upper_arm.R", p(0.30, 0, 1.70), p(0.68, 0, 1.42), spine)
    add_bone(data, "forearm.R", p(0.68, 0, 1.42), p(0.88, 0, 1.08), upper_r)
    thigh_l = add_bone(data, "thigh.L", p(-0.18, 0, 0.96), p(-0.20, 0, 0.52), hips)
    add_bone(data, "shin.L", p(-0.20, 0, 0.52), p(-0.20, 0, 0.08), thigh_l)
    thigh_r = add_bone(data, "thigh.R", p(0.18, 0, 0.96), p(0.20, 0, 0.52), hips)
    add_bone(data, "shin.R", p(0.20, 0, 0.52), p(0.20, 0, 0.08), thigh_r)
    bpy.ops.object.mode_set(mode="POSE")
    for bone in armature.pose.bones:
        bone.rotation_mode = "XYZ"
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    return armature


def parent_to_bone(value, armature, bone_name):
    world = value.matrix_world.copy()
    value.parent = armature
    value.parent_type = "BONE"
    value.parent_bone = bone_name
    value.matrix_world = world


def finish_part(value, name, material, armature, bone, bevel=0.025):
    value.name = name
    if bevel > 0:
        modifier = value.modifiers.new("Forged edge", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
    value.data.materials.append(material)
    parent_to_bone(value, armature, bone)
    return value


def cube(name, location, scale, material, armature, bone, rotation=(0, 0, 0), bevel=0.025):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    value = bpy.context.object
    value.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish_part(value, name, material, armature, bone, bevel)


def sphere(name, location, scale, material, armature, bone, subdivisions=3):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions, location=location)
    value = bpy.context.object
    value.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish_part(value, name, material, armature, bone, 0)


def cylinder(name, location, radius, depth, material, armature, bone, rotation=(0, 0, 0), vertices=12):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation
    )
    return finish_part(bpy.context.object, name, material, armature, bone, radius * 0.08)


def cone(name, location, radius1, radius2, depth, material, armature, bone, rotation=(0, 0, 0), vertices=8):
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=radius1,
        radius2=radius2,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    return finish_part(bpy.context.object, name, material, armature, bone, radius1 * 0.06)


def palette(spec):
    colors = spec["palette"]
    return {
        "skin": make_material("Skin", colors["skin"], roughness=0.88),
        "cloth": make_material("Muted Cloth", colors["cloth"], roughness=0.92),
        "cloth_dark": make_material("Dark Cloth", colors["cloth_dark"], roughness=0.94),
        "leather": make_material("Worn Leather", colors["leather"], roughness=0.9),
        "metal": make_material("Worn Metal", colors["metal"], metallic=0.5, roughness=0.42),
        "metal_edge": make_material("Metal Edges", colors["metal_edge"], metallic=0.55, roughness=0.3),
        "gold": make_material("Gold Trim", colors.get("gold", "#c9973f"), metallic=0.8, roughness=0.32),
        "bone": make_material("Bone", colors.get("bone", "#c7b58b"), roughness=0.84),
        "eye": make_material("Eyes", colors.get("eye", "#e67b2e"), roughness=0.42, emission=0.8),
        "ember": make_material("Ember Accent", colors["ember"], roughness=0.45, emission=1.15),
    }


def common_body(spec, armature, materials):
    s = float(spec.get("scale", 1.0))
    bulk = float(spec.get("bulk", 1.0))
    limb = float(spec.get("limb_bulk", bulk))
    head_scale = float(spec.get("head_scale", 1.0))
    archetype = spec["archetype"]

    # Chunky heroic read: broad chest over a narrower waist, not one vertical slab.
    cube("Chest", (0, 0, 1.60 * s), (0.44 * bulk * s, 0.28 * bulk * s, 0.26 * s), materials["cloth"], armature, "spine", bevel=0.09 * s)
    cube("Waist", (0, 0, 1.22 * s), (0.30 * bulk * s, 0.23 * bulk * s, 0.20 * s), materials["cloth"], armature, "spine", bevel=0.06 * s)
    cube("Belt", (0, 0, 1.02 * s), (0.36 * bulk * s, 0.27 * bulk * s, 0.085 * s), materials["leather"], armature, "hips", bevel=0.035 * s)
    cube("Belt Buckle", (0, 0.26 * bulk * s, 1.02 * s), (0.09 * s, 0.03 * s, 0.07 * s), materials["gold"], armature, "hips", bevel=0.015 * s)
    cube("Front Tabard", (0, 0.26 * s, 1.18 * s), (0.22 * s, 0.03 * s, 0.34 * s), materials["cloth_dark"], armature, "spine", bevel=0.018 * s)

    head_z = (2.16 if archetype == "footman" else 2.08) * s
    sphere("Head", (0, 0.025 * s, head_z), (0.30 * head_scale * s, 0.275 * head_scale * s, 0.32 * head_scale * s), materials["skin"], armature, "head")
    for side, x in (("L", -1), ("R", 1)):
        cube(f"Upper Arm {side}", (0.50 * x * s, 0, 1.53 * s), (0.19 * limb * s, 0.20 * limb * s, 0.30 * s), materials["cloth"], armature, f"upper_arm.{side}", rotation=(0, math.radians(34 * x), 0), bevel=0.06 * s)
        cube(f"Bracer {side}", (0.78 * x * s, 0, 1.24 * s), (0.165 * limb * s, 0.18 * limb * s, 0.26 * s), materials["leather"], armature, f"forearm.{side}", rotation=(0, math.radians(25 * x), 0), bevel=0.05 * s)
        # Oversized mitt hands: the game reads gestures at ~100 px, fingers never resolve.
        sphere(f"Hand {side}", (0.92 * x * s, 0.02 * s, 1.04 * s), (0.15 * limb * s, 0.14 * limb * s, 0.16 * limb * s), materials["skin"] if archetype == "goblin" else materials["leather"], armature, f"forearm.{side}", subdivisions=2)
        cube(f"Thigh {side}", (0.19 * x * s, 0, 0.76 * s), (0.175 * limb * s, 0.21 * limb * s, 0.29 * s), materials["cloth_dark"], armature, f"thigh.{side}", bevel=0.05 * s)
        cube(f"Shin {side}", (0.20 * x * s, 0, 0.30 * s), (0.16 * limb * s, 0.19 * limb * s, 0.27 * s), materials["leather"], armature, f"shin.{side}", bevel=0.045 * s)
        cube(f"Boot {side}", (0.20 * x * s, 0.12 * s, 0.10 * s), (0.19 * limb * s, 0.33 * s, 0.115 * s), materials["leather"], armature, f"shin.{side}", bevel=0.045 * s)
        cube(f"Boot Cap {side}", (0.20 * x * s, 0.40 * s, 0.09 * s), (0.15 * limb * s, 0.075 * s, 0.09 * s), materials["cloth_dark"], armature, f"shin.{side}", bevel=0.03 * s)


def build_footman(spec, armature, m):
    s = float(spec.get("scale", 1.0))
    common_body(spec, armature, m)
    cube("Breastplate", (0, 0.29 * s, 1.60 * s), (0.40 * s, 0.06 * s, 0.27 * s), m["metal"], armature, "spine", bevel=0.075 * s)
    cube("Breastplate Trim", (0, 0.30 * s, 1.44 * s), (0.36 * s, 0.05 * s, 0.045 * s), m["gold"], armature, "spine", bevel=0.02 * s)
    cube("Backplate", (0, -0.29 * s, 1.60 * s), (0.40 * s, 0.05 * s, 0.27 * s), m["metal"], armature, "spine", bevel=0.075 * s)
    for side, x in (("L", -1), ("R", 1)):
        # Oversized pauldrons carry the heroic silhouette.
        sphere(f"Pauldron {side}", (0.42 * x * s, 0, 1.74 * s), (0.25 * s, 0.26 * s, 0.20 * s), m["metal"], armature, f"upper_arm.{side}", subdivisions=2)
        cube(f"Pauldron Rim {side}", (0.42 * x * s, 0, 1.62 * s), (0.23 * s, 0.23 * s, 0.03 * s), m["gold"], armature, f"upper_arm.{side}", bevel=0.02 * s)
        cube(f"Gauntlet {side}", (0.90 * x * s, 0.02 * s, 1.08 * s), (0.17 * s, 0.16 * s, 0.15 * s), m["metal"], armature, f"forearm.{side}", bevel=0.04 * s)
        cube(f"Knee Plate {side}", (0.20 * x * s, 0.20 * s, 0.49 * s), (0.17 * s, 0.04 * s, 0.13 * s), m["metal"], armature, f"shin.{side}", bevel=0.035 * s)
    sphere("Helmet Dome", (0, 0.01 * s, 2.28 * s), (0.37 * s, 0.35 * s, 0.32 * s), m["metal"], armature, "head")
    cube("Helmet Face Gap", (0, 0.315 * s, 2.13 * s), (0.25 * s, 0.04 * s, 0.10 * s), m["cloth_dark"], armature, "head", bevel=0.015 * s)
    cube("Helmet Brow", (0, 0.345 * s, 2.24 * s), (0.37 * s, 0.06 * s, 0.06 * s), m["gold"], armature, "head", bevel=0.025 * s)
    cube("Nose Guard", (0, 0.375 * s, 2.10 * s), (0.05 * s, 0.05 * s, 0.19 * s), m["metal_edge"], armature, "head", bevel=0.012 * s)
    cone("Helmet Crest", (0, -0.06 * s, 2.56 * s), 0.075 * s, 0.02 * s, 0.24 * s, m["ember"], armature, "head", vertices=6)
    cube("Cape", (0, -0.37 * s, 1.38 * s), (0.42 * s, 0.04 * s, 0.72 * s), m["cloth_dark"], armature, "spine", rotation=(math.radians(-6), 0, 0), bevel=0.03 * s)

    # Heater shield on the left forearm, plate facing the side-view camera.
    cube("Shield Plate", (-0.99 * s, 0.10 * s, 1.34 * s), (0.05 * s, 0.44 * s, 0.34 * s), m["cloth"], armature, "forearm.L", bevel=0.03 * s)
    cube("Shield Point", (-0.99 * s, 0.10 * s, 0.98 * s), (0.05 * s, 0.32 * s, 0.32 * s), m["cloth"], armature, "forearm.L", rotation=(math.radians(45), 0, 0), bevel=0.03 * s)
    cube("Shield Rim Top", (-0.99 * s, 0.10 * s, 1.68 * s), (0.06 * s, 0.46 * s, 0.05 * s), m["gold"], armature, "forearm.L", bevel=0.02 * s)
    sphere("Shield Boss", (-1.06 * s, 0.10 * s, 1.30 * s), (0.07 * s, 0.15 * s, 0.15 * s), m["gold"], armature, "forearm.L", subdivisions=2)
    cube("Shield Sigil", (-1.05 * s, 0.10 * s, 1.52 * s), (0.025 * s, 0.10 * s, 0.10 * s), m["ember"], armature, "forearm.L", rotation=(math.radians(45), 0, 0), bevel=0.012 * s)

    # Arming sword gripped in the right mitt, blade up: pommel, gold crossguard, bright blade.
    sphere("Sword Pommel", (0.92 * s, 0.02 * s, 0.84 * s), (0.07 * s, 0.07 * s, 0.07 * s), m["gold"], armature, "forearm.R", subdivisions=2)
    cylinder("Sword Grip", (0.92 * s, 0.02 * s, 1.02 * s), 0.05 * s, 0.30 * s, m["leather"], armature, "forearm.R", vertices=8)
    cube("Sword Guard", (0.92 * s, 0.02 * s, 1.20 * s), (0.20 * s, 0.07 * s, 0.045 * s), m["gold"], armature, "forearm.R", bevel=0.02 * s)
    cube("Sword Blade", (0.92 * s, 0.02 * s, 1.66 * s), (0.075 * s, 0.035 * s, 0.44 * s), m["metal_edge"], armature, "forearm.R", bevel=0.02 * s)
    cone("Sword Tip", (0.92 * s, 0.02 * s, 2.18 * s), 0.075 * s, 0.005 * s, 0.18 * s, m["metal_edge"], armature, "forearm.R", vertices=4)
    cube("Ember Sigil", (0, 0.355 * s, 1.60 * s), (0.09 * s, 0.028 * s, 0.09 * s), m["ember"], armature, "spine", rotation=(0, 0, math.radians(45)), bevel=0.012 * s)


def build_goblin(spec, armature, m):
    s = float(spec.get("scale", 0.78))
    common_body(spec, armature, m)
    sphere("Hunched Torso", (0, 0.015 * s, 1.45 * s), (0.43 * s, 0.30 * s, 0.48 * s), m["cloth"], armature, "spine", subdivisions=2)
    sphere("Heavy Jaw", (0, 0.10 * s, 2.02 * s), (0.25 * s, 0.245 * s, 0.20 * s), m["skin"], armature, "head", subdivisions=2)
    # Large ears and bright, asymmetric face make front/back readable at mobile scale.
    for side, x in (("L", -1), ("R", 1)):
        cone(f"Ear {side}", (0.43 * x * s, 0.02 * s, 2.09 * s), 0.19 * s, 0.025 * s, 0.55 * s, m["skin"], armature, "head", rotation=(0, math.radians(90 * x), 0), vertices=6)
        sphere(f"Eye {side}", (0.105 * x * s, 0.305 * s, 2.17 * s), (0.078 * s, 0.042 * s, 0.062 * s), m["eye"], armature, "head", subdivisions=2)
        cube(f"Brow {side}", (0.105 * x * s, 0.335 * s, 2.235 * s), (0.105 * s, 0.025 * s, 0.035 * s), m["cloth_dark"], armature, "head", rotation=(0, 0, math.radians(-12 * x)), bevel=0.012 * s)
        cone(f"Tusk {side}", (0.10 * x * s, 0.33 * s, 1.98 * s), 0.055 * s, 0.0, 0.22 * s, m["bone"], armature, "head", rotation=(math.radians(82), 0, 0), vertices=6)
        sphere(f"Bone Pauldron {side}", (0.41 * x * s, 0, 1.67 * s), (0.22 * s, 0.23 * s, 0.16 * s), m["bone"], armature, f"upper_arm.{side}", subdivisions=1)
        for claw in range(3):
            cone(
                f"Claw {side} {claw}",
                (0.92 * x * s + (claw - 1) * 0.05 * s, 0.14 * s, 0.98 * s),
                0.03 * s, 0.0, 0.14 * s, m["bone"], armature, f"forearm.{side}",
                rotation=(math.radians(95), 0, 0), vertices=4,
            )
    sphere("Goblin Nose", (0, 0.31 * s, 2.08 * s), (0.105 * s, 0.095 * s, 0.09 * s), m["skin"], armature, "head", subdivisions=2)
    cube("Goblin Mouth", (0, 0.315 * s, 1.995 * s), (0.13 * s, 0.025 * s, 0.025 * s), m["cloth_dark"], armature, "head", bevel=0.01 * s)
    cube("Chest Harness", (0, 0.28 * s, 1.46 * s), (0.34 * s, 0.04 * s, 0.065 * s), m["leather"], armature, "spine", rotation=(0, 0, math.radians(-16)), bevel=0.025 * s)
    cube("Loincloth", (0, 0.27 * s, 0.94 * s), (0.23 * s, 0.035 * s, 0.29 * s), m["cloth_dark"], armature, "hips", rotation=(math.radians(8), 0, 0), bevel=0.02 * s)

    # Heavy spiked club gripped in the right claw, head up: the goblin's whole read is
    # "small body, big crude weapon".
    cylinder("Club Grip", (0.92 * s, 0.02 * s, 1.00 * s), 0.055 * s, 0.34 * s, m["leather"], armature, "forearm.R", vertices=8)
    cone("Club Head", (0.92 * s, 0.02 * s, 1.58 * s), 0.11 * s, 0.20 * s, 0.72 * s, m["cloth_dark"], armature, "forearm.R", vertices=8)
    cylinder("Club Band", (0.92 * s, 0.02 * s, 1.78 * s), 0.205 * s, 0.07 * s, m["metal"], armature, "forearm.R", vertices=8)
    for index, (angle, spike_z) in enumerate(((20, 1.62), (110, 1.76), (200, 1.68), (290, 1.84))):
        radians = math.radians(angle)
        cone(
            f"Club Spike {index}",
            (0.92 * s + math.cos(radians) * 0.21 * s, 0.02 * s + math.sin(radians) * 0.21 * s, spike_z * s),
            0.05 * s, 0.0, 0.20 * s, m["metal_edge"], armature, "forearm.R",
            rotation=(math.radians(-90) * math.sin(radians), math.radians(90) * math.cos(radians), 0), vertices=5,
        )
    cube("Ember Token", (0.14 * s, 0.33 * s, 1.31 * s), (0.065 * s, 0.02 * s, 0.065 * s), m["ember"], armature, "spine", rotation=(0, 0, math.radians(45)), bevel=0.012 * s)


def deg(values):
    return {name: tuple(math.radians(axis) for axis in rotation) for name, rotation in values.items()}


def merge(base, extra):
    value = dict(base)
    value.update(extra)
    return value


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
    action.frame_start = min(item[0] for item in frames)
    action.frame_end = max(item[0] for item in frames)
    # Blender 5 actions use slotted channel bags and no longer expose action.fcurves.
    # Keyframe insertion already defaults to Bezier interpolation, which is the desired
    # eased motion here; avoid version-specific channel traversal.
    armature.animation_data.action = None


def build_actions(armature, spec):
    s = float(spec.get("scale", 1.0))
    goblin = spec["archetype"] == "goblin"
    neutral = deg({"spine": (8, 0, 0), "neck": (-6, 0, 0)}) if goblin else {}

    def r(values):
        return merge(neutral, deg(values))

    create_action(armature, "idle", [
        (1, neutral, {}, {}),
        (8, r({"spine": (8 if goblin else 2, 0, 1), "upper_arm.L": (-4, 0, -2), "upper_arm.R": (-4, 0, 2), "neck": (-8 if goblin else -2, 0, 0)}), {"root": (0, 0, 0.03 * s)}, {}),
        (16, r({"spine": (12 if goblin else 3, 0, -1), "upper_arm.L": (-2, 0, 0), "upper_arm.R": (-2, 0, 0)}), {"root": (0, 0, 0.015 * s)}, {}),
        (24, neutral, {}, {}),
    ])
    walk_a = r({"thigh.L": (-34, 0, 0), "shin.L": (20, 0, 0), "thigh.R": (34, 0, 0), "upper_arm.L": (26, 0, 0), "upper_arm.R": (-26, 0, 0), "hips": (0, 0, 4), "spine": (4, 0, -4)})
    walk_pass_a = r({"thigh.L": (8, 0, 0), "shin.L": (30, 0, 0), "thigh.R": (-6, 0, 0), "upper_arm.L": (4, 0, 0), "upper_arm.R": (-4, 0, 0)})
    walk_b = r({"thigh.L": (34, 0, 0), "thigh.R": (-34, 0, 0), "shin.R": (20, 0, 0), "upper_arm.L": (-26, 0, 0), "upper_arm.R": (26, 0, 0), "hips": (0, 0, -4), "spine": (4, 0, 4)})
    walk_pass_b = r({"thigh.R": (8, 0, 0), "shin.R": (30, 0, 0), "thigh.L": (-6, 0, 0), "upper_arm.L": (-4, 0, 0), "upper_arm.R": (4, 0, 0)})
    create_action(armature, "walk", [
        (1, walk_a, {}, {}), (7, walk_pass_a, {"root": (0, 0, 0.06 * s)}, {}),
        (13, walk_b, {}, {}), (19, walk_pass_b, {"root": (0, 0, 0.06 * s)}, {}), (25, walk_a, {}, {}),
    ])
    create_action(armature, "attack", [
        (1, neutral, {}, {}),
        (6, r({"hips": (0, 0, -14), "spine": (-12, 0, -26), "upper_arm.R": (-70, -10, -50), "forearm.R": (-40, 0, 0), "upper_arm.L": (10, 0, -8), "neck": (0, 0, 14)}), {"root": (0, -0.06 * s, -0.02 * s)}, {}),
        (9, r({"hips": (0, 0, -16), "spine": (-14, 0, -28), "upper_arm.R": (-76, -10, -54), "forearm.R": (-44, 0, 0), "upper_arm.L": (12, 0, -8), "neck": (0, 0, 16)}), {"root": (0, -0.06 * s, -0.02 * s)}, {}),
        (11, r({"hips": (0, 0, 14), "spine": (18, 0, 30), "upper_arm.R": (95, 8, 62), "forearm.R": (50, 0, 0), "upper_arm.L": (-22, 0, -12), "neck": (0, 0, -12)}), {"root": (0, 0.16 * s, 0.02 * s)}, {}),
        (13, r({"hips": (0, 0, 15), "spine": (20, 0, 32), "upper_arm.R": (102, 8, 66), "forearm.R": (54, 0, 0), "upper_arm.L": (-24, 0, -12), "neck": (0, 0, -12)}), {"root": (0, 0.17 * s, 0.01 * s)}, {}),
        (18, r({"spine": (8, 0, 12), "upper_arm.R": (42, 0, 24), "forearm.R": (18, 0, 0)}), {"root": (0, 0.04 * s, 0)}, {}),
        (24, neutral, {}, {}),
    ])
    create_action(armature, "heavy_attack", [
        (1, neutral, {}, {}),
        (8, r({"hips": (0, 0, -14), "spine": (-18, 0, -25), "upper_arm.R": (-78, -10, -54), "forearm.R": (-48, 0, 0), "upper_arm.L": (-28, 0, -18)}), {}, {}),
        (15, r({"hips": (0, 0, 14), "spine": (20, 0, 30), "upper_arm.R": (96, 8, 62), "forearm.R": (52, 0, 0)}), {"root": (0, 0.16 * s, 0)}, {}),
        (22, r({"spine": (8, 0, 12), "upper_arm.R": (38, 0, 24)}), {}, {}),
        (30, neutral, {}, {}),
    ])
    create_action(armature, "defend", [
        (1, neutral, {}, {}),
        (6, r({"spine": (-8, 0, -8), "upper_arm.L": (-58, -10, 38), "forearm.L": (-40, 0, 0), "upper_arm.R": (16, 0, 8)}), {}, {}),
        (14, r({"spine": (-10, 0, -10), "upper_arm.L": (-64, -10, 42), "forearm.L": (-44, 0, 0), "upper_arm.R": (20, 0, 10)}), {"root": (0, -0.035 * s, -0.025 * s)}, {}),
        (22, neutral, {}, {}),
    ])
    create_action(armature, "dodge", [
        (1, neutral, {}, {}),
        (6, r({"hips": (-14, 0, -24), "spine": (-24, 0, 28), "thigh.L": (30, 0, 0), "thigh.R": (-22, 0, 0)}), {"root": (0, -0.08 * s, -0.10 * s)}, {}),
        (12, r({"hips": (-8, 0, 16), "spine": (-14, 0, -20), "thigh.L": (-18, 0, 0), "thigh.R": (24, 0, 0)}), {"root": (0, 0.04 * s, -0.04 * s)}, {}),
        (18, neutral, {}, {}),
    ])
    create_action(armature, "hit", [
        (1, neutral, {}, {}),
        (4, r({"hips": (-12, 0, 0), "spine": (-28, 0, 12), "neck": (-14, 0, 0), "upper_arm.L": (22, 0, -20), "upper_arm.R": (22, 0, 20)}), {"root": (0, -0.12 * s, -0.04 * s)}, {"root": (0.95, 1.0, 0.93)}),
        (8, r({"hips": (-5, 0, 0), "spine": (-12, 0, 5), "upper_arm.L": (8, 0, -8), "upper_arm.R": (8, 0, 8)}), {"root": (0, -0.04 * s, -0.01 * s)}, {}),
        (12, neutral, {}, {}),
    ])
    create_action(armature, "taunt", [
        (1, neutral, {}, {}),
        (8, r({"spine": (-8, 0, 0), "upper_arm.L": (-70, 0, -45), "forearm.L": (-30, 0, 0), "upper_arm.R": (-70, 0, 45), "forearm.R": (-30, 0, 0)}), {"root": (0, 0, 0.05 * s)}, {}),
        (16, r({"spine": (14, 0, 0), "upper_arm.L": (-42, 0, -30), "upper_arm.R": (-42, 0, 30)}), {}, {}),
        (24, neutral, {}, {}),
    ])
    create_action(armature, "victory", [
        (1, neutral, {}, {}),
        (8, r({"spine": (-5, 0, 0), "upper_arm.R": (-105, 0, 8), "forearm.R": (-10, 0, 0), "upper_arm.L": (-35, 0, -20)}), {"root": (0, 0, 0.05 * s)}, {}),
        (16, r({"spine": (5, 0, 0), "upper_arm.R": (-92, 0, 5), "upper_arm.L": (-28, 0, -15)}), {}, {}),
        (24, neutral, {}, {}),
    ])
    create_action(armature, "death", [
        (1, neutral, {}, {}),
        (8, r({"hips": (-14, 0, -8), "spine": (-22, 0, -16), "upper_arm.L": (28, 0, -20), "upper_arm.R": (24, 0, 24)}), {"root": (0, 0, -0.04 * s)}, {}),
        (16, r({"hips": (-22, 0, -12), "spine": (-28, 0, -14), "thigh.L": (36, 0, 0), "thigh.R": (32, 0, 0), "shin.L": (-50, 0, 0), "shin.R": (-46, 0, 0)}), {"root": (0, 0, -0.12 * s)}, {}),
        (24, r({"hips": (-30, 0, -18), "spine": (-38, 0, -12), "thigh.L": (50, 0, 0), "thigh.R": (46, 0, 0), "shin.L": (-64, 0, 0), "shin.R": (-60, 0, 0), "upper_arm.L": (42, 0, -32), "upper_arm.R": (38, 0, 35)}), {"root": (0, 0, -0.22 * s)}, {}),
        (28, r({"hips": (-40, 0, -23), "spine": (-48, 0, -10), "thigh.L": (62, 0, 0), "thigh.R": (56, 0, 0), "shin.L": (-76, 0, 0), "shin.R": (-70, 0, 0), "upper_arm.L": (50, 0, -39), "upper_arm.R": (46, 0, 41)}), {"root": (0, 0, -0.315 * s)}, {}),
        (30, r({"hips": (-38, 0, -22), "spine": (-46, 0, -10), "thigh.L": (60, 0, 0), "thigh.R": (54, 0, 0), "shin.L": (-74, 0, 0), "shin.R": (-68, 0, 0), "upper_arm.L": (48, 0, -38), "upper_arm.R": (44, 0, 40)}), {"root": (0, 0, -0.29 * s)}, {}),
        (32, r({"hips": (-40, 0, -22), "spine": (-47, 0, -10), "thigh.L": (61, 0, 0), "thigh.R": (55, 0, 0), "shin.L": (-75, 0, 0), "shin.R": (-69, 0, 0), "upper_arm.L": (49, 0, -38), "upper_arm.R": (45, 0, 40)}), {"root": (0, 0, -0.30 * s)}, {}),
    ])


def main():
    args = parse_args()
    with open(args.spec, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("archetype") not in {"footman", "goblin"}:
        raise RuntimeError("archetype must be 'footman' or 'goblin'")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scale = float(spec.get("scale", 1.0))
    armature = create_armature(scale)
    materials = palette(spec)
    if spec["archetype"] == "footman":
        build_footman(spec, armature, materials)
    else:
        build_goblin(spec, armature, materials)
    build_actions(armature, spec)
    armature["text2model_forge_asset_id"] = spec["id"]
    armature["text2model_forge_source_kind"] = "owned_blender_master"
    armature["text2model_forge_generator"] = "modular_character_v1"
    destination = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=destination)
    print(f"TEXT2MODEL_FORGE_CHARACTER={destination}")


main()
