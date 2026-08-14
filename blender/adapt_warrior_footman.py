"""Adapt the CC0 Quaternius Warrior into the Ember Defense footman master.

The Warrior contributes what the primitive factory could not: real anatomy, a
deformation rig, and professionally keyed actions (Sword_Attack, RecieveHit, Death,
Idle_Attacking). This adapter adds the one thing the game brief requires that the
pack lacks — an authored heater shield, socketed to the left forearm bone exactly
like the pack sockets its sword to Weapon.R — and saves the result as an owned master.

Usage: blender --background <warrior.blend> --python adapt_warrior_footman.py -- --output <master.blend>
"""

import argparse
import math
import os
import sys

import bpy


def parse_args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args(raw)


def make_material(name, color, metallic=0.0, roughness=0.8, emission=0.0):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    node = value.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    if emission > 0:
        emission_input = node.inputs.get("Emission Color") or node.inputs.get("Emission")
        if emission_input is not None:
            emission_input.default_value = (*color, 1.0)
        strength = node.inputs.get("Emission Strength")
        if strength is not None:
            strength.default_value = emission
    value.diffuse_color = (*color, 1.0)
    return value


def bone_parent(value, armature, bone_name):
    world = value.matrix_world.copy()
    value.parent = armature
    value.parent_type = "BONE"
    value.parent_bone = bone_name
    value.matrix_world = world


def add_part(name, primitive, location, scale, material, armature, bone, rotation=(0, 0, 0), bevel=0.02):
    primitive(location=location, rotation=rotation)
    value = bpy.context.object
    value.name = name
    value.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0:
        modifier = value.modifiers.new("Forged edge", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
    value.data.materials.append(material)
    bone_parent(value, armature, bone)
    return value


def main():
    args = parse_args()
    armature = bpy.data.objects["CharacterArmature"]

    # The pack authors the character facing -Y; Ember Defense fronts face +Y (south
    # camera). Rotate the whole rig so direction labels stay semantically true.
    armature.rotation_euler = (0.0, 0.0, math.radians(180.0))
    bpy.context.view_layer.update()

    # Pose the rig in its dominant stance BEFORE measuring socket positions: bone
    # parenting bakes the offset against the CURRENT pose, and idle/walk/attack all
    # keep the arm near the Idle_Weapon stance. Resting-pose offsets float in space.
    armature.animation_data_create()
    armature.animation_data.action = bpy.data.actions["Idle_Weapon"]
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    arm_bone = armature.pose.bones["LowerArm.L"]
    mid = armature.matrix_world @ ((arm_bone.head + arm_bone.tail) / 2)
    outward = 1.0 if mid.x >= 0 else -1.0
    head_bone = armature.pose.bones["Head"]
    head = armature.matrix_world @ ((head_bone.head + head_bone.tail) / 2)

    navy = make_material("Shield Navy", (0.086, 0.16, 0.30), roughness=0.85)
    steel = make_material("Helm Steel", (0.10, 0.12, 0.16), metallic=0.12, roughness=0.6)
    gold = make_material("Shield Gold", (0.62, 0.45, 0.15), metallic=0.8, roughness=0.32)
    ember = make_material("Shield Ember", (0.95, 0.38, 0.10), roughness=0.45, emission=1.1)

    x = mid.x + outward * 0.24
    y = mid.y + 0.06
    z = mid.z
    cube = bpy.ops.mesh.primitive_cube_add
    add_part("Shield Plate", cube, (x, y, z + 0.16), (0.06, 0.52, 0.42), navy, armature, "LowerArm.L", bevel=0.035)
    add_part("Shield Point", cube, (x, y, z - 0.28), (0.06, 0.38, 0.38), navy, armature, "LowerArm.L", rotation=(math.radians(45), 0, 0), bevel=0.035)
    add_part("Shield Rim Top", cube, (x, y, z + 0.58), (0.07, 0.54, 0.055), gold, armature, "LowerArm.L", bevel=0.02)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, location=(x + outward * 0.08, y, z + 0.12))
    boss = bpy.context.object
    boss.name = "Shield Boss"
    boss.scale = (0.075, 0.16, 0.16)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    boss.data.materials.append(gold)
    bone_parent(boss, armature, "LowerArm.L")
    add_part("Shield Sigil", cube, (x + outward * 0.065, y, z + 0.38), (0.028, 0.11, 0.11), ember, armature, "LowerArm.L", rotation=(math.radians(45), 0, 0), bevel=0.012)

    # Simple authored helmet: dome + brow band + nose guard on the Head bone. Paint
    # can weather a helmet, but it cannot be trusted to invent one on a bare head.
    # Sized to swallow the hair bun (which pokes through anything smaller); sits
    # above the brow, not on the face.
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, location=(head.x, head.y - 0.09, head.z + 0.26))
    dome = bpy.context.object
    dome.name = "Helmet Dome"
    dome.scale = (0.43, 0.54, 0.44)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.shade_smooth()
    dome.data.materials.append(steel)
    bone_parent(dome, armature, "Head")
    add_part("Helmet Brow", cube, (head.x, head.y - 0.02, head.z + 0.10), (0.40, 0.46, 0.05), gold, armature, "Head", bevel=0.02)
    add_part("Nose Guard", cube, (head.x, head.y + 0.40, head.z - 0.02), (0.045, 0.05, 0.13), steel, armature, "Head", bevel=0.012)
    # Kettle-helm neck guard: the pack's ponytail exits low at the back of the skull
    # where no believable dome can reach; an angled back plate covers it instead.
    add_part(
        "Helmet Neck Guard", cube, (head.x, head.y - 0.33, head.z - 0.04),
        (0.34, 0.10, 0.24), steel, armature, "Head",
        rotation=(math.radians(-34), 0, 0), bevel=0.02,
    )

    armature.animation_data.action = None

    armature["asset_forge_asset_id"] = "footman"
    armature["asset_forge_source_kind"] = "owned_blender_master"
    armature["asset_forge_generator"] = "warrior_adapter_v1"
    armature["asset_forge_base_license"] = "CC0-1.0 Quaternius RPG Characters"

    destination = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    # The pack ships with auto-pack pointing at the author's machine; already-packed
    # data stays packed, but re-packing missing paths would abort the save.
    bpy.data.use_autopack = False
    bpy.ops.wm.save_as_mainfile(filepath=destination)
    print(f"ASSET_FORGE_CHARACTER={destination}")


main()
