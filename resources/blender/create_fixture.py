import math
import os
import sys

import bpy


def script_args():
    args = sys.argv
    return args[args.index("--") + 1 :] if "--" in args else []


def output_path():
    args = script_args()
    if len(args) != 2 or args[0] != "--output":
        raise RuntimeError("Usage: blender --background --python create_fixture.py -- --output <fixture.blend>")
    return os.path.abspath(args[1])


def material(name, color, metallic=0.0, roughness=0.7):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1.0)
    value.metallic = metallic
    value.roughness = roughness
    return value


def add_cube(name, location, scale, mat, parent):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = obj.modifiers.new("SoftEdges", "BEVEL")
    bevel.width = 0.08
    bevel.segments = 2
    obj.data.materials.append(mat)
    obj.parent = parent
    return obj


def add_sphere(name, location, scale, mat, parent):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    obj.parent = parent
    return obj


def key(root, frame, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    root.location = location
    root.rotation_euler = rotation
    root.scale = scale
    root.keyframe_insert("location", frame=frame)
    root.keyframe_insert("rotation_euler", frame=frame)
    root.keyframe_insert("scale", frame=frame)


def action(root, name, keys):
    value = bpy.data.actions.new(name=name)
    value.use_fake_user = True
    root.animation_data_create()
    root.animation_data.action = value
    for frame, location, rotation, scale in keys:
        key(root, frame, location, rotation, scale)
    value.frame_start = min(item[0] for item in keys)
    value.frame_end = max(item[0] for item in keys)
    root.animation_data.action = None


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    root = bpy.data.objects.new("CharacterRoot", None)
    bpy.context.collection.objects.link(root)

    steel = material("TarnishedSteel", (0.16, 0.21, 0.25), metallic=0.55, roughness=0.48)
    blue = material("MutedBlue", (0.055, 0.13, 0.20), roughness=0.82)
    leather = material("Leather", (0.16, 0.075, 0.035), roughness=0.9)
    ember = material("Ember", (0.9, 0.18, 0.025), metallic=0.1, roughness=0.45)

    add_cube("Torso", (0.0, 0.0, 1.55), (0.46, 0.25, 0.62), blue, root)
    add_sphere("Helmet", (0.0, 0.0, 2.35), (0.38, 0.34, 0.42), steel, root)
    add_cube("LeftLeg", (-0.22, 0.0, 0.62), (0.16, 0.18, 0.62), leather, root)
    add_cube("RightLeg", (0.22, 0.0, 0.62), (0.16, 0.18, 0.62), leather, root)
    add_cube("Shield", (-0.58, -0.03, 1.55), (0.10, 0.34, 0.55), steel, root)
    sword = add_cube("Sword", (0.66, 0.0, 1.55), (0.07, 0.08, 0.72), steel, root)
    sword.rotation_euler.y = math.radians(-18)
    add_sphere("EmberBadge", (0.0, -0.27, 1.55), (0.10, 0.05, 0.10), ember, root)

    action(root, "idle", [
        (1, (0, 0, 0), (0, 0, 0), (1, 1, 1)),
        (12, (0, 0, 0.04), (0, 0, 0), (1.0, 1.0, 1.015)),
        (24, (0, 0, 0), (0, 0, 0), (1, 1, 1)),
    ])
    action(root, "walk", [
        (1, (0, 0, 0), (0, 0, math.radians(-3)), (1, 1, 1)),
        (7, (0, 0, 0.08), (0, 0, math.radians(3)), (1, 1, 1)),
        (13, (0, 0, 0), (0, 0, math.radians(-3)), (1, 1, 1)),
        (19, (0, 0, 0.08), (0, 0, math.radians(3)), (1, 1, 1)),
        (25, (0, 0, 0), (0, 0, math.radians(-3)), (1, 1, 1)),
    ])
    action(root, "attack", [
        (1, (0, 0, 0), (0, 0, 0), (1, 1, 1)),
        (6, (0, -0.10, 0), (math.radians(-5), 0, math.radians(-18)), (1, 1, 1)),
        (12, (0, 0.28, 0.03), (math.radians(8), 0, math.radians(24)), (1, 1, 1)),
        (18, (0, 0.16, 0), (0, 0, math.radians(8)), (1, 1, 1)),
        (24, (0, 0, 0), (0, 0, 0), (1, 1, 1)),
    ])
    action(root, "hit", [
        (1, (0, 0, 0), (0, 0, 0), (1, 1, 1)),
        (8, (0, -0.16, 0), (math.radians(-9), 0, 0), (0.96, 1.0, 0.96)),
    ])
    action(root, "death", [
        (1, (0, 0, 0), (0, 0, 0), (1, 1, 1)),
        (12, (0.12, 0, -0.25), (0, math.radians(28), math.radians(-30)), (1, 1, 1)),
        (24, (0.35, 0, -1.0), (0, math.radians(78), math.radians(-82)), (1, 1, 1)),
    ])

    destination = output_path()
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=destination)
    print(f"TEXT2MODEL_FORGE_FIXTURE={destination}")


main()
