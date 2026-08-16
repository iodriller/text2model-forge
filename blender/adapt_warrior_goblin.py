"""Adapt the CC0 Quaternius Warrior into a generic short-biped raider fixture.

Same rig/action library as the footman adapter (the CC0 pack shares one skeleton and
one professional action set across characters), but authored differently to read as
a distinct, lesser-armored raider rather than a knight:

- the pack's plate ShoulderPad.L/R are removed (crude leather straps, not plate);
- the pack's arming sword is replaced by an authored spiked club, built directly in
  the sword's own validated grip transform so it lands in the hand with zero manual
  bone-vector math;
- the whole rig is scaled down uniformly (smaller frame than the footman knight);
- torn pointed ears, a brow ridge, an upturned snout, an underbite jaw with tusks,
  and a leather skullcap are added to the head, since the donor's human face/hair
  read as human no matter what color it is painted;
- a predatory hunch (forward stoop through Abdomen/Torso/Neck, head craned back up)
  is baked into EVERY shared action's keyframes, not just the rest pose, so idle,
  walk, attack, and death all inherit the goblin's posture instead of the knight's
  upright stance.

Skin tone, cloth color, and grime are NOT authored here — the texture-master bake
prompt paints the surface, so this script only needs to get geometry and equipment
right.

Usage: blender --background <warrior.blend> --python adapt_warrior_goblin.py -- --output <master.blend>
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


def make_material(name, color, metallic=0.0, roughness=0.8):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    node = value.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    value.diffuse_color = (*color, 1.0)
    return value


def bone_parent(value, armature, bone_name):
    world = value.matrix_world.copy()
    value.parent = armature
    value.parent_type = "BONE"
    value.parent_bone = bone_name
    value.matrix_world = world


def add_part(name, primitive, location, scale, material, armature, bone, rotation=(0, 0, 0), bevel=0.015):
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


ACTIONS_USED = (
    "Idle_Weapon", "Walk", "Sword_Attack", "Sword_AttackFast", "Roll", "RecieveHit", "Idle_Attacking", "Death",
)
HUNCH_DEGREES = {
    "Abdomen": 4.0,
    "Torso": 5.0,
    "Neck": 6.0,
    "Head": -15.0,  # counters the spine's cumulative forward stoop so the face still peers up/forward
}


def bake_hunch(armature, action_names, bone_angles_degrees):
    """Compose a constant forward-stoop rotation into EVERY keyframe of the given
    actions, for the given bones, so every animation this rig ever plays inherits a
    goblin's posture instead of the knight's upright stance.

    A rest-pose (edit-bone) edit would reinterpret every existing action's stored
    rotations against a new zero point and break their deformation; baking the same
    corrective quaternion into each action's own keyframes (in bone-local pose space,
    exactly how Blender's own "Bake Action" works) is what actually carries the
    posture into idle, walk, attack, and death alike.
    """
    from mathutils import Quaternion, Vector as MVector

    axis = MVector((1.0, 0.0, 0.0))
    hunches = {bone: Quaternion(axis, math.radians(angle)) for bone, angle in bone_angles_degrees.items()}
    for action_name in action_names:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Cannot bake hunch: action {action_name!r} not found")
        armature.animation_data.action = action
        start, end = action.frame_range
        for frame in range(int(math.floor(start)), int(math.ceil(end)) + 1):
            bpy.context.scene.frame_set(frame)
            for bone_name, hunch in hunches.items():
                pose_bone = armature.pose.bones[bone_name]
                pose_bone.rotation_quaternion = hunch @ pose_bone.rotation_quaternion
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def add_head_part(
    name, primitive, head_origin, head_forward, head_up, head_right,
    forward, up, right, tilt_deg, twist_deg, scale, material, armature,
):
    """Place a face/head part using the Head bone's OWN current local axes rather
    than fixed world offsets.

    Fixed world offsets (head.y + constant) only work while the head is close to
    upright; once a hunch tilts the head by any real amount, "world +Y" no longer
    points out of the face, and every face part silently ends up buried in the wrong
    place (this broke the first hunch attempt completely). Deriving forward/up/right
    from the bone's own posed matrix keeps every part correctly anchored to the face
    regardless of how much the neck is craned.
    """
    from mathutils import Quaternion

    location = head_origin + head_forward * forward + head_up * up + head_right * right
    base = head_forward.to_track_quat("Z", "Y")
    tilt = Quaternion((1.0, 0.0, 0.0), math.radians(tilt_deg))
    twist = Quaternion((0.0, 0.0, 1.0), math.radians(twist_deg))
    rotation = tuple((base @ tilt @ twist).to_euler())
    return add_part(name, primitive, location, scale, material, armature, "Head", rotation=rotation)


def add_grip_part(name, primitive, grip_origin, grip_up, distance, scale, material, armature, bone, lateral=(0.0, 0.0, 0.0)):
    """Place a part `distance` meters up the grip's own held-item axis (derived from
    the pack's own sword, so it matches the rig's real grip direction) plus a small
    lateral offset for spikes ringing the club head.

    Deliberately uses location+rotation only (the add_part() pattern already proven
    for the footman's shield/helmet) rather than assigning matrix_world from the
    grip's full matrix: the grip matrix carries its own baked scale, and reusing it
    directly double-applies that scale on top of each part's own authored scale.
    """
    from mathutils import Vector

    location = grip_origin + grip_up * distance + Vector(lateral)
    rotation = tuple(grip_up.to_track_quat("Z", "Y").to_euler())
    return add_part(name, primitive, location, scale, material, armature, bone, rotation=rotation)


def main():
    args = parse_args()
    armature = bpy.data.objects["CharacterArmature"]

    # Same front-facing correction as the footman adapter (pack faces -Y).
    armature.rotation_euler = (0.0, 0.0, math.radians(180.0))
    # A goblin raider is a smaller frame than an armored knight; uniform scale keeps
    # every bone-parented offset (measured below) automatically consistent.
    armature.scale = (0.85, 0.85, 0.85)
    bpy.context.view_layer.update()

    armature.animation_data_create()
    bake_hunch(armature, ACTIONS_USED, HUNCH_DEGREES)

    # Pose BEFORE measuring: bone parenting bakes offsets against the current pose,
    # and every action shares this rig's Idle_Weapon stance for its held-item grip.
    # The hunch baked above is already part of Idle_Weapon's keyframes, so measuring
    # here automatically places equipment/face parts at the goblin's new posture.
    armature.animation_data.action = bpy.data.actions["Idle_Weapon"]
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    from mathutils import Vector

    sword = bpy.data.objects["Warrior_Sword"]
    grip_matrix = sword.matrix_world.copy()
    grip_origin = grip_matrix.translation.copy()
    grip_up = (grip_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    bpy.data.objects.remove(sword, do_unlink=True)
    # Plate pauldrons read as a knight, not a raider; a goblin wears crude straps.
    for name in ("ShoulderPad.L", "ShoulderPad.R"):
        part = bpy.data.objects.get(name)
        if part is not None:
            bpy.data.objects.remove(part, do_unlink=True)

    head_bone = armature.pose.bones["Head"]
    head_pose_matrix = armature.matrix_world @ head_bone.matrix
    head_rot = head_pose_matrix.to_3x3()
    # Measured once (probe_head_axes.py) against the rig's own rest orientation:
    # local Z is face-forward, local Y is up-the-skull, local X is left/right.
    head_origin = armature.matrix_world @ ((head_bone.head + head_bone.tail) / 2)
    head_forward = (head_rot @ Vector((0.0, 0.0, 1.0))).normalized()
    head_up = (head_rot @ Vector((0.0, 1.0, 0.0))).normalized()
    head_right = (head_rot @ Vector((1.0, 0.0, 0.0))).normalized()

    wood = make_material("Club Wood", (0.16, 0.11, 0.07), roughness=0.9)
    band = make_material("Club Iron Band", (0.08, 0.09, 0.10), metallic=0.2, roughness=0.55)
    spike = make_material("Club Spike", (0.10, 0.11, 0.13), metallic=0.25, roughness=0.5)
    ear = make_material("Goblin Ear", (0.32, 0.34, 0.20), roughness=0.75)
    import os
    debug_colors = os.environ.get("ASSET_FORGE_DEBUG_FACE") == "1"
    if debug_colors:
        brow_mat = make_material("Goblin Face Brow", (1.0, 0.0, 0.0))
        snout_mat = make_material("Goblin Face Snout", (1.0, 0.55, 0.0))
        jaw_mat = make_material("Goblin Face Jaw", (0.0, 1.0, 1.0))
        tusk = make_material("Goblin Tusk", (1.0, 1.0, 0.0))
        ear = make_material("Goblin Ear", (0.0, 0.3, 1.0))
        hood_material = make_material("Goblin Hood", (1.0, 0.0, 1.0))
    else:
        brow_mat = snout_mat = jaw_mat = make_material("Goblin Face", (0.30, 0.32, 0.19), roughness=0.7)
        tusk = make_material("Goblin Tusk", (0.72, 0.68, 0.56), roughness=0.5)
        hood_material = make_material("Goblin Hood", (0.14, 0.10, 0.07), roughness=0.85)

    # Club, gripped along the pack's own held-item axis (derived from where its
    # sword actually pointed when posed) so it lands correctly in the hand without
    # re-deriving bone vectors from scratch.
    cone = bpy.ops.mesh.primitive_cone_add
    cylinder = bpy.ops.mesh.primitive_cylinder_add
    add_grip_part("Club Haft", cylinder, grip_origin, grip_up, 0.20, (0.030, 0.030, 0.26), wood, armature, "Weapon.R")
    add_grip_part("Club Band Lower", cylinder, grip_origin, grip_up, 0.20, (0.038, 0.038, 0.02), band, armature, "Weapon.R")
    add_grip_part("Club Band Upper", cylinder, grip_origin, grip_up, 0.44, (0.038, 0.038, 0.02), band, armature, "Weapon.R")
    add_grip_part("Club Head", cylinder, grip_origin, grip_up, 0.50, (0.095, 0.095, 0.13), wood, armature, "Weapon.R")
    for index, (dx, dy) in enumerate([(0.06, 0.0), (-0.06, 0.0), (0.0, 0.06), (0.0, -0.06), (0.042, 0.042), (-0.042, -0.042)]):
        add_grip_part(
            f"Club Spike {index:02d}", cone, grip_origin, grip_up, 0.52, (0.024, 0.024, 0.10), spike, armature, "Weapon.R",
            lateral=(dx, dy, 0.0),
        )

    # Long torn pointed ears on the bare head (this raider wears no helmet), placed
    # from the head bone's OWN axes (forward/up/right) rather than world offsets, so
    # they stay correctly anchored to the skull at any hunch angle.
    for side, x_sign in (("L", -1.0), ("R", 1.0)):
        add_head_part(
            f"Goblin Ear.{side}", cone, head_origin, head_forward, head_up, head_right,
            forward=-0.05, up=0.10, right=x_sign * 0.15,
            tilt_deg=125.0, twist_deg=x_sign * 32.0,
            scale=(0.055, 0.055, 0.34), material=ear, armature=armature,
        )

    # A goblin's face is what makes it a goblin, not the paint on top of it: a heavy
    # scowling brow, an upturned snout, and a jutting underbite jaw with tusks. Paint
    # can weather these features; it cannot invent them on a human skull. Spaced
    # generously along "up" (brow near eye level, snout at nose height, jaw at chin)
    # so the three read as distinct features instead of merging into one plate.
    cube = bpy.ops.mesh.primitive_cube_add
    add_head_part(
        "Goblin Brow", cube, head_origin, head_forward, head_up, head_right,
        forward=0.18, up=0.05, right=0.0, tilt_deg=60.0, twist_deg=0.0,
        scale=(0.17, 0.045, 0.06), material=brow_mat, armature=armature,
    )
    add_head_part(
        "Goblin Snout", cube, head_origin, head_forward, head_up, head_right,
        forward=0.34, up=0.02, right=0.0, tilt_deg=100.0, twist_deg=0.0,
        scale=(0.055, 0.05, 0.10), material=snout_mat, armature=armature,
    )
    add_head_part(
        "Goblin Jaw", cube, head_origin, head_forward, head_up, head_right,
        forward=0.20, up=-0.24, right=0.0, tilt_deg=75.0, twist_deg=0.0,
        scale=(0.14, 0.055, 0.09), material=jaw_mat, armature=armature,
    )
    for side, x_sign in (("L", -1.0), ("R", 1.0)):
        add_head_part(
            f"Goblin Tusk.{side}", cone, head_origin, head_forward, head_up, head_right,
            forward=0.30, up=-0.20, right=x_sign * 0.11, tilt_deg=160.0, twist_deg=x_sign * 18.0,
            scale=(0.018, 0.018, 0.08), material=tusk, armature=armature,
        )

    # Crude leather skullcap over the donor's human hair, pulled well back so its
    # front edge sits clear of the brow even at the hood's own radius, wrapping the
    # crown/back of the skull instead of poking up like a mohawk peak.
    hood_center = head_origin + head_forward * -0.32 + head_up * 0.06
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, location=hood_center)
    hood = bpy.context.object
    hood.name = "Goblin Hood"
    hood.rotation_euler = head_forward.to_track_quat("Z", "Y").to_euler()
    hood.scale = (0.24, 0.20, 0.24)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.shade_smooth()
    hood.data.materials.append(hood_material)
    bone_parent(hood, armature, "Head")

    armature.animation_data.action = None

    armature["asset_forge_asset_id"] = "goblin"
    armature["asset_forge_source_kind"] = "owned_blender_master"
    armature["asset_forge_generator"] = "warrior_adapter_goblin_v1"
    armature["asset_forge_base_license"] = "CC0-1.0 Quaternius RPG Characters"

    destination = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    bpy.data.use_autopack = False
    bpy.ops.wm.save_as_mainfile(filepath=destination)
    print(f"ASSET_FORGE_CHARACTER={destination}")


main()
