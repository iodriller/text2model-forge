"""Retarget a small, declared humanoid motion set onto a Darkness short-biped rig.

This script runs inside Blender.  It deliberately treats an external library as
seed material: source rotations are sampled, converted between rest-bone axes,
baked into new Darkness-owned actions, and the imported source objects/actions
are then removed from the resulting checkpoint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as worker


CLIPS = {
    "idle": {"source": "Idle_Loop", "loop": True},
    "walk": {"source": "Walk_Loop", "loop": True},
    "attack": {"source": "Sword_Attack", "loop": False},
    "death": {"source": "Death01", "loop": False},
}

# Several source bones can feed one lower-resolution Darkness bone.  This keeps
# the source library's clavicle and multi-spine motion instead of silently
# dropping it when the short-biped family has fewer joints.
BONE_CHAINS = {
    "hips": ("pelvis",),
    "spine": ("spine_01", "spine_02"),
    "chest": ("spine_03",),
    "neck": ("neck_01",),
    "head": ("Head",),
    # The research Darkness rig's historical suffixes describe its X side,
    # while Quaternius uses anatomical sides. Match limbs by rest-space side
    # rather than equal-looking suffixes: Quaternius right is negative X and
    # therefore maps to the Darkness ``*_l`` chain.
    "shoulder_l": ("clavicle_r", "upperarm_r"),
    "elbow_l": ("lowerarm_r",),
    "wrist_l": ("hand_r",),
    "shoulder_r": ("clavicle_l", "upperarm_l"),
    "elbow_r": ("lowerarm_l",),
    "wrist_r": ("hand_l",),
    "hip_l": ("thigh_r",),
    "knee_l": ("calf_r",),
    "ankle_l": ("foot_r",),
    "foot_l": ("ball_r",),
    "hip_r": ("thigh_l",),
    "knee_r": ("calf_l",),
    "ankle_r": ("foot_l",),
    "foot_r": ("ball_l",),
}

CRITICAL_BONES = {
    "idle": ("hips", "chest", "head"),
    "walk": ("hips", "hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"),
    "attack": ("hips", "chest", "shoulder_l", "elbow_l", "wrist_l"),
    "death": ("hips", "spine", "chest", "neck", "head", "hip_l", "hip_r"),
}

EQUIPMENT = {
    "iteration": 8,
    "component_id": "fixture.weapon.short_biped.v1",
    "archetype": "one_handed_club",
    "socket": "hand_right",
    "bone": "hand_l",
    "grip_socket_bone": "grip_socket_right",
    "semantic_role": "source_anatomical_right_weapon_hand",
    "source_weapon_bone": "hand_r",
    "side_resolution_method": "source_motion_dominance_plus_rest_x_alignment_v1",
    "rig_policy": "articulated_digit_grip_v1",
    "object_name": "VettedMeshWeapon",
    "shield": None,
}

HANDLE_RADIUS_FRACTION = 0.018


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=448)
    parser.add_argument(
        "--character-spec",
        type=Path,
        help="Studio character_spec.json for equipment semantics; omit only for the research fixture.",
    )
    return parser.parse_args(argv)


def _load_equipment_spec(path: Path | None) -> None:
    if path is None:
        return
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    equipment = list(value.get("equipment") or [])
    weapons = [item for item in equipment if item.get("category") == "weapon"]
    if len(weapons) != 1:
        raise ValueError("Studio humanoid retarget currently requires exactly one weapon")
    weapon = weapons[0]
    if weapon.get("side") != "right" or weapon.get("socket") != "hand_right.grip":
        raise ValueError("Studio weapon contract requires the anatomical right-hand grip")
    description = str(weapon.get("description", "")).lower()
    archetype = "one_handed_sword" if "sword" in description or "sword" in str(weapon.get("equipment_id")) else "one_handed_club"
    shields = [item for item in equipment if item.get("category") == "shield"]
    if len(shields) > 1:
        raise ValueError("Studio humanoid retarget supports at most one shield")
    shield = shields[0] if shields else None
    if shield is not None and (
        shield.get("side") != "left"
        or shield.get("socket") != "forearm_left.shield"
        or shield.get("grip") != "forearm_strap"
    ):
        raise ValueError("Studio shield contract requires the anatomical left forearm strap")
    EQUIPMENT.update(
        {
            "component_id": str(weapon["equipment_id"]),
            "archetype": archetype,
            "object_name": "DarknessWeapon",
            "shield": shield,
            "studio_character_spec": str(path.resolve()),
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _armature(name: str) -> bpy.types.Object:
    item = bpy.data.objects.get(name)
    if item is None or item.type != "ARMATURE":
        raise ValueError(f"required armature is missing: {name}")
    return item


def _orthonormal(matrix: Matrix) -> Matrix:
    return matrix.to_3x3().normalized()


def _axis_correction(
    source_armature: bpy.types.Object,
    source_bone: bpy.types.PoseBone,
    target_armature: bpy.types.Object,
    target_bone: bpy.types.PoseBone,
) -> Quaternion:
    source_rest_world = _orthonormal(source_armature.matrix_world) @ _orthonormal(
        source_bone.bone.matrix_local
    )
    target_rest_world = _orthonormal(target_armature.matrix_world) @ _orthonormal(
        target_bone.bone.matrix_local
    )
    return (target_rest_world.inverted() @ source_rest_world).to_quaternion().normalized()


def _converted_rotation(
    source_armature: bpy.types.Object,
    source_name: str,
    target_armature: bpy.types.Object,
    target_name: str,
    reference_rotation: Quaternion,
) -> Quaternion:
    source_bone = source_armature.pose.bones[source_name]
    target_bone = target_armature.pose.bones[target_name]
    correction = _axis_correction(source_armature, source_bone, target_armature, target_bone)
    source_current = source_bone.matrix_basis.to_quaternion().normalized()
    source_delta = (source_current @ reference_rotation.inverted()).normalized()
    return (correction @ source_delta @ correction.inverted()).normalized()


def _reset_pose(armature: bpy.types.Object) -> None:
    for bone in armature.pose.bones:
        bone.matrix_basis.identity()


def _target_height(meshes: set[bpy.types.Object]) -> float:
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    return max(point.z for point in points) - min(point.z for point in points)


def _final_vertex_height(meshes: set[bpy.types.Object]) -> float:
    """Use the same authoritative vertex-space measure as motion validation."""
    obj = max(meshes, key=lambda item: len(item.data.vertices))
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    return max(point.z for point in points) - min(point.z for point in points)


def _source_motion_hand_analysis(
    source_armature: bpy.types.Object,
    source_action: bpy.types.Action,
    target_armature: bpy.types.Object,
) -> dict[str, object]:
    """Resolve the weapon side from motion and rest-space evidence.

    Equal suffixes are not a sufficient side contract: the imported library's
    anatomical right hand is negative X, while the current research Darkness
    rig calls its negative-X hand ``hand_l``.  The dominant source hand is
    found from the whole attack, then matched to the target hand on the same
    rest-space X side.
    """
    source_armature.animation_data.action = source_action
    start = int(math.ceil(float(source_action.frame_range[0])))
    end = int(math.floor(float(source_action.frame_range[1])))
    hands: dict[str, dict[str, object]] = {}
    for name in ("hand_l", "hand_r"):
        points: list[Vector] = []
        rotations: list[float] = []
        for frame in range(start, end + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            bone = source_armature.pose.bones[name]
            points.append(source_armature.matrix_world @ bone.head)
            rotations.append(float(bone.matrix_basis.to_quaternion().angle))
        path_length = sum(
            (points[index] - points[index - 1]).length for index in range(1, len(points))
        )
        rotation_range = max(rotations) - min(rotations)
        bone_length = float(source_armature.data.bones[name].length)
        score = path_length + bone_length * rotation_range
        rest_x = float(
            (source_armature.matrix_world @ source_armature.data.bones[name].head_local).x
        )
        hands[name] = {
            "path_length": path_length,
            "maximum_displacement": max((point - points[0]).length for point in points),
            "rotation_range_radians": rotation_range,
            "motion_score": score,
            "rest_x": rest_x,
        }

    ranked = sorted(hands, key=lambda name: float(hands[name]["motion_score"]), reverse=True)
    source_weapon = ranked[0]
    source_x = float(hands[source_weapon]["rest_x"])
    target_rest_x = {
        name: float((target_armature.matrix_world @ target_armature.data.bones[name].head_local).x)
        for name in ("hand_l", "hand_r")
    }
    target_weapon = min(
        target_rest_x,
        key=lambda name: (
            0 if math.copysign(1.0, target_rest_x[name]) == math.copysign(1.0, source_x) else 1,
            abs(target_rest_x[name]),
        ),
    )
    primary_score = float(hands[ranked[0]]["motion_score"])
    secondary_score = float(hands[ranked[1]]["motion_score"])
    dominance_ratio = primary_score / max(secondary_score, 1e-9)
    passed = (
        source_weapon == EQUIPMENT["source_weapon_bone"]
        and target_weapon == EQUIPMENT["bone"]
        and dominance_ratio >= 1.15
        and math.copysign(1.0, target_rest_x[target_weapon]) == math.copysign(1.0, source_x)
    )
    if not passed:
        raise RuntimeError(
            "weapon-hand resolution failed: "
            f"source={source_weapon}, target={target_weapon}, dominance={dominance_ratio}"
        )
    return {
        "source_action": source_action.name,
        "source_hands": hands,
        "selected_source_weapon_bone": source_weapon,
        "selected_target_weapon_bone": target_weapon,
        "target_rest_x": target_rest_x,
        "dominance_ratio": dominance_ratio,
        "automatic_gate_passed": passed,
    }


def _smoothstep(start: float, end: float, value: float) -> float:
    if end <= start:
        return float(value >= end)
    normalized = max(0.0, min(1.0, (value - start) / (end - start)))
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _build_articulated_grip(
    armature: bpy.types.Object,
    meshes: set[bpy.types.Object],
    actions: dict[str, bpy.types.Action],
    ranges: dict[str, tuple[int, int]],
    body_height: float,
    handle_radius: float,
) -> dict[str, object]:
    """Landmark, rig, skin, and pose the target's modeled digit branches.

    The generated target has two real topology branches but no digit bones.
    Treating those branches as a generic vertex cloud produced a fist-shaped
    blob.  This stage instead creates base/joint/tip landmarks, adds two deform
    bones per branch, transfers only the local hand weights to those bones, and
    solves a two-link wrap around a dedicated grip socket.  The closed pose is
    baked into every equipped action, while the open rest topology remains
    editable in the Blender master.
    """
    obj = max(meshes, key=lambda item: len(item.data.vertices))
    bone_name = str(EQUIPMENT["bone"])
    group = obj.vertex_groups.get(bone_name)
    if group is None:
        raise ValueError(f"weapon hand has no skin-weight group: {bone_name}")
    bone = armature.data.bones[bone_name]
    mesh_to_bone = bone.matrix_local.inverted() @ armature.matrix_world.inverted() @ obj.matrix_world
    hand_matrix = bone.matrix_local.copy()
    original_areas = [float(polygon.area) for polygon in obj.data.polygons]
    local_positions = {vertex.index: mesh_to_bone @ vertex.co for vertex in obj.data.vertices}
    skin_weights = {
        vertex.index: next(
            (element.weight for element in vertex.groups if element.group == group.index),
            0.0,
        )
        for vertex in obj.data.vertices
    }
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in obj.data.edges:
        first, second = edge.vertices
        adjacency[first].add(second)
        adjacency[second].add(first)

    distal = {
        index
        for index, local in local_positions.items()
        if skin_weights[index] >= 0.15 and local.y >= bone.length
    }
    unseen = set(distal)
    distal_components: list[set[int]] = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        if len(component) >= 20:
            distal_components.append(component)
    distal_components.sort(key=len, reverse=True)
    if not 2 <= len(distal_components) <= 4:
        raise RuntimeError(
            f"grip topology did not resolve the expected target digit branches: "
            f"{[len(component) for component in distal_components]}"
        )

    region = {
        index
        for index, local in local_positions.items()
        if skin_weights[index] >= 0.15 and local.y >= bone.length * 0.32
    }
    labels: dict[int, int] = {}
    distances: dict[int, int] = {}
    frontier: deque[tuple[int, int, int]] = deque()
    for label, component in enumerate(distal_components):
        for index in component:
            labels[index] = label
            distances[index] = 0
            frontier.append((index, label, 0))
    while frontier:
        current, label, distance = frontier.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in region:
                continue
            candidate_distance = distance + 1
            if neighbor in distances and distances[neighbor] <= candidate_distance:
                continue
            labels[neighbor] = label
            distances[neighbor] = candidate_distance
            frontier.append((neighbor, label, candidate_distance))

    branches: list[dict[str, object]] = []
    for label, component in enumerate(distal_components):
        assigned = [index for index, value in labels.items() if value == label]
        base_band = [
            index
            for index in assigned
            if bone.length * 0.32 <= local_positions[index].y <= bone.length * 0.58
        ]
        if not base_band:
            base_band = sorted(assigned, key=lambda index: local_positions[index].y)[: max(8, len(assigned) // 8)]
        pivot = sum((local_positions[index] for index in base_band), Vector()) / len(base_band)
        tip = sum((local_positions[index] for index in component), Vector()) / len(component)
        span = max(tip.y - pivot.y, bone.length * 0.20)
        joint_band = [
            index
            for index in assigned
            if pivot.y + span * 0.38 <= local_positions[index].y <= pivot.y + span * 0.70
        ]
        if not joint_band:
            joint_band = sorted(
                assigned,
                key=lambda index: abs(local_positions[index].y - (pivot.y + span * 0.54)),
            )[: max(8, len(assigned) // 10)]
        joint = sum((local_positions[index] for index in joint_band), Vector()) / len(joint_band)
        if (joint - pivot).length <= body_height * 0.005 or (tip - joint).length <= body_height * 0.005:
            joint = pivot.lerp(tip, 0.52)
        transverse_offsets = sorted(
            math.hypot(local_positions[index].x - tip.x, local_positions[index].z - tip.z)
            for index in component
        )
        transverse_radius = transverse_offsets[int((len(transverse_offsets) - 1) * 0.65)]
        branches.append(
            {
                "label": label,
                "assigned": assigned,
                "distal": sorted(component),
                "pivot": pivot,
                "joint": joint,
                "tip_before": tip,
                "transverse_radius": transverse_radius,
            }
        )

    # Place the shaft through the opening actually formed by the two digit-tip
    # landmarks. The legacy hand-bone origin is not the center of this generated
    # target's palm, which is why earlier socket-only attempts visibly missed.
    grip_center = Vector(
        (
            sum(float(branch["tip_before"].x) for branch in branches) / len(branches),
            sum(float(branch["pivot"].y) for branch in branches) / len(branches),
            sum(float(branch["tip_before"].z) for branch in branches) / len(branches),
        )
    )
    for branch in branches:
        pivot = branch["pivot"]
        joint = branch["joint"]
        tip = branch["tip_before"]
        tip_radial = Vector((tip.x - grip_center.x, 0.0, tip.z - grip_center.z))
        if tip_radial.length <= 1e-8:
            tip_radial = Vector((pivot.x - grip_center.x, 0.0, pivot.z - grip_center.z))
        branch_target_radius = handle_radius + max(
            body_height * 0.004,
            float(branch["transverse_radius"]) * 0.82,
        ) + body_height * 0.002
        desired_radial = tip_radial.normalized() * branch_target_radius
        desired_tip = Vector(
            (
                grip_center.x + desired_radial.x,
                grip_center.y + bone.length * 0.08,
                grip_center.z + desired_radial.z,
            )
        )

        first_length = (joint - pivot).length
        second_length = (tip - joint).length
        target_delta = desired_tip - pivot
        minimum_reach = abs(first_length - second_length) + body_height * 0.001
        maximum_reach = first_length + second_length - body_height * 0.001
        target_distance = min(max(target_delta.length, minimum_reach), maximum_reach)
        solved_tip = pivot + target_delta.normalized() * target_distance
        direction = (solved_tip - pivot).normalized()
        along = (
            first_length * first_length
            - second_length * second_length
            + target_distance * target_distance
        ) / (2.0 * target_distance)
        height = math.sqrt(max(first_length * first_length - along * along, 0.0))
        plane_normal = (joint - pivot).cross(tip - joint)
        if plane_normal.length <= 1e-8:
            plane_normal = direction.cross(Vector((0.0, 1.0, 0.0)))
        if plane_normal.length <= 1e-8:
            plane_normal = direction.cross(Vector((0.0, 0.0, 1.0)))
        perpendicular = plane_normal.normalized().cross(direction).normalized()
        candidates = (
            pivot + direction * along + perpendicular * height,
            pivot + direction * along - perpendicular * height,
        )
        branch["desired_tip"] = desired_tip
        branch["solved_joint"] = min(candidates, key=lambda value: (value - joint).length)
        branch["solved_tip"] = solved_tip
        branch["first_length"] = first_length
        branch["second_length"] = second_length
        branch["target_tip_radius"] = branch_target_radius

    # Add a non-deforming grip socket and an explicit two-link deform chain for
    # each topology branch. All coordinates are derived from the detected mesh
    # landmarks in the legacy hand bone's local space.
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    parent = armature.data.edit_bones[bone_name]
    socket = armature.data.edit_bones.new(str(EQUIPMENT["grip_socket_bone"]))
    socket.head = hand_matrix @ grip_center
    socket.tail = hand_matrix @ (grip_center + Vector((0.0, body_height * 0.05, 0.0)))
    socket.parent = parent
    socket.use_deform = False
    for branch in branches:
        label = int(branch["label"]) + 1
        base_name = f"grip_digit_{label:02d}_base"
        tip_name = f"grip_digit_{label:02d}_tip"
        base = armature.data.edit_bones.new(base_name)
        base.head = hand_matrix @ branch["pivot"]
        base.tail = hand_matrix @ branch["joint"]
        base.parent = parent
        base.use_deform = True
        tip_bone = armature.data.edit_bones.new(tip_name)
        tip_bone.head = base.tail
        tip_bone.tail = hand_matrix @ branch["tip_before"]
        tip_bone.parent = base
        tip_bone.use_connect = True
        tip_bone.use_deform = True
        branch["base_bone"] = base_name
        branch["tip_bone"] = tip_name
    bpy.ops.object.mode_set(mode="OBJECT")

    armature.data.bones[str(EQUIPMENT["grip_socket_bone"])]["darkness_landmark_role"] = "grip_axis"
    for branch in branches:
        armature.data.bones[str(branch["base_bone"])]["darkness_landmark_role"] = "digit_base_to_joint"
        armature.data.bones[str(branch["tip_bone"])]["darkness_landmark_role"] = "digit_joint_to_tip"

    # Replace only each branch's share of the hand weight. The palm side keeps
    # hand influence, then smoothly transfers through base and tip phalanges.
    maximum_weight_delta = 0.0
    total_weight_before = {
        vertex.index: sum(float(element.weight) for element in vertex.groups)
        for vertex in obj.data.vertices
    }
    for branch in branches:
        base_group = obj.vertex_groups.new(name=str(branch["base_bone"]))
        tip_group = obj.vertex_groups.new(name=str(branch["tip_bone"]))
        assigned = list(branch["assigned"])
        minimum_y = min(local_positions[index].y for index in assigned)
        maximum_y = max(local_positions[index].y for index in assigned)
        for index in assigned:
            original_hand_weight = skin_weights[index]
            if original_hand_weight <= 0.0:
                continue
            progress = (local_positions[index].y - minimum_y) / max(maximum_y - minimum_y, 1e-8)
            digit_blend = _smoothstep(0.05, 0.36, progress)
            tip_blend = _smoothstep(0.38, 0.76, progress)
            if index in branch["distal"]:
                digit_blend = 1.0
                tip_blend = 1.0
            digit_weight = original_hand_weight * digit_blend
            remaining_hand_weight = original_hand_weight - digit_weight
            group.remove([index])
            if remaining_hand_weight > 1e-6:
                group.add([index], remaining_hand_weight, "REPLACE")
            base_weight = digit_weight * (1.0 - tip_blend)
            tip_weight = digit_weight * tip_blend
            if base_weight > 1e-6:
                base_group.add([index], base_weight, "REPLACE")
            if tip_weight > 1e-6:
                tip_group.add([index], tip_weight, "REPLACE")
    obj.data.update()
    for vertex in obj.data.vertices:
        after = sum(float(element.weight) for element in vertex.groups)
        maximum_weight_delta = max(maximum_weight_delta, abs(after - total_weight_before[vertex.index]))

    def oriented_matrix(rest_bone: bpy.types.Bone, head: Vector, tail: Vector) -> Matrix:
        rest_direction = (rest_bone.tail_local - rest_bone.head_local).normalized()
        desired_direction = (tail - head).normalized()
        rotation = rest_direction.rotation_difference(desired_direction)
        result = (rotation.to_matrix() @ rest_bone.matrix_local.to_3x3()).to_4x4()
        result.translation = head
        return result

    for branch in branches:
        solved_base = hand_matrix @ branch["pivot"]
        solved_joint = hand_matrix @ branch["solved_joint"]
        solved_tip = hand_matrix @ branch["solved_tip"]
        branch["closed_base_matrix"] = oriented_matrix(
            armature.data.bones[str(branch["base_bone"])], solved_base, solved_joint
        )
        branch["closed_tip_matrix"] = oriented_matrix(
            armature.data.bones[str(branch["tip_bone"])], solved_joint, solved_tip
        )

    # The grip is constant relative to the animated hand, so two identical keys
    # per clip make the ownership/export contract explicit without oversampling.
    for clip_name, action in actions.items():
        armature.animation_data.action = action
        for frame in ranges[clip_name]:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            hand_pose = armature.pose.bones[bone_name]
            hand_deformation = hand_pose.matrix @ hand_pose.bone.matrix_local.inverted()
            for branch in branches:
                base_pose = armature.pose.bones[str(branch["base_bone"])]
                tip_pose = armature.pose.bones[str(branch["tip_bone"])]
                base_pose.rotation_mode = "QUATERNION"
                tip_pose.rotation_mode = "QUATERNION"
                base_pose.matrix = hand_deformation @ branch["closed_base_matrix"]
                bpy.context.view_layer.update()
                tip_pose.matrix = hand_deformation @ branch["closed_tip_matrix"]
                bpy.context.view_layer.update()
                for pose_bone in (base_pose, tip_pose):
                    pose_bone.keyframe_insert(
                        data_path="rotation_quaternion", frame=frame, group=pose_bone.name
                    )
                    pose_bone.keyframe_insert(data_path="location", frame=frame, group=pose_bone.name)

    # Validate the deformed mesh, not merely the control bones. Contact is
    # measured against the dedicated shaft axis in a representative attack pose.
    attack_start, attack_end = ranges["attack"]
    armature.animation_data.action = actions["attack"]
    bpy.context.scene.frame_set(attack_start + (attack_end - attack_start) // 2)
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    affected_indices = set(labels)
    collapsed_polygons = sum(
        1
        for index, polygon in enumerate(evaluated_mesh.polygons)
        if affected_indices.intersection(polygon.vertices)
        and float(polygon.area)
        <= max(original_areas[index] * 0.02, body_height * body_height * 1e-10)
    )
    socket_pose = armature.pose.bones[str(EQUIPMENT["grip_socket_bone"])]
    socket_world_inverse = (armature.matrix_world @ socket_pose.matrix).inverted()
    branch_reports: list[dict[str, object]] = []
    for branch in branches:
        local_surface = [
            socket_world_inverse @ (evaluated.matrix_world @ evaluated_mesh.vertices[index].co)
            for index in branch["distal"]
        ]
        radii = [math.hypot(point.x, point.z) for point in local_surface]
        centroid = sum(local_surface, Vector()) / len(local_surface)
        base_pose = armature.pose.bones[str(branch["base_bone"])]
        tip_pose = armature.pose.bones[str(branch["tip_bone"])]
        branch_reports.append(
            {
                "label": branch["label"],
                "vertices": len(branch["assigned"]),
                "distal_vertices": len(branch["distal"]),
                "base_landmark": list(branch["pivot"]),
                "joint_landmark": list(branch["joint"]),
                "tip_landmark": list(branch["tip_before"]),
                "desired_tip": list(branch["desired_tip"]),
                "solved_joint": list(branch["solved_joint"]),
                "solved_tip": list(branch["solved_tip"]),
                "target_tip_radius": branch["target_tip_radius"],
                "transverse_radius": branch["transverse_radius"],
                "base_bone": branch["base_bone"],
                "tip_bone": branch["tip_bone"],
                "base_rotation_degrees": math.degrees(float(base_pose.matrix_basis.to_quaternion().angle)),
                "tip_rotation_degrees": math.degrees(float(tip_pose.matrix_basis.to_quaternion().angle)),
                "surface_centroid_in_grip_space": list(centroid),
                "minimum_surface_radius": min(radii),
                "maximum_surface_radius": max(radii),
                "closest_surface_error": min(abs(radius - handle_radius) for radius in radii),
                "deep_penetrating_vertices": sum(radius < handle_radius * 0.55 for radius in radii),
            }
        )
    evaluated.to_mesh_clear()
    minimum_tip_separation = min(
        (
            (
                Vector(first["surface_centroid_in_grip_space"])
                - Vector(second["surface_centroid_in_grip_space"])
            ).length
            for offset, first in enumerate(branch_reports)
            for second in branch_reports[offset + 1 :]
        ),
        default=0.0,
    )
    passed = (
        len(branches) == 2
        and len([bone for bone in armature.data.bones if bone.name.startswith("grip_digit_")]) == 4
        and maximum_weight_delta <= 1e-5
        and collapsed_polygons == 0
        and minimum_tip_separation >= body_height * 0.012
        and all(
            int(report["deep_penetrating_vertices"]) == 0
            and float(report["closest_surface_error"]) <= body_height * 0.012
            for report in branch_reports
        )
    )
    if not passed:
        raise RuntimeError(
            "articulated digit grip gate failed: "
            f"weights={maximum_weight_delta}, collapsed={collapsed_polygons}, "
            f"separation={minimum_tip_separation}, branches={branch_reports}"
        )

    obj["darkness_grip_corrective"] = "articulated_digit_landmark_grip_v1"
    obj["darkness_grip_bone"] = bone_name
    obj["darkness_grip_affected_vertices"] = len(affected_indices)
    return {
        "method": "articulated_digit_landmark_grip_v1",
        "bone": bone_name,
        "grip_socket_bone": EQUIPMENT["grip_socket_bone"],
        "grip_center_in_hand_space": list(grip_center),
        "handle_radius": handle_radius,
        "target_tip_radii": [float(branch["target_tip_radius"]) for branch in branches],
        "detected_digit_branches": len(distal_components),
        "distal_component_sizes": [len(component) for component in distal_components],
        "generated_deform_bones": [
            name
            for branch in branches
            for name in (str(branch["base_bone"]), str(branch["tip_bone"]))
        ],
        "affected_vertices": len(affected_indices),
        "maximum_weight_sum_delta": maximum_weight_delta,
        "branches": branch_reports,
        "minimum_tip_separation": minimum_tip_separation,
        "collapsed_polygons": collapsed_polygons,
        "automatic_gate_passed": passed,
    }


def _create_club(armature: bpy.types.Object, body_height: float) -> bpy.types.Object:
    """Create a body-relative one-handed weapon at the resolved anatomical-right socket.

    The mesh is authored along bone-local +Y. Copy Transforms binds the grip to
    the declared hand bone, so no world-space coordinates or per-frame edits are
    involved. The dimensions are short-biped proportions, not target-specific
    absolute guesses.
    """
    socket_bone = str(EQUIPMENT["grip_socket_bone"])
    if socket_bone not in armature.pose.bones:
        raise ValueError(f"equipment socket bone is missing: {socket_bone}")

    is_sword = EQUIPMENT["archetype"] == "one_handed_sword"
    handle_length = body_height * (0.16 if is_sword else 0.30)
    handle_radius = body_height * HANDLE_RADIUS_FRACTION
    head_length = body_height * (0.43 if is_sword else 0.26)
    head_radius = body_height * (0.032 if is_sword else 0.085)
    handle_center = body_height * (0.035 if is_sword else 0.12)
    head_center = body_height * (0.335 if is_sword else 0.36)
    along_bone = Matrix.Rotation(-math.pi / 2.0, 4, "X")

    mesh = bpy.data.meshes.new("DarknessWeaponMesh" if is_sword else "DarknessClubMesh")
    geometry = bmesh.new()
    bmesh.ops.create_cone(
        geometry,
        cap_ends=True,
        cap_tris=False,
        segments=10,
        radius1=handle_radius * 0.82,
        radius2=handle_radius,
        depth=handle_length,
        matrix=Matrix.Translation((0.0, handle_center, 0.0)) @ along_bone,
    )
    if is_sword:
        # Broad low-poly blade, guard, and pommel; all dimensions remain body-relative.
        blade = bmesh.ops.create_cube(geometry, size=1.0)["verts"]
        bmesh.ops.transform(
            geometry,
            verts=blade,
            matrix=Matrix.Translation((0.0, head_center, 0.0))
            @ Matrix.Diagonal((body_height * 0.055, head_length, body_height * 0.012, 1.0)),
        )
        guard = bmesh.ops.create_cube(geometry, size=1.0)["verts"]
        bmesh.ops.transform(
            geometry,
            verts=guard,
            matrix=Matrix.Translation((0.0, body_height * 0.125, 0.0))
            @ Matrix.Diagonal((body_height * 0.18, body_height * 0.025, body_height * 0.025, 1.0)),
        )
        bmesh.ops.create_icosphere(
            geometry,
            subdivisions=1,
            radius=body_height * 0.035,
            matrix=Matrix.Translation((0.0, -body_height * 0.065, 0.0)),
        )
    else:
        bmesh.ops.create_cone(
            geometry,
            cap_ends=True,
            cap_tris=False,
            segments=9,
            radius1=head_radius * 0.56,
            radius2=head_radius,
            depth=head_length,
            matrix=Matrix.Translation((0.0, head_center, 0.0)) @ along_bone,
        )
        # An uneven cap and restrained knots keep the silhouette hand-made at sprite scale.
        bmesh.ops.create_icosphere(
            geometry,
            subdivisions=1,
            radius=head_radius * 0.94,
            matrix=Matrix.Translation((head_radius * 0.06, body_height * 0.49, -head_radius * 0.05))
            @ Matrix.Diagonal((1.0, 0.72, 0.92, 1.0)),
        )
        for x, y, z, radius in (
            (head_radius * 0.78, head_center + head_length * 0.03, head_radius * 0.16, head_radius * 0.30),
            (-head_radius * 0.68, head_center - head_length * 0.12, -head_radius * 0.22, head_radius * 0.25),
        ):
            bmesh.ops.create_icosphere(
                geometry,
                subdivisions=1,
                radius=radius,
                matrix=Matrix.Translation((x, y, z)),
            )
    geometry.to_mesh(mesh)
    geometry.free()
    mesh.update()

    club = bpy.data.objects.new(str(EQUIPMENT["object_name"]), mesh)
    bpy.context.scene.collection.objects.link(club)
    club["darkness_component_id"] = EQUIPMENT["component_id"]
    club["darkness_socket"] = EQUIPMENT["socket"]
    club["darkness_socket_bone"] = socket_bone
    club["darkness_rig_policy"] = EQUIPMENT["rig_policy"]
    club["darkness_equipment_category"] = "weapon"
    club["darkness_tip_reach_fraction"] = 0.55 if is_sword else 0.52
    constraint = club.constraints.new(type="COPY_TRANSFORMS")
    constraint.name = "DarknessHandSocket"
    constraint.target = armature
    constraint.subtarget = socket_bone
    constraint.target_space = "WORLD"
    constraint.owner_space = "WORLD"
    return club


def _create_shield(armature: bpy.types.Object, body_height: float) -> bpy.types.Object | None:
    shield_spec = EQUIPMENT.get("shield")
    if not shield_spec:
        return None
    socket_bone = "hand_r"  # target negative/positive suffix convention: hand_r is anatomical left.
    if socket_bone not in armature.pose.bones:
        raise ValueError(f"shield socket bone is missing: {socket_bone}")
    mesh = bpy.data.meshes.new("DarknessShieldMesh")
    geometry = bmesh.new()
    bmesh.ops.create_icosphere(
        geometry,
        subdivisions=2,
        radius=1.0,
        matrix=Matrix.Translation((0.0, body_height * 0.05, body_height * 0.055))
        @ Matrix.Diagonal((body_height * 0.22, body_height * 0.34, body_height * 0.028, 1.0)),
    )
    # A visible inner strap communicates the forearm attachment in close-up review.
    strap = bmesh.ops.create_cube(geometry, size=1.0)["verts"]
    bmesh.ops.transform(
        geometry,
        verts=strap,
        matrix=Matrix.Translation((0.0, body_height * 0.02, -body_height * 0.04))
        @ Matrix.Diagonal((body_height * 0.055, body_height * 0.18, body_height * 0.025, 1.0)),
    )
    geometry.to_mesh(mesh)
    geometry.free()
    mesh.update()
    shield = bpy.data.objects.new("DarknessShield", mesh)
    bpy.context.scene.collection.objects.link(shield)
    shield["darkness_component_id"] = str(shield_spec["equipment_id"])
    shield["darkness_socket"] = "forearm_left.shield"
    shield["darkness_socket_bone"] = socket_bone
    shield["darkness_rig_policy"] = "rigid_forearm_strap_v1"
    shield["darkness_equipment_category"] = "shield"
    constraint = shield.constraints.new(type="COPY_TRANSFORMS")
    constraint.name = "DarknessLeftForearmShieldSocket"
    constraint.target = armature
    constraint.subtarget = socket_bone
    constraint.target_space = "WORLD"
    constraint.owner_space = "WORLD"
    return shield


def _validate_equipment_binding(
    armature: bpy.types.Object,
    club: bpy.types.Object,
    actions: dict[str, bpy.types.Action],
    ranges: dict[str, tuple[int, int]],
    body_height: float,
    *,
    source_hand_analysis: dict[str, object],
    grip_corrective: dict[str, object],
) -> dict[str, object]:
    sampled_errors: list[float] = []
    maximum_tip_reach = 0.0
    tip_reach_fraction = float(club.get("darkness_tip_reach_fraction", 0.52))
    local_tip = Vector((0.0, body_height * tip_reach_fraction, 0.0))
    socket_bone = str(EQUIPMENT["grip_socket_bone"])
    for clip_name, action in actions.items():
        armature.animation_data.action = action
        start, end = ranges[clip_name]
        for frame in _frame_samples(start, end):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            evaluated = club.evaluated_get(bpy.context.evaluated_depsgraph_get())
            socket = armature.matrix_world @ armature.pose.bones[socket_bone].head
            grip = evaluated.matrix_world.translation
            sampled_errors.append((grip - socket).length)
            maximum_tip_reach = max(maximum_tip_reach, (evaluated.matrix_world @ local_tip - socket).length)

    maximum_error = max(sampled_errors, default=float("inf"))
    expected_tip_reach = body_height * tip_reach_fraction
    passed = (
        len(club.data.vertices) > 0
        and maximum_error <= body_height * 0.0001
        and abs(maximum_tip_reach - expected_tip_reach) <= body_height * 0.01
        and source_hand_analysis.get("automatic_gate_passed") is True
        and grip_corrective.get("automatic_gate_passed") is True
    )
    if not passed:
        raise RuntimeError(
            f"weapon socket gate failed: vertices={len(club.data.vertices)}, "
            f"maximum_grip_error={maximum_error}, maximum_tip_reach={maximum_tip_reach}"
        )
    return {
        **EQUIPMENT,
        "mesh": club.name,
        "vertices": len(club.data.vertices),
        "body_relative_dimensions": {
            "handle_length": 0.16 if EQUIPMENT["archetype"] == "one_handed_sword" else 0.30,
            "handle_radius": HANDLE_RADIUS_FRACTION,
            "head_length": 0.26,
            "head_radius": 0.085,
            "tip_reach": tip_reach_fraction,
        },
        "sampled_attachment_frames": len(sampled_errors),
        "maximum_grip_error": maximum_error,
        "maximum_tip_reach": maximum_tip_reach,
        "source_hand_analysis": source_hand_analysis,
        "grip_corrective": grip_corrective,
        "automatic_gate_passed": passed,
        "human_approval_required": True,
        "human_approved": False,
    }


def _validate_shield_binding(
    armature: bpy.types.Object,
    shield: bpy.types.Object | None,
    actions: dict[str, bpy.types.Action],
    ranges: dict[str, tuple[int, int]],
    body_height: float,
) -> dict[str, object] | None:
    if shield is None:
        return None
    socket_bone = str(shield["darkness_socket_bone"])
    errors = []
    for clip_name, action in actions.items():
        armature.animation_data.action = action
        start, end = ranges[clip_name]
        for frame in _frame_samples(start, end):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            evaluated = shield.evaluated_get(bpy.context.evaluated_depsgraph_get())
            socket = armature.matrix_world @ armature.pose.bones[socket_bone].head
            errors.append((evaluated.matrix_world.translation - socket).length)
    maximum_error = max(errors, default=float("inf"))
    passed = len(shield.data.vertices) > 0 and maximum_error <= body_height * 0.0001
    if not passed:
        raise RuntimeError(
            f"shield socket gate failed: vertices={len(shield.data.vertices)}, maximum_error={maximum_error}"
        )
    return {
        "component_id": shield["darkness_component_id"],
        "mesh": shield.name,
        "socket": shield["darkness_socket"],
        "socket_bone": socket_bone,
        "rig_policy": shield["darkness_rig_policy"],
        "sampled_attachment_frames": len(errors),
        "maximum_attachment_error": maximum_error,
        "automatic_gate_passed": True,
        "human_approval_required": True,
        "human_approved": False,
    }


def _source_height(armature: bpy.types.Object) -> float:
    heads = [armature.matrix_world @ bone.head_local for bone in armature.data.bones]
    tails = [armature.matrix_world @ bone.tail_local for bone in armature.data.bones]
    points = heads + tails
    return max(point.z for point in points) - min(point.z for point in points)


def _sample_action(
    source_armature: bpy.types.Object,
    source_action: bpy.types.Action,
    target_armature: bpy.types.Object,
    target_action: bpy.types.Action,
    *,
    location_scale: float,
    reference_pose: dict[str, tuple[Quaternion, Vector]],
) -> tuple[int, int]:
    source_start = int(math.ceil(float(source_action.frame_range[0])))
    source_end = int(math.floor(float(source_action.frame_range[1])))
    if source_end <= source_start:
        raise ValueError(f"source action has no usable frame span: {source_action.name}")
    source_armature.animation_data.action = source_action
    target_armature.animation_data.action = target_action
    for output_frame, source_frame in enumerate(range(source_start, source_end + 1), start=1):
        bpy.context.scene.frame_set(source_frame)
        bpy.context.view_layer.update()
        _reset_pose(target_armature)
        for target_name, source_chain in BONE_CHAINS.items():
            target_bone = target_armature.pose.bones[target_name]
            target_bone.rotation_mode = "QUATERNION"
            rotation = Quaternion()
            for source_name in source_chain:
                rotation = rotation @ _converted_rotation(
                    source_armature,
                    source_name,
                    target_armature,
                    target_name,
                    reference_pose[source_name][0],
                )
            target_bone.rotation_quaternion = rotation.normalized()
            if target_name == "hips":
                source_bone = source_armature.pose.bones[source_chain[0]]
                correction = _axis_correction(
                    source_armature,
                    source_bone,
                    target_armature,
                    target_bone,
                )
                reference_location = reference_pose[source_chain[0]][1]
                target_bone.location = correction @ (
                    (source_bone.location - reference_location) * location_scale
                )
            target_bone.keyframe_insert(
                data_path="rotation_quaternion",
                frame=output_frame,
                group=target_name,
            )
            target_bone.keyframe_insert(
                data_path="location",
                frame=output_frame,
                group=target_name,
            )
    return 1, source_end - source_start + 1


def _reference_pose(
    source_armature: bpy.types.Object,
    action: bpy.types.Action,
) -> dict[str, tuple[Quaternion, Vector]]:
    """Capture a natural idle as the neutral pose instead of the library T-pose."""
    source_armature.animation_data.action = action
    bpy.context.scene.frame_set(int(math.ceil(float(action.frame_range[0]))))
    bpy.context.view_layer.update()
    return {
        name: (
            source_armature.pose.bones[name].matrix_basis.to_quaternion().normalized().copy(),
            source_armature.pose.bones[name].location.copy(),
        )
        for chain in BONE_CHAINS.values()
        for name in chain
    }


def _remove_objects(objects: set[bpy.types.Object]) -> None:
    for obj in objects:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Armature):
                bpy.data.armatures.remove(data)


def _frame_samples(start: int, end: int) -> list[int]:
    span = end - start
    return sorted({start, start + span // 4, start + span // 2, start + 3 * span // 4, end})


def _bone_point(armature: bpy.types.Object, name: str, *, tail: bool = False) -> Vector:
    bone = armature.pose.bones[name]
    return armature.matrix_world @ (bone.tail if tail else bone.head)


def _rotation_degrees(bone: bpy.types.PoseBone) -> float:
    rotation = bone.matrix_basis.to_quaternion()
    angle = min(float(rotation.angle), math.tau - float(rotation.angle))
    return math.degrees(angle)


def _validate_and_render(
    output_root: Path,
    armature: bpy.types.Object,
    actions: dict[str, bpy.types.Action],
    ranges: dict[str, tuple[int, int]],
    *,
    render_size: int,
) -> dict[str, object]:
    obj = max(worker._mesh_objects(), key=lambda item: len(item.data.vertices))
    rest_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    rest_areas = [polygon.area for polygon in obj.data.polygons]
    height = max(point.z for point in rest_positions) - min(point.z for point in rest_positions)
    clips: dict[str, object] = {}
    finite = True
    evidence_frames: list[tuple[str, bpy.types.Action, int]] = []
    evidence_minimum: Vector | None = None
    evidence_maximum: Vector | None = None
    for clip_name, action in actions.items():
        start, end = ranges[clip_name]
        samples = _frame_samples(start, end)
        excursion = {name: 0.0 for name in CRITICAL_BONES[clip_name]}
        pose_reports: list[dict[str, object]] = []
        positions: dict[str, dict[str, list[float]]] = {}
        for frame in samples:
            armature.animation_data.action = action
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            positions[str(frame)] = {}
            for name in CRITICAL_BONES[clip_name]:
                head = _bone_point(armature, name)
                tail = _bone_point(armature, name, tail=True)
                values = (*head, *tail)
                finite = finite and all(math.isfinite(value) for value in values)
                positions[str(frame)][name] = [*head, *tail]
                excursion[name] = max(excursion[name], _rotation_degrees(armature.pose.bones[name]))
            pose_reports.append(
                worker._evaluated_pose_report(
                    obj,
                    pose_name=f"{clip_name}@{frame}",
                    rest_positions=rest_positions,
                    rest_areas=rest_areas,
                )
            )
            frame_minimum, frame_maximum = worker._scene_bounds()
            if evidence_minimum is None:
                evidence_minimum = frame_minimum.copy()
                evidence_maximum = frame_maximum.copy()
            else:
                evidence_minimum = Vector(
                    min(evidence_minimum[axis], frame_minimum[axis]) for axis in range(3)
                )
                evidence_maximum = Vector(
                    max(evidence_maximum[axis], frame_maximum[axis]) for axis in range(3)
                )
            evidence_frames.append((clip_name, action, frame))
        clips[clip_name] = {
            "source_action": CLIPS[clip_name]["source"],
            "frame_start": start,
            "frame_end": end,
            "sampled_frames": samples,
            "critical_bones": list(CRITICAL_BONES[clip_name]),
            "critical_joint_excursion_degrees": excursion,
            "critical_joint_positions": positions,
            "collapsed_faces": sum(int(item["collapsed_faces"]) for item in pose_reports),
            "severely_compressed_faces": sum(
                int(item["severely_compressed_faces"]) for item in pose_reports
            ),
            "maximum_vertex_displacement": max(
                float(item["maximum_vertex_displacement"]) for item in pose_reports
            ),
            "minimum_z": min(float(item["bounds_minimum"][2]) for item in pose_reports),
        }

    if evidence_minimum is None or evidence_maximum is None:
        raise RuntimeError("retarget evidence frame plan is empty")
    fixed_bounds = (evidence_minimum, evidence_maximum)
    for clip_name, action, frame in evidence_frames:
        armature.animation_data.action = action
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        worker.render_diagnostics(
            output_root,
            f"motion_{clip_name}_{frame:03d}",
            size=render_size,
            bounds_override=fixed_bounds,
        )

    grip_evidence: list[str] = []
    attack_samples = _frame_samples(*ranges["attack"])
    for frame in (attack_samples[1], attack_samples[2]):
        armature.animation_data.action = actions["attack"]
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        grip_head = _bone_point(armature, str(EQUIPMENT["bone"]))
        grip_tail = _bone_point(armature, str(EQUIPMENT["bone"]), tail=True)
        grip_center = (grip_head + grip_tail) * 0.5
        half_extent = Vector((height * 0.13,) * 3)
        grip_evidence.extend(
            str(path)
            for path in worker.render_diagnostics(
                output_root,
                f"grip_attack_{frame:03d}",
                size=max(render_size, 640),
                bounds_override=(grip_center - half_extent, grip_center + half_extent),
            )
        )

    walk = clips["walk"]
    walk_frames = walk["sampled_frames"]
    foot_heights = {side: [] for side in ("foot_l", "foot_r")}
    for frame in walk_frames:
        armature.animation_data.action = actions["walk"]
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for side in foot_heights:
            foot_heights[side].append(_bone_point(armature, side, tail=True).z)
    attack_frames = ranges["attack"]
    armature.animation_data.action = actions["attack"]
    bpy.context.scene.frame_set(attack_frames[0])
    attack_wrist = str(EQUIPMENT["bone"]).replace("hand_", "wrist_")
    attack_start = _bone_point(armature, attack_wrist, tail=True)
    attack_travel = 0.0
    for frame in range(attack_frames[0], attack_frames[1] + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        attack_travel = max(
            attack_travel,
            (_bone_point(armature, attack_wrist, tail=True) - attack_start).length,
        )
    death_frames = ranges["death"]
    armature.animation_data.action = actions["death"]
    bpy.context.scene.frame_set(death_frames[0])
    death_hips_start = _bone_point(armature, "hips")
    death_head_start = _bone_point(armature, "head")
    bpy.context.scene.frame_set(death_frames[1])
    bpy.context.view_layer.update()
    death_hips_end = _bone_point(armature, "hips")
    death_head_end = _bone_point(armature, "head")

    return {
        "schema_version": 1,
        "clips": clips,
        "finite_critical_joint_transforms": finite,
        "walk_swing_ranges": {
            name: max(values) - min(values) for name, values in foot_heights.items()
        },
        "attack_weapon_wrist": attack_wrist,
        "attack_maximum_hand_travel": attack_travel,
        "grip_evidence": grip_evidence,
        "death_hips_descent": death_hips_start.z - death_hips_end.z,
        "death_head_descent": death_head_start.z - death_head_end.z,
        "body_height": height,
        "retarget_gate_passed": finite and set(actions) == set(CLIPS),
        "human_approval_required": True,
        "human_approved": False,
    }


def main() -> int:
    args = _arguments()
    _load_equipment_spec(args.character_spec)
    target_path = args.target.resolve()
    source_path = args.source.resolve()
    output_root = args.output_directory.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(target_path), load_ui=False, use_scripts=False)
    target = _armature("DarknessShortBipedRig")
    target_objects = set(bpy.context.scene.objects)
    target_meshes = {item for item in target_objects if item.type == "MESH"}
    target.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)

    bpy.ops.import_scene.gltf(filepath=str(source_path))
    imported_objects = set(bpy.context.scene.objects) - target_objects
    source_armatures = sorted(
        (item for item in imported_objects if item.type == "ARMATURE"),
        key=lambda item: item.name,
    )
    if len(source_armatures) != 1:
        raise ValueError(f"expected one source armature, found {len(source_armatures)}")
    source = source_armatures[0]
    source.animation_data_create()
    for track in source.animation_data.nla_tracks:
        track.mute = True

    source_actions = {action.name: action for action in bpy.data.actions}
    missing_actions = sorted(
        str(spec["source"]) for spec in CLIPS.values() if spec["source"] not in source_actions
    )
    missing_source_bones = sorted(
        {name for chain in BONE_CHAINS.values() for name in chain}
        - {bone.name for bone in source.pose.bones}
    )
    missing_target_bones = sorted(set(BONE_CHAINS) - {bone.name for bone in target.pose.bones})
    if missing_actions or missing_source_bones or missing_target_bones:
        raise ValueError(
            f"retarget contract mismatch: actions={missing_actions}, "
            f"source_bones={missing_source_bones}, target_bones={missing_target_bones}"
        )

    source_hand_analysis = _source_motion_hand_analysis(
        source,
        source_actions["Sword_Attack"],
        target,
    )
    location_scale = _target_height(target_meshes) / _source_height(source)
    reference_pose = _reference_pose(source, source_actions["Idle_Loop"])
    target.animation_data_create()
    target_actions: dict[str, bpy.types.Action] = {}
    ranges: dict[str, tuple[int, int]] = {}
    for clip_name, spec in CLIPS.items():
        action = bpy.data.actions.new(name=clip_name)
        action.use_fake_user = True
        ranges[clip_name] = _sample_action(
            source,
            source_actions[str(spec["source"])],
            target,
            action,
            location_scale=location_scale,
            reference_pose=reference_pose,
        )
        action["darkness_clip"] = clip_name
        action["darkness_loop"] = bool(spec["loop"])
        action["darkness_source"] = "Quaternius Universal Animation Library Standard"
        target_actions[clip_name] = action

    source.animation_data_clear()
    _remove_objects(imported_objects)
    for action in list(bpy.data.actions):
        if action not in target_actions.values():
            bpy.data.actions.remove(action)
    body_height = _final_vertex_height(target_meshes)
    grip_corrective = _build_articulated_grip(
        target,
        target_meshes,
        target_actions,
        ranges,
        body_height,
        body_height * HANDLE_RADIUS_FRACTION,
    )
    club = _create_club(target, body_height)
    shield = _create_shield(target, body_height)
    equipment_report = _validate_equipment_binding(
        target,
        club,
        target_actions,
        ranges,
        body_height,
        source_hand_analysis=source_hand_analysis,
        grip_corrective=grip_corrective,
    )
    equipment_report["shield"] = _validate_shield_binding(
        target, shield, target_actions, ranges, body_height
    )
    _reset_pose(target)
    target.animation_data.action = None
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    report = _validate_and_render(
        output_root,
        target,
        target_actions,
        ranges,
        render_size=args.render_size,
    )
    report["source"] = {
        "name": "Quaternius Universal Animation Library Standard",
        "url": "https://quaternius.com/packs/universalanimationlibrary.html",
        "license": "CC0-1.0",
        "path": str(source_path),
        "sha256": _sha256(source_path),
    }
    report["target"] = {"path": str(target_path), "sha256": _sha256(target_path)}
    report["bone_chains"] = {name: list(chain) for name, chain in BONE_CHAINS.items()}
    report["location_scale"] = location_scale
    report["equipment"] = equipment_report
    report_path = output_root / "retarget_validation.json"
    _write_json(report_path, report)

    target.animation_data.action = None
    bpy.context.scene.frame_set(1)
    worker._clear_pose(target)
    worker._remove_diagnostic_objects()
    checkpoint = output_root / "quaternius_retargeted_candidate.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(checkpoint))
    worker._triangulate_scene_for_export()
    exported = output_root / "quaternius_retargeted_candidate.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(exported),
        export_format="GLB",
        export_materials="NONE",
        export_normals=False,
        export_tangents=False,
        export_skins=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_anim_single_armature=True,
        export_reset_pose_bones=True,
        export_anim_slide_to_zero=True,
        export_rest_position_armature=True,
        export_shared_accessors=True,
    )
    _write_json(
        output_root / "retarget_contract.json",
        {
            "schema_version": 1,
            "anatomy_family": "short_biped_v1",
            "source_library": "Quaternius Universal Animation Library Standard",
            "source_license": "CC0-1.0",
            "equipment": equipment_report,
            "clips": {
                name: {
                    "source_action": CLIPS[name]["source"],
                    "frame_start": ranges[name][0],
                    "frame_end": ranges[name][1],
                    "loop": CLIPS[name]["loop"],
                }
                for name in CLIPS
            },
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        if "--output-directory" in argv:
            error_root = Path(argv[argv.index("--output-directory") + 1]).resolve()
            error_root.mkdir(parents=True, exist_ok=True)
            (error_root / "retarget_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
