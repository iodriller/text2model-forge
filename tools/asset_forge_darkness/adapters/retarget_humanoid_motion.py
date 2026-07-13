"""Retarget a small, declared humanoid motion set onto a Darkness short-biped rig.

This script runs inside Blender.  It deliberately treats an external library as
seed material: source rotations are sampled, converted between rest-bone axes,
baked into new Darkness-owned actions, and the imported source objects/actions
are then removed from the resulting checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

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
    "shoulder_l": ("clavicle_l", "upperarm_l"),
    "elbow_l": ("lowerarm_l",),
    "wrist_l": ("hand_l",),
    "shoulder_r": ("clavicle_r", "upperarm_r"),
    "elbow_r": ("lowerarm_r",),
    "wrist_r": ("hand_r",),
    "hip_l": ("thigh_l",),
    "knee_l": ("calf_l",),
    "ankle_l": ("foot_l",),
    "foot_l": ("ball_l",),
    "hip_r": ("thigh_r",),
    "knee_r": ("calf_r",),
    "ankle_r": ("foot_r",),
    "foot_r": ("ball_r",),
}

CRITICAL_BONES = {
    "idle": ("hips", "chest", "head"),
    "walk": ("hips", "hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"),
    "attack": ("hips", "chest", "shoulder_r", "elbow_r", "wrist_r"),
    "death": ("hips", "spine", "chest", "neck", "head", "hip_l", "hip_r"),
}


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=448)
    return parser.parse_args(argv)


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
    attack_start = _bone_point(armature, "wrist_r", tail=True)
    attack_travel = 0.0
    for frame in range(attack_frames[0], attack_frames[1] + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        attack_travel = max(attack_travel, (_bone_point(armature, "wrist_r", tail=True) - attack_start).length)
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
        "attack_maximum_hand_travel": attack_travel,
        "death_hips_descent": death_hips_start.z - death_hips_end.z,
        "death_head_descent": death_head_start.z - death_head_end.z,
        "body_height": height,
        "retarget_gate_passed": finite and set(actions) == set(CLIPS),
        "human_approval_required": True,
        "human_approved": False,
    }


def main() -> int:
    args = _arguments()
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
    report_path = output_root / "retarget_validation.json"
    _write_json(report_path, report)

    target.animation_data.action = None
    bpy.context.scene.frame_set(1)
    worker._clear_pose(target)
    worker._remove_diagnostic_objects()
    checkpoint = output_root / "quaternius_retargeted_goblin.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(checkpoint))
    worker._triangulate_scene_for_export()
    exported = output_root / "quaternius_retargeted_goblin.glb"
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
