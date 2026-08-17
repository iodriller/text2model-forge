"""Rename a Rigify-named donor animation onto the Unreal bone names D7 expects.

Why this is needed
------------------
`resources/adapters/retarget_humanoid_motion.py` retargets against a fixed `BONE_CHAINS`
contract written for the **Unreal-Godot** build of Quaternius's Universal
Animation Library (`pelvis`, `spine_01`, `thigh_l`, `ball_r`, `Head`, ...).
That build is distributed as an itch.io zip. The CC0 glTF mirror that can be
fetched non-interactively ships the **Godot/Rigify** build instead, whose
skeleton is identical in structure but named `DEF-hips`, `DEF-spine.001`,
`DEF-thigh.L` and so on. Same rig, same 45 clips, different labels.

So the contract is not actually unmet -- only the naming is. This script does
the one-to-one rename and re-exports a `.glb`, rather than loosening a
qualified retarget adapter to accept two vocabularies.

Handedness
----------
The retarget contract deliberately maps Quaternius *right* to the Text2Model
`*_l` chain, because Quaternius's right side sits at negative X. Verified on
the Rigify build before writing this: `DEF-shoulder.R` is at x=-0.0188 and
`DEF-shoulder.L` at x=+0.0188, so the anatomical rename below preserves that
same sign convention exactly and the adapter's existing side logic keeps
applying unchanged. Getting this backwards would silently mirror every clip,
which is why it was measured rather than assumed.

Usage
-----
    blender --background --factory-startup --python resources/adapters/prepare_donor_motion.py \
        -- --source <rigify.glb|.gltf> --output <unreal-named.glb>
"""
from __future__ import annotations

import argparse
import sys

import bpy


# Rigify DEF- name -> the Unreal name retarget_humanoid_motion.BONE_CHAINS wants.
# Only bones that contract names need to be exact; the finger chain is carried
# through untouched because the retarget never addresses it.
RIGIFY_TO_UNREAL = {
    "DEF-hips": "pelvis",
    "DEF-spine.001": "spine_01",
    "DEF-spine.002": "spine_02",
    "DEF-spine.003": "spine_03",
    "DEF-neck": "neck_01",
    "DEF-head": "Head",
    "DEF-shoulder.L": "clavicle_l",
    "DEF-shoulder.R": "clavicle_r",
    "DEF-upper_arm.L": "upperarm_l",
    "DEF-upper_arm.R": "upperarm_r",
    "DEF-forearm.L": "lowerarm_l",
    "DEF-forearm.R": "lowerarm_r",
    "DEF-hand.L": "hand_l",
    "DEF-hand.R": "hand_r",
    "DEF-thigh.L": "thigh_l",
    "DEF-thigh.R": "thigh_r",
    "DEF-shin.L": "calf_l",
    "DEF-shin.R": "calf_r",
    "DEF-foot.L": "foot_l",
    "DEF-foot.R": "foot_r",
    "DEF-toe.L": "ball_l",
    "DEF-toe.R": "ball_r",
}


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _arguments(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if args.source.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=args.source)
    else:
        raise SystemExit("source must be .glb or .gltf")

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise SystemExit(f"expected exactly one armature, found {len(armatures)}")
    armature = armatures[0]

    present = {bone.name for bone in armature.data.bones}
    missing = sorted(set(RIGIFY_TO_UNREAL) - present)
    if missing:
        raise SystemExit(f"source is not the expected Rigify skeleton; missing {missing}")

    # Renaming the bone renames it everywhere Blender references it by name --
    # vertex groups, constraints, and crucially every action's fcurve data
    # path -- so the 45 clips follow the rename without being touched.
    for rigify_name, unreal_name in RIGIFY_TO_UNREAL.items():
        armature.data.bones[rigify_name].name = unreal_name

    renamed = {bone.name for bone in armature.data.bones}
    still_missing = sorted(set(RIGIFY_TO_UNREAL.values()) - renamed)
    if still_missing:
        raise SystemExit(f"rename did not produce {still_missing}")

    bpy.ops.export_scene.gltf(
        filepath=args.output, export_format="GLB", export_animations=True
    )
    print(f"prepared donor motion: {args.output}")
    print(f"  actions carried: {len(bpy.data.actions)}")
    print(f"  bones renamed:   {len(RIGIFY_TO_UNREAL)}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    raise SystemExit(main(argv))
