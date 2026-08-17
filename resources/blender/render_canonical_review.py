"""Render representative action frames from a canonical master for owner review."""

from __future__ import annotations

import argparse
import os
import sys

import bpy


def arguments():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(values)


def main():
    args = arguments()
    output = os.path.abspath(args.output_dir)
    os.makedirs(output, exist_ok=True)
    scene = bpy.context.scene
    rig = bpy.data.objects.get("AF_Rig")
    if rig is None or scene.camera is None:
        raise RuntimeError("canonical master is missing AF_Rig or its review camera")
    choices = [("idle", 1), ("walk", 13), ("attack", 11)]
    if bpy.data.actions.get("fly"):
        choices.append(("fly", 8))
    else:
        choices.append(("death", 20))
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    for action_name, frame in choices:
        value = bpy.data.actions.get(action_name)
        if value is None:
            continue
        rig.animation_data.action = value
        scene.frame_set(frame)
        scene.render.filepath = os.path.join(output, f"{action_name}-{frame:02d}.png")
        bpy.ops.render.render(write_still=True)
    rig.animation_data.action = None
    print("TEXT2MODEL_FORGE_CANONICAL_REVIEW=" + output)


main()
