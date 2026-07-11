"""Print the armatures, actions, and frame ranges of a .blend/.fbx/.glb source.

Usage: blender --background --python probe_actions.py -- --source <file>
"""

import argparse
import os
import sys

import bpy


def main():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args(raw)

    path = os.path.abspath(args.source)
    extension = os.path.splitext(path)[1].lower()
    if extension == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
    elif extension == ".fbx":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if hasattr(bpy.ops.wm, "fbx_import"):
            bpy.ops.wm.fbx_import(filepath=path)
        else:
            bpy.ops.import_scene.fbx(filepath=path)
    elif extension in (".glb", ".gltf"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise RuntimeError(f"Unsupported extension {extension}")

    print("PROBE_ARMATURES=" + ", ".join(obj.name for obj in bpy.data.objects if obj.type == "ARMATURE"))
    print("PROBE_MESHES=" + ", ".join(obj.name for obj in bpy.data.objects if obj.type == "MESH")[:400])
    for action in bpy.data.actions:
        print(f"PROBE_ACTION={action.name} range={tuple(action.frame_range)}")
    dimensions = [obj.dimensions.z for obj in bpy.data.objects if obj.type == "MESH"]
    if dimensions:
        print(f"PROBE_MAX_MESH_HEIGHT={max(dimensions):.3f}")


main()
