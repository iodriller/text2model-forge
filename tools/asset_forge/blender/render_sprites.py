import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def parse_args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--source")
    return parser.parse_args(raw)


def absolute(repo_root, value):
    return value if os.path.isabs(value) else os.path.join(repo_root, value)


def import_source(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".blend":
        return
    if extension == ".fbx":
        if hasattr(bpy.ops.wm, "fbx_import"):
            bpy.ops.wm.fbx_import(filepath=path)
        else:
            bpy.ops.import_scene.fbx(filepath=path)
        return
    if extension in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
        return
    raise RuntimeError(f"Unsupported source extension: {extension}")


def find_object(name):
    result = bpy.data.objects.get(name)
    if result is None:
        available = ", ".join(sorted(obj.name for obj in bpy.data.objects)[:30])
        raise RuntimeError(f"Animation object '{name}' was not found. Available: {available}")
    return result


def find_action(names):
    by_lower = {action.name.lower(): action for action in bpy.data.actions}
    for name in names:
        action = by_lower.get(name.lower())
        if action is not None:
            return action
    available = ", ".join(sorted(action.name for action in bpy.data.actions))
    raise RuntimeError(f"None of the actions {names} were found. Available: {available}")


def add_sun(rotation, energy, color):
    light_data = bpy.data.lights.new(name="AssetForgeKey", type="AREA")
    light_data.energy = energy
    light_data.color = color
    light_data.shape = "DISK"
    light_data.size = 5.0
    light = bpy.data.objects.new(name="AssetForgeKey", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (rotation[0], rotation[1], rotation[2])
    light.rotation_euler = (math.radians(25), 0, math.radians(145))
    return light


def configure_scene(config):
    scene = bpy.context.scene
    render_width, render_height = config["render_size"]
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(render_width)
    scene.render.resolution_y = int(render_height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.image_settings.color_depth = "8"
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    if scene.world is None:
        scene.world = bpy.data.worlds.new("AssetForgeWorld")
    scene.world.color = (0.018, 0.022, 0.03)

    camera_data = bpy.data.cameras.new("AssetForgeCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(config.get("orthographic_scale", 4.6))
    camera = bpy.data.objects.new("AssetForgeCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    add_sun((-4.0, -5.0, 8.0), 1050.0, (1.0, 0.82, 0.64))
    fill_data = bpy.data.lights.new(name="AssetForgeFill", type="AREA")
    fill_data.energy = 500.0
    fill_data.color = (0.18, 0.30, 0.52)
    fill_data.size = 6.0
    fill = bpy.data.objects.new(name="AssetForgeFill", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (4.0, 3.0, 5.0)
    return scene, camera


def position_camera(camera, direction, config):
    distance = float(config.get("camera_distance", 10.0))
    height = float(config.get("camera_height", 7.0))
    target_height = float(config.get("camera_target_height", 1.15))
    positions = {
        "north": (0.0, -distance, height),
        "south": (0.0, distance, height),
        "east": (-distance, 0.0, height),
        "west": (distance, 0.0, height),
    }
    if direction not in positions:
        raise RuntimeError(f"Unsupported direction: {direction}")
    camera.location = positions[direction]
    target = Vector((0.0, 0.0, target_height))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def sampled_frame(action, index, count, loop):
    start, end = float(action.frame_range[0]), float(action.frame_range[1])
    if count <= 1:
        return start
    denominator = count if loop else count - 1
    return start + ((end - start) * index / denominator)


def main():
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    source = absolute(repo_root, args.source or config["source"])
    if not os.path.isfile(source):
        raise RuntimeError(f"Character source not found: {source}")
    import_source(source)

    animation_object = None
    if config.get("animation_object"):
        animation_object = find_object(config["animation_object"])
        animation_object.animation_data_create()
    scene, camera = configure_scene(config)
    frames_root = os.path.abspath(args.frames_root)

    for animation_name, animation_config in config["animations"].items():
        action_names = animation_config.get("actions", [])
        action = find_action(action_names) if action_names else None
        if action is not None and animation_object is None:
            raise RuntimeError(f"Animation '{animation_name}' has actions but config has no animation_object")
        if animation_object is not None:
            animation_object.animation_data.action = action
        frame_count = int(animation_config["frames"])
        loop = bool(animation_config.get("loop", False))
        for direction in config["directions"]:
            position_camera(camera, direction, config)
            output_folder = os.path.join(frames_root, config["id"], animation_name, direction)
            os.makedirs(output_folder, exist_ok=True)
            for index in range(frame_count):
                frame = sampled_frame(action, index, frame_count, loop) if action is not None else float(index + 1)
                scene.frame_set(int(math.floor(frame)), subframe=frame - math.floor(frame))
                scene.render.filepath = os.path.join(output_folder, f"{index:02d}.png")
                bpy.ops.render.render(write_still=True)

    if animation_object is not None:
        animation_object.animation_data.action = None
    print(f"ASSET_FORGE_FRAMES={frames_root}")


main()
