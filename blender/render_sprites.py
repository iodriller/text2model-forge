import argparse
import json
import math
import os
import shutil
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
    parser.add_argument("--actions", help="optional comma-separated incremental render subset")
    parser.add_argument(
        "--skip-aux-passes",
        action="store_true",
        help="skip depth/equipment/protection passes (baked-texture masters need no overpaint support)",
    )
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
        # Blender 4.2 called this BLENDER_EEVEE_NEXT; 5.x renamed it back to
        # BLENDER_EEVEE. Pick whichever this build offers.
        scene.render.engine = next(
            (name for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
             if name in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()),
            "BLENDER_EEVEE",
        )
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(render_width)
    scene.render.resolution_y = int(render_height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.image_settings.color_depth = "8"
    scene.render.use_freestyle = bool(config.get("outline", True))
    if scene.render.use_freestyle:
        scene.render.line_thickness = float(config.get("outline_thickness", 0.72))
        line_set = scene.view_layers[0].freestyle_settings.linesets[0]
        if line_set.linestyle is None:
            line_set.linestyle = bpy.data.linestyles.new("Asset Forge Outline")
        line_style = line_set.linestyle
        line_style.color = (0.012, 0.016, 0.02)
        line_style.alpha = 0.92
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.view_settings.exposure = float(config.get("render_exposure", 0.0))
    if scene.world is None:
        scene.world = bpy.data.worlds.new("AssetForgeWorld")
    scene.world.color = (0.018, 0.022, 0.03)

    camera_data = bpy.data.cameras.new("AssetForgeCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(config.get("orthographic_scale", 4.6))
    camera = bpy.data.objects.new("AssetForgeCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    add_sun((-4.0, -5.0, 8.0), 1400.0, (1.0, 0.86, 0.68))
    fill_data = bpy.data.lights.new(name="AssetForgeFill", type="AREA")
    fill_data.energy = 650.0
    fill_data.color = (0.24, 0.38, 0.60)
    fill_data.size = 6.0
    fill = bpy.data.objects.new(name="AssetForgeFill", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (4.0, 3.0, 5.0)

    # Cool top rim pulls the silhouette off dark battlefield backgrounds; pointing
    # straight down keeps all four camera directions lit identically.
    rim_data = bpy.data.lights.new(name="AssetForgeRim", type="AREA")
    rim_data.energy = 850.0
    rim_data.color = (0.55, 0.70, 0.95)
    rim_data.shape = "DISK"
    rim_data.size = 3.5
    rim = bpy.data.objects.new(name="AssetForgeRim", object_data=rim_data)
    bpy.context.collection.objects.link(rim)
    rim.location = (0.0, 0.0, 7.5)
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


def required_orthographic_scale(camera, config):
    """Measure evaluated meshes in camera space so equipment cannot clip silently."""
    inverse_camera = camera.matrix_world.inverted()
    aspect = float(config["render_size"][0]) / float(config["render_size"][1])
    maximum_x = 0.0
    maximum_y = 0.0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            for vertex in mesh.vertices:
                camera_point = inverse_camera @ (evaluated.matrix_world @ vertex.co)
                maximum_x = max(maximum_x, abs(float(camera_point.x)))
                maximum_y = max(maximum_y, abs(float(camera_point.y)))
        finally:
            evaluated.to_mesh_clear()
    if maximum_x <= 0.0 or maximum_y <= 0.0:
        raise RuntimeError("Auto-frame found no visible evaluated mesh bounds")
    return max(maximum_y * 2.0, maximum_x * 2.0 / aspect)


def auto_frame_animation(camera, animation_object, action, animation_config, config):
    """Choose one stable scale for every frame and direction in a clip."""
    frame_count = int(animation_config["frames"])
    loop = bool(animation_config.get("loop", False))
    required = 0.0
    for direction in config["directions"]:
        position_camera(camera, direction, config)
        for index in range(frame_count):
            frame = sampled_frame(action, index, frame_count, loop, animation_config) if action is not None else float(index + 1)
            bpy.context.scene.frame_set(int(math.floor(frame)), subframe=frame - math.floor(frame))
            bpy.context.view_layer.update()
            required = max(required, required_orthographic_scale(camera, config))
    margin = float(config.get("auto_frame_margin_fraction", 0.10))
    if margin < 0.0 or margin >= 0.4:
        raise RuntimeError("auto_frame_margin_fraction must be in [0, 0.4)")
    minimum = float(config.get("orthographic_scale", 0.0))
    return max(minimum, required * (1.0 + 2.0 * margin))


def build_depth_tree(scene, config):
    """Compositor tree whose OUTPUT is normalized depth (white = near).

    Assigned to the scene only during the dedicated depth render pass, so the
    grayscale is written through the ordinary PNG path (File Output nodes inside
    Blender 5 node groups only speak multilayer EXR).
    """
    scene.view_layers[0].use_pass_z = True
    tree = bpy.data.node_groups.new("AssetForgeDepth", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.scene = bpy.context.scene
    group_output = tree.nodes.new("NodeGroupOutput")
    if "Depth" not in layers.outputs:
        raise RuntimeError(
            "Render Layers node exposes no Depth output; passes: "
            + ", ".join(s.name for s in layers.outputs)
        )

    distance = float(config.get("camera_distance", 10.0))
    height = float(config.get("camera_height", 7.0))
    target_height = float(config.get("camera_target_height", 1.15))
    camera_span = math.sqrt(distance * distance + (height - target_height) ** 2)
    # Blender 5 unified the utility nodes: Map Range is a shader-style node here.
    map_range = tree.nodes.new("ShaderNodeMapRange")
    map_range.clamp = True
    map_range.inputs["From Min"].default_value = camera_span - 2.6
    map_range.inputs["From Max"].default_value = camera_span + 2.6
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.0
    tree.links.new(layers.outputs["Depth"], map_range.inputs["Value"])
    tree.links.new(map_range.outputs["Result"], group_output.inputs[0])
    return tree


def render_depth(scene, depth_tree, depth_path):
    freestyle = scene.render.use_freestyle
    look = scene.view_settings.look
    view_transform = scene.view_settings.view_transform
    scene.render.use_freestyle = False
    scene.compositing_node_group = depth_tree
    scene.render.use_compositing = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.filepath = depth_path
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        scene.render.use_freestyle = freestyle
        scene.render.use_compositing = False
        scene.compositing_node_group = None
        scene.view_settings.view_transform = view_transform
        scene.view_settings.look = look


def equipment_objects(patterns):
    import fnmatch

    matched = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if any(fnmatch.fnmatchcase(obj.name, pattern) for pattern in patterns):
            matched.append(obj)
    if patterns and not matched:
        available = ", ".join(sorted(obj.name for obj in bpy.data.objects if obj.type == "MESH"))[:400]
        raise RuntimeError(f"No meshes match equipment_objects {patterns}. Meshes: {available}")
    return matched


def mask_material(name, color):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Roughness"].default_value = 1.0
    emission = node.inputs.get("Emission Color") or node.inputs.get("Emission")
    if emission is not None:
        emission.default_value = (*color, 1.0)
    strength = node.inputs.get("Emission Strength")
    if strength is not None:
        strength.default_value = 1.0
    return material


def render_equipment_mask(scene, equipment, mask_path):
    """Visibility-aware white/black material mask; body occlusion is preserved."""
    white = mask_material("AssetForge Mask White", (1.0, 1.0, 1.0))
    black = mask_material("AssetForge Mask Black", (0.0, 0.0, 0.0))
    slot_backups = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if not obj.material_slots:
            obj.data.materials.append(None)
            added_slot = True
        else:
            added_slot = False
        target = white if obj in equipment else black
        for slot in obj.material_slots:
            slot_backups.append((obj, slot, slot.link, slot.material, added_slot))
            slot.link = "OBJECT"
            slot.material = target
    freestyle = scene.render.use_freestyle
    compositing = scene.render.use_compositing
    look = scene.view_settings.look
    view_transform = scene.view_settings.view_transform
    scene.render.use_freestyle = False
    scene.render.use_compositing = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.filepath = mask_path
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        scene.render.use_freestyle = freestyle
        scene.render.use_compositing = compositing
        scene.view_settings.view_transform = view_transform
        scene.view_settings.look = look
        added_objects = set()
        for obj, slot, link, material, added_slot in slot_backups:
            slot.material = material
            slot.link = link
            if added_slot:
                added_objects.add(obj)
        for obj in added_objects:
            if obj.data.materials and obj.data.materials[-1] is None:
                obj.data.materials.pop(index=len(obj.data.materials) - 1)


def sampled_frame(action, index, count, loop, animation_config):
    explicit = animation_config.get("sample_frames")
    if explicit is not None:
        if len(explicit) != count:
            raise RuntimeError(
                f"sample_frames has {len(explicit)} values but frames is {count}"
            )
        return float(explicit[index])
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
    depth_tree = build_depth_tree(scene, config) if config.get("depth_pass") and not args.skip_aux_passes else None
    scene.render.use_compositing = False
    equipment = [] if args.skip_aux_passes else equipment_objects(config.get("equipment_objects") or [])
    protected_patterns = config.get("protected_objects") or config.get("equipment_objects") or []
    protected = [] if args.skip_aux_passes else equipment_objects(protected_patterns)

    selected_actions = None
    if args.actions:
        selected_actions = {value.strip() for value in args.actions.split(",") if value.strip()}
        unknown = selected_actions.difference(config["animations"])
        if unknown:
            raise RuntimeError(f"Unknown incremental actions: {sorted(unknown)}")

    for animation_name, animation_config in config["animations"].items():
        if selected_actions is not None and animation_name not in selected_actions:
            continue
        action_names = animation_config.get("actions", [])
        action = find_action(action_names) if action_names else None
        if action is not None and animation_object is None:
            raise RuntimeError(f"Animation '{animation_name}' has actions but config has no animation_object")
        if animation_object is not None:
            animation_object.animation_data.action = action
        frame_count = int(animation_config["frames"])
        loop = bool(animation_config.get("loop", False))
        view_config = dict(config)
        view_config.update(animation_config.get("camera", {}))
        if view_config.get("auto_frame", False):
            camera.data.ortho_scale = auto_frame_animation(
                camera,
                animation_object,
                action,
                animation_config,
                view_config,
            )
        else:
            camera.data.ortho_scale = float(view_config.get("orthographic_scale", 4.6))
        for direction in config["directions"]:
            position_camera(camera, direction, view_config)
            output_folder = os.path.join(frames_root, config["id"], animation_name, direction)
            os.makedirs(output_folder, exist_ok=True)
            depth_folder = os.path.join(frames_root, config["id"] + "-depth", animation_name, direction)
            equip_folder = os.path.join(frames_root, config["id"] + "-equip", animation_name, direction)
            protect_folder = os.path.join(frames_root, config["id"] + "-protect", animation_name, direction)
            if depth_tree is not None:
                os.makedirs(depth_folder, exist_ok=True)
            if equipment:
                os.makedirs(equip_folder, exist_ok=True)
            if protected:
                os.makedirs(protect_folder, exist_ok=True)
            for index in range(frame_count):
                frame = sampled_frame(action, index, frame_count, loop, animation_config) if action is not None else float(index + 1)
                scene.frame_set(int(math.floor(frame)), subframe=frame - math.floor(frame))
                scene.render.filepath = os.path.join(output_folder, f"{index:02d}.png")
                bpy.ops.render.render(write_still=True)
                if depth_tree is not None:
                    render_depth(scene, depth_tree, os.path.join(depth_folder, f"{index:02d}.png"))
                if equipment:
                    equip_path = os.path.join(equip_folder, f"{index:02d}.png")
                    render_equipment_mask(scene, equipment, equip_path)
                if protected:
                    protect_path = os.path.join(protect_folder, f"{index:02d}.png")
                    if {obj.name for obj in protected} == {obj.name for obj in equipment}:
                        shutil.copy2(equip_path, protect_path)
                    else:
                        render_equipment_mask(scene, protected, protect_path)

    if animation_object is not None:
        animation_object.animation_data.action = None
    print(f"ASSET_FORGE_FRAMES={frames_root}")


main()
