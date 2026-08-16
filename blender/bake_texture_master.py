"""Texture master bake: paint once, render forever.

Per-frame diffusion (overpaint v3) proved the style but pays for consistency with GPU
hours and per-frame protection machinery.  This module moves the paint from frames to
the SURFACE: a handful of canonical views are rendered once, painted once by SDXL, then
projected back onto the mesh and baked into a UV texture atlas.  Every animation frame
afterwards is an ordinary deterministic Blender render of the same painted surface, so
frame-to-frame and rebuild-to-rebuild consistency is structural, not statistical.

Three modes, all run against the character master .blend given on the Blender CLI:

  render-views   render N orbit views (flat neutral light) + true-depth maps + metadata
  project-bake   given painted_XX.png views, bake a weighted multi-view projection into
                 a new UV atlas per mesh and save <master>_baked.blend
  icons          render a bust portrait and a full-body icon from the (baked) master

Projection weighting per view: facing^2 * visibility * paint alpha.  Visibility compares
the fragment's camera-space depth against that view's rendered depth map, so a shield
cannot stamp its paint onto the torso it occludes.  Texels no view can see fall back to
the master's original material color, never to black.

Usage:
  blender --background master.blend --python bake_texture_master.py -- \
      --mode render-views --config unit.character.json --work <bake work dir>
"""

import argparse
import json
import math
import os
import re
import sys

import bpy
from mathutils import Vector

DEPTH_HALF_RANGE = 2.8
OCCLUSION_EPSILON = 0.03
FACING_SHARPNESS = 2.0
FALLBACK_RAMP = 4.0


def parse_args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["render-views", "project-bake", "icons"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--work", required=True, help="bake working folder (views/, atlases/, manifest)")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", help="baked .blend destination (project-bake) or icon folder (icons)")
    return parser.parse_args(raw)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def bake_block(config):
    block = config.get("texture_master")
    if not isinstance(block, dict) or not block.get("enabled"):
        raise RuntimeError("Character config has no enabled 'texture_master' block")
    return block


def apply_bake_pose(config):
    """Views and projection bake must agree on ONE pose.  Rigs whose equipment sockets
    were measured in an action pose (bone parenting bakes offsets against the pose at
    parenting time) must bake in that same action; plain rigs use rest pose."""
    name = config.get("animation_object")
    if not name:
        return None
    armature = bpy.data.objects.get(name)
    if armature is None:
        raise RuntimeError(f"Animation object '{name}' was not found")
    block = config.get("texture_master") or {}
    pose_action = block.get("pose_action")
    if pose_action:
        action = next((a for a in bpy.data.actions if a.name.lower() == str(pose_action).lower()), None)
        if action is None:
            available = ", ".join(sorted(a.name for a in bpy.data.actions))
            raise RuntimeError(f"texture_master.pose_action {pose_action!r} not found. Actions: {available}")
        armature.animation_data_create()
        armature.animation_data.action = action
        armature.data.pose_position = "POSE"
        bpy.context.scene.frame_set(int(block.get("pose_frame", 1)))
    else:
        if armature.animation_data:
            armature.animation_data.action = None
        armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    return armature


def visible_meshes():
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if len(obj.data.polygons) == 0:
            continue
        meshes.append(obj)
    if not meshes:
        raise RuntimeError("Master contains no renderable meshes")
    return meshes


def flat_scene(resolution):
    """Neutral, even-light setup: the paint should carry the drama, the production
    render rig re-lights every frame, so the texture must not bake in a hard key."""
    scene = bpy.context.scene
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
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_freestyle = False
    scene.render.use_compositing = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("AssetForgeBakeWorld")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (0.68, 0.68, 0.68, 1.0)
        background.inputs[1].default_value = 1.0

    light_data = bpy.data.lights.new("AssetForgeBakeTop", type="AREA")
    light_data.energy = 420.0
    light_data.color = (1.0, 0.98, 0.95)
    light_data.shape = "DISK"
    light_data.size = 8.0
    light = bpy.data.objects.new("AssetForgeBakeTop", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, 0.0, 8.0)

    camera_data = bpy.data.cameras.new("AssetForgeBakeCamera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("AssetForgeBakeCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def orbit_camera(camera, yaw_deg, elevation_deg, distance, target, ortho_scale):
    yaw = math.radians(yaw_deg)
    elevation = math.radians(elevation_deg)
    offset = Vector((
        math.sin(yaw) * math.cos(elevation),
        -math.cos(yaw) * math.cos(elevation),
        math.sin(elevation),
    )) * distance
    camera.location = target + offset
    camera.data.ortho_scale = ortho_scale
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def camera_axes(camera):
    matrix = camera.matrix_world
    right = Vector((matrix[0][0], matrix[1][0], matrix[2][0])).normalized()
    up = Vector((matrix[0][1], matrix[1][1], matrix[2][1])).normalized()
    forward = -Vector((matrix[0][2], matrix[1][2], matrix[2][2])).normalized()
    return right, up, forward


def build_depth_tree(distance):
    scene = bpy.context.scene
    is_group_api = hasattr(scene, "compositing_node_group")
    if is_group_api:
        tree = bpy.data.node_groups.new("AssetForgeBakeDepth", "CompositorNodeTree")
        scene.compositing_node_group = tree
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    else:
        # Blender 4.5 owns compositor nodes directly on Scene.node_tree.
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.scene = bpy.context.scene
    output = tree.nodes.new("NodeGroupOutput" if is_group_api else "CompositorNodeComposite")
    if "Depth" not in layers.outputs:
        raise RuntimeError("Render Layers node exposes no Depth output")
    map_range = tree.nodes.new("ShaderNodeMapRange")
    map_range.clamp = True
    map_range.inputs["From Min"].default_value = distance - DEPTH_HALF_RANGE
    map_range.inputs["From Max"].default_value = distance + DEPTH_HALF_RANGE
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.0
    tree.links.new(layers.outputs["Depth"], map_range.inputs["Value"])
    tree.links.new(map_range.outputs["Result"], output.inputs[0])
    return tree


def render_still(scene, filepath):
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def render_views(args, config):
    block = bake_block(config)
    views = int(block.get("views", 8))
    resolution = int(block.get("view_size", 768))
    ortho_scale = float(block.get("view_scale", 4.2))
    elevation = float(block.get("view_elevation", 18.0))
    distance = 10.0
    target = Vector((0.0, 0.0, float(block.get("view_target_height", 1.4))))

    apply_bake_pose(config)
    scene, camera = flat_scene(resolution)
    scene.view_layers[0].use_pass_z = True
    depth_tree = build_depth_tree(distance)

    # 8-bit PNG output passes through the display view transform; "Standard" applies the
    # sRGB curve, which would silently break the exact depth comparison at projection
    # time.  "Raw" writes the normalized depth values as-is.
    depth_encoding = "linear"
    color_transform = scene.view_settings.view_transform
    try:
        scene.view_settings.view_transform = "Raw"
        scene.view_settings.view_transform = color_transform
    except TypeError:
        depth_encoding = "srgb"

    views_folder = os.path.join(args.work, "views")
    os.makedirs(views_folder, exist_ok=True)
    metadata = {
        "views": [],
        "resolution": resolution,
        "ortho_scale": ortho_scale,
        "depth_encoding": depth_encoding,
    }
    # Masters front +Y (the game's south camera sees the face), so yaw 180 is the
    # front; view 0 faces the character for anchor creation and human review.  The
    # orbit ring alone leaves helmet/shoulder tops uncovered, so two elevated views
    # are always added on top of the ring.
    specs = [((180.0 + index * 360.0 / views) % 360.0, elevation) for index in range(views)]
    for extra in block.get("extra_views", [{"yaw": 210.0, "elevation": 58.0}, {"yaw": 330.0, "elevation": 58.0}]):
        specs.append((float(extra["yaw"]), float(extra["elevation"])))
    for index, (yaw, view_elevation) in enumerate(specs):
        orbit_camera(camera, yaw, view_elevation, distance, target, ortho_scale)
        right, up, forward = camera_axes(camera)
        render_still(scene, os.path.join(views_folder, f"view_{index:02d}.png"))
        if hasattr(scene, "compositing_node_group"):
            scene.compositing_node_group = depth_tree
        else:
            scene.use_nodes = True
        scene.render.use_compositing = True
        if depth_encoding == "linear":
            scene.view_settings.view_transform = "Raw"
        render_still(scene, os.path.join(views_folder, f"depth_{index:02d}.png"))
        scene.view_settings.view_transform = color_transform
        scene.render.use_compositing = False
        if hasattr(scene, "compositing_node_group"):
            scene.compositing_node_group = None
        else:
            scene.use_nodes = False
        metadata["views"].append({
            "index": index,
            "yaw_deg": yaw,
            "camera_location": list(camera.location),
            "right": list(right),
            "up": list(up),
            "forward": list(forward),
            "ortho_scale": ortho_scale,
            "depth_from_min": distance - DEPTH_HALF_RANGE,
            "depth_from_max": distance + DEPTH_HALF_RANGE,
        })
    with open(os.path.join(args.work, "views.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"ASSET_FORGE_BAKE_VIEWS={views_folder}")


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def ensure_bake_uv(obj):
    """Quaternius-style masters map many faces onto a few flat-color texels; painted
    surfaces need a real one-to-one layout, so every mesh gets a fresh smart unwrap."""
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    mesh = obj.data
    existing = mesh.uv_layers.get("AssetForgeBake")
    if existing is not None:
        mesh.uv_layers.remove(existing)
    layer = mesh.uv_layers.new(name="AssetForgeBake")
    mesh.uv_layers.active = layer
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.02, correct_aspect=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    # The pre-edit-mode layer reference is dangling after the mode round-trip; never
    # touch it again.  The layer is addressed by name from here on.
    return "AssetForgeBake"


def bake_original_diffuse(copy, target):
    """Bake the original materials' diffuse color into the atlas layout.

    Texels no painted view can see (armpits, seams, occluded backs) fall back to this
    layer, so they read as the authored master instead of a default-gray scar.  Baking
    handles every material construction — palette textures, vertex colors, constants —
    without guessing at node graphs."""
    added = []
    for slot in copy.material_slots:
        material = slot.material
        if material is None or not material.use_nodes:
            continue
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = target
        material.node_tree.nodes.active = node
        node.select = True
        added.append((material, node))
    if not added:
        return False
    try:
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, margin=12, use_clear=True)
    finally:
        for material, node in added:
            material.node_tree.nodes.remove(node)
    return True


def posed_copy(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
    copy = bpy.data.objects.new(obj.name + ".bakecopy", mesh)
    copy.matrix_world = evaluated.matrix_world.copy()
    bpy.context.collection.objects.link(copy)
    return copy


def build_projection_material(name, view_records, fallback_image, target_image, depth_encoding="linear"):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    geometry = nodes.new("ShaderNodeNewGeometry")

    def value_node(default):
        node = nodes.new("ShaderNodeValue")
        node.outputs[0].default_value = default
        return node.outputs[0]

    def math_node(operation, first, second):
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        _plug(links, first, node.inputs[0])
        _plug(links, second, node.inputs[1])
        return node.outputs[0]

    def vector_math(operation, first, second):
        node = nodes.new("ShaderNodeVectorMath")
        node.operation = operation
        _plug(links, first, node.inputs[0])
        _plug(links, second, node.inputs[1])
        return node.outputs["Value"] if operation == "DOT_PRODUCT" else node.outputs[0]

    def scale_color(color, scalar):
        node = nodes.new("ShaderNodeVectorMath")
        node.operation = "SCALE"
        _plug(links, color, node.inputs[0])
        _plug(links, scalar, node.inputs["Scale"])
        return node.outputs[0]

    color_sum = None
    weight_sum = None
    for record in view_records:
        cam_loc = Vector(record["camera_location"])
        right = Vector(record["right"])
        up = Vector(record["up"])
        forward = Vector(record["forward"])
        scale = float(record["ortho_scale"])

        relative = vector_math("SUBTRACT", geometry.outputs["Position"], tuple(cam_loc))
        u = math_node("ADD", math_node("DIVIDE", vector_math("DOT_PRODUCT", relative, tuple(right)), value_node(scale)), value_node(0.5))
        v = math_node("ADD", math_node("DIVIDE", vector_math("DOT_PRODUCT", relative, tuple(up)), value_node(scale)), value_node(0.5))
        combine = nodes.new("ShaderNodeCombineXYZ")
        _plug(links, u, combine.inputs["X"])
        _plug(links, v, combine.inputs["Y"])

        painted = nodes.new("ShaderNodeTexImage")
        painted.image = record["painted_image"]
        painted.extension = "CLIP"
        painted.interpolation = "Linear"
        links.new(combine.outputs[0], painted.inputs["Vector"])

        depth_tex = nodes.new("ShaderNodeTexImage")
        depth_tex.image = record["depth_image"]
        depth_tex.extension = "CLIP"
        depth_tex.interpolation = "Linear"
        depth_tex.image.colorspace_settings.name = "Non-Color"
        links.new(combine.outputs[0], depth_tex.inputs["Vector"])

        fragment_depth = vector_math("DOT_PRODUCT", relative, tuple(forward))
        map_range = nodes.new("ShaderNodeMapRange")
        map_range.clamp = True
        map_range.inputs["From Min"].default_value = float(record["depth_from_min"])
        map_range.inputs["From Max"].default_value = float(record["depth_from_max"])
        map_range.inputs["To Min"].default_value = 1.0
        map_range.inputs["To Max"].default_value = 0.0
        _plug(links, fragment_depth, map_range.inputs["Value"])
        stored = depth_tex.outputs["Color"]
        if depth_encoding == "srgb":
            # Legacy fallback: depth PNGs written through the Standard display
            # transform carry the sRGB curve; approximate the inverse.
            stored = math_node("POWER", stored, value_node(2.2))
        stored_minus_fragment = math_node("SUBTRACT", stored, map_range.outputs["Result"])
        visible = math_node("LESS_THAN", stored_minus_fragment, value_node(OCCLUSION_EPSILON))

        facing_raw = vector_math("DOT_PRODUCT", geometry.outputs["Normal"], tuple(-forward))
        facing = math_node("POWER", math_node("MAXIMUM", facing_raw, value_node(0.0)), value_node(FACING_SHARPNESS))

        weight = math_node("MULTIPLY", math_node("MULTIPLY", facing, visible), painted.outputs["Alpha"])
        contribution = scale_color(painted.outputs["Color"], weight)

        color_sum = contribution if color_sum is None else vector_math("ADD", color_sum, contribution)
        weight_sum = weight if weight_sum is None else math_node("ADD", weight_sum, weight)

    total = math_node("MAXIMUM", weight_sum, value_node(1e-4))
    total_vector = nodes.new("ShaderNodeCombineXYZ")
    for socket_name in ("X", "Y", "Z"):
        _plug(links, total, total_vector.inputs[socket_name])
    average = vector_math("DIVIDE", color_sum, total_vector.outputs[0])

    if fallback_image is not None:
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.uv_map = "AssetForgeBake"
        fallback_tex = nodes.new("ShaderNodeTexImage")
        fallback_tex.image = fallback_image
        links.new(uv_node.outputs[0], fallback_tex.inputs["Vector"])
        fallback_color = fallback_tex.outputs["Color"]
    else:
        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (0.18, 0.18, 0.20, 1.0)
        fallback_color = rgb.outputs[0]

    presence = math_node("MINIMUM", math_node("MULTIPLY", weight_sum, value_node(FALLBACK_RAMP)), value_node(1.0))
    inverse = math_node("SUBTRACT", value_node(1.0), presence)
    final_color = vector_math("ADD", scale_color(average, presence), scale_color(fallback_color, inverse))
    _plug(links, final_color, emission.inputs["Color"])

    target_node = nodes.new("ShaderNodeTexImage")
    target_node.image = target_image
    nodes.active = target_node
    target_node.select = True
    return material


def _plug(links, source, destination):
    """Wire a node socket, or set a constant when a plain value is provided."""
    if source is None:
        return
    if isinstance(source, (int, float)):
        destination.default_value = float(source)
        return
    if isinstance(source, (tuple, list, Vector)):
        destination.default_value = tuple(source)
        return
    links.new(source, destination)


def project_bake(args, config):
    block = bake_block(config)
    unit = str(config["id"])
    views_folder = os.path.join(args.work, "views")
    with open(os.path.join(args.work, "views.json"), "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    view_records = []
    for record in metadata["views"]:
        index = int(record["index"])
        painted_path = os.path.join(views_folder, f"painted_{index:02d}.png")
        depth_path = os.path.join(views_folder, f"depth_{index:02d}.png")
        if not os.path.isfile(painted_path):
            raise RuntimeError(f"Painted view is missing: {painted_path}")
        record = dict(record)
        record["painted_image"] = bpy.data.images.load(painted_path)
        record["depth_image"] = bpy.data.images.load(depth_path)
        view_records.append(record)

    apply_bake_pose(config)
    meshes = visible_meshes()
    body = max(meshes, key=lambda obj: obj.dimensions.x * obj.dimensions.y * obj.dimensions.z)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.render.bake.margin = 12

    atlases_folder = os.path.join(args.work, "atlases")
    os.makedirs(atlases_folder, exist_ok=True)
    manifest = {"unit": unit, "objects": []}
    default_resolution = int(block.get("bake_resolution", 2048))
    for obj in meshes:
        ensure_bake_uv(obj)
        resolution = default_resolution if obj is body else max(512, default_resolution // 2)
        image_name = sanitize(f"{unit}_{obj.name}_baked")
        target = bpy.data.images.new(image_name, width=resolution, height=resolution, alpha=False)
        target.generated_color = (0.0, 0.0, 0.0, 1.0)

        copy = posed_copy(obj)
        copy.data.uv_layers.active = copy.data.uv_layers["AssetForgeBake"]
        bpy.ops.object.select_all(action="DESELECT")
        copy.select_set(True)
        bpy.context.view_layer.objects.active = copy

        fallback_image = bpy.data.images.new(image_name + "_base", width=resolution, height=resolution, alpha=False)
        if not bake_original_diffuse(copy, fallback_image):
            bpy.data.images.remove(fallback_image)
            fallback_image = None

        material = build_projection_material(
            f"AssetForgeProjection_{obj.name}", view_records, fallback_image, target,
            depth_encoding=str(metadata.get("depth_encoding", "linear")),
        )
        copy.data.materials.clear()
        copy.data.materials.append(material)
        bpy.ops.object.bake(type="EMIT", margin=12, use_clear=True)

        atlas_path = os.path.join(atlases_folder, image_name + ".png")
        target.filepath_raw = atlas_path
        target.file_format = "PNG"
        target.save()
        target.pack()

        baked = bpy.data.materials.new(f"AssetForge Baked {obj.name}")
        baked.use_nodes = True
        principled = next(n for n in baked.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        principled.inputs["Roughness"].default_value = 0.75
        principled.inputs["Metallic"].default_value = 0.0
        texture = baked.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = target
        uv_node = baked.node_tree.nodes.new("ShaderNodeUVMap")
        uv_node.uv_map = "AssetForgeBake"
        baked.node_tree.links.new(uv_node.outputs[0], texture.inputs["Vector"])
        baked.node_tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
        obj.data.materials.clear()
        obj.data.materials.append(baked)

        bpy.data.objects.remove(copy, do_unlink=True)
        manifest["objects"].append({"object": obj.name, "atlas": atlas_path, "resolution": resolution})

    armature_name = config.get("animation_object")
    if armature_name and bpy.data.objects.get(armature_name):
        bpy.data.objects[armature_name].data.pose_position = "POSE"

    destination = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    bpy.data.use_autopack = False
    bpy.ops.wm.save_as_mainfile(filepath=destination)
    with open(os.path.join(args.work, "bake-manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"ASSET_FORGE_BAKED_MASTER={destination}")


def render_icons(args, config):
    """Bust portrait + full-body icon from the styled master; UI needs both per unit."""
    name = config.get("animation_object")
    armature = bpy.data.objects.get(name) if name else None
    idle = config.get("animations", {}).get("idle", {})
    actions = idle.get("actions") or []
    if armature is not None and actions:
        action = next((a for a in bpy.data.actions if a.name.lower() == actions[0].lower()), None)
        if action is not None:
            armature.animation_data_create()
            armature.animation_data.action = action
            samples = idle.get("sample_frames") or [1]
            bpy.context.scene.frame_set(int(samples[0]))
            bpy.context.view_layer.update()

    scene, camera = flat_scene(int(config.get("render_size", [768, 768])[0]))
    # Icons carry presentation lighting, unlike bake views.
    key_data = bpy.data.lights.new("AssetForgeIconKey", type="AREA")
    key_data.energy = 1000.0
    key_data.color = (1.0, 0.9, 0.75)
    key_data.size = 5.0
    key = bpy.data.objects.new("AssetForgeIconKey", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (-3.5, -4.5, 6.5)
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue

    folder = os.path.abspath(args.output)
    os.makedirs(folder, exist_ok=True)
    unit = str(config["id"])
    block = config.get("texture_master") or {}

    portrait_height = float(block.get("portrait_height", 2.2))
    orbit_camera(camera, 210.0, 8.0, 10.0, Vector((0.0, 0.0, portrait_height)), float(block.get("portrait_scale", 2.0)))
    render_still(scene, os.path.join(folder, f"{unit}_portrait.png"))

    orbit_camera(camera, 210.0, 16.0, 10.0, Vector((0.0, 0.0, 1.35)), float(config.get("orthographic_scale", 5.0)) * 0.72)
    render_still(scene, os.path.join(folder, f"{unit}_icon.png"))
    print(f"ASSET_FORGE_ICONS={folder}")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "render-views":
        render_views(args, config)
    elif args.mode == "project-bake":
        if not args.output:
            raise RuntimeError("--output (baked .blend path) is required for project-bake")
        project_bake(args, config)
    else:
        if not args.output:
            raise RuntimeError("--output (icon folder) is required for icons")
        render_icons(args, config)


try:
    main()
except Exception:
    # Microsoft Store Blender does not forward background stdout/stderr through its
    # app-execution alias. Persist the traceback beside the stage so orchestrators can
    # still diagnose a failed, resumable bake.
    import traceback

    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if "--work" in raw:
        folder = os.path.abspath(raw[raw.index("--work") + 1])
        os.makedirs(folder, exist_ok=True)
        mode = raw[raw.index("--mode") + 1] if "--mode" in raw else "unknown"
        with open(os.path.join(folder, f"blender-{mode}-error.txt"), "w", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
    raise
