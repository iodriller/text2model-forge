"""Deterministic Blender background worker for Darkness mesh evidence and bounded cleanup."""
from __future__ import annotations

import argparse
from array import array
import json
import math
import sys
import time
import traceback
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


SUPPORTED_OPERATIONS = {
    "blender.analyze",
    "blender.repair",
    "blender.render_diagnostics",
    "blender.export",
}


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args(argv)


def _mesh_objects() -> list[bpy.types.Object]:
    return sorted(
        (item for item in bpy.context.scene.objects if item.type == "MESH"),
        key=lambda item: item.name,
    )


def _import_asset(path: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    suffix = path.suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"unsupported Blender input type: {suffix}")
    if not _mesh_objects():
        raise ValueError("input contains no mesh objects")


def _face_components(bm: bmesh.types.BMesh) -> list[list[bmesh.types.BMFace]]:
    remaining = set(bm.faces)
    components: list[list[bmesh.types.BMFace]] = []
    while remaining:
        seed = min(remaining, key=lambda item: item.index)
        remaining.remove(seed)
        stack = [seed]
        component: list[bmesh.types.BMFace] = []
        while stack:
            face = stack.pop()
            component.append(face)
            for edge in face.edges:
                for neighbor in edge.link_faces:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        components.append(sorted(component, key=lambda item: item.index))
    return sorted(components, key=lambda item: (-len(item), item[0].index if item else -1))


def _object_analysis(obj: bpy.types.Object) -> tuple[dict[str, object], list[list[int]]]:
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        components = _face_components(bm)
        used = {vertex for face in bm.faces for vertex in face.verts}
        finite = all(
            math.isfinite(value)
            for vertex in bm.verts
            for value in tuple(obj.matrix_world @ vertex.co)
        )
        inconsistent = 0
        for edge in bm.edges:
            if len(edge.link_loops) != 2:
                continue
            first, second = edge.link_loops
            first_direction = (first.vert.index, first.link_loop_next.vert.index)
            second_direction = (second.vert.index, second.link_loop_next.vert.index)
            if first_direction == second_direction:
                inconsistent += 1
        report = {
            "name": obj.name,
            "vertices": len(bm.verts),
            "edges": len(bm.edges),
            "faces": len(bm.faces),
            "components": len(components),
            "component_faces": [len(component) for component in components],
            "isolated_vertices": len(bm.verts) - len(used),
            "degenerate_faces": sum(face.calc_area() <= 1e-12 for face in bm.faces),
            "boundary_edges": sum(len(edge.link_faces) == 1 for edge in bm.edges),
            "non_manifold_edges": sum(len(edge.link_faces) > 2 for edge in bm.edges),
            "inconsistent_winding_edges": inconsistent,
            "finite_coordinates": finite,
        }
        return report, [[face.index for face in component] for component in components]
    finally:
        bm.free()


def analyze_scene(*, include_face_indices: bool = False) -> dict[str, object]:
    objects: list[dict[str, object]] = []
    component_records: list[dict[str, object]] = []
    for obj in _mesh_objects():
        report, components = _object_analysis(obj)
        objects.append(report)
        for local_index, faces in enumerate(components):
            record: dict[str, object] = {
                "object": obj.name,
                "local_index": local_index,
                "faces": len(faces),
            }
            if include_face_indices:
                record["face_indices"] = faces
            component_records.append(record)
    component_records.sort(key=lambda item: (-int(item["faces"]), str(item["object"]), int(item["local_index"])))
    return {
        "blender_version": bpy.app.version_string,
        "mesh_objects": len(objects),
        "vertices": sum(int(item["vertices"]) for item in objects),
        "faces": sum(int(item["faces"]) for item in objects),
        "connected_components": len(component_records),
        "isolated_vertices": sum(int(item["isolated_vertices"]) for item in objects),
        "degenerate_faces": sum(int(item["degenerate_faces"]) for item in objects),
        "boundary_edges": sum(int(item["boundary_edges"]) for item in objects),
        "non_manifold_edges": sum(int(item["non_manifold_edges"]) for item in objects),
        "inconsistent_winding_edges": sum(int(item["inconsistent_winding_edges"]) for item in objects),
        "finite_coordinates": all(bool(item["finite_coordinates"]) for item in objects),
        "objects": objects,
        "components": component_records,
    }


def keep_largest_component(*, weld_distance: float = 0.0) -> dict[str, object]:
    internal = analyze_scene(include_face_indices=True)
    components = list(internal["components"])
    if not components:
        raise ValueError("scene has no connected face component")
    selected = components[0]
    selected_object = str(selected["object"])
    selected_faces = set(int(item) for item in selected["face_indices"])

    removed_faces = 0
    removed_objects = 0
    for obj in list(_mesh_objects()):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        keep = selected_faces if obj.name == selected_object else set()
        rejected = [face for face in bm.faces if face.index not in keep]
        removed_faces += len(rejected)
        if rejected:
            bmesh.ops.delete(bm, geom=rejected, context="FACES")
        unused = [vertex for vertex in bm.verts if not vertex.link_faces]
        if unused:
            bmesh.ops.delete(bm, geom=unused, context="VERTS")
        if weld_distance > 0 and bm.verts:
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=weld_distance)
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
            bm.to_mesh(obj.data)
            obj.data.update()
            bm.free()
        else:
            bm.free()
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            removed_objects += 1

    before = {
        **internal,
        "components": [
            {key: value for key, value in component.items() if key != "face_indices"}
            for component in components
        ],
    }
    after = analyze_scene()
    return {
        "operation": "keep_largest_component",
        "weld_distance": weld_distance,
        "removed_faces": removed_faces,
        "removed_objects": removed_objects,
        "selected_source_object": selected_object,
        "selected_source_component": int(selected["local_index"]),
        "before": before,
        "after": after,
    }


def _scene_bounds() -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in _mesh_objects() for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _remove_diagnostic_objects() -> None:
    scene = bpy.context.scene
    for obj in list(scene.objects):
        if obj.name.startswith("DarknessDiagnostic") or obj.name.startswith("DarknessKey") or obj.name.startswith("DarknessFill"):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                if isinstance(data, bpy.types.Camera):
                    bpy.data.cameras.remove(data)
                elif isinstance(data, bpy.types.Light):
                    bpy.data.lights.remove(data)


def render_diagnostics(output_root: Path, prefix: str, *, size: int = 512) -> list[Path]:
    scene = bpy.context.scene
    _remove_diagnostic_objects()
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("DarknessDiagnosticWorld")
    scene.world.color = (0.015, 0.018, 0.023)

    material = bpy.data.materials.get("DarknessDiagnosticMaterial") or bpy.data.materials.new("DarknessDiagnosticMaterial")
    material.diffuse_color = (0.20, 0.38, 0.16, 1.0)
    material.metallic = 0.0
    material.roughness = 0.82
    bpy.context.view_layer.material_override = material

    minimum, maximum = _scene_bounds()
    center = (minimum + maximum) * 0.5
    extents = maximum - minimum
    radius = max(extents) * 2.4

    camera_data = bpy.data.cameras.new("DarknessDiagnosticCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(extents.x, extents.z) * 1.24
    camera = bpy.data.objects.new("DarknessDiagnosticCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    for name, direction, energy, size_value in (
        ("DarknessKey", Vector((1.5, -2.0, 2.2)), 1100.0, 4.0),
        ("DarknessFill", Vector((-1.5, 1.0, 1.2)), 650.0, 3.0),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size_value
        light = bpy.data.objects.new(name, light_data)
        light.location = center + direction.normalized() * radius
        _look_at(light, center)
        scene.collection.objects.link(light)

    outputs: list[Path] = []
    for view_name, direction in (
        ("front", Vector((0, -1, 0))),
        ("right", Vector((1, 0, 0))),
        ("back", Vector((0, 1, 0))),
        ("left", Vector((-1, 0, 0))),
    ):
        camera.location = center + direction * radius
        _look_at(camera, center)
        path = output_root / f"{prefix}_{view_name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        outputs.append(path)
    return outputs


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_render_pixels(path: Path) -> tuple[tuple[int, int], array]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = array("f", [0.0]) * (len(image.pixels))
        image.pixels.foreach_get(pixels)
        return (int(image.size[0]), int(image.size[1])), pixels
    finally:
        bpy.data.images.remove(image)


def compare_renders(
    source_paths: list[Path],
    candidate_paths: list[Path],
    *,
    maximum_material_change_fraction: float,
) -> dict[str, object]:
    if len(source_paths) != len(candidate_paths):
        raise ValueError("source and candidate diagnostic sets must have equal length")
    views: dict[str, object] = {}
    material_fractions: list[float] = []
    for source_path, candidate_path in zip(source_paths, candidate_paths):
        source_size, source = _read_render_pixels(source_path)
        candidate_size, candidate = _read_render_pixels(candidate_path)
        if source_size != candidate_size or len(source) != len(candidate):
            raise ValueError("source and candidate diagnostic dimensions differ")
        pixels = source_size[0] * source_size[1]
        changed = 0
        material_changed = 0
        absolute_sum = 0.0
        for index in range(pixels):
            offset = index * 4
            difference = [abs(float(source[offset + channel]) - float(candidate[offset + channel])) for channel in range(3)]
            peak = max(difference)
            absolute_sum += sum(difference)
            if peak > (0.5 / 255.0):
                changed += 1
            if peak > (8.0 / 255.0):
                material_changed += 1
        name = source_path.stem.rsplit("_", 1)[-1]
        material_fraction = material_changed / max(pixels, 1)
        material_fractions.append(material_fraction)
        views[name] = {
            "linear_mean_absolute_error_255": absolute_sum * 255.0 / max(pixels * 3, 1),
            "changed_fraction": changed / max(pixels, 1),
            "material_change_fraction": material_fraction,
        }
    observed_maximum = max(material_fractions, default=0.0)
    return {
        "schema_version": 1,
        "views": views,
        "maximum_material_change_fraction": observed_maximum,
        "maximum_material_change_fraction_threshold": maximum_material_change_fraction,
        "automatic_visual_gate_passed": observed_maximum <= maximum_material_change_fraction,
        "human_approval_required": True,
        "approved": False,
    }


def _output(path: Path, media_type: str, role: str, **metadata: object) -> dict[str, object]:
    return {"path": str(path), "media_type": media_type, "role": role, "metadata": metadata}


def execute(request: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    operation_id = str(request.get("operation_id", ""))
    if operation_id not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported Blender operation: {operation_id}")
    input_paths = dict(request.get("input_paths") or {})
    if len(input_paths) != 1:
        raise ValueError("Blender worker requires exactly one input path")
    source_path = Path(next(iter(input_paths.values()))).resolve()
    output_root = Path(str(request["output_directory"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parameters = dict(request.get("parameters") or {})
    render_size = int(parameters.get("render_size", 512))
    if not 128 <= render_size <= 2048:
        raise ValueError("render_size must be between 128 and 2048")

    _import_asset(source_path)
    source_analysis = analyze_scene()
    source_analysis_path = output_root / "source_analysis.json"
    _write_json(source_analysis_path, source_analysis)
    source_checkpoint = output_root / "source_import.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(source_checkpoint))
    source_renders = render_diagnostics(output_root, "source", size=render_size)
    _remove_diagnostic_objects()

    outputs = [
        _output(source_analysis_path, "application/json", "source_analysis"),
        _output(source_checkpoint, "application/x-blender", "source_checkpoint"),
    ]
    outputs.extend(_output(path, "image/png", f"source_{path.stem.rsplit('_', 1)[-1]}") for path in source_renders)
    warnings = ["Fixed camera labels describe Blender world axes; human confirmation of semantic front/back is required."]

    if operation_id in {"blender.repair", "blender.export"}:
        component_policy = str(parameters.get("component_policy", "none"))
        weld_distance = float(parameters.get("weld_distance", 0.0))
        if not 0 <= weld_distance <= 0.01:
            raise ValueError("weld_distance must be between zero and 0.01")
        if component_policy == "keep_largest":
            repair = keep_largest_component(weld_distance=weld_distance)
        elif component_policy == "none":
            repair = {"operation": "none", "before": source_analysis, "after": source_analysis}
        else:
            raise ValueError("component_policy must be 'none' or 'keep_largest'")
        repair["human_approval_required"] = True
        repair["source_overwritten"] = False
        repair_path = output_root / "repair_report.json"
        candidate_checkpoint = output_root / "candidate.blend"
        _remove_diagnostic_objects()
        bpy.ops.wm.save_as_mainfile(filepath=str(candidate_checkpoint))
        candidate_glb = output_root / "candidate.glb"
        bpy.ops.export_scene.gltf(
            filepath=str(candidate_glb),
            export_format="GLB",
            export_normals=False,
            export_tangents=False,
            export_shared_accessors=True,
        )
        candidate_renders = render_diagnostics(output_root, "candidate", size=render_size)
        comparison_threshold = float(parameters.get("maximum_material_change_fraction", 0.02))
        if not 0 <= comparison_threshold <= 1:
            raise ValueError("maximum_material_change_fraction must be between zero and one")
        comparison = compare_renders(
            source_renders,
            candidate_renders,
            maximum_material_change_fraction=comparison_threshold,
        )
        comparison_path = output_root / "comparison_report.json"
        _write_json(comparison_path, comparison)
        repair["comparison"] = comparison
        _import_asset(candidate_glb)
        export_analysis = analyze_scene()
        export_hard_gate_passed = (
            bool(export_analysis["finite_coordinates"])
            and int(export_analysis["connected_components"]) == 1
            and int(export_analysis["boundary_edges"]) == 0
            and int(export_analysis["non_manifold_edges"]) == 0
            and int(export_analysis["inconsistent_winding_edges"]) == 0
        )
        export_validation = {
            "schema_version": 1,
            "hard_gate_passed": export_hard_gate_passed,
            "analysis": export_analysis,
        }
        export_validation_path = output_root / "export_validation.json"
        _write_json(export_validation_path, export_validation)
        repair["export_validation"] = export_validation
        _write_json(repair_path, repair)
        if not export_hard_gate_passed:
            raise RuntimeError("exported candidate failed Blender topology hard gates")
        if not bool(comparison["automatic_visual_gate_passed"]):
            raise RuntimeError("candidate exceeded the locked visual-change budget")
        outputs.extend(
            [
                _output(repair_path, "application/json", "repair_report"),
                _output(comparison_path, "application/json", "comparison_report"),
                _output(export_validation_path, "application/json", "export_validation"),
                _output(candidate_checkpoint, "application/x-blender", "candidate_checkpoint"),
                _output(candidate_glb, "model/gltf-binary", "candidate_geometry"),
            ]
        )
        outputs.extend(
            _output(path, "image/png", f"candidate_{path.stem.rsplit('_', 1)[-1]}")
            for path in candidate_renders
        )
        warnings.append("The candidate is staged research output and is never automatically promoted.")

    diagnostics = {
        "blender_version": bpy.app.version_string,
        "source_vertices": int(source_analysis["vertices"]),
        "source_faces": int(source_analysis["faces"]),
        "source_components": int(source_analysis["connected_components"]),
        "source_finite": bool(source_analysis["finite_coordinates"]),
    }
    return outputs, diagnostics, warnings


def main() -> int:
    args = _arguments()
    started = time.monotonic()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8-sig"))
        outputs, diagnostics, warnings = execute(request)
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 3)
        response = {
            "schema_version": 1,
            "job_id": request["job_id"],
            "status": "succeeded",
            "outputs": outputs,
            "diagnostics": diagnostics,
            "warnings": warnings,
            "error": None,
        }
    except Exception as exc:
        request = locals().get("request", {})
        response = {
            "schema_version": 1,
            "job_id": request.get("job_id", "unknown"),
            "status": "failed",
            "outputs": [],
            "diagnostics": {"elapsed_seconds": round(time.monotonic() - started, 3)},
            "warnings": [],
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if response["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
