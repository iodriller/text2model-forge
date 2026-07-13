"""Deterministic Blender background worker for Darkness mesh evidence and bounded cleanup."""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import bmesh
import bpy
from mathutils import Quaternion, Vector


SUPPORTED_OPERATIONS = {
    "blender.analyze",
    "blender.repair",
    "blender.repair_retopology",
    "blender.propose_short_biped_rig",
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


def _coordinate_digest(bm: bmesh.types.BMesh) -> str:
    digest = hashlib.sha256()
    for vertex in sorted(bm.verts, key=lambda item: item.index):
        digest.update(f"{vertex.co.x:.17g},{vertex.co.y:.17g},{vertex.co.z:.17g}\n".encode("ascii"))
    return digest.hexdigest()


def _duplicate_faces(bm: bmesh.types.BMesh) -> list[bmesh.types.BMFace]:
    seen: set[tuple[int, ...]] = set()
    duplicates: list[bmesh.types.BMFace] = []
    for face in bm.faces:
        key = tuple(sorted(vertex.index for vertex in face.verts))
        if key in seen:
            duplicates.append(face)
        else:
            seen.add(key)
    return duplicates


def _fill_simple_boundary_loops(
    bm: bmesh.types.BMesh,
    *,
    maximum_loop_sides: int,
) -> list[bmesh.types.BMFace]:
    boundary = {edge for edge in bm.edges if len(edge.link_faces) == 1}
    if not boundary:
        return []
    neighbors: dict[bmesh.types.BMVert, list[bmesh.types.BMVert]] = {}
    for edge in boundary:
        first, second = edge.verts
        neighbors.setdefault(first, []).append(second)
        neighbors.setdefault(second, []).append(first)
    if any(len(items) != 2 for items in neighbors.values()):
        return []
    created: list[bmesh.types.BMFace] = []
    while boundary:
        seed = min(boundary, key=lambda edge: tuple(sorted(vertex.index for vertex in edge.verts)))
        first, second = seed.verts
        loop = [first, second]
        boundary.remove(seed)
        previous, current = first, second
        while current != first:
            following = next(vertex for vertex in neighbors[current] if vertex != previous)
            edge = bm.edges.get((current, following))
            if edge is None:
                raise RuntimeError("boundary-loop edge disappeared during topology repair")
            boundary.discard(edge)
            previous, current = current, following
            if current != first:
                loop.append(current)
            if len(loop) > maximum_loop_sides:
                raise RuntimeError("boundary loop exceeds the bounded repair side limit")
        created.append(bm.faces.new(loop))
    return created


def repair_quad_dominant_manifold(
    *,
    minimum_quad_fraction: float,
    maximum_removed_faces: int,
    maximum_created_faces: int,
    maximum_boundary_loop_sides: int,
) -> dict[str, object]:
    objects = _mesh_objects()
    if len(objects) != 1:
        raise ValueError("quad-dominant topology repair requires exactly one mesh object")
    obj = objects[0]
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        coordinate_digest_before = _coordinate_digest(bm)
        vertices_before = len(bm.verts)
        faces_before = len(bm.faces)
        nonquads = [face for face in bm.faces if len(face.verts) != 4]
        bmesh.ops.triangulate(bm, faces=nonquads, quad_method="BEAUTY", ngon_method="BEAUTY")
        bmesh.ops.join_triangles(
            bm,
            faces=[face for face in bm.faces if len(face.verts) == 3],
            cmp_seam=False,
            cmp_sharp=False,
            cmp_uvs=False,
            cmp_vcols=False,
            cmp_materials=False,
            angle_face_threshold=math.pi,
            angle_shape_threshold=math.pi,
        )

        removed_faces = 0
        created_faces = 0
        iterations: list[dict[str, int]] = []
        for iteration in range(8):
            bm.faces.ensure_lookup_table()
            rejected = set(_duplicate_faces(bm))
            for edge in bm.edges:
                survivors = [face for face in edge.link_faces if face not in rejected]
                if len(survivors) > 2:
                    rejected.update(
                        sorted(survivors, key=lambda face: (face.calc_area(), face.index))[: len(survivors) - 2]
                    )
            if removed_faces + len(rejected) > maximum_removed_faces:
                raise RuntimeError("topology repair exceeds the bounded removed-face limit")
            if rejected:
                bmesh.ops.delete(bm, geom=list(rejected), context="FACES_ONLY")
                removed_faces += len(rejected)

            boundary_before = [edge for edge in bm.edges if len(edge.link_faces) == 1]
            filled = bmesh.ops.holes_fill(bm, edges=boundary_before, sides=0).get("faces", []) if boundary_before else []
            created_faces += len(filled)
            boundary_after_fill = [edge for edge in bm.edges if len(edge.link_faces) == 1]
            manual = (
                _fill_simple_boundary_loops(bm, maximum_loop_sides=maximum_boundary_loop_sides)
                if boundary_after_fill
                else []
            )
            created_faces += len(manual)
            if created_faces > maximum_created_faces:
                raise RuntimeError("topology repair exceeds the bounded created-face limit")
            boundary_after = sum(len(edge.link_faces) == 1 for edge in bm.edges)
            non_manifold_after = sum(len(edge.link_faces) > 2 for edge in bm.edges)
            iterations.append(
                {
                    "iteration": iteration,
                    "removed_faces": len(rejected),
                    "boundary_before": len(boundary_before),
                    "filled_faces": len(filled),
                    "manual_loop_faces": len(manual),
                    "boundary_after": boundary_after,
                    "non_manifold_after": non_manifold_after,
                }
            )
            if boundary_after == 0 and non_manifold_after == 0:
                break
            if not rejected and not filled and not manual:
                break

        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        coordinate_digest_after = _coordinate_digest(bm)
        quad_faces = sum(len(face.verts) == 4 for face in bm.faces)
        nonquad_faces = len(bm.faces) - quad_faces
        quad_fraction = quad_faces / max(len(bm.faces), 1)
        boundary_edges = sum(len(edge.link_faces) == 1 for edge in bm.edges)
        non_manifold_edges = sum(len(edge.link_faces) > 2 for edge in bm.edges)
        vertex_coordinates_unchanged = (
            len(bm.verts) == vertices_before and coordinate_digest_after == coordinate_digest_before
        )
        topology_gate_passed = (
            boundary_edges == 0
            and non_manifold_edges == 0
            and vertex_coordinates_unchanged
            and quad_fraction >= minimum_quad_fraction
        )
        result = {
            "operation": "quad_dominant_manifold_repair",
            "minimum_quad_fraction": minimum_quad_fraction,
            "maximum_removed_faces": maximum_removed_faces,
            "maximum_created_faces": maximum_created_faces,
            "maximum_boundary_loop_sides": maximum_boundary_loop_sides,
            "vertices_before": vertices_before,
            "faces_before": faces_before,
            "vertices_after": len(bm.verts),
            "faces_after": len(bm.faces),
            "quad_faces": quad_faces,
            "nonquad_faces": nonquad_faces,
            "quad_fraction": quad_fraction,
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold_edges,
            "removed_faces": removed_faces,
            "created_faces": created_faces,
            "vertex_coordinate_sha256_before": coordinate_digest_before,
            "vertex_coordinate_sha256_after": coordinate_digest_after,
            "vertex_coordinates_unchanged": vertex_coordinates_unchanged,
            "topology_gate_passed": topology_gate_passed,
            "iterations": iterations,
        }
        bm.to_mesh(obj.data)
        obj.data.update()
        return result
    finally:
        bm.free()


SHORT_BIPED_BONES = (
    ("hips", None, "hips", "spine"),
    ("spine", "hips", "spine", "chest"),
    ("chest", "spine", "chest", "neck"),
    ("neck", "chest", "neck", "head"),
    ("head", "neck", "head", "head_end"),
    ("shoulder_l", "chest", "shoulder_l", "elbow_l"),
    ("elbow_l", "shoulder_l", "elbow_l", "wrist_l"),
    ("wrist_l", "elbow_l", "wrist_l", "hand_l"),
    ("hand_l", "wrist_l", "hand_l", "hand_end_l"),
    ("shoulder_r", "chest", "shoulder_r", "elbow_r"),
    ("elbow_r", "shoulder_r", "elbow_r", "wrist_r"),
    ("wrist_r", "elbow_r", "wrist_r", "hand_r"),
    ("hand_r", "wrist_r", "hand_r", "hand_end_r"),
    ("hip_l", "hips", "hip_l", "knee_l"),
    ("knee_l", "hip_l", "knee_l", "ankle_l"),
    ("ankle_l", "knee_l", "ankle_l", "foot_l"),
    ("foot_l", "ankle_l", "foot_l", "toe_l"),
    ("hip_r", "hips", "hip_r", "knee_r"),
    ("knee_r", "hip_r", "knee_r", "ankle_r"),
    ("ankle_r", "knee_r", "ankle_r", "foot_r"),
    ("foot_r", "ankle_r", "foot_r", "toe_r"),
)


def _nearest_vertex(points: list[Vector], target: Vector) -> tuple[int, Vector, float]:
    index = min(range(len(points)), key=lambda item: (points[item] - target).length_squared)
    point = points[index]
    return index, point, (point - target).length


def infer_short_biped_landmarks() -> dict[str, object]:
    objects = _mesh_objects()
    if len(objects) != 1:
        raise ValueError("short-biped landmark inference requires exactly one mesh object")
    obj = objects[0]
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    extents = maximum - minimum
    upright_gate_passed = extents.z > extents.x * 1.2 and extents.z > extents.y * 2.0
    if not upright_gate_passed:
        raise RuntimeError("short-biped landmark proposal requires an upright Blender Z-up body")
    center = (minimum + maximum) * 0.5

    def point(x: float, depth: float, height: float) -> Vector:
        return Vector((center.x + x * extents.x, center.y + depth * extents.y, minimum.z + height * extents.z))

    landmarks = {
        "hips": point(0.0, 0.00, 0.40),
        "spine": point(0.0, 0.00, 0.50),
        "chest": point(0.0, 0.00, 0.63),
        "neck": point(0.0, 0.00, 0.76),
        "head": point(0.0, -0.03, 0.87),
        "head_end": point(0.0, -0.03, 0.97),
        "shoulder_l": point(-0.20, 0.00, 0.70),
        "elbow_l": point(-0.36, 0.00, 0.57),
        "wrist_l": point(-0.44, -0.01, 0.40),
        "hand_l": point(-0.47, -0.03, 0.32),
        "hand_end_l": point(-0.49, -0.05, 0.28),
        "shoulder_r": point(0.20, 0.00, 0.70),
        "elbow_r": point(0.36, 0.00, 0.57),
        "wrist_r": point(0.44, -0.01, 0.40),
        "hand_r": point(0.47, -0.03, 0.32),
        "hand_end_r": point(0.49, -0.05, 0.28),
        "hip_l": point(-0.13, 0.00, 0.39),
        "knee_l": point(-0.14, 0.00, 0.22),
        "ankle_l": point(-0.15, 0.00, 0.075),
        "foot_l": point(-0.15, -0.04, 0.055),
        "toe_l": point(-0.15, -0.43, 0.035),
        "hip_r": point(0.13, 0.00, 0.39),
        "knee_r": point(0.14, 0.00, 0.22),
        "ankle_r": point(0.15, 0.00, 0.075),
        "foot_r": point(0.15, -0.04, 0.055),
        "toe_r": point(0.15, -0.43, 0.035),
    }
    surface_targets = {
        "crown": point(0.0, -0.02, 1.0),
        "palm_l": point(-0.50, -0.20, 0.31),
        "palm_r": point(0.50, -0.20, 0.31),
        "sole_l": point(-0.15, 0.0, 0.0),
        "sole_r": point(0.15, 0.0, 0.0),
        "toe_surface_l": point(-0.15, -0.50, 0.04),
        "toe_surface_r": point(0.15, -0.50, 0.04),
    }
    surface_landmarks: dict[str, object] = {}
    for name, target in surface_targets.items():
        index, found, distance = _nearest_vertex(points, target)
        surface_landmarks[name] = {
            "vertex_index": index,
            "world_position": list(found),
            "proposal_distance": distance,
        }
    symmetry_errors = []
    for left_name, right_name in (
        ("shoulder_l", "shoulder_r"),
        ("elbow_l", "elbow_r"),
        ("wrist_l", "wrist_r"),
        ("hand_l", "hand_r"),
        ("hip_l", "hip_r"),
        ("knee_l", "knee_r"),
        ("ankle_l", "ankle_r"),
        ("foot_l", "foot_r"),
    ):
        left = landmarks[left_name]
        right = landmarks[right_name]
        symmetry_errors.append(abs((left.x - center.x) + (right.x - center.x)))
        symmetry_errors.append(abs(left.y - right.y))
        symmetry_errors.append(abs(left.z - right.z))
    maximum_symmetry_error = max(symmetry_errors, default=0.0)
    landmark_gate_passed = upright_gate_passed and maximum_symmetry_error <= max(extents) * 1e-6
    return {
        "schema_version": 1,
        "method": "bounded_short_biped_proportions_v1",
        "coordinate_system": "Blender Z-up; X left/right; -Y semantic front",
        "bounds": {"minimum": list(minimum), "maximum": list(maximum), "extents": list(extents)},
        "landmarks": {name: list(value) for name, value in landmarks.items()},
        "surface_landmarks": surface_landmarks,
        "confidence": {
            "anatomy_family": 0.8,
            "bilateral_symmetry": 1.0,
            "internal_joint_centers": 0.55,
            "functional_contacts": 0.6,
        },
        "upright_gate_passed": upright_gate_passed,
        "maximum_bilateral_symmetry_error": maximum_symmetry_error,
        "landmark_gate_passed": landmark_gate_passed,
        "human_approval_required": True,
        "human_approved": False,
    }


def _mesh_coordinate_digest(obj: bpy.types.Object) -> str:
    digest = hashlib.sha256()
    for vertex in obj.data.vertices:
        digest.update(f"{vertex.co.x:.17g},{vertex.co.y:.17g},{vertex.co.z:.17g}\n".encode("ascii"))
    return digest.hexdigest()


def _mesh_topology_digest(obj: bpy.types.Object) -> str:
    digest = hashlib.sha256()
    for polygon in obj.data.polygons:
        digest.update((",".join(str(index) for index in polygon.vertices) + "\n").encode("ascii"))
    return digest.hexdigest()


def _create_short_biped_armature(landmark_report: dict[str, object]) -> bpy.types.Object:
    landmarks = {name: Vector(value) for name, value in dict(landmark_report["landmarks"]).items()}
    armature_data = bpy.data.armatures.new("DarknessShortBipedRig")
    armature = bpy.data.objects.new("DarknessShortBipedRig", armature_data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bones: dict[str, bpy.types.EditBone] = {}
    for name, parent_name, head_name, tail_name in SHORT_BIPED_BONES:
        bone = armature_data.edit_bones.new(name)
        bone.head = landmarks[head_name]
        bone.tail = landmarks[tail_name]
        bone.use_deform = True
        if parent_name is not None:
            bone.parent = bones[parent_name]
            bone.use_connect = (bone.head - bone.parent.tail).length <= 1e-6
        bones[name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.select_set(False)
    return armature


def _bind_short_biped_weights_bone_heat(
    obj: bpy.types.Object,
    armature: bpy.types.Object,
) -> dict[str, object]:
    for group in list(obj.vertex_groups):
        obj.vertex_groups.remove(group)
    for modifier in list(obj.modifiers):
        if modifier.type == "ARMATURE":
            obj.modifiers.remove(modifier)
    obj.parent = None
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    result = bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    if "FINISHED" not in result:
        raise RuntimeError("Blender bone-heat weighting did not finish")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.vertex_group_clean(group_select_mode="ALL", limit=1e-6, keep_single=True)
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    bpy.context.view_layer.update()

    counts = [len(vertex.groups) for vertex in obj.data.vertices]
    sums = [sum(element.weight for element in vertex.groups) for vertex in obj.data.vertices]
    weights = [element.weight for vertex in obj.data.vertices for element in vertex.groups]
    hard_failures: list[str] = []
    if not counts or min(counts) == 0:
        hard_failures.append("unweighted_vertices")
    if counts and max(counts) > 4:
        hard_failures.append("too_many_influences")
    maximum_sum_error = max((abs(value - 1.0) for value in sums), default=1.0)
    if maximum_sum_error > 1e-5:
        hard_failures.append("weights_not_normalized")
    deform_bone_names = {bone.name for bone in armature.data.bones if bone.use_deform}
    missing_groups = sorted(deform_bone_names - {group.name for group in obj.vertex_groups})
    if missing_groups:
        hard_failures.append("missing_deform_bone_groups")
    return {
        "schema_version": 1,
        "method": "blender_bone_heat_pruned_to_four_v1",
        "vertices": len(obj.data.vertices),
        "bones": len(deform_bone_names),
        "minimum_influences": min(counts, default=0),
        "maximum_influences": max(counts, default=0),
        "maximum_weight_sum_error": maximum_sum_error,
        "minimum_assigned_weight": min(weights, default=0.0),
        "missing_deform_bone_groups": missing_groups,
        "hard_failures": hard_failures,
        "gate_passed": not hard_failures,
    }


def _clear_pose(armature: bpy.types.Object) -> None:
    for bone in armature.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        bone.location = Vector((0.0, 0.0, 0.0))
        bone.scale = Vector((1.0, 1.0, 1.0))
    bpy.context.view_layer.update()


def _rotate_bone_world_axis(armature: bpy.types.Object, name: str, axis: Vector, degrees: float) -> None:
    pose_bone = armature.pose.bones[name]
    local_axis = pose_bone.bone.matrix_local.to_3x3().inverted() @ axis.normalized()
    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = Quaternion(local_axis.normalized(), math.radians(degrees))


def _evaluated_pose_report(
    obj: bpy.types.Object,
    *,
    pose_name: str,
    rest_positions: list[Vector],
    rest_areas: list[float],
) -> dict[str, object]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_object.to_mesh()
    try:
        matrix = evaluated_object.matrix_world
        positions = [matrix @ vertex.co for vertex in evaluated_mesh.vertices]
        finite = all(math.isfinite(value) for point in positions for value in point)
        areas = [polygon.area for polygon in evaluated_mesh.polygons]
        collapsed_faces = sum(
            area <= 1e-12 or area <= rest_area * 1e-4
            for area, rest_area in zip(areas, rest_areas)
        )
        severely_compressed_faces = sum(
            area <= rest_area * 0.05
            for area, rest_area in zip(areas, rest_areas)
            if rest_area > 1e-12
        )
        maximum_displacement = max(
            ((position - rest).length for position, rest in zip(positions, rest_positions)),
            default=0.0,
        )
        minimum = Vector(tuple(min(point[axis] for point in positions) for axis in range(3)))
        maximum = Vector(tuple(max(point[axis] for point in positions) for axis in range(3)))
        extents = maximum - minimum
        hard_failures: list[str] = []
        if len(positions) != len(rest_positions) or len(areas) != len(rest_areas):
            hard_failures.append("topology_changed")
        if not finite:
            hard_failures.append("non_finite_coordinates")
        if collapsed_faces:
            hard_failures.append("collapsed_faces")
        return {
            "pose": pose_name,
            "vertices": len(positions),
            "faces": len(areas),
            "finite_coordinates": finite,
            "collapsed_faces": collapsed_faces,
            "severely_compressed_faces": severely_compressed_faces,
            "maximum_vertex_displacement": maximum_displacement,
            "bounds_extents": list(extents),
            "hard_failures": hard_failures,
            "gate_passed": not hard_failures,
        }
    finally:
        evaluated_object.to_mesh_clear()


def _run_short_biped_rig_probe(
    output_root: Path,
    source_renders: list[Path],
    *,
    render_size: int,
    maximum_material_change_fraction: float,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    obj = _mesh_objects()[0]
    coordinate_digest_before = _mesh_coordinate_digest(obj)
    topology_digest_before = _mesh_topology_digest(obj)
    landmark_report = infer_short_biped_landmarks()
    if not bool(landmark_report["landmark_gate_passed"]):
        raise RuntimeError("short-biped landmark proposal failed its automatic gate")
    armature = _create_short_biped_armature(landmark_report)
    skinning_report = _bind_short_biped_weights_bone_heat(obj, armature)
    coordinate_digest_after = _mesh_coordinate_digest(obj)
    topology_digest_after = _mesh_topology_digest(obj)
    skinning_report["vertex_coordinates_unchanged"] = coordinate_digest_before == coordinate_digest_after
    skinning_report["topology_unchanged"] = topology_digest_before == topology_digest_after
    if not skinning_report["vertex_coordinates_unchanged"]:
        skinning_report["hard_failures"].append("rest_vertex_coordinates_changed")
    if not skinning_report["topology_unchanged"]:
        skinning_report["hard_failures"].append("rest_topology_changed")
    skinning_report["gate_passed"] = not skinning_report["hard_failures"]
    if not bool(skinning_report["gate_passed"]):
        raise RuntimeError("short-biped skinning proposal failed its automatic gate")

    _clear_pose(armature)
    neutral_renders = render_diagnostics(output_root, "rig_neutral", size=render_size)
    neutral_comparison = compare_renders(
        source_renders,
        neutral_renders,
        maximum_material_change_fraction=maximum_material_change_fraction,
    )
    if not bool(neutral_comparison["automatic_visual_gate_passed"]):
        raise RuntimeError("neutral rig changed the locked rest-pose appearance")

    rest_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    rest_areas = [polygon.area for polygon in obj.data.polygons]
    poses: list[dict[str, object]] = []

    _clear_pose(armature)
    _rotate_bone_world_axis(armature, "shoulder_l", Vector((0, 1, 0)), 50.0)
    _rotate_bone_world_axis(armature, "shoulder_r", Vector((0, 1, 0)), -50.0)
    _rotate_bone_world_axis(armature, "elbow_l", Vector((1, 0, 0)), -55.0)
    _rotate_bone_world_axis(armature, "elbow_r", Vector((1, 0, 0)), -55.0)
    bpy.context.view_layer.update()
    poses.append(_evaluated_pose_report(obj, pose_name="shoulder_elbow_stress", rest_positions=rest_positions, rest_areas=rest_areas))
    shoulder_renders = render_diagnostics(output_root, "rig_shoulder_stress", size=render_size)

    _clear_pose(armature)
    _rotate_bone_world_axis(armature, "hip_l", Vector((1, 0, 0)), -22.0)
    _rotate_bone_world_axis(armature, "hip_r", Vector((1, 0, 0)), 22.0)
    _rotate_bone_world_axis(armature, "knee_l", Vector((1, 0, 0)), 58.0)
    bpy.context.view_layer.update()
    poses.append(_evaluated_pose_report(obj, pose_name="hip_knee_stress", rest_positions=rest_positions, rest_areas=rest_areas))
    leg_renders = render_diagnostics(output_root, "rig_leg_stress", size=render_size)
    _clear_pose(armature)
    _remove_diagnostic_objects()

    deformation_report = {
        "schema_version": 1,
        "poses": poses,
        "hard_failures": [failure for pose in poses for failure in pose["hard_failures"]],
        "gate_passed": all(bool(pose["gate_passed"]) for pose in poses),
        "human_approval_required": True,
        "human_approved": False,
    }
    rig_contract = {
        "schema_version": 1,
        "anatomy_family": "short_biped_v1",
        "proposal_method": "bounded_geometric_landmarks_and_blender_bone_heat_v1",
        "joints": [
            {
                "name": name,
                "parent": parent,
                "head_landmark": head,
                "tail_landmark": tail,
            }
            for name, parent, head, tail in SHORT_BIPED_BONES
        ],
        "contacts": {"foot_left": "toe_l", "foot_right": "toe_r"},
        "sockets": {"hand_left": "hand_l", "hand_right": "hand_r", "projectile_origin": "chest"},
        "human_approval_required": True,
        "human_approved": False,
    }
    landmark_path = output_root / "landmarks_contract.json"
    rig_path = output_root / "rig_contract.json"
    skinning_path = output_root / "skinning_report.json"
    deformation_path = output_root / "deformation_report.json"
    comparison_path = output_root / "neutral_comparison_report.json"
    _write_json(landmark_path, landmark_report)
    _write_json(rig_path, rig_contract)
    _write_json(skinning_path, skinning_report)
    _write_json(deformation_path, deformation_report)
    _write_json(comparison_path, neutral_comparison)

    rigged_blend = output_root / "rigged_candidate.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(rigged_blend))
    _triangulate_scene_for_export()
    rigged_glb = output_root / "rigged_candidate.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(rigged_glb),
        export_format="GLB",
        export_materials="NONE",
        export_normals=False,
        export_tangents=False,
        export_skins=True,
        export_animations=False,
        export_shared_accessors=True,
    )
    _import_asset(rigged_glb)
    export_scene_analysis = analyze_scene()
    export_primary = max(_mesh_objects(), key=lambda item: len(item.data.vertices))
    export_primary_analysis, _ = _object_analysis(export_primary)
    export_armatures = sum(item.type == "ARMATURE" for item in bpy.context.scene.objects)
    export_skin_modifiers = sum(
        modifier.type == "ARMATURE"
        for mesh in _mesh_objects()
        for modifier in mesh.modifiers
    )
    export_gate_passed = (
        export_armatures == 1
        and export_skin_modifiers == 1
        and bool(export_primary_analysis["finite_coordinates"])
        and int(export_primary_analysis["components"]) == 1
        and int(export_primary_analysis["boundary_edges"]) == 0
        and int(export_primary_analysis["non_manifold_edges"]) == 0
    )
    export_report = {
        "schema_version": 1,
        "analysis": export_primary_analysis,
        "scene_mesh_objects": int(export_scene_analysis["mesh_objects"]),
        "auxiliary_mesh_objects": [item.name for item in _mesh_objects() if item != export_primary],
        "armatures": export_armatures,
        "skin_modifiers": export_skin_modifiers,
        "gate_passed": export_gate_passed,
    }
    export_path = output_root / "rigged_export_validation.json"
    _write_json(export_path, export_report)
    automatic_gate_passed = (
        bool(landmark_report["landmark_gate_passed"])
        and bool(skinning_report["gate_passed"])
        and bool(deformation_report["gate_passed"])
        and bool(neutral_comparison["automatic_visual_gate_passed"])
        and export_gate_passed
    )
    if not automatic_gate_passed:
        raise RuntimeError("short-biped rig/deformation probe failed an automatic gate")

    outputs = [
        _output(landmark_path, "application/json", "landmarks_contract"),
        _output(rig_path, "application/json", "rig_contract"),
        _output(skinning_path, "application/json", "skinning_report"),
        _output(deformation_path, "application/json", "deformation_report"),
        _output(comparison_path, "application/json", "neutral_comparison_report"),
        _output(export_path, "application/json", "rigged_export_validation"),
        _output(rigged_blend, "application/x-blender", "rigged_candidate_checkpoint"),
        _output(rigged_glb, "model/gltf-binary", "rigged_candidate"),
    ]
    for prefix, paths in (
        ("rig_neutral", neutral_renders),
        ("rig_shoulder_stress", shoulder_renders),
        ("rig_leg_stress", leg_renders),
    ):
        outputs.extend(
            _output(path, "image/png", f"{prefix}_{path.stem.rsplit('_', 1)[-1]}")
            for path in paths
        )
    diagnostics = {
        "landmarks": len(landmark_report["landmarks"]),
        "bones": len(SHORT_BIPED_BONES),
        "automatic_rig_probe_gate_passed": automatic_gate_passed,
        "human_approved": False,
    }
    warnings = [
        "This is a staged deformation probe on generated topology, not an approved canonical production mesh.",
        "Proportional joint centers and pruned Blender bone-heat weights require visual review and corrective refinement.",
        "Self-collision, twist distribution, facial deformation, clothing rigidity, and motion contacts are not yet gated.",
    ]
    return outputs, diagnostics, warnings


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


def _triangulate_scene_for_export() -> None:
    for obj in _mesh_objects():
        source_mesh = obj.data
        vertices = [tuple(vertex.co) for vertex in source_mesh.vertices]
        vertex_weights = [
            [(element.group, element.weight) for element in vertex.groups]
            for vertex in source_mesh.vertices
        ]
        group_names = {group.index: group.name for group in obj.vertex_groups}
        triangles: list[tuple[int, int, int]] = []
        for polygon in source_mesh.polygons:
            indices = list(polygon.vertices)
            if len(indices) == 3:
                triangles.append(tuple(indices))
                continue
            center = sum((Vector(vertices[index]) for index in indices), Vector()) / len(indices)
            center_index = len(vertices)
            vertices.append(tuple(center))
            accumulated: dict[int, float] = {}
            for index in indices:
                for group_index, weight in vertex_weights[index]:
                    accumulated[group_index] = accumulated.get(group_index, 0.0) + weight / len(indices)
            selected = sorted(accumulated.items(), key=lambda item: (-item[1], item[0]))[:4]
            weight_total = sum(weight for _, weight in selected)
            vertex_weights.append(
                [(group_index, weight / weight_total) for group_index, weight in selected]
                if weight_total > 0
                else []
            )
            triangles.extend(
                (index, indices[(offset + 1) % len(indices)], center_index)
                for offset, index in enumerate(indices)
            )
        export_mesh = bpy.data.meshes.new(f"{source_mesh.name}_ExportTriangles")
        export_mesh.from_pydata(vertices, [], triangles)
        export_mesh.update()
        obj.data = export_mesh
        recreated_groups = {
            old_index: obj.vertex_groups.new(name=name)
            for old_index, name in sorted(group_names.items())
        }
        for vertex_index, assignments in enumerate(vertex_weights):
            for group_index, weight in assignments:
                recreated_groups[group_index].add([vertex_index], weight, "REPLACE")
        if source_mesh.users == 0:
            bpy.data.meshes.remove(source_mesh)


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

    if operation_id == "blender.propose_short_biped_rig":
        comparison_threshold = float(parameters.get("maximum_material_change_fraction", 0.02))
        if not 0 <= comparison_threshold <= 1:
            raise ValueError("maximum_material_change_fraction must be between zero and one")
        probe_outputs, probe_diagnostics, probe_warnings = _run_short_biped_rig_probe(
            output_root,
            source_renders,
            render_size=render_size,
            maximum_material_change_fraction=comparison_threshold,
        )
        outputs.extend(probe_outputs)
        warnings.extend(probe_warnings)
    elif operation_id in {"blender.repair", "blender.repair_retopology", "blender.export"}:
        component_policy = str(parameters.get("component_policy", "none"))
        weld_distance = float(parameters.get("weld_distance", 0.0))
        if not 0 <= weld_distance <= 0.01:
            raise ValueError("weld_distance must be between zero and 0.01")
        if operation_id == "blender.repair_retopology":
            minimum_quad_fraction = float(parameters.get("minimum_quad_fraction", 0.99))
            maximum_removed_faces = int(parameters.get("maximum_removed_faces", 16))
            maximum_created_faces = int(parameters.get("maximum_created_faces", 16))
            maximum_boundary_loop_sides = int(parameters.get("maximum_boundary_loop_sides", 64))
            if not 0.95 <= minimum_quad_fraction <= 1.0:
                raise ValueError("minimum_quad_fraction must be between 0.95 and 1.0")
            if not 0 <= maximum_removed_faces <= 128:
                raise ValueError("maximum_removed_faces must be between zero and 128")
            if not 0 <= maximum_created_faces <= 128:
                raise ValueError("maximum_created_faces must be between zero and 128")
            if not 3 <= maximum_boundary_loop_sides <= 256:
                raise ValueError("maximum_boundary_loop_sides must be between 3 and 256")
            repair = repair_quad_dominant_manifold(
                minimum_quad_fraction=minimum_quad_fraction,
                maximum_removed_faces=maximum_removed_faces,
                maximum_created_faces=maximum_created_faces,
                maximum_boundary_loop_sides=maximum_boundary_loop_sides,
            )
        elif component_policy == "keep_largest":
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
        _triangulate_scene_for_export()
        pre_export_analysis = analyze_scene()
        candidate_glb = output_root / "candidate.glb"
        bpy.ops.export_scene.gltf(
            filepath=str(candidate_glb),
            export_format="GLB",
            export_materials="NONE",
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
            "pre_export_analysis": pre_export_analysis,
            "analysis": export_analysis,
        }
        export_validation_path = output_root / "export_validation.json"
        _write_json(export_validation_path, export_validation)
        repair["export_validation"] = export_validation
        repair["automatic_promotion_gate_passed"] = (
            export_hard_gate_passed
            and bool(comparison["automatic_visual_gate_passed"])
            and bool(repair.get("topology_gate_passed", True))
        )
        _write_json(repair_path, repair)
        if not export_hard_gate_passed:
            raise RuntimeError("exported candidate failed Blender topology hard gates")
        if not bool(comparison["automatic_visual_gate_passed"]):
            raise RuntimeError("candidate exceeded the locked visual-change budget")
        if not bool(repair.get("topology_gate_passed", True)):
            raise RuntimeError("candidate failed the bounded quad-dominant topology gate")
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
    if operation_id == "blender.propose_short_biped_rig":
        diagnostics.update(probe_diagnostics)
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
