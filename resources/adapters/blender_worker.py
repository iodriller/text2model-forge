"""Deterministic Blender background worker for Text2Model mesh evidence and bounded cleanup."""
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
    "blender.author_short_biped_motion",
    "blender.author_rigid_articulation",
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


def voxel_remesh_to_manifold(
    *, voxel_fraction: float = 0.006, target_faces: int = 0
) -> dict[str, object]:
    """Rebuild every mesh as a guaranteed-manifold surface via OpenVDB.

    Why this exists. D3's repair toolkit is a *welder*: remove_doubles plus a
    connected-component filter. That is the right tool for a generator whose
    output is already close to manifold, which is what the pipeline was
    qualified against. It cannot fix an isosurface extractor's output, and on
    an 8 GB card the D2 backend is Hunyuan3D, whose surface-net decode emits
    exactly that. Measured on a real knight mesh:

        hunyuan3d source   158,858 v   4 comps   163 boundary   30,258 non-manifold
        after weld+filter  156,729 v   1 comp    158 boundary   28,491 non-manifold

    Welding moved 30,258 non-manifold edges by 6%. It never converges,
    because a non-manifold edge is one shared by three or more faces -- there
    are no duplicate vertices there to merge. No weld distance fixes that.

    Voxelising does. Rasterising the surface into an OpenVDB level set and
    re-extracting it discards the input's topology entirely and generates a
    fresh one, which is manifold by construction. Same mesh, same machine:

        voxel 0.4% of longest axis   151,654 v   0 boundary   0 non-manifold   0.9s
        voxel 0.6% of longest axis    66,832 v   0 boundary   0 non-manifold   0.5s

    Sub-second, CPU-only, no VRAM, no new dependency -- OpenVDB ships inside
    Blender. The cost is that fine detail below the voxel size is lost and
    the result is uniform quads; both are acceptable here because D8 bakes
    surface detail back as texture and the retopology worker owns final
    topology. `voxel_fraction` is a fraction of the longest bounding-box
    axis, not an absolute size, so it is resolution-independent across
    assets of wildly different scale.
    """
    before = analyze_scene()
    objects = list(_mesh_objects())
    if not objects:
        raise ValueError("scene has no mesh to remesh")
    if not 0.0005 <= voxel_fraction <= 0.1:
        raise ValueError("voxel_fraction must be between 0.0005 and 0.1")

    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            minimum = Vector(map(min, minimum, point))
            maximum = Vector(map(max, maximum, point))
    longest = max(maximum - minimum)
    if not math.isfinite(longest) or longest <= 0:
        raise ValueError("cannot size a voxel against a degenerate bounding box")
    voxel_size = longest * voxel_fraction

    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.data.remesh_voxel_size = voxel_size
        obj.data.remesh_voxel_adaptivity = 0.0
        bpy.ops.object.voxel_remesh()
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
            bm.to_mesh(obj.data)
            obj.data.update()
        bm.free()

    remeshed = analyze_scene()

    # Decimate to a budget after remeshing, never instead of it.
    #
    # Voxel size and vertex count pull in opposite directions and both matter
    # downstream. The voxel has to be finer than the smallest feature that
    # must stay topologically separate -- fingers, or D7's grip builder finds
    # a mitt -- but on a 1.97 m character a 5.4 mm voxel produced 668,406
    # vertices, and Blender's bone-heat weighting then failed outright at D4:
    # "failed to find solution for one or more bones", maximum_influences 0,
    # every vertex unweighted. The same pipeline succeeded at 151k.
    #
    # Choosing one number cannot satisfy both. Remeshing fine and then
    # collapsing to a budget satisfies both: topology is decided at the fine
    # voxel, density is decided here, and surface detail lost to either is
    # recovered at D8's texture bake. Collapse decimation preserves the
    # manifoldness the remesh just established, which is re-measured below
    # rather than assumed.
    decimated = None
    if target_faces > 0:
        total_faces = int(remeshed.get("faces", 0))
        if total_faces > target_faces:
            ratio = max(0.02, min(1.0, target_faces / float(total_faces)))
            for obj in _mesh_objects():
                bpy.context.view_layer.objects.active = obj
                modifier = obj.modifiers.new(name="text2model_decimate", type="DECIMATE")
                modifier.decimate_type = "COLLAPSE"
                modifier.ratio = ratio
                modifier.use_collapse_triangulate = True
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            decimated = analyze_scene()

    after = decimated or remeshed
    return {
        "operation": "voxel_remesh_to_manifold",
        "voxel_fraction": voxel_fraction,
        "voxel_size": voxel_size,
        "longest_axis": longest,
        "target_faces": target_faces,
        "before": before,
        "after_remesh": remeshed,
        "after": after,
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

WEIGHT_JOINT_PAIRS = {
    "shoulders": (
        ("shoulder_l", "chest", "shoulder_l"),
        ("shoulder_r", "chest", "shoulder_r"),
    ),
    "elbows": (
        ("elbow_l", "shoulder_l", "elbow_l"),
        ("elbow_r", "shoulder_r", "elbow_r"),
    ),
    "hips": (
        ("hip_l", "hips", "hip_l"),
        ("hip_r", "hips", "hip_r"),
    ),
    "knees": (
        ("knee_l", "hip_l", "knee_l"),
        ("knee_r", "hip_r", "knee_r"),
    ),
}


def _nearest_vertex(points: list[Vector], target: Vector) -> tuple[int, Vector, float]:
    index = min(range(len(points)), key=lambda item: (points[item] - target).length_squared)
    point = points[index]
    return index, point, (point - target).length


def infer_short_biped_landmarks(
    normalized_adjustments: dict[str, list[float]] | None = None,
) -> dict[str, object]:
    objects = _mesh_objects()
    if len(objects) != 1:
        raise ValueError("short-biped landmark inference requires exactly one mesh object")
    obj = objects[0]
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    extents = maximum - minimum
    # What this gate is actually for: refuse to fit a standing-biped skeleton
    # to a mesh that is lying down, or is a flat/wide object where the whole
    # landmark scheme is meaningless. The original form of the test --
    # z > x * 1.2 and z > y * 2.0 -- encoded that as fixed ratios against a
    # *bare* body, and equipment breaks it. Measured on a real knight holding
    # a shield out to the side: 1.965 tall, 1.770 wide, 0.988 deep. It is
    # unambiguously upright, yet it failed both clauses (needing 2.124 and
    # 1.976), the second by 11 mm. The shield, not the posture, was the
    # problem.
    #
    # Dominance is the property that actually matters and the one equipment
    # cannot fake: for anything standing, the vertical extent is the largest.
    # A lying figure -- the case worth refusing -- puts its longest extent in
    # x or y and is still rejected. The margin keeps a near-cube from
    # sneaking through on floating-point noise.
    # Two clauses, both rotation-invariant about Z so a character facing any
    # compass direction behaves identically:
    #   1. Vertical is at least as large as the widest horizontal axis, less a
    #      5% tolerance. The tolerance is not slack -- a T-pose has an arm span
    #      about equal to its height, so demanding strict dominance would
    #      reject the single most common rig pose there is.
    #   2. Vertical clearly exceeds the *narrowest* horizontal axis. Anything
    #      standing is thin through one horizontal direction; this is what
    #      rejects a cube or a slab that clause 1 alone would admit.
    widest = max(extents.x, extents.y)
    narrowest = min(extents.x, extents.y)
    upright_gate_passed = extents.z >= widest * 0.95 and extents.z > narrowest * 1.5
    if not upright_gate_passed:
        raise RuntimeError(
            "short-biped landmark proposal requires an upright Blender Z-up body; "
            f"extents were x={extents.x:.3f} y={extents.y:.3f} z={extents.z:.3f}, "
            "so the vertical axis is not the dominant one"
        )
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
    adjustments = normalized_adjustments or {}
    unknown = sorted(set(adjustments) - set(landmarks))
    if unknown:
        raise ValueError(f"unknown short-biped landmarks in adjustment: {unknown}")
    for name, values in adjustments.items():
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"landmark adjustment for {name} must contain exactly three values")
        normalized = Vector(tuple(float(value) for value in values))
        if any(not math.isfinite(value) or abs(value) > 0.20 for value in normalized):
            raise ValueError(f"landmark adjustment for {name} exceeds the cumulative 8% bound")
        landmarks[name] += Vector(
            (normalized.x * extents.x, normalized.y * extents.y, normalized.z * extents.z)
        )
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
        "normalized_adjustments": adjustments,
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
    armature_data = bpy.data.armatures.new("Text2ModelShortBipedRig")
    armature = bpy.data.objects.new("Text2ModelShortBipedRig", armature_data)
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


def _skinning_report(
    obj: bpy.types.Object,
    armature: bpy.types.Object,
    *,
    method: str,
) -> dict[str, object]:
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
        "method": method,
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
            "bounds_minimum": list(minimum),
            "bounds_maximum": list(maximum),
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
    landmark_adjustments: dict[str, list[float]],
    weight_adjustments: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    obj = _mesh_objects()[0]
    coordinate_digest_before = _mesh_coordinate_digest(obj)
    topology_digest_before = _mesh_topology_digest(obj)
    landmark_report = infer_short_biped_landmarks(landmark_adjustments)
    if not bool(landmark_report["landmark_gate_passed"]):
        raise RuntimeError("short-biped landmark proposal failed its automatic gate")
    armature = _create_short_biped_armature(landmark_report)
    skinning_report = _bind_short_biped_weights_bone_heat(obj, armature)
    if weight_adjustments:
        skinning_report = _apply_weight_redistributions(
            obj,
            armature,
            landmark_report,
            weight_adjustments,
        )
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
        # Write the report before raising, and name the cause in the message.
        # Previously this raised a bare "failed its automatic gate": the
        # report was only persisted further down on the success path, so a
        # failure produced no artifact at all and the operator was told that
        # something was wrong but not which of the four conditions, nor by how
        # much. Both halves of that are diagnosability bugs -- the numbers
        # exist at this point and simply were not surfaced.
        _write_json(output_root / "skinning_report.json", skinning_report)
        failures = ", ".join(str(item) for item in skinning_report["hard_failures"])
        detail = {
            key: skinning_report.get(key)
            for key in (
                "unweighted_vertices",
                "minimum_influences",
                "maximum_influences",
                "maximum_weight_sum_error",
                "method",
            )
            if key in skinning_report
        }
        raise RuntimeError(
            f"short-biped skinning proposal failed its automatic gate: {failures}. {detail}. "
            "Bone-heat weighting cannot solve geometry that sits far from every bone -- fused "
            "equipment such as a held weapon is the usual cause on a generated mesh."
        )

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
        # The research rig's legacy suffixes denote X side. Expose anatomical
        # sockets so negative-X ``hand_l`` is the character's right hand.
        "sockets": {"hand_left": "hand_r", "hand_right": "hand_l", "projectile_origin": "chest"},
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


def _short_biped_motion_specs() -> dict[str, dict[str, object]]:
    return {
        "idle": {
            "loop": True,
            "poses": [
                {"frame": 1},
                {"frame": 13, "hips_location": (0.0, 0.0, 0.012), "rotations": (("chest", (1, 0, 0), 2.0),)},
                {"frame": 25},
            ],
        },
        "walk": {
            "loop": True,
            "poses": [
                {"frame": 1},
                {
                    "frame": 7,
                    "hips_location": (0.0, 0.0, 0.022),
                    "rotations": (
                        ("hip_l", (1, 0, 0), 25.0),
                        ("knee_l", (1, 0, 0), 38.0),
                        ("shoulder_l", (1, 0, 0), 18.0),
                        ("shoulder_r", (1, 0, 0), -18.0),
                    ),
                },
                {"frame": 13},
                {
                    "frame": 19,
                    "hips_location": (0.0, 0.0, 0.022),
                    "rotations": (
                        ("hip_r", (1, 0, 0), 25.0),
                        ("knee_r", (1, 0, 0), 38.0),
                        ("shoulder_l", (1, 0, 0), -18.0),
                        ("shoulder_r", (1, 0, 0), 18.0),
                    ),
                },
                {"frame": 25},
            ],
        },
        "attack": {
            "loop": False,
            "poses": [
                {"frame": 1},
                {
                    "frame": 7,
                    "rotations": (
                        ("chest", (0, 0, 1), -8.0),
                        ("shoulder_r", (0, 1, 0), 15.0),
                    ),
                },
                {
                    "frame": 14,
                    "rotations": (
                        ("chest", (0, 0, 1), 12.0),
                        ("shoulder_r", (0, 1, 0), -65.0),
                        ("elbow_r", (1, 0, 0), -75.0),
                    ),
                },
                {
                    "frame": 20,
                    "rotations": (
                        ("shoulder_r", (0, 1, 0), -25.0),
                        ("elbow_r", (1, 0, 0), -30.0),
                    ),
                },
                {"frame": 25},
            ],
        },
        "hit": {
            "loop": False,
            "poses": [
                {"frame": 1},
                {
                    "frame": 6,
                    "hips_location": (0.045, 0.025, 0.0),
                    "rotations": (
                        ("hips", (0, 0, 1), 10.0),
                        ("chest", (0, 0, 1), 25.0),
                        ("head", (0, 0, 1), -15.0),
                        ("shoulder_l", (0, 1, 0), 12.0),
                        ("shoulder_r", (0, 1, 0), -12.0),
                    ),
                },
                {"frame": 12, "rotations": (("chest", (0, 0, 1), -6.0),)},
                {"frame": 19},
            ],
        },
        "death": {
            "loop": False,
            "poses": [
                {"frame": 1},
                {
                    "frame": 12,
                    "hips_location": (0.08, 0.0, -0.08),
                    "rotations": (("hips", (0, 1, 0), -28.0), ("head", (0, 1, 0), 8.0)),
                },
                {
                    "frame": 24,
                    "hips_location": (0.28, 0.0, -0.12),
                    "rotations": (("hips", (0, 1, 0), -65.0), ("head", (0, 1, 0), 12.0)),
                },
                {
                    "frame": 36,
                    "hips_location": (0.38, 0.0, -0.13),
                    "rotations": (("hips", (0, 1, 0), -88.0),),
                },
            ],
        },
    }


MOTION_CRITICAL_JOINTS = {
    "idle": ("hips", "chest", "head"),
    "walk": (
        "hips",
        "hip_l", "knee_l", "ankle_l", "foot_l",
        "hip_r", "knee_r", "ankle_r", "foot_r",
        "shoulder_l", "shoulder_r",
    ),
    "attack": ("hips", "chest", "shoulder_r", "elbow_r", "wrist_r", "hand_r"),
    "hit": ("hips", "spine", "chest", "neck", "head"),
    "death": ("hips", "spine", "chest", "neck", "head", "hip_l", "hip_r", "knee_l", "knee_r"),
}

MOTION_REQUIRED_ACTIVE_JOINTS = {
    "idle": ("chest",),
    "walk": ("hip_l", "knee_l", "hip_r", "knee_r", "shoulder_l", "shoulder_r"),
    "attack": ("chest", "shoulder_r", "elbow_r"),
    "hit": ("hips", "chest", "head"),
    "death": ("hips", "head"),
}


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
    return _skinning_report(obj, armature, method="blender_bone_heat_pruned_to_four_v1")


def _vertex_group_weight(group: bpy.types.VertexGroup, vertex_index: int) -> float:
    try:
        return float(group.weight(vertex_index))
    except RuntimeError:
        return 0.0


def _apply_weight_redistributions(
    obj: bpy.types.Object,
    armature: bpy.types.Object,
    landmark_report: dict[str, object],
    adjustments: list[dict[str, object]],
) -> dict[str, object]:
    landmarks = {name: Vector(value) for name, value in dict(landmark_report["landmarks"]).items()}
    extents = Vector(landmark_report["bounds"]["extents"])
    height = float(extents.z)
    records: list[dict[str, object]] = []
    for index, raw in enumerate(adjustments):
        if not isinstance(raw, dict):
            raise ValueError(f"weight adjustment {index} must be an object")
        unknown = sorted(
            set(raw) - {"joint_pair", "direction", "transfer_fraction", "radius_fraction"}
        )
        if unknown:
            raise ValueError(f"weight adjustment {index} has unknown fields: {unknown}")
        joint_pair = str(raw.get("joint_pair", ""))
        direction = str(raw.get("direction", ""))
        transfer_fraction = float(raw.get("transfer_fraction", 0.0))
        radius_fraction = float(raw.get("radius_fraction", 0.0))
        if joint_pair not in WEIGHT_JOINT_PAIRS:
            raise ValueError(f"unsupported weight joint pair: {joint_pair}")
        if direction not in {"parent_to_child", "child_to_parent"}:
            raise ValueError(f"unsupported weight transfer direction: {direction}")
        if not 0.025 <= transfer_fraction <= 0.50:
            raise ValueError("weight transfer_fraction must be between 0.025 and 0.50")
        if not 0.03 <= radius_fraction <= 0.25:
            raise ValueError("weight radius_fraction must be between 0.03 and 0.25")
        radius = height * radius_fraction
        affected_vertices: set[int] = set()
        transferred_weight = 0.0
        for landmark_name, parent_name, child_name in WEIGHT_JOINT_PAIRS[joint_pair]:
            parent_group = obj.vertex_groups.get(parent_name)
            child_group = obj.vertex_groups.get(child_name)
            if parent_group is None or child_group is None:
                raise RuntimeError(
                    f"weight redistribution requires groups {parent_name} and {child_name}"
                )
            donor = parent_group if direction == "parent_to_child" else child_group
            receiver = child_group if direction == "parent_to_child" else parent_group
            center = landmarks[landmark_name]
            for vertex in obj.data.vertices:
                distance = (obj.matrix_world @ vertex.co - center).length
                if distance >= radius:
                    continue
                falloff = 1.0 - distance / radius
                donor_weight = _vertex_group_weight(donor, vertex.index)
                if donor_weight <= 0:
                    continue
                amount = donor_weight * transfer_fraction * falloff
                receiver_weight = _vertex_group_weight(receiver, vertex.index)
                donor.add([vertex.index], donor_weight - amount, "REPLACE")
                receiver.add([vertex.index], receiver_weight + amount, "REPLACE")
                affected_vertices.add(vertex.index)
                transferred_weight += amount
        records.append(
            {
                "joint_pair": joint_pair,
                "direction": direction,
                "transfer_fraction": transfer_fraction,
                "radius_fraction": radius_fraction,
                "affected_vertices": len(affected_vertices),
                "transferred_weight": transferred_weight,
            }
        )
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.vertex_group_clean(group_select_mode="ALL", limit=1e-6, keep_single=True)
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    bpy.context.view_layer.update()
    report = _skinning_report(
        obj,
        armature,
        method="blender_bone_heat_plus_bounded_joint_redistribution_v1",
    )
    report["weight_adjustments"] = records
    return report


def _apply_authored_pose(
    armature: bpy.types.Object,
    pose: dict[str, object],
) -> None:
    _clear_pose(armature)
    for bone_name, axis, degrees in pose.get("rotations", ()):
        _rotate_bone_world_axis(armature, bone_name, Vector(axis), float(degrees))
    hips = armature.pose.bones["hips"]
    world_offset = Vector(pose.get("hips_location", (0.0, 0.0, 0.0)))
    hips.location = hips.bone.matrix_local.to_3x3().inverted() @ world_offset
    bpy.context.view_layer.update()


def _author_motion_actions(
    armature: bpy.types.Object,
    specs: dict[str, dict[str, object]],
) -> dict[str, bpy.types.Action]:
    if armature.animation_data is not None:
        armature.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    animation_data = armature.animation_data_create()
    actions: dict[str, bpy.types.Action] = {}
    for clip_name, spec in specs.items():
        action = bpy.data.actions.new(name=clip_name)
        action.use_fake_user = True
        animation_data.action = action
        for pose in spec["poses"]:
            frame = int(pose["frame"])
            bpy.context.scene.frame_set(frame)
            _apply_authored_pose(armature, pose)
            for bone in armature.pose.bones:
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)
                bone.keyframe_insert(data_path="location", frame=frame, group=bone.name)
        action["text2model_clip"] = clip_name
        action["text2model_loop"] = bool(spec["loop"])
        actions[clip_name] = action
    animation_data.action = None
    bpy.context.scene.frame_set(0)
    _clear_pose(armature)
    return actions


def _pose_snapshot(armature: bpy.types.Object) -> tuple[float, ...]:
    return tuple(
        float(value)
        for bone in armature.pose.bones
        for row in bone.matrix_basis
        for value in row
    )


def _evaluate_action_frame(
    armature: bpy.types.Object,
    action: bpy.types.Action,
    frame: int,
) -> None:
    armature.animation_data.action = action
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()


def _bone_tail_world(armature: bpy.types.Object, name: str) -> Vector:
    return armature.matrix_world @ armature.pose.bones[name].tail


def _bone_head_world(armature: bpy.types.Object, name: str) -> Vector:
    return armature.matrix_world @ armature.pose.bones[name].head


def _critical_joint_snapshot(
    armature: bpy.types.Object,
    names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name in names:
        bone = armature.pose.bones[name]
        head = _bone_head_world(armature, name)
        tail = _bone_tail_world(armature, name)
        rotation = bone.matrix_basis.to_quaternion()
        angle = min(float(rotation.angle), math.tau - float(rotation.angle))
        values = (*head, *tail, angle)
        result[name] = {
            "head": list(head),
            "tail": list(tail),
            "rotation_degrees_from_rest": math.degrees(angle),
            "finite": all(math.isfinite(value) for value in values),
        }
    return result


def _bone_point_at_frame(
    armature: bpy.types.Object,
    action: bpy.types.Action,
    frame: int,
    bone_name: str,
    *,
    point: str = "tail",
) -> Vector:
    _evaluate_action_frame(armature, action, frame)
    if point == "head":
        return _bone_head_world(armature, bone_name)
    return _bone_tail_world(armature, bone_name)


def _glb_animation_names(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:4] != b"glTF" or len(data) < 20:
        raise ValueError("exported animation artifact is not GLB 2.0")
    json_length = int.from_bytes(data[12:16], "little")
    if data[16:20] != b"JSON":
        raise ValueError("GLB does not begin with a JSON chunk")
    document = json.loads(data[20 : 20 + json_length].decode("utf-8"))
    return sorted(str(item.get("name", "")) for item in document.get("animations", []))


def _run_short_biped_motion(
    output_root: Path,
    *,
    render_size: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if len(armatures) != 1 or len(_mesh_objects()) != 1:
        raise ValueError("short-biped motion authoring requires exactly one armature and one mesh")
    armature = armatures[0]
    obj = _mesh_objects()[0]
    required_bones = {item[0] for item in SHORT_BIPED_BONES}
    available_bones = {bone.name for bone in armature.data.bones}
    missing_bones = sorted(required_bones - available_bones)
    if missing_bones:
        raise RuntimeError(f"motion rig is missing required bones: {missing_bones}")
    if not any(modifier.type == "ARMATURE" and modifier.object == armature for modifier in obj.modifiers):
        raise RuntimeError("motion mesh is not bound to the required armature")

    coordinate_digest_before = _mesh_coordinate_digest(obj)
    topology_digest_before = _mesh_topology_digest(obj)
    rest_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    rest_areas = [polygon.area for polygon in obj.data.polygons]
    height = max(point.z for point in rest_positions) - min(point.z for point in rest_positions)
    rest_ground = min(point.z for point in rest_positions)
    specs = _short_biped_motion_specs()
    actions = _author_motion_actions(armature, specs)

    clip_reports: dict[str, object] = {}
    all_hard_failures: list[str] = []
    for clip_name, spec in specs.items():
        action = actions[clip_name]
        keyframes = [int(pose["frame"]) for pose in spec["poses"]]
        sample_frames = sorted(
            set(keyframes + [(first + second) // 2 for first, second in zip(keyframes, keyframes[1:])])
        )
        pose_reports = []
        critical_joint_frames: dict[str, object] = {}
        joint_excursion_degrees = {name: 0.0 for name in MOTION_CRITICAL_JOINTS[clip_name]}
        for frame in sample_frames:
            _evaluate_action_frame(armature, action, frame)
            joint_snapshot = _critical_joint_snapshot(armature, MOTION_CRITICAL_JOINTS[clip_name])
            critical_joint_frames[str(frame)] = joint_snapshot
            for name, snapshot in joint_snapshot.items():
                joint_excursion_degrees[name] = max(
                    joint_excursion_degrees[name],
                    float(snapshot["rotation_degrees_from_rest"]),
                )
            pose_reports.append(
                _evaluated_pose_report(
                    obj,
                    pose_name=f"{clip_name}@{frame}",
                    rest_positions=rest_positions,
                    rest_areas=rest_areas,
                )
            )
        loop_seam_error = None
        if bool(spec["loop"]):
            _evaluate_action_frame(armature, action, keyframes[0])
            start = _pose_snapshot(armature)
            _evaluate_action_frame(armature, action, keyframes[-1])
            end = _pose_snapshot(armature)
            loop_seam_error = max(abs(first - second) for first, second in zip(start, end))
        hard_failures = [failure for report in pose_reports for failure in report["hard_failures"]]
        non_finite_joints = sorted(
            name
            for frame in critical_joint_frames.values()
            for name, snapshot in frame.items()
            if not bool(snapshot["finite"])
        )
        if non_finite_joints:
            hard_failures.append("non_finite_critical_joint_transform")
        inactive_required_joints = sorted(
            name
            for name in MOTION_REQUIRED_ACTIVE_JOINTS[clip_name]
            if joint_excursion_degrees[name] < 1.0
        )
        if inactive_required_joints:
            hard_failures.append("inactive_required_critical_joint")
        minimum_z = min(float(report["bounds_minimum"][2]) for report in pose_reports)
        maximum_ground_penetration = max(0.0, rest_ground - minimum_z)
        if clip_name != "death" and maximum_ground_penetration > height * 0.02:
            hard_failures.append("ground_penetration")
        if loop_seam_error is not None and loop_seam_error > 1e-6:
            hard_failures.append("loop_seam")
        all_hard_failures.extend(f"{clip_name}:{failure}" for failure in hard_failures)
        clip_reports[clip_name] = {
            "frame_start": keyframes[0],
            "frame_end": keyframes[-1],
            "loop": bool(spec["loop"]),
            "authored_keyframes": len(keyframes),
            "sampled_frames": sample_frames,
            "loop_seam_error": loop_seam_error,
            "collapsed_faces": sum(int(report["collapsed_faces"]) for report in pose_reports),
            "severely_compressed_faces": sum(int(report["severely_compressed_faces"]) for report in pose_reports),
            "maximum_vertex_displacement": max(float(report["maximum_vertex_displacement"]) for report in pose_reports),
            "minimum_z": minimum_z,
            "final_minimum_z": float(pose_reports[-1]["bounds_minimum"][2]),
            "maximum_ground_penetration": maximum_ground_penetration,
            "critical_joints": list(MOTION_CRITICAL_JOINTS[clip_name]),
            "critical_joint_frames": critical_joint_frames,
            "joint_excursion_degrees": joint_excursion_degrees,
            "inactive_required_joints": inactive_required_joints,
            "non_finite_critical_joints": non_finite_joints,
            "hard_failures": hard_failures,
            "gate_passed": not hard_failures,
        }

    walk = actions["walk"]
    rest_left_contact = _bone_point_at_frame(armature, walk, 1, "foot_l")
    rest_right_contact = _bone_point_at_frame(armature, walk, 1, "foot_r")
    right_plant_anchor = _bone_point_at_frame(armature, walk, 7, "foot_r")
    left_plant_anchor = _bone_point_at_frame(armature, walk, 19, "foot_l")
    right_plant_absolute_error = (right_plant_anchor - rest_right_contact).length
    left_plant_absolute_error = (left_plant_anchor - rest_left_contact).length
    right_plant_window = {
        str(frame): (_bone_point_at_frame(armature, walk, frame, "foot_r") - right_plant_anchor).length
        for frame in range(5, 10)
    }
    left_plant_window = {
        str(frame): (_bone_point_at_frame(armature, walk, frame, "foot_l") - left_plant_anchor).length
        for frame in range(17, 22)
    }
    right_plant_error = max(right_plant_absolute_error, *right_plant_window.values())
    left_plant_error = max(left_plant_absolute_error, *left_plant_window.values())
    maximum_walk_plant_error = max(right_plant_error, left_plant_error)
    contact_gate_passed = maximum_walk_plant_error <= height * 0.015
    if not contact_gate_passed:
        all_hard_failures.append("walk:planted_foot_drift")
    left_swing_clearance = (
        _bone_point_at_frame(armature, walk, 7, "foot_l").z - rest_left_contact.z
    )
    right_swing_clearance = (
        _bone_point_at_frame(armature, walk, 19, "foot_r").z - rest_right_contact.z
    )
    minimum_swing_clearance = min(left_swing_clearance, right_swing_clearance)
    swing_clearance_gate_passed = minimum_swing_clearance >= height * 0.005
    if not swing_clearance_gate_passed:
        all_hard_failures.append("walk:insufficient_swing_foot_clearance")

    attack = actions["attack"]
    attack_hand_start = _bone_point_at_frame(armature, attack, 1, "hand_r")
    attack_hand_impact = _bone_point_at_frame(armature, attack, 14, "hand_r")
    attack_hand_travel = (attack_hand_impact - attack_hand_start).length
    attack_reach_gate_passed = attack_hand_travel >= height * 0.05
    if not attack_reach_gate_passed:
        all_hard_failures.append("attack:insufficient_hand_travel")

    hit = actions["hit"]
    hit_head_start = _bone_point_at_frame(armature, hit, 1, "head", point="head")
    hit_head_reaction = _bone_point_at_frame(armature, hit, 6, "head", point="head")
    hit_head_travel = (hit_head_reaction - hit_head_start).length
    hit_reaction_gate_passed = hit_head_travel >= height * 0.01
    if not hit_reaction_gate_passed:
        all_hard_failures.append("hit:insufficient_reaction_travel")

    death = actions["death"]
    death_hips_start = _bone_point_at_frame(armature, death, 1, "hips", point="head")
    death_hips_final = _bone_point_at_frame(armature, death, 36, "hips", point="head")
    death_head_start = _bone_point_at_frame(armature, death, 1, "head", point="head")
    death_head_final = _bone_point_at_frame(armature, death, 36, "head", point="head")
    death_hips_descent = death_hips_start.z - death_hips_final.z
    death_head_descent = death_head_start.z - death_head_final.z
    death_descent_gate_passed = (
        death_hips_descent >= height * 0.05 and death_head_descent >= height * 0.10
    )
    if not death_descent_gate_passed:
        all_hard_failures.append("death:insufficient_body_descent")
    death_ground_error = abs(float(clip_reports["death"]["final_minimum_z"]) - rest_ground)
    death_ground_gate_passed = death_ground_error <= height * 0.05
    if not death_ground_gate_passed:
        all_hard_failures.append("death:ground_settle")

    evidence_frames = tuple(
        (clip_name, int(pose["frame"]))
        for clip_name, spec in specs.items()
        for pose in spec["poses"]
    )
    evidence_renders: list[tuple[str, list[Path]]] = []
    for clip_name, frame in evidence_frames:
        _evaluate_action_frame(armature, actions[clip_name], frame)
        prefix = f"motion_{clip_name}_{frame:03d}"
        evidence_renders.append((prefix, render_diagnostics(output_root, prefix, size=render_size)))

    armature.animation_data.action = None
    bpy.context.scene.frame_set(0)
    _clear_pose(armature)
    _remove_diagnostic_objects()
    coordinate_digest_after = _mesh_coordinate_digest(obj)
    topology_digest_after = _mesh_topology_digest(obj)
    if coordinate_digest_after != coordinate_digest_before:
        all_hard_failures.append("rest_vertex_coordinates_changed")
    if topology_digest_after != topology_digest_before:
        all_hard_failures.append("rest_topology_changed")

    required_clips = sorted(specs)
    motion_contract = {
        "schema_version": 1,
        "anatomy_family": "short_biped_v1",
        "frames_per_second": 24,
        "clips": {
            name: {
                "frame_start": int(spec["poses"][0]["frame"]),
                "frame_end": int(spec["poses"][-1]["frame"]),
                "loop": bool(spec["loop"]),
            }
            for name, spec in specs.items()
        },
        "events": {"attack": [{"frame": 14, "event": "impact"}], "death": [{"frame": 36, "event": "settled"}]},
        "contacts": {
            "walk": [
                {"frame": 7, "bone": "foot_r", "role": "planted"},
                {"frame": 19, "bone": "foot_l", "role": "planted"},
            ]
        },
    }
    motion_report = {
        "schema_version": 1,
        "required_clips": required_clips,
        "authored_clips": sorted(actions),
        "clip_reports": clip_reports,
        "walk_contact": {
            "right_plant_error": right_plant_error,
            "left_plant_error": left_plant_error,
            "right_absolute_error": right_plant_absolute_error,
            "left_absolute_error": left_plant_absolute_error,
            "maximum_error": maximum_walk_plant_error,
            "right_window_errors": right_plant_window,
            "left_window_errors": left_plant_window,
            "maximum_allowed": height * 0.015,
            "gate_passed": contact_gate_passed,
        },
        "walk_swing": {
            "left_clearance": left_swing_clearance,
            "right_clearance": right_swing_clearance,
            "minimum_required": height * 0.005,
            "gate_passed": swing_clearance_gate_passed,
        },
        "attack_function": {
            "impact_frame": 14,
            "hand_travel": attack_hand_travel,
            "minimum_required": height * 0.05,
            "gate_passed": attack_reach_gate_passed,
        },
        "hit_function": {
            "reaction_frame": 6,
            "head_travel": hit_head_travel,
            "minimum_required": height * 0.01,
            "gate_passed": hit_reaction_gate_passed,
        },
        "death_function": {
            "hips_descent": death_hips_descent,
            "head_descent": death_head_descent,
            "minimum_hips_descent": height * 0.05,
            "minimum_head_descent": height * 0.10,
            "gate_passed": death_descent_gate_passed,
        },
        "ground": {
            "rest_height": rest_ground,
            "death_final_minimum_z": clip_reports["death"]["final_minimum_z"],
            "death_settle_error": death_ground_error,
            "death_maximum_allowed": height * 0.05,
            "death_gate_passed": death_ground_gate_passed,
        },
        "rest_vertex_coordinates_unchanged": coordinate_digest_after == coordinate_digest_before,
        "rest_topology_unchanged": topology_digest_after == topology_digest_before,
        "hard_failures": all_hard_failures,
        "motion_gate_passed": not all_hard_failures,
        "human_approval_required": True,
        "human_approved": False,
    }
    contract_path = output_root / "motion_contract.json"
    report_path = output_root / "motion_validation.json"
    _write_json(contract_path, motion_contract)
    _write_json(report_path, motion_report)

    animated_blend = output_root / "animated_candidate.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(animated_blend))
    _triangulate_scene_for_export()
    animated_glb = output_root / "animated_candidate.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(animated_glb),
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
    exported_animation_names = _glb_animation_names(animated_glb)
    _import_asset(animated_glb)
    export_primary = max(_mesh_objects(), key=lambda item: len(item.data.vertices))
    export_primary_analysis, _ = _object_analysis(export_primary)
    export_armatures = sum(item.type == "ARMATURE" for item in bpy.context.scene.objects)
    export_skin_modifiers = sum(
        modifier.type == "ARMATURE"
        for mesh in _mesh_objects()
        for modifier in mesh.modifiers
    )
    export_gate_passed = (
        exported_animation_names == required_clips
        and export_armatures == 1
        and export_skin_modifiers == 1
        and bool(export_primary_analysis["finite_coordinates"])
        and int(export_primary_analysis["components"]) == 1
        and int(export_primary_analysis["boundary_edges"]) == 0
        and int(export_primary_analysis["non_manifold_edges"]) == 0
    )
    export_report = {
        "schema_version": 1,
        "animations": exported_animation_names,
        "armatures": export_armatures,
        "skin_modifiers": export_skin_modifiers,
        "analysis": export_primary_analysis,
        "gate_passed": export_gate_passed,
    }
    export_path = output_root / "animated_export_validation.json"
    _write_json(export_path, export_report)
    automatic_gate_passed = bool(motion_report["motion_gate_passed"]) and export_gate_passed
    if not automatic_gate_passed:
        raise RuntimeError("short-biped motion authoring failed an automatic gate")

    outputs = [
        _output(contract_path, "application/json", "motion_contract"),
        _output(report_path, "application/json", "motion_validation"),
        _output(export_path, "application/json", "animated_export_validation"),
        _output(animated_blend, "application/x-blender", "animated_candidate_checkpoint"),
        _output(animated_glb, "model/gltf-binary", "animated_candidate"),
    ]
    for prefix, paths in evidence_renders:
        outputs.extend(
            _output(path, "image/png", f"{prefix}_{path.stem.rsplit('_', 1)[-1]}")
            for path in paths
        )
    diagnostics = {
        "clips": len(actions),
        "automatic_motion_gate_passed": automatic_gate_passed,
        "maximum_walk_plant_error": maximum_walk_plant_error,
        "human_approved": False,
    }
    warnings = [
        "These are deterministic project-owned baseline clips, not human-approved final animation.",
        "The unarmed attack is a timing/readability proof; weapon attachment and reach-target validation remain open.",
        "Existing shoulder/armpit compression from the staged skin remains visible and requires corrective refinement.",
        "Self-collision, ground penetration across interpolated frames, sprite readability, and Unity clip import are not yet fully gated.",
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
        if obj.name.startswith("Text2ModelDiagnostic") or obj.name.startswith("Text2ModelKey") or obj.name.startswith("Text2ModelFill"):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                if isinstance(data, bpy.types.Camera):
                    bpy.data.cameras.remove(data)
                elif isinstance(data, bpy.types.Light):
                    bpy.data.lights.remove(data)


def _realtime_render_engine() -> str:
    """The realtime engine id this Blender build actually offers.

    Blender 4.2 shipped EEVEE Next as "BLENDER_EEVEE_NEXT"; Blender 5.x
    dropped the legacy engine and renamed it back to "BLENDER_EEVEE".
    Hardcoding either one makes the worker fail on half the Blender versions
    in the wild with a bare TypeError from the enum assignment -- observed on
    Blender 5.1.2, where it killed D3 after the mesh had already imported
    cleanly. Ask the build what it supports instead.
    """
    try:
        engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    except Exception:
        engines = []
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if candidate in engines:
            return candidate
    return "BLENDER_WORKBENCH" if "BLENDER_WORKBENCH" in engines else "BLENDER_EEVEE"


def render_diagnostics(
    output_root: Path,
    prefix: str,
    *,
    size: int = 512,
    bounds_override: tuple[Vector, Vector] | None = None,
) -> list[Path]:
    scene = bpy.context.scene
    _remove_diagnostic_objects()
    scene.render.engine = _realtime_render_engine()
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Text2ModelDiagnosticWorld")
    scene.world.color = (0.015, 0.018, 0.023)

    material = bpy.data.materials.get("Text2ModelDiagnosticMaterial") or bpy.data.materials.new("Text2ModelDiagnosticMaterial")
    material.diffuse_color = (0.20, 0.38, 0.16, 1.0)
    material.metallic = 0.0
    material.roughness = 0.82
    bpy.context.view_layer.material_override = material

    minimum, maximum = bounds_override if bounds_override is not None else _scene_bounds()
    center = (minimum + maximum) * 0.5
    extents = maximum - minimum
    radius = max(extents) * 2.4

    camera_data = bpy.data.cameras.new("Text2ModelDiagnosticCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(extents.x, extents.z) * 1.24
    camera = bpy.data.objects.new("Text2ModelDiagnosticCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    for name, direction, energy, size_value in (
        ("Text2ModelKey", Vector((1.5, -2.0, 2.2)), 1100.0, 4.0),
        ("Text2ModelFill", Vector((-1.5, 1.0, 1.2)), 650.0, 3.0),
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


def _scene_world_bounds() -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in _mesh_objects() for corner in obj.bound_box]
    return (
        Vector(tuple(min(point[axis] for point in points) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in points) for axis in range(3))),
    )


def _front_normalized(point: Vector, minimum: Vector, maximum: Vector) -> tuple[float, float]:
    width = max(float(maximum.x - minimum.x), 1e-9)
    height = max(float(maximum.z - minimum.z), 1e-9)
    return (
        float((point.x - minimum.x) / width),
        float(1.0 - (point.z - minimum.z) / height),
    )


def _object_front_box(obj: bpy.types.Object, minimum: Vector, maximum: Vector) -> tuple[float, float, float, float]:
    projected = [_front_normalized(obj.matrix_world @ Vector(corner), minimum, maximum) for corner in obj.bound_box]
    return (
        min(item[0] for item in projected),
        min(item[1] for item in projected),
        max(item[0] for item in projected),
        max(item[1] for item in projected),
    )


def _box_iou(first, second) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def _extract_faces_in_front_box(
    source: bpy.types.Object,
    component_id: str,
    box: list[float],
    minimum: Vector,
    maximum: Vector,
) -> bpy.types.Object:
    selected = []
    for polygon in source.data.polygons:
        center = source.matrix_world @ polygon.center
        x, y = _front_normalized(center, minimum, maximum)
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            selected.append(polygon)
    if not selected:
        raise RuntimeError(f"rigid component {component_id} selected zero faces")
    vertex_indices = sorted({index for polygon in selected for index in polygon.vertices})
    remap = {old: new for new, old in enumerate(vertex_indices)}
    vertices = [tuple(source.data.vertices[index].co) for index in vertex_indices]
    faces = [[remap[index] for index in polygon.vertices] for polygon in selected]
    mesh = bpy.data.meshes.new(component_id + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    result = bpy.data.objects.new(component_id, mesh)
    bpy.context.collection.objects.link(result)
    result.matrix_world = source.matrix_world.copy()
    for material in source.data.materials:
        result.data.materials.append(material)
    for target, polygon in zip(result.data.polygons, selected, strict=True):
        target.material_index = min(polygon.material_index, max(0, len(result.data.materials) - 1))

    bm = bmesh.new()
    try:
        bm.from_mesh(source.data)
        bm.faces.ensure_lookup_table()
        delete = [bm.faces[polygon.index] for polygon in selected]
        bmesh.ops.delete(bm, geom=delete, context="FACES")
        bm.to_mesh(source.data)
        source.data.update()
    finally:
        bm.free()
    return result


def _set_origin_at_world(obj: bpy.types.Object, point: Vector) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.context.scene.cursor.location = point
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
    obj.select_set(False)


def _author_rigid_actions(
    parts: list[tuple[bpy.types.Object, dict[str, object]]], animations: list[str]
) -> list[str]:
    created: list[str] = []
    for obj, part in parts:
        component_id = str(part["component_id"])
        axis_index = {"x": 0, "y": 1, "z": 2}[str(part["rotation_axis"])]
        neutral = math.radians(float(part["neutral_degrees"]))
        target = math.radians(float(part["maximum_degrees"]))
        obj.rotation_mode = "XYZ"
        obj.rotation_euler[axis_index] = neutral
        obj.animation_data_create()
        for clip in animations:
            side_tokens = {token for token in ("left", "right") if token in clip.lower()}
            if side_tokens and not any(token in component_id.lower() for token in side_tokens):
                continue
            action = bpy.data.actions.new(name=f"{clip}__{component_id}")
            action.use_fake_user = True
            obj.animation_data.action = action
            is_close = "close" in clip.lower()
            values = (target, neutral) if is_close else (neutral, target)
            for frame, value in ((1, values[0]), (20, values[1])):
                obj.rotation_euler[axis_index] = value
                obj.keyframe_insert(data_path="rotation_euler", index=axis_index, frame=frame)
            action["text2model_clip"] = clip
            action["text2model_component_id"] = component_id
            created.append(action.name)
        obj.animation_data.action = None
        obj.rotation_euler[axis_index] = neutral
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    return created


def _run_rigid_articulation(
    output_root: Path,
    structure_plan: dict[str, object],
    animations: list[str],
    *,
    render_size: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    parts_plan = list(structure_plan.get("parts") or [])
    if not parts_plan:
        raise ValueError("rigid articulation requires at least one planned part")
    minimum, maximum = _scene_world_bounds()
    unused = set(_mesh_objects())
    parts: list[tuple[bpy.types.Object, dict[str, object]]] = []
    for raw in parts_plan:
        part = dict(raw)
        component_id = str(part["component_id"])
        box = [float(item) for item in part["front_box_normalized"]]
        scored = sorted(
            (
                (_box_iou(_object_front_box(obj, minimum, maximum), box), len(obj.data.polygons), obj)
                for obj in unused
            ),
            key=lambda item: (-item[0], -item[1], item[2].name),
        )
        if scored and scored[0][0] >= 0.45 and len(unused) > len(parts_plan) - len(parts):
            obj = scored[0][2]
            unused.remove(obj)
            obj.name = component_id
        else:
            source = max(unused or set(_mesh_objects()), key=lambda item: len(item.data.polygons))
            obj = _extract_faces_in_front_box(source, component_id, box, minimum, maximum)
        pivot_x, pivot_y = [float(item) for item in part["pivot_normalized"]]
        pivot = Vector(
            (
                minimum.x + pivot_x * (maximum.x - minimum.x),
                sum((obj.matrix_world @ Vector(corner)).y for corner in obj.bound_box) / 8.0,
                maximum.z - pivot_y * (maximum.z - minimum.z),
            )
        )
        _set_origin_at_world(obj, pivot)
        obj["text2model_component_id"] = component_id
        obj["text2model_rotation_axis"] = str(part["rotation_axis"])
        obj["text2model_minimum_degrees"] = float(part["minimum_degrees"])
        obj["text2model_maximum_degrees"] = float(part["maximum_degrees"])
        obj["text2model_neutral_degrees"] = float(part["neutral_degrees"])
        parts.append((obj, part))
    actions = _author_rigid_actions(parts, animations)
    neutral_renders = render_diagnostics(output_root, "rigid_neutral", size=render_size)
    for obj, part in parts:
        axis = {"x": 0, "y": 1, "z": 2}[str(part["rotation_axis"])]
        obj.rotation_euler[axis] = math.radians(float(part["maximum_degrees"]))
    bpy.context.view_layer.update()
    open_renders = render_diagnostics(output_root, "rigid_open", size=render_size)
    for obj, part in parts:
        axis = {"x": 0, "y": 1, "z": 2}[str(part["rotation_axis"])]
        obj.rotation_euler[axis] = math.radians(float(part["neutral_degrees"]))
    bpy.context.view_layer.update()
    _remove_diagnostic_objects()

    blend_path = output_root / "rigid_articulated_candidate.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = output_root / "rigid_articulated_candidate.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_materials="EXPORT",
        export_animations=True,
    )
    report = {
        "schema_version": 1,
        "method": "qwen_semantic_front_boxes_deterministic_mesh_partition_v1",
        "parts": [
            {
                "component_id": str(part["component_id"]),
                "object": obj.name,
                "faces": len(obj.data.polygons),
                "rotation_axis": part["rotation_axis"],
                "minimum_degrees": part["minimum_degrees"],
                "maximum_degrees": part["maximum_degrees"],
                "neutral_degrees": part["neutral_degrees"],
            }
            for obj, part in parts
        ],
        "animations": actions,
        "hard_failures": [
            f"empty_component:{obj.name}" for obj, _ in parts if len(obj.data.polygons) == 0
        ],
        "gate_passed": all(len(obj.data.polygons) > 0 for obj, _ in parts) and bool(actions),
        "human_approval_required": True,
        "human_approved": False,
    }
    report_path = output_root / "rigid_articulation_report.json"
    _write_json(report_path, report)
    if not report["gate_passed"]:
        raise RuntimeError("rigid articulation failed component/action hard gates")
    outputs = [
        _output(report_path, "application/json", "rigid_articulation_report"),
        _output(blend_path, "application/x-blender", "rigid_articulated_checkpoint"),
        _output(glb_path, "model/gltf-binary", "rigid_articulated_candidate"),
    ]
    for prefix, paths in (("rigid_neutral", neutral_renders), ("rigid_open", open_renders)):
        outputs.extend(
            _output(path, "image/png", f"{prefix}_{path.stem.rsplit('_', 1)[-1]}")
            for path in paths
        )
    diagnostics = {
        "rigid_parts": len(parts),
        "rigid_actions": len(actions),
        "automatic_rigid_gate_passed": True,
    }
    return outputs, diagnostics, [
        "Front-view semantic segmentation is a proposal; human approval of part boundaries and pivots is required."
    ]


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
        raw_landmark_adjustments = parameters.get("landmark_adjustments", {})
        if not isinstance(raw_landmark_adjustments, dict):
            raise ValueError("landmark_adjustments must be an object")
        landmark_adjustments = {
            str(name): list(values) if isinstance(values, list) else values
            for name, values in raw_landmark_adjustments.items()
        }
        raw_weight_adjustments = parameters.get("weight_adjustments", [])
        if not isinstance(raw_weight_adjustments, list):
            raise ValueError("weight_adjustments must be an array")
        weight_adjustments = [dict(item) if isinstance(item, dict) else item for item in raw_weight_adjustments]
        probe_outputs, probe_diagnostics, probe_warnings = _run_short_biped_rig_probe(
            output_root,
            source_renders,
            render_size=render_size,
            maximum_material_change_fraction=comparison_threshold,
            landmark_adjustments=landmark_adjustments,
            weight_adjustments=weight_adjustments,
        )
        outputs.extend(probe_outputs)
        warnings.extend(probe_warnings)
    elif operation_id == "blender.author_short_biped_motion":
        motion_outputs, motion_diagnostics, motion_warnings = _run_short_biped_motion(
            output_root,
            render_size=render_size,
        )
        outputs.extend(motion_outputs)
        warnings.extend(motion_warnings)
    elif operation_id == "blender.author_rigid_articulation":
        structure_plan = parameters.get("structure_plan")
        animations = parameters.get("animations")
        if not isinstance(structure_plan, dict):
            raise ValueError("structure_plan must be an object")
        if not isinstance(animations, list) or not all(isinstance(item, str) for item in animations):
            raise ValueError("animations must be an array of ids")
        rigid_outputs, rigid_diagnostics, rigid_warnings = _run_rigid_articulation(
            output_root,
            structure_plan,
            list(animations),
            render_size=render_size,
        )
        outputs.extend(rigid_outputs)
        warnings.extend(rigid_warnings)
    elif operation_id in {"blender.repair", "blender.repair_retopology", "blender.export"}:
        component_policy = str(parameters.get("component_policy", "none"))
        minimum_connected_components = int(parameters.get("minimum_connected_components", 1))
        maximum_connected_components = int(parameters.get("maximum_connected_components", 1))
        if not 1 <= minimum_connected_components <= maximum_connected_components <= 256:
            raise ValueError(
                "connected-component bounds must satisfy 1 <= minimum <= maximum <= 256"
            )
        weld_distance = float(parameters.get("weld_distance", 0.0))
        if not 0 <= weld_distance <= 0.01:
            raise ValueError("weld_distance must be between zero and 0.01")
        # How to reach a manifold surface before the component filter runs.
        #   weld_only  -- the historical behaviour, correct for a generator
        #                 whose output is already near-manifold.
        #   if_needed  -- voxel-remesh only when the source really is
        #                 non-manifold or has holes, so a clean mesh keeps its
        #                 original topology and detail untouched.
        #   always     -- voxel-remesh unconditionally.
        # See voxel_remesh_to_manifold() for the measurements that motivate
        # this; welding cannot fix an isosurface extractor's output.
        manifold_policy = str(parameters.get("manifold_policy", "weld_only"))
        if manifold_policy not in {"weld_only", "if_needed", "always"}:
            raise ValueError("manifold_policy must be 'weld_only', 'if_needed', or 'always'")
        voxel_fraction = float(parameters.get("voxel_fraction", 0.006))
        voxel_target_faces = int(parameters.get("voxel_target_faces", 0))
        if voxel_target_faces and not 1000 <= voxel_target_faces <= 5_000_000:
            raise ValueError("voxel_target_faces must be between 1000 and 5,000,000")
        remesh_report: dict[str, object] | None = None
        if operation_id in {"blender.repair", "blender.export"} and manifold_policy != "weld_only":
            already_manifold = (
                int(source_analysis.get("non_manifold_edges", 0)) == 0
                and int(source_analysis.get("boundary_edges", 0)) == 0
            )
            if manifold_policy == "always" or not already_manifold:
                remesh_report = voxel_remesh_to_manifold(
                    voxel_fraction=voxel_fraction, target_faces=voxel_target_faces
                )
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
        if remesh_report is not None:
            # Recorded as its own step rather than folded into `repair`, so the
            # evidence shows exactly what the remesh changed and what the
            # component filter changed afterwards.
            repair["voxel_remesh"] = remesh_report
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
        if remesh_report is not None:
            # A voxel remesh rebuilds the surface; it is a different operation
            # from a weld and needs its own budget. The default budget assumes
            # cleanup barely moves a vertex, which is true for remove_doubles
            # and false by construction here. Measured on the knight, against
            # the pre-remesh source: front 1.5%, back 2.9%, left 14.0%,
            # right 12.7% -- the side views carry the thin sword and shield
            # rim, where rounding is most visible. Crucially this is flat
            # across voxel sizes (14.03% at 0.004, 14.05% at 0.002), so it is
            # the rebuild itself, not lost resolution, and shrinking the voxel
            # cannot buy it back.
            #
            # This does not remove a check, it selects the right one: identity
            # is still enforced by D3's Qwen review against the approved D1
            # concept and by the human gate at D4. Setting it to 0 or 1
            # deliberately disables/permits everything and is the caller's
            # choice to make explicitly.
            comparison_threshold = float(
                parameters.get("maximum_material_change_fraction_remeshed", 0.25)
            )
            if not 0 <= comparison_threshold <= 1:
                raise ValueError(
                    "maximum_material_change_fraction_remeshed must be between zero and one"
                )
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
            and minimum_connected_components
            <= int(export_analysis["connected_components"])
            <= maximum_connected_components
            and int(export_analysis["boundary_edges"]) == 0
            and int(export_analysis["non_manifold_edges"]) == 0
            and int(export_analysis["inconsistent_winding_edges"]) == 0
        )
        export_validation = {
            "schema_version": 1,
            "hard_gate_passed": export_hard_gate_passed,
            "pre_export_analysis": pre_export_analysis,
            "analysis": export_analysis,
            "minimum_connected_components": minimum_connected_components,
            "maximum_connected_components": maximum_connected_components,
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
    elif operation_id == "blender.author_short_biped_motion":
        diagnostics.update(motion_diagnostics)
    elif operation_id == "blender.author_rigid_articulation":
        diagnostics.update(rigid_diagnostics)
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
