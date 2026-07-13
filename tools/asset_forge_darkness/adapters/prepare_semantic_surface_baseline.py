"""Apply a deterministic semantic palette before diffusion refines the surface."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy


def _arguments() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _material(name: str, dark: tuple[float, float, float], light: tuple[float, float, float], roughness: float):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = 0.0
    coordinates = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 7.0
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.62
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.24
    ramp.color_ramp.elements[0].color = (*dark, 1.0)
    ramp.color_ramp.elements[1].position = 0.76
    ramp.color_ramp.elements[1].color = (*light, 1.0)
    links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _semantic_materials():
    return {
        "skin": _material("Darkness Skin Olive", (0.09, 0.16, 0.045), (0.30, 0.43, 0.12), 0.82),
        "cloth": _material("Darkness Cloth Charcoal", (0.025, 0.022, 0.030), (0.105, 0.075, 0.095), 0.92),
        "leather": _material("Darkness Leather Worn", (0.055, 0.025, 0.012), (0.24, 0.105, 0.035), 0.84),
        "wood": _material("Darkness Club Wood", (0.065, 0.022, 0.008), (0.30, 0.12, 0.025), 0.88),
        "iron": _material("Darkness Club Iron", (0.018, 0.022, 0.026), (0.16, 0.19, 0.20), 0.58),
    }


def _assign_body(obj, materials) -> dict[str, int]:
    obj.data.materials.clear()
    keys = ("skin", "cloth", "leather")
    for key in keys:
        obj.data.materials.append(materials[key])
    indices = {key: index for index, key in enumerate(keys)}
    group_names = {group.index: group.name.lower() for group in obj.vertex_groups}
    world_z = [float((obj.matrix_world @ vertex.co).z) for vertex in obj.data.vertices]
    minimum, maximum = min(world_z), max(world_z)
    height = max(maximum - minimum, 1e-6)
    counts = {key: 0 for key in keys}
    for polygon in obj.data.polygons:
        vertices = [obj.data.vertices[index] for index in polygon.vertices]
        normalized_z = (sum(world_z[item.index] for item in vertices) / len(vertices) - minimum) / height
        weights: dict[str, float] = {}
        for vertex in vertices:
            for assignment in vertex.groups:
                name = group_names.get(assignment.group, "")
                weights[name] = weights.get(name, 0.0) + float(assignment.weight)
        relevant = {name for name, value in weights.items() if value >= 0.15}
        is_leg = any(token in name for name in relevant for token in ("foot", "ankle", "shin", "calf"))
        is_pelvis = any(token in name for name in relevant for token in ("hip", "pelvis", "thigh"))
        if normalized_z < 0.19 or (is_leg and normalized_z < 0.31):
            semantic = "leather"
        elif 0.43 <= normalized_z <= 0.485:
            semantic = "leather"
        elif (is_pelvis and normalized_z < 0.56) or 0.31 <= normalized_z < 0.43:
            semantic = "cloth"
        else:
            semantic = "skin"
        polygon.material_index = indices[semantic]
        counts[semantic] += 1
    return counts


def _assign_club(obj, materials) -> dict[str, int]:
    obj.data.materials.clear()
    keys = ("wood", "iron")
    for key in keys:
        obj.data.materials.append(materials[key])
    indices = {key: index for index, key in enumerate(keys)}
    local_y = [float(vertex.co.y) for vertex in obj.data.vertices]
    minimum, maximum = min(local_y), max(local_y)
    length = max(maximum - minimum, 1e-6)
    counts = {key: 0 for key in keys}
    for polygon in obj.data.polygons:
        normalized_y = (
            sum(local_y[index] for index in polygon.vertices) / len(polygon.vertices) - minimum
        ) / length
        semantic = "iron" if 0.53 <= normalized_y <= 0.63 or normalized_y >= 0.93 else "wood"
        polygon.material_index = indices[semantic]
        counts[semantic] += 1
    return counts


def main() -> int:
    args = _arguments()
    source = args.source.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    error_path = report_path.parent / "semantic_surface_error.txt"
    if error_path.is_file():
        error_path.unlink()
    materials = _semantic_materials()
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise RuntimeError("surface baseline found no visible meshes")
    club = bpy.data.objects.get("DarknessClub")
    if club is None or club.type != "MESH":
        raise RuntimeError("surface baseline requires DarknessClub")
    bodies = [obj for obj in meshes if obj is not club]
    body = max(bodies, key=lambda obj: len(obj.data.polygons))
    assignments = {body.name: _assign_body(body, materials), club.name: _assign_club(club, materials)}
    body["darkness_surface_semantics"] = "skin,cloth,leather"
    club["darkness_surface_semantics"] = "wood,iron"
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report = {
        "schema_version": 1,
        "stage": "semantic_surface_baseline",
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "body": body.name,
        "club": club.name,
        "material_face_counts": assignments,
        "palette": {
            "skin": "desaturated_olive",
            "cloth": "charcoal_plum",
            "leather": "worn_brown",
            "wood": "dark_warm_wood",
            "iron": "cold_dark_iron",
        },
        "automatic_gate_passed": all(sum(values.values()) > 0 for values in assignments.values()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        if "--report" in raw:
            report = Path(raw[raw.index("--report") + 1]).resolve()
            (report.parent / "semantic_surface_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
