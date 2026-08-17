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
    parser.add_argument("--asset-spec", type=Path)
    return parser.parse_args(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _material(
    name: str,
    dark: tuple[float, float, float],
    light: tuple[float, float, float],
    roughness: float,
    metallic: float = 0.0,
):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
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


def _material_descriptor(description: str, index: int):
    value = description.lower()
    if any(token in value for token in ("steel", "iron", "metal", "silver")):
        colors = ((0.025, 0.035, 0.045), (0.20, 0.24, 0.27), 0.46, 0.74)
    elif "stone" in value:
        colors = ((0.08, 0.075, 0.07), (0.34, 0.32, 0.28), 0.91, 0.0)
    elif any(token in value for token in ("wood", "timber")):
        colors = ((0.065, 0.022, 0.008), (0.30, 0.12, 0.025), 0.88, 0.0)
    elif "leather" in value:
        colors = ((0.055, 0.025, 0.012), (0.24, 0.105, 0.035), 0.84, 0.0)
    elif any(token in value for token in ("blue", "cobalt")):
        colors = ((0.018, 0.035, 0.075), (0.055, 0.16, 0.34), 0.91, 0.0)
    elif any(token in value for token in ("red", "crimson")):
        colors = ((0.075, 0.012, 0.012), (0.34, 0.045, 0.025), 0.9, 0.0)
    elif "skin" in value and any(token in value for token in ("olive", "green", "goblin")):
        colors = ((0.09, 0.16, 0.045), (0.30, 0.43, 0.12), 0.82, 0.0)
    elif "skin" in value:
        colors = ((0.18, 0.075, 0.045), (0.54, 0.28, 0.17), 0.78, 0.0)
    elif any(token in value for token in ("cloth", "fabric", "linen")):
        colors = ((0.025, 0.022, 0.030), (0.105, 0.075, 0.095), 0.92, 0.0)
    else:
        hue = (index * 0.173) % 1.0
        colors = ((0.04 + hue * 0.05, 0.045, 0.055), (0.15 + hue * 0.12, 0.16, 0.18), 0.84, 0.0)
    dark, light, roughness, metallic = colors
    key = f"surface_{index:02d}"
    material = _material(
        f"Text2Model {description.title()}", dark, light, roughness, metallic
    )
    return key, material


def _semantic_materials(spec):
    descriptions = [str(item) for item in spec.get("materials", []) if str(item).strip()]
    if not descriptions:
        descriptions = ["neutral worn material"]
    return {
        key: {"description": description, "material": material}
        for index, description in enumerate(descriptions)
        for key, material in [_material_descriptor(description, index)]
    }


def _select_index(materials, *tokens: str) -> int:
    records = list(materials.values())
    for index, record in enumerate(records):
        description = record["description"].lower()
        if any(token in description for token in tokens):
            return index
    return 0


def _assign_mesh(obj, materials, *, asset_kind: str) -> dict[str, int]:
    records = list(materials.items())
    obj.data.materials.clear()
    for _, record in records:
        obj.data.materials.append(record["material"])
    counts = {key: 0 for key, _ in records}
    if len(records) == 1:
        for polygon in obj.data.polygons:
            polygon.material_index = 0
            counts[records[0][0]] += 1
        return counts

    name = obj.name.lower()
    if any(token in name for token in ("weapon", "sword", "blade", "club")):
        preferred = _select_index(materials, "steel", "iron", "metal", "wood")
    elif "shield" in name:
        preferred = _select_index(materials, "wood", "steel", "iron", "metal")
    elif asset_kind not in {"character", "creature"}:
        preferred = next(
            (
                index
                for index, (_, record) in enumerate(records)
                if any(token in name for token in record["description"].lower().split())
            ),
            0,
        )
    else:
        preferred = -1

    world_z = [float((obj.matrix_world @ vertex.co).z) for vertex in obj.data.vertices]
    minimum, maximum = min(world_z), max(world_z)
    height = max(maximum - minimum, 1e-6)
    cloth = _select_index(materials, "cloth", "fabric", "linen", "blue", "cobalt")
    leather = _select_index(materials, "leather")
    for polygon in obj.data.polygons:
        normalized_z = (
            sum(world_z[index] for index in polygon.vertices) / len(polygon.vertices) - minimum
        ) / height
        if preferred >= 0:
            index = preferred
        elif normalized_z < 0.20 or 0.43 <= normalized_z <= 0.49:
            index = leather
        elif 0.28 <= normalized_z < 0.57:
            index = cloth
        else:
            index = 0
        polygon.material_index = index
        counts[records[index][0]] += 1
    return counts


def main() -> int:
    args = _arguments()
    source = args.source.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    error_path = report_path.parent / "semantic_surface_error.txt"
    if error_path.is_file():
        error_path.unlink()
    spec = (
        json.loads(args.asset_spec.resolve().read_text(encoding="utf-8"))
        if args.asset_spec is not None
        else {
            "asset_kind": "character",
            "materials": ["olive skin", "charcoal cloth", "worn leather", "dark wood", "iron"],
        }
    )
    materials = _semantic_materials(spec)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise RuntimeError("surface baseline found no visible meshes")
    assignments = {
        obj.name: _assign_mesh(obj, materials, asset_kind=str(spec.get("asset_kind", "character")))
        for obj in meshes
    }
    for obj in meshes:
        obj["text2model_surface_semantics"] = ",".join(materials)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report = {
        "schema_version": 1,
        "stage": "semantic_surface_baseline",
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "asset_spec": str(args.asset_spec.resolve()) if args.asset_spec is not None else None,
        "asset_spec_sha256": _sha256(args.asset_spec.resolve()) if args.asset_spec is not None else None,
        "asset_kind": spec.get("asset_kind", "character"),
        "meshes": [obj.name for obj in meshes],
        "material_face_counts": assignments,
        "palette": {key: record["description"] for key, record in materials.items()},
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
