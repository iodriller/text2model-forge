"""Numerically validate embedded glTF skinning and animation key poses."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import trimesh


COMPONENTS = {5123: (np.uint16, 2), 5125: (np.uint32, 4), 5126: (np.float32, 4)}
WIDTHS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_glb(path):
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2 or total != len(data):
        raise ValueError("invalid GLB header")
    chunks, offset = {}, 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks[kind] = data[offset : offset + length]
        offset += length
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def accessor(doc, binary, index):
    item = doc["accessors"][index]
    view = doc["bufferViews"][item["bufferView"]]
    dtype, _ = COMPONENTS[item["componentType"]]
    width = WIDTHS[item["type"]]
    offset = view.get("byteOffset", 0) + item.get("byteOffset", 0)
    result = np.frombuffer(binary, dtype=dtype, count=item["count"] * width, offset=offset)
    return result.reshape(item["count"], width) if width > 1 else result.copy()


def quat_matrix(value):
    x, y, z, w = value
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def local_matrix(node, rotation=None, translation=None):
    matrix = quat_matrix(rotation if rotation is not None else node.get("rotation", [0, 0, 0, 1]))
    matrix[:3, 3] = translation if translation is not None else node.get("translation", [0, 0, 0])
    return matrix


def parents(doc):
    result = [-1] * len(doc["nodes"])
    for parent, node in enumerate(doc["nodes"]):
        for child in node.get("children", []):
            result[child] = parent
    return result


def pose_vertices(doc, binary, animation, keyframe_index=1):
    primitive = doc["meshes"][0]["primitives"][0]
    positions = accessor(doc, binary, primitive["attributes"]["POSITION"]).astype(np.float32)
    joints = accessor(doc, binary, primitive["attributes"]["JOINTS_0"]).astype(np.int32)
    weights = accessor(doc, binary, primitive["attributes"]["WEIGHTS_0"]).astype(np.float32)
    skin = doc["skins"][0]
    inverse = accessor(doc, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    overrides = {}
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        output = accessor(doc, binary, sampler["output"])
        overrides[(channel["target"]["node"], channel["target"]["path"])] = output[keyframe_index]
    parent_map = parents(doc)
    world = [np.eye(4, dtype=np.float32) for _ in doc["nodes"]]
    for index, node in enumerate(doc["nodes"]):
        local = local_matrix(
            node,
            overrides.get((index, "rotation")),
            overrides.get((index, "translation")),
        )
        parent = parent_map[index]
        world[index] = world[parent] @ local if parent >= 0 else local
    skin_matrices = np.array([world[node] @ inverse[i] for i, node in enumerate(skin["joints"])])
    homogeneous = np.column_stack([positions, np.ones(len(positions), dtype=np.float32)])
    posed = np.zeros((len(positions), 3), dtype=np.float32)
    for influence in range(4):
        transformed = np.einsum("nij,nj->ni", skin_matrices[joints[:, influence]], homogeneous)[:, :3]
        posed += transformed * weights[:, influence, None]
    return positions, posed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    doc, binary = load_glb(args.input)
    primitive = doc["meshes"][0]["primitives"][0]
    faces = accessor(doc, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
    reports = []
    for animation in doc.get("animations", []):
        base, posed = pose_vertices(doc, binary, animation)
        edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
        base_lengths = np.linalg.norm(base[edges[:, 0]] - base[edges[:, 1]], axis=1)
        posed_lengths = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
        meaningful = base_lengths > 1e-4
        ratio = posed_lengths[meaningful] / base_lengths[meaningful]
        base_mesh = trimesh.Trimesh(base, faces, process=False)
        posed_mesh = trimesh.Trimesh(posed, faces, process=False)
        reports.append(
            {
                "animation": animation["name"],
                "finite": bool(np.isfinite(posed).all()),
                "edge_stretch_p95": float(np.percentile(ratio, 95)),
                "edge_stretch_p99": float(np.percentile(ratio, 99)),
                "edge_stretch_max": float(ratio.max()),
                "severe_edge_fraction": float(np.mean(ratio > 3.0)),
                "volume_ratio": float(abs(posed_mesh.volume) / max(abs(base_mesh.volume), 1e-9)),
                "bounds": posed_mesh.bounds.tolist(),
            }
        )
    result = {
        "animations": reports,
        "hard_failures": [
            item["animation"]
            for item in reports
            if not item["finite"]
            or item["edge_stretch_p99"] > 3.0
            or item["severe_edge_fraction"] > 0.005
            or not 0.25 <= item["volume_ratio"] <= 2.0
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if result["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
