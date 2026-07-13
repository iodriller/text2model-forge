"""Generate an original connected, skinned short-biped GLB without Blender."""
from __future__ import annotations

import argparse
import io
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import trimesh
from skimage import measure


def ellipsoid_sdf(x, y, z, center, radii):
    q = np.sqrt(
        ((x - center[0]) / radii[0]) ** 2
        + ((y - center[1]) / radii[1]) ** 2
        + ((z - center[2]) / radii[2]) ** 2
    )
    return (q - 1.0) * min(radii)


def capsule_sdf(x, y, z, start, end, radius):
    px, py, pz = x - start[0], y - start[1], z - start[2]
    bx, by, bz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
    denominator = bx * bx + by * by + bz * bz
    h = np.clip((px * bx + py * by + pz * bz) / denominator, 0.0, 1.0)
    return np.sqrt((px - bx * h) ** 2 + (py - by * h) ** 2 + (pz - bz * h) ** 2) - radius


JOINTS = [
    ("hips", -1, (0.0, 0.78, 0.0)),
    ("spine", 0, (0.0, 1.02, 0.0)),
    ("chest", 1, (0.0, 1.30, 0.0)),
    ("neck", 2, (0.0, 1.49, 0.0)),
    ("head", 3, (0.0, 1.69, 0.0)),
    ("shoulder_l", 2, (-0.29, 1.36, 0.0)),
    ("elbow_l", 5, (-0.52, 1.14, 0.0)),
    ("wrist_l", 6, (-0.68, 0.93, 0.0)),
    ("hand_l", 7, (-0.73, 0.84, 0.0)),
    ("shoulder_r", 2, (0.29, 1.36, 0.0)),
    ("elbow_r", 9, (0.52, 1.14, 0.0)),
    ("wrist_r", 10, (0.68, 0.93, 0.0)),
    ("hand_r", 11, (0.73, 0.84, 0.0)),
    ("hip_l", 0, (-0.17, 0.73, 0.0)),
    ("knee_l", 13, (-0.18, 0.39, 0.0)),
    ("ankle_l", 14, (-0.18, 0.10, 0.0)),
    ("foot_l", 15, (-0.18, 0.05, 0.14)),
    ("hip_r", 0, (0.17, 0.73, 0.0)),
    ("knee_r", 17, (0.18, 0.39, 0.0)),
    ("ankle_r", 18, (0.18, 0.10, 0.0)),
    ("foot_r", 19, (0.18, 0.05, 0.14)),
]


def generate_mesh(resolution=(96, 112, 72)):
    mins = np.array([-0.92, -0.08, -0.48], dtype=np.float32)
    maxs = np.array([0.92, 2.06, 0.58], dtype=np.float32)
    axes = [np.linspace(mins[i], maxs[i], resolution[i], dtype=np.float32) for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    fields = [
        ellipsoid_sdf(x, y, z, (0, 1.16, 0), (0.38, 0.47, 0.25)),
        ellipsoid_sdf(x, y, z, (0, 0.78, 0), (0.33, 0.28, 0.24)),
        capsule_sdf(x, y, z, (0, 1.35, 0), (0, 1.53, 0), 0.18),
        ellipsoid_sdf(x, y, z, (0, 1.72, 0), (0.31, 0.34, 0.29)),
        ellipsoid_sdf(x, y, z, (-0.36, 1.73, 0), (0.25, 0.10, 0.11)),
        ellipsoid_sdf(x, y, z, (0.36, 1.73, 0), (0.25, 0.10, 0.11)),
    ]
    for sign in (-1, 1):
        shoulder = (sign * 0.29, 1.36, 0)
        elbow = (sign * 0.52, 1.14, 0)
        wrist = (sign * 0.68, 0.93, 0)
        fields.extend(
            [
                capsule_sdf(x, y, z, shoulder, elbow, 0.145),
                capsule_sdf(x, y, z, elbow, wrist, 0.12),
                ellipsoid_sdf(x, y, z, (sign * 0.73, 0.84, 0), (0.13, 0.16, 0.10)),
            ]
        )
        hip = (sign * 0.17, 0.73, 0)
        knee = (sign * 0.18, 0.39, 0)
        ankle = (sign * 0.18, 0.10, 0)
        fields.extend(
            [
                capsule_sdf(x, y, z, hip, knee, 0.175),
                capsule_sdf(x, y, z, knee, ankle, 0.135),
                ellipsoid_sdf(x, y, z, (sign * 0.18, 0.06, 0.14), (0.17, 0.11, 0.28)),
            ]
        )
    sdf = np.minimum.reduce(fields)
    spacing = tuple((maxs - mins) / (np.array(resolution) - 1))
    vertices, faces, normals, _ = measure.marching_cubes(sdf, level=0, spacing=spacing)
    vertices = vertices + mins
    faces = np.ascontiguousarray(faces[:, ::-1], dtype=np.uint32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh


def skin_weights(vertices, joint_specs=JOINTS):
    positions = np.array([joint[2] for joint in joint_specs], dtype=np.float32)
    distances = np.linalg.norm(vertices[:, None, :] - positions[None, :, :], axis=2)
    nearest = np.argpartition(distances, 4, axis=1)[:, :4]
    nearest_distances = np.take_along_axis(distances, nearest, axis=1)
    raw = 1.0 / np.maximum(nearest_distances, 0.035) ** 3
    weights = raw / raw.sum(axis=1, keepdims=True)
    return nearest.astype(np.uint16), weights.astype(np.float32)


def load_target_mesh(path):
    loaded = trimesh.load(io.BytesIO(Path(path).read_bytes()), file_type="glb", force="scene")
    return trimesh.util.concatenate(tuple(loaded.dump()))


def cross_section_fit(mesh, target, *, profile=None, blend=0.72):
    """Fit a connected owned topology to target cross-section envelopes.

    This intentionally transfers only coarse shape. It cannot copy target
    topology or disconnected internal parts, and therefore preserves the
    canonical vertex/face graph and deformation contract.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    source_bounds = np.asarray(mesh.bounds, dtype=np.float32)
    target_vertices = np.asarray(target.vertices, dtype=np.float32)
    target_bounds = np.asarray(target.bounds, dtype=np.float32)
    if profile is None:
        levels = np.linspace(source_bounds[0, 1], source_bounds[1, 1], 33, dtype=np.float32)
        target_levels = target_bounds[0, 1] + (
            (levels - source_bounds[0, 1]) / max(np.ptp(source_bounds[:, 1]), 1e-6)
        ) * np.ptp(target_bounds[:, 1])
        source_values, target_values = [], []
        source_window = np.ptp(source_bounds[:, 1]) / 20
        target_window = np.ptp(target_bounds[:, 1]) / 20
        for source_y, target_y in zip(levels, target_levels):
            source_slice = vertices[np.abs(vertices[:, 1] - source_y) <= source_window]
            target_slice = target_vertices[np.abs(target_vertices[:, 1] - target_y) <= target_window]
            if len(source_slice) < 8:
                source_slice = vertices
            if len(target_slice) < 8:
                target_slice = target_vertices

            def envelope(points):
                low = np.quantile(points[:, [0, 2]], 0.01, axis=0)
                high = np.quantile(points[:, [0, 2]], 0.99, axis=0)
                return (low + high) / 2, np.maximum((high - low) / 2, 0.025)

            source_values.append(envelope(source_slice))
            target_values.append(envelope(target_slice))
        profile = {
            "levels": levels,
            "target_levels": target_levels,
            "source_centers": np.array([item[0] for item in source_values]),
            "source_radii": np.array([item[1] for item in source_values]),
            "target_centers": np.array([item[0] for item in target_values]),
            "target_radii": np.array([item[1] for item in target_values]),
        }

    def interpolate(values, y):
        return np.column_stack(
            [np.interp(y, profile["levels"], values[:, axis]) for axis in range(values.shape[1])]
        )

    source_center = interpolate(profile["source_centers"], vertices[:, 1])
    source_radius = interpolate(profile["source_radii"], vertices[:, 1])
    target_center = interpolate(profile["target_centers"], vertices[:, 1])
    target_radius = interpolate(profile["target_radii"], vertices[:, 1])
    scale = np.clip(target_radius / np.maximum(source_radius, 0.025), 0.60, 1.45)
    xz = vertices[:, [0, 2]]
    fitted_xz = target_center + (xz - source_center) * scale
    vertices[:, [0, 2]] = xz * (1 - blend) + fitted_xz * blend
    normalized_y = (vertices[:, 1] - profile["levels"][0]) / max(np.ptp(profile["levels"]), 1e-6)
    fitted_y = profile["target_levels"][0] + normalized_y * np.ptp(profile["target_levels"])
    vertices[:, 1] = vertices[:, 1] * (1 - blend) + fitted_y * blend
    fitted = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    fitted.remove_unreferenced_vertices()
    return fitted, profile


def fit_joint_specs(joint_specs, profile, *, blend=0.72):
    points = np.array([item[2] for item in joint_specs], dtype=np.float32)

    def interpolate(values):
        return np.column_stack(
            [np.interp(points[:, 1], profile["levels"], values[:, axis]) for axis in range(values.shape[1])]
        )

    source_center = interpolate(profile["source_centers"])
    source_radius = interpolate(profile["source_radii"])
    target_center = interpolate(profile["target_centers"])
    target_radius = interpolate(profile["target_radii"])
    scale = np.clip(target_radius / np.maximum(source_radius, 0.025), 0.60, 1.45)
    fitted_xz = target_center + (points[:, [0, 2]] - source_center) * scale
    points[:, [0, 2]] = points[:, [0, 2]] * (1 - blend) + fitted_xz * blend
    normalized_y = (points[:, 1] - profile["levels"][0]) / max(np.ptp(profile["levels"]), 1e-6)
    fitted_y = profile["target_levels"][0] + normalized_y * np.ptp(profile["target_levels"])
    points[:, 1] = points[:, 1] * (1 - blend) + fitted_y * blend
    return [(name, parent, tuple(float(value) for value in points[index])) for index, (name, parent, _) in enumerate(joint_specs)]


def quat(axis, degrees):
    radians = math.radians(degrees) / 2
    scale = math.sin(radians)
    return [axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(radians)]


class GlbBuilder:
    def __init__(self):
        self.binary = bytearray()
        self.views = []
        self.accessors = []

    def accessor(self, array, component_type, accessor_type, *, target=None, include_bounds=False):
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        data = np.ascontiguousarray(array).tobytes()
        self.binary.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        view_index = len(self.views)
        self.views.append(view)
        count = int(len(array))
        accessor = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
        }
        if include_bounds:
            accessor["min"] = np.asarray(array).min(axis=0).tolist()
            accessor["max"] = np.asarray(array).max(axis=0).tolist()
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index

    def write(self, path, document):
        document["buffers"] = [{"byteLength": len(self.binary)}]
        document["bufferViews"] = self.views
        document["accessors"] = self.accessors
        json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        binary = bytes(self.binary) + b"\0" * ((4 - len(self.binary) % 4) % 4)
        total = 12 + 8 + len(json_bytes) + 8 + len(binary)
        with Path(path).open("wb") as stream:
            stream.write(struct.pack("<III", 0x46546C67, 2, total))
            stream.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
            stream.write(json_bytes)
            stream.write(struct.pack("<II", len(binary), 0x004E4942))
            stream.write(binary)


def build_glb(mesh, path, joint_specs=JOINTS):
    builder = GlbBuilder()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)
    joints, weights = skin_weights(vertices, joint_specs)
    colors = np.zeros((len(vertices), 4), dtype=np.float32)
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    normalized_y = (vertices[:, 1] - bounds[0, 1]) / max(np.ptp(bounds[:, 1]), 1e-6)
    normalized_x = np.abs(vertices[:, 0] - bounds[:, 0].mean()) / max(np.ptp(bounds[:, 0]), 1e-6)
    skin_region = (normalized_y > 0.68) | (normalized_x > 0.23) | (normalized_y < 0.38)
    colors[:] = (0.20, 0.12, 0.07, 1.0)
    colors[skin_region] = (0.32, 0.48, 0.18, 1.0)
    position_accessor = builder.accessor(vertices, 5126, "VEC3", target=34962, include_bounds=True)
    normal_accessor = builder.accessor(normals, 5126, "VEC3", target=34962)
    joint_accessor = builder.accessor(joints, 5123, "VEC4", target=34962)
    weight_accessor = builder.accessor(weights, 5126, "VEC4", target=34962)
    color_accessor = builder.accessor(colors, 5126, "VEC4", target=34962)
    index_accessor = builder.accessor(faces, 5125, "SCALAR", target=34963)

    world = np.array([joint[2] for joint in joint_specs], dtype=np.float32)
    inverse_bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(joint_specs), axis=0)
    inverse_bind[:, 3, :3] = -world
    inverse_accessor = builder.accessor(inverse_bind, 5126, "MAT4")

    nodes = []
    for index, (name, parent, position) in enumerate(joint_specs):
        local = np.array(position) - (world[parent] if parent >= 0 else 0)
        nodes.append({"name": name, "translation": local.tolist(), "children": []})
    for index, (_, parent, _) in enumerate(joint_specs):
        if parent >= 0:
            nodes[parent]["children"].append(index)
    for node in nodes:
        if not node["children"]:
            node.pop("children")
    mesh_node = len(nodes)
    nodes.append({"name": "canonical_short_biped", "mesh": 0, "skin": 0})

    animations = []
    name_to_index = {joint[0]: index for index, joint in enumerate(joint_specs)}

    def animation(name, tracks):
        times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        time_accessor = builder.accessor(times, 5126, "SCALAR", include_bounds=True)
        samplers, channels = [], []
        for joint_name, path_name, values in tracks:
            values = np.asarray(values, dtype=np.float32)
            output_accessor = builder.accessor(values, 5126, "VEC4" if path_name == "rotation" else "VEC3")
            sampler_index = len(samplers)
            samplers.append({"input": time_accessor, "output": output_accessor, "interpolation": "LINEAR"})
            channels.append({"sampler": sampler_index, "target": {"node": name_to_index[joint_name], "path": path_name}})
        animations.append({"name": name, "samplers": samplers, "channels": channels})

    identity = quat((1, 0, 0), 0)
    animation("idle", [("hips", "translation", [[0, 0, 0], [0, 0.015, 0], [0, 0, 0]])])
    animation(
        "walk",
        [
            ("hip_l", "rotation", [quat((1, 0, 0), 25), quat((1, 0, 0), -25), quat((1, 0, 0), 25)]),
            ("hip_r", "rotation", [quat((1, 0, 0), -25), quat((1, 0, 0), 25), quat((1, 0, 0), -25)]),
            ("shoulder_l", "rotation", [quat((1, 0, 0), -18), quat((1, 0, 0), 18), quat((1, 0, 0), -18)]),
            ("shoulder_r", "rotation", [quat((1, 0, 0), 18), quat((1, 0, 0), -18), quat((1, 0, 0), 18)]),
        ],
    )
    animation("attack", [("shoulder_r", "rotation", [identity, quat((1, 0, 0), -52), quat((1, 0, 0), 18)])])
    animation("hit", [("chest", "rotation", [identity, quat((0, 0, 1), 14), identity])])
    animation("death", [("hips", "rotation", [identity, quat((0, 0, 1), 45), quat((0, 0, 1), 90)])])

    document = {
        "asset": {"version": "2.0", "generator": "Asset Forge Darkness canonical short-biped v1"},
        "scene": 0,
        "scenes": [{"nodes": [0, mesh_node]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "canonical_short_biped_mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "JOINTS_0": joint_accessor,
                            "WEIGHTS_0": weight_accessor,
                            "COLOR_0": color_accessor,
                        },
                        "indices": index_accessor,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [{"name": "vertex_color", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.9}}],
        "skins": [{"name": "canonical_short_biped_rig", "inverseBindMatrices": inverse_accessor, "skeleton": 0, "joints": list(range(len(joint_specs)))}],
        "animations": animations,
    }
    builder.write(path, document)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    started = time.monotonic()
    try:
        output_root = Path(request["output_directory"]).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        mesh = generate_mesh()
        original_extents = np.asarray(mesh.extents, dtype=np.float64)
        joint_specs = JOINTS
        target_fit = "not_requested"
        profile = None
        fit_metrics = {"method": target_fit, "target_artifact_id": None}
        if request.get("input_paths"):
            target_path = next(iter(request["input_paths"].values()))
            target = load_target_mesh(target_path)
            mesh, profile = cross_section_fit(mesh, target)
            joint_specs = fit_joint_specs(JOINTS, profile)
            target_fit = "cross_section_envelope_v1"
            target_extents = np.asarray(target.extents, dtype=np.float64)
            fitted_extents = np.asarray(mesh.extents, dtype=np.float64)
            before_error = np.abs(original_extents - target_extents) / np.maximum(target_extents, 1e-6)
            after_error = np.abs(fitted_extents - target_extents) / np.maximum(target_extents, 1e-6)
            input_record = request.get("inputs", [{}])[0]
            fit_metrics = {
                "method": target_fit,
                "target_artifact_id": input_record.get("artifact_id"),
                "target_sha256": input_record.get("sha256"),
                "target_connected_components": int(len(target.split(only_watertight=False))),
                "original_extents": original_extents.tolist(),
                "target_extents": target_extents.tolist(),
                "fitted_extents": fitted_extents.tolist(),
                "mean_relative_extent_error_before": float(before_error.mean()),
                "mean_relative_extent_error_after": float(after_error.mean()),
                "topology_preserved": True,
                "fine_detail_transferred": False,
            }
        glb_path = output_root / "canonical_short_biped_v1.glb"
        build_glb(mesh, glb_path, joint_specs)
        lod1_path = output_root / "canonical_short_biped_v1_lod1.glb"
        lod2_path = output_root / "canonical_short_biped_v1_lod2.glb"
        lod1 = generate_mesh((72, 84, 54))
        lod2 = generate_mesh((48, 56, 36))
        if profile is not None:
            lod1, _ = cross_section_fit(lod1, target, profile=profile)
            lod2, _ = cross_section_fit(lod2, target, profile=profile)
        build_glb(lod1, lod1_path, joint_specs)
        build_glb(lod2, lod2_path, joint_specs)
        rig_path = output_root / "rig_contract.json"
        rig_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "anatomy_family": "short_biped_v1",
                    "joints": [
                        {"name": name, "parent": JOINTS[parent][0] if parent >= 0 else None, "world_position": position}
                        for name, parent, position in joint_specs
                    ],
                    "sockets": {
                        "hand_right": {"joint": "hand_r", "local_position": [0, 0, 0]},
                        "hand_left": {"joint": "hand_l", "local_position": [0, 0, 0]},
                        "projectile_origin": {"joint": "chest", "local_position": [0, 0.08, 0.18]},
                    },
                    "contacts": {"foot_left": "foot_l", "foot_right": "foot_r"},
                    "animations": ["idle", "walk", "attack", "hit", "death"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        physics_path = output_root / "physics_contract.json"
        physics_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "body_colliders": [
                        {"type": "capsule", "joint": "hips", "radius": 0.28, "height": 0.62, "axis": "y"},
                        {"type": "capsule", "joint": "chest", "radius": 0.30, "height": 0.55, "axis": "y"},
                        {"type": "sphere", "joint": "head", "radius": 0.32},
                    ],
                    "hitboxes": [
                        {"id": "body", "joint": "spine", "shape": "capsule"},
                        {"id": "head", "joint": "head", "shape": "sphere"},
                    ],
                    "renderer_bounds_padding": 0.25,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        fit_path = output_root / "fit_contract.json"
        fit_path.write_text(json.dumps({"schema_version": 1, **fit_metrics}, indent=2), encoding="utf-8")
        loaded = trimesh.load(glb_path, force="scene")
        combined = trimesh.util.concatenate(list(loaded.geometry.values()))
        response = {
            "schema_version": 1,
            "job_id": request["job_id"],
            "status": "succeeded",
            "outputs": [
                {"path": str(glb_path), "media_type": "model/gltf-binary", "role": "canonical_master", "metadata": {"anatomy_family": "short_biped_v1", "animations": ["idle", "walk", "attack", "hit", "death"]}},
                {"path": str(lod1_path), "media_type": "model/gltf-binary", "role": "lod1", "metadata": {"level": 1}},
                {"path": str(lod2_path), "media_type": "model/gltf-binary", "role": "lod2", "metadata": {"level": 2}},
                {"path": str(rig_path), "media_type": "application/json", "role": "rig_contract", "metadata": {}},
                {"path": str(physics_path), "media_type": "application/json", "role": "physics_contract", "metadata": {}},
                {"path": str(fit_path), "media_type": "application/json", "role": "target_fit_contract", "metadata": {}},
            ],
            "diagnostics": {
                "vertices": int(len(combined.vertices)),
                "faces": int(len(combined.faces)),
                "watertight": bool(combined.is_watertight),
                "components": int(len(combined.split(only_watertight=False))),
                "joints": len(joint_specs),
                "animations": 5,
                "weights_normalized": True,
                "target_fit": target_fit,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            "warnings": ["Cross-section fitting transfers coarse target envelopes only; details, UVs, and component semantics are not transferred."],
        }
    except Exception as exc:
        response = {"schema_version": 1, "job_id": request.get("job_id", "unknown"), "status": "failed", "outputs": [], "diagnostics": {}, "warnings": [], "error": f"{type(exc).__name__}: {exc}"}
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
