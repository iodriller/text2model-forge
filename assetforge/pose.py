"""Deterministic pose packs: authored OpenPose skeletons for game animation frames.

A pose pack holds keyframed COCO-18 skeletons per action. Frames are tweened between
keyframes and rendered procedurally to OpenPose-style control images, so no pose
estimation model is involved: poses are authored data, hash-recorded in provenance,
and editable per workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .core import SCHEMA_VERSION, ForgeError, read_json, sha256_json, write_json

KEYPOINTS = (
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
)

# Standard OpenPose body-18 limb pairs and colors, used by SDXL openpose ControlNets.
LIMBS = (
    ("neck", "r_shoulder"), ("neck", "l_shoulder"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("neck", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("neck", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("neck", "nose"), ("nose", "r_eye"), ("r_eye", "r_ear"),
    ("nose", "l_eye"), ("l_eye", "l_ear"),
)
LIMB_COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 170), (255, 0, 85),
)
JOINT_COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 170), (255, 0, 85), (255, 0, 255),
)

Pose = dict[str, list[float] | None]


def _pose(base: Pose, **overrides: list[float] | None) -> Pose:
    result = {name: (list(value) if value is not None else None) for name, value in base.items()}
    for name, value in overrides.items():
        if name not in KEYPOINTS:
            raise ForgeError(f"Unknown keypoint {name!r}")
        result[name] = list(value) if value is not None else None
    return result


# Side view, character faces EAST (screen right). West is flipX in the engine.
_BASE: Pose = {
    "nose": [0.565, 0.235], "neck": [0.500, 0.320],
    "r_shoulder": [0.487, 0.325], "r_elbow": [0.472, 0.420], "r_wrist": [0.498, 0.500],
    "l_shoulder": [0.515, 0.315], "l_elbow": [0.532, 0.410], "l_wrist": [0.560, 0.490],
    "r_hip": [0.487, 0.550], "r_knee": [0.478, 0.700], "r_ankle": [0.468, 0.860],
    "l_hip": [0.515, 0.550], "l_knee": [0.528, 0.700], "l_ankle": [0.545, 0.860],
    "r_eye": [0.585, 0.215], "l_eye": None, "r_ear": [0.545, 0.222], "l_ear": None,
}


def default_pack() -> dict[str, Any]:
    idle_down = _pose(
        _BASE,
        nose=[0.565, 0.245], neck=[0.500, 0.330], r_eye=[0.585, 0.225], r_ear=[0.545, 0.232],
        r_shoulder=[0.487, 0.335], l_shoulder=[0.515, 0.325],
        r_wrist=[0.494, 0.512], l_wrist=[0.556, 0.502],
    )
    walk_a = _pose(
        _BASE,
        l_knee=[0.560, 0.690], l_ankle=[0.600, 0.845],
        r_knee=[0.452, 0.705], r_ankle=[0.408, 0.865],
        l_wrist=[0.462, 0.500], l_elbow=[0.492, 0.415],
        r_wrist=[0.560, 0.470], r_elbow=[0.512, 0.402],
        neck=[0.503, 0.315], nose=[0.568, 0.230], r_eye=[0.588, 0.210], r_ear=[0.548, 0.217],
    )
    walk_pass = _pose(
        _BASE,
        neck=[0.500, 0.308], nose=[0.565, 0.223], r_eye=[0.585, 0.203], r_ear=[0.545, 0.210],
        r_knee=[0.492, 0.688], r_ankle=[0.482, 0.852],
        l_knee=[0.522, 0.700], l_ankle=[0.530, 0.862],
    )
    walk_b = _pose(
        _BASE,
        r_knee=[0.552, 0.690], r_ankle=[0.592, 0.845],
        l_knee=[0.460, 0.705], l_ankle=[0.416, 0.865],
        r_wrist=[0.458, 0.502], r_elbow=[0.488, 0.418],
        l_wrist=[0.565, 0.468], l_elbow=[0.518, 0.400],
        neck=[0.503, 0.315], nose=[0.568, 0.230], r_eye=[0.588, 0.210], r_ear=[0.548, 0.217],
    )
    attack_windup = _pose(
        _BASE,
        neck=[0.482, 0.318], nose=[0.545, 0.238], r_eye=[0.565, 0.218], r_ear=[0.525, 0.225],
        r_elbow=[0.430, 0.360], r_wrist=[0.400, 0.268],
        l_elbow=[0.522, 0.408], l_wrist=[0.565, 0.452],
        l_knee=[0.545, 0.695], l_ankle=[0.572, 0.858],
    )
    attack_strike = _pose(
        _BASE,
        neck=[0.532, 0.330], nose=[0.600, 0.252], r_eye=[0.618, 0.232], r_ear=[0.578, 0.240],
        r_elbow=[0.610, 0.360], r_wrist=[0.700, 0.408],
        l_elbow=[0.548, 0.420], l_wrist=[0.520, 0.480],
        l_knee=[0.585, 0.700], l_ankle=[0.628, 0.858],
        r_knee=[0.462, 0.712], r_ankle=[0.415, 0.868],
        r_hip=[0.500, 0.555], l_hip=[0.528, 0.555],
    )
    shoot_raise = _pose(
        _BASE,
        l_elbow=[0.578, 0.348], l_wrist=[0.652, 0.330],
        r_elbow=[0.508, 0.352], r_wrist=[0.562, 0.332],
    )
    shoot_draw = _pose(
        _BASE,
        neck=[0.495, 0.318],
        l_elbow=[0.582, 0.345], l_wrist=[0.660, 0.328],
        r_elbow=[0.462, 0.348], r_wrist=[0.505, 0.330],
    )
    shoot_release = _pose(
        _BASE,
        l_elbow=[0.582, 0.345], l_wrist=[0.662, 0.326],
        r_elbow=[0.448, 0.360], r_wrist=[0.462, 0.320],
    )
    defend_brace = _pose(
        _BASE,
        neck=[0.492, 0.328], nose=[0.552, 0.248], r_eye=[0.572, 0.228], r_ear=[0.532, 0.235],
        l_elbow=[0.552, 0.375], l_wrist=[0.610, 0.330],
        r_elbow=[0.475, 0.408], r_wrist=[0.520, 0.372],
        l_knee=[0.548, 0.700], l_ankle=[0.575, 0.860],
        r_knee=[0.462, 0.708], r_ankle=[0.432, 0.865],
    )
    cast_gather = _pose(
        _BASE,
        neck=[0.488, 0.322],
        r_elbow=[0.452, 0.395], r_wrist=[0.478, 0.320],
        l_elbow=[0.535, 0.390], l_wrist=[0.512, 0.318],
    )
    cast_thrust = _pose(
        _BASE,
        neck=[0.522, 0.325], nose=[0.588, 0.245], r_eye=[0.608, 0.225], r_ear=[0.568, 0.232],
        r_elbow=[0.588, 0.368], r_wrist=[0.672, 0.352],
        l_elbow=[0.598, 0.358], l_wrist=[0.678, 0.375],
        l_knee=[0.562, 0.700], l_ankle=[0.598, 0.858],
        r_knee=[0.468, 0.710], r_ankle=[0.428, 0.865],
    )
    hit_recoil = _pose(
        _BASE,
        neck=[0.472, 0.332], nose=[0.525, 0.255], r_eye=[0.545, 0.235], r_ear=[0.505, 0.242],
        r_wrist=[0.520, 0.452], r_elbow=[0.470, 0.408],
        l_wrist=[0.585, 0.445], l_elbow=[0.545, 0.400],
        r_hip=[0.478, 0.555], l_hip=[0.506, 0.555],
    )
    death_stagger = _pose(
        _BASE,
        neck=[0.458, 0.395], nose=[0.505, 0.320], r_eye=[0.525, 0.300], r_ear=[0.488, 0.308],
        r_hip=[0.472, 0.600], l_hip=[0.500, 0.600],
        r_knee=[0.442, 0.735], l_knee=[0.512, 0.740],
        r_ankle=[0.428, 0.872], l_ankle=[0.548, 0.870],
        r_wrist=[0.428, 0.520], l_wrist=[0.552, 0.512],
    )
    death_falling = _pose(
        _BASE,
        neck=[0.408, 0.620], nose=[0.352, 0.588], r_eye=[0.362, 0.568], r_ear=[0.392, 0.585],
        r_shoulder=[0.402, 0.628], l_shoulder=[0.418, 0.612],
        r_elbow=[0.372, 0.700], r_wrist=[0.340, 0.760],
        l_elbow=[0.398, 0.690], l_wrist=[0.372, 0.752],
        r_hip=[0.512, 0.680], l_hip=[0.528, 0.668],
        r_knee=[0.565, 0.762], l_knee=[0.585, 0.748],
        r_ankle=[0.612, 0.852], l_ankle=[0.635, 0.840],
    )
    death_down = _pose(
        _BASE,
        nose=[0.298, 0.845], neck=[0.362, 0.838],
        r_eye=[0.292, 0.828], r_ear=[0.318, 0.832],
        r_shoulder=[0.368, 0.845], l_shoulder=[0.372, 0.828],
        r_elbow=[0.330, 0.878], r_wrist=[0.288, 0.888],
        l_elbow=[0.412, 0.795], l_wrist=[0.452, 0.778],
        r_hip=[0.520, 0.842], l_hip=[0.525, 0.830],
        r_knee=[0.588, 0.858], l_knee=[0.598, 0.842],
        r_ankle=[0.658, 0.870], l_ankle=[0.668, 0.852],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "humanoid_side_east",
        "view": "east",
        "canvas": [768, 768],
        "bone_width_fraction": 0.016,
        "joint_radius_fraction": 0.009,
        "actions": {
            "idle": {
                "frames": 4, "loop": True,
                "keyframes": [{"t": 0.0, "pose": _BASE}, {"t": 0.5, "pose": idle_down}],
            },
            "walk": {
                "frames": 6, "loop": True,
                "keyframes": [
                    {"t": 0.0, "pose": walk_a}, {"t": 0.25, "pose": walk_pass},
                    {"t": 0.5, "pose": walk_b}, {"t": 0.75, "pose": walk_pass},
                ],
            },
            "attack": {
                "frames": 6, "loop": False,
                "keyframes": [
                    {"t": 0.0, "pose": _BASE}, {"t": 0.35, "pose": attack_windup},
                    {"t": 0.6, "pose": attack_strike}, {"t": 1.0, "pose": _BASE},
                ],
            },
            "shoot": {
                "frames": 6, "loop": False,
                "keyframes": [
                    {"t": 0.0, "pose": _BASE}, {"t": 0.25, "pose": shoot_raise},
                    {"t": 0.5, "pose": shoot_draw}, {"t": 0.7, "pose": shoot_release},
                    {"t": 1.0, "pose": _BASE},
                ],
            },
            "defend": {
                "frames": 4, "loop": False,
                "keyframes": [
                    {"t": 0.0, "pose": _BASE}, {"t": 0.35, "pose": defend_brace},
                    {"t": 0.7, "pose": defend_brace}, {"t": 1.0, "pose": _BASE},
                ],
            },
            "cast": {
                "frames": 6, "loop": False,
                "keyframes": [
                    {"t": 0.0, "pose": _BASE}, {"t": 0.35, "pose": cast_gather},
                    {"t": 0.65, "pose": cast_thrust}, {"t": 1.0, "pose": _BASE},
                ],
            },
            "hit": {
                "frames": 2, "loop": False,
                "keyframes": [{"t": 0.0, "pose": hit_recoil}, {"t": 1.0, "pose": _BASE}],
            },
            "death": {
                "frames": 6, "loop": False,
                "keyframes": [
                    {"t": 0.0, "pose": _BASE}, {"t": 0.3, "pose": death_stagger},
                    {"t": 0.65, "pose": death_falling}, {"t": 1.0, "pose": death_down},
                ],
            },
        },
    }


def pack_path(workspace: Path, pack_id: str) -> Path:
    return workspace.resolve() / "poses" / f"{pack_id}.pose.json"


def load_pack(workspace: Path, pack_id: str = "humanoid_side_east") -> dict[str, Any]:
    path = pack_path(workspace, pack_id)
    if path.is_file():
        pack = read_json(path)
        if pack_id == "humanoid_side_east":
            # Upgrade-safe: newly shipped actions appear without clobbering local edits.
            for name, action in default_pack()["actions"].items():
                pack.setdefault("actions", {}).setdefault(name, action)
        return pack
    if pack_id == "humanoid_side_east":
        return default_pack()
    raise ForgeError(f"Pose pack {pack_id!r} was not found at {path}")


def materialize_pack(workspace: Path, pack_id: str = "humanoid_side_east") -> Path:
    path = pack_path(workspace, pack_id)
    if not path.is_file():
        write_json(path, load_pack(workspace, pack_id), overwrite=False)
    return path


def pack_sha256(pack: dict[str, Any]) -> str:
    return sha256_json(pack)


def standing_extent(pack: dict[str, Any]) -> tuple[float, float]:
    """(head_y, ankle_y) canvas fractions of the standing figure, from the idle keyframes."""
    actions = pack.get("actions", {})
    source = actions.get("idle") or next(iter(actions.values()), None)
    if not source:
        raise ForgeError("Pose pack has no actions to derive the standing extent from")
    tops: list[float] = []
    ankles: list[float] = []
    for keyframe in source["keyframes"]:
        pose = keyframe["pose"]
        for name, value in pose.items():
            if value:
                tops.append(value[1])
        for name in ("r_ankle", "l_ankle"):
            if pose.get(name):
                ankles.append(pose[name][1])
    if not tops or not ankles:
        raise ForgeError("Pose pack keyframes have no usable keypoints")
    return min(tops), max(ankles)


def _tween_point(a: list[float] | None, b: list[float] | None, t: float) -> list[float] | None:
    if a is None and b is None:
        return None
    if a is None:
        return list(b) if t >= 0.5 else None
    if b is None:
        return list(a) if t < 0.5 else None
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def tween_pose(a: Pose, b: Pose, t: float) -> Pose:
    return {name: _tween_point(a.get(name), b.get(name), t) for name in KEYPOINTS}


def frame_pose(action: dict[str, Any], index: int) -> Pose:
    frames = int(action["frames"])
    if not 0 <= index < frames:
        raise ForgeError(f"Frame index {index} outside 0..{frames - 1}")
    keyframes = sorted(action["keyframes"], key=lambda item: float(item["t"]))
    if not keyframes:
        raise ForgeError("Action has no keyframes")
    loop = bool(action.get("loop", False))
    if frames == 1:
        t = 0.0
    else:
        t = index / frames if loop else index / (frames - 1)
    if loop and keyframes[-1]["t"] < 1.0:
        keyframes = keyframes + [{"t": 1.0, "pose": keyframes[0]["pose"]}]
    previous = keyframes[0]
    for keyframe in keyframes:
        if float(keyframe["t"]) <= t:
            previous = keyframe
            continue
        span = float(keyframe["t"]) - float(previous["t"])
        local = 0.0 if span <= 0 else (t - float(previous["t"])) / span
        return tween_pose(previous["pose"], keyframe["pose"], local)
    return {name: (list(value) if value is not None else None) for name, value in previous["pose"].items()}


def render_pose(pose: Pose, canvas: tuple[int, int], bone_width_fraction: float = 0.016,
                joint_radius_fraction: float = 0.009) -> Image.Image:
    width, height = canvas
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    bone_width = max(3, round(min(width, height) * bone_width_fraction))
    joint_radius = max(2, round(min(width, height) * joint_radius_fraction))

    def point(name: str) -> tuple[float, float] | None:
        value = pose.get(name)
        if value is None:
            return None
        return (value[0] * width, value[1] * height)

    for (start, end), color in zip(LIMBS, LIMB_COLORS):
        a, b = point(start), point(end)
        if a is None or b is None:
            continue
        draw.line([a, b], fill=color, width=bone_width)
    for name, color in zip(KEYPOINTS, JOINT_COLORS):
        value = point(name)
        if value is None:
            continue
        x, y = value
        draw.ellipse([x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius], fill=color)
    return image


def render_action_frames(pack: dict[str, Any], action_name: str) -> list[Image.Image]:
    actions = pack.get("actions", {})
    if action_name not in actions:
        raise ForgeError(f"Pose pack {pack.get('id')!r} has no action {action_name!r}; available: {sorted(actions)}")
    action = actions[action_name]
    canvas = tuple(int(value) for value in pack.get("canvas", [768, 768]))
    return [
        render_pose(
            frame_pose(action, index),
            canvas,
            float(pack.get("bone_width_fraction", 0.016)),
            float(pack.get("joint_radius_fraction", 0.009)),
        )
        for index in range(int(action["frames"]))
    ]
