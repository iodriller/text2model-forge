"""Per-action deformation QA: the gate that would have caught the ball-goblin.

Mechanical geometry validation (morphology.py) and the art critic (critic.py) both
judge a small number of static frames. Neither one runs the mesh through motion. A
construction where body parts are independent objects rigidly parented to individual
bones (rather than one skinned mesh) can pass every existing gate and still fall apart
the moment an action rotates a joint -- exactly what shipped undetected in the first
Creature DNA goblin (disconnected limbs on walk/taunt/death).

This module renders geometry-only frames of every action and checks that the
silhouette stays one dominant 2D blob and doesn't
balloon or shatter frame to frame, BEFORE any GPU time is spent on the texture-master
bake. It intentionally reuses the existing per-animation `baseline_tolerance` concept
(qa.py's silhouette-bottom drift check) rather than inventing a parallel notion of
"grounded" -- it just applies that same idea earlier, on raw geometry, at a different
working resolution (scaled proportionally). This is a render-space preflight, not
proof of connected 3D topology, correct skin weights, self-intersection freedom,
foot contact, or weapon grip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .core import utc_now, write_json

DEFAULTS = {
    "alpha_threshold": 128,
    "analysis_size": 160,               # downscale before analysis: cheap, still catches real defects
    "min_component_ratio": 0.94,        # largest connected blob must be this fraction of all foreground pixels
    "max_area_ratio_jump": 1.55,        # frame-to-frame foreground pixel count must stay within this ratio
    "grounded_baseline_tolerance": 14,  # px at analysis_size, used when an animation sets none of its own
    "exempt_from_grounding": ["dodge", "death"],  # actions that legitimately leave / collapse the ground plane
}


def _mask(frame: Image.Image, threshold: int, size: int) -> Image.Image:
    alpha = frame.getchannel("A").resize((size, size), Image.Resampling.NEAREST)
    return alpha.point(lambda a: 255 if a >= threshold else 0)


def _largest_component_ratio(mask: Image.Image) -> float:
    """Fraction of foreground pixels belonging to the single largest 4-connected blob."""
    width, height = mask.size
    data = mask.tobytes()
    total = sum(1 for value in data if value)
    if total == 0:
        return 0.0
    visited = bytearray(len(data))
    largest = 0
    for start in range(len(data)):
        if data[start] == 0 or visited[start]:
            continue
        stack = [start]
        visited[start] = 1
        size_count = 0
        while stack:
            index = stack.pop()
            size_count += 1
            y, x = divmod(index, width)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    if not visited[neighbor] and data[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)
        largest = max(largest, size_count)
    return largest / total


def _area(mask: Image.Image) -> float:
    return ImageStat.Stat(mask).sum[0] / 255.0


def _bottom(mask: Image.Image) -> int | None:
    box = mask.getbbox()
    return box[3] if box else None


def _frame_path(frames_root: Path, unit: str, action: str, direction: str, index: int) -> Path:
    return frames_root / unit / action / direction / f"{index:02d}.png"


def critique_deformation(config_path: Path, repo_root: Path, frames_root: Path, report_path: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    unit = str(config["id"])
    thresholds = dict(DEFAULTS)
    thresholds.update(config.get("deformation") or {})
    exempt = set(thresholds["exempt_from_grounding"])
    analysis_size = int(thresholds["analysis_size"])
    cell_height = int((config.get("cell_size") or [256, 256])[1]) or 256
    frames_root = Path(frames_root).resolve()

    directions = config.get("directions", [])
    silhouette = config.get("silhouette_contract")
    contract_views = set(silhouette.get("proof_views", [])) if isinstance(silhouette, dict) else set()
    check_directions = [d for d in directions if not contract_views or d in contract_views] or directions

    failures: list[dict[str, Any]] = []
    actions_report: list[dict[str, Any]] = []

    for animation, animation_config in config.get("animations", {}).items():
        frame_count = int(animation_config.get("frames", 0))
        if frame_count <= 0:
            continue
        raw_tolerance = int(animation_config.get("baseline_tolerance", thresholds["grounded_baseline_tolerance"]))
        action_tolerance = raw_tolerance * (analysis_size / cell_height)

        for direction in check_directions:
            areas: list[float] = []
            bottoms: list[int] = []
            component_ratios: list[float] = []

            for index in range(frame_count):
                path = _frame_path(frames_root, unit, animation, direction, index)
                if not path.is_file():
                    failures.append({
                        "animation": animation, "direction": direction, "frame": index,
                        "check": "exists", "detail": f"deformation-proof frame missing: {path}",
                    })
                    continue
                with Image.open(path) as source:
                    frame = source.convert("RGBA")
                mask = _mask(frame, int(thresholds["alpha_threshold"]), analysis_size)
                area = _area(mask)
                if area <= 0:
                    failures.append({
                        "animation": animation, "direction": direction, "frame": index,
                        "check": "content", "detail": "frame has no visible silhouette",
                    })
                    continue
                ratio = _largest_component_ratio(mask)
                areas.append(area)
                component_ratios.append(ratio)
                bottom = _bottom(mask)
                if bottom is not None:
                    bottoms.append(bottom)
                if ratio < thresholds["min_component_ratio"]:
                    failures.append({
                        "animation": animation, "direction": direction, "frame": index, "check": "connectivity",
                        "detail": (
                            f"largest connected silhouette piece is {ratio:.3f} of visible pixels "
                            f"(< {thresholds['min_component_ratio']}); the rendered silhouette has split"
                        ),
                    })

            for index in range(1, len(areas)):
                previous, current = areas[index - 1], areas[index]
                if previous <= 0:
                    continue
                jump = max(current / previous, previous / current)
                if jump > thresholds["max_area_ratio_jump"]:
                    failures.append({
                        "animation": animation, "direction": direction, "frame": index, "check": "stability",
                        "detail": (
                            f"silhouette area changed {jump:.2f}x between frames {index - 1} and {index} "
                            f"(> {thresholds['max_area_ratio_jump']}x)"
                        ),
                    })

            if animation not in exempt and len(bottoms) >= 2:
                drift = max(bottoms) - min(bottoms)
                if drift > action_tolerance:
                    failures.append({
                        "animation": animation, "direction": direction, "check": "grounding",
                        "detail": (
                            f"silhouette baseline drifted {drift:.1f}px across frames "
                            f"(> {action_tolerance:.1f}px); feet are not staying planted"
                        ),
                    })

            actions_report.append({
                "animation": animation,
                "direction": direction,
                "frames_checked": len(areas),
                "min_component_ratio": round(min(component_ratios), 4) if component_ratios else None,
                "baseline_drift_px": round(max(bottoms) - min(bottoms), 1) if len(bottoms) >= 2 else None,
            })

    report = {
        "schema_version": 1,
        "stage": "deformation",
        "unit": unit,
        "passed": not failures,
        "thresholds": thresholds,
        "actions": actions_report,
        "failures": failures,
        "completed_utc": utc_now(),
    }
    write_json(Path(report_path), report)
    return report
