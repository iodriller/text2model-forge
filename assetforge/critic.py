"""Automated art direction: deterministic quality scores every candidate must pass.

Mechanical QA (validate_sheets) proves frames exist, stand on a shared baseline, and
move.  It cannot tell whether the unit still LOOKS right.  This module encodes the
owner's standing art direction — dark grim palette, gameplay-size readability, one
identity across frames and sheets — as measurable gates, so a build that drifts toward
neon, washes out, flickers identity, or turns to mush at 96 px fails IN the pipeline
instead of in front of the owner.  The human reviews winners, not iterations.

Taste that cannot be measured lives in docs/asset-forge-rubric.md and is applied by
the art-directing agent on the acceptance board; every owner rejection must be added
there so no complaint is ever made twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from .core import ForgeError, utc_now, write_json

DEFAULTS = {
    "min_value": 0.10,          # masked mean brightness floor: not a black blob
    "max_value": 0.62,          # ceiling: dark fantasy, not pastel
    "max_saturation": 0.60,     # grim muted palette, no neon
    "min_edge": 5.0,            # edge energy at 96 px: forms must survive gameplay size
    "min_hue_coherence": 0.45,  # consecutive frames keep one palette (no identity flicker)
    "min_identity": 0.30,       # every sheet agrees with idle/south frame 0
    "alpha_threshold": 128,
    "gameplay_height": 96,
    "hue_bins": 16,
}


def _sheet_paths(config: dict[str, Any], repo_root: Path) -> list[tuple[str, str, Path]]:
    pattern = str(config["output_path_pattern"])
    entries = []
    for animation in config.get("animations", {}):
        for direction in config.get("directions", []):
            relative = pattern.format(animation=animation, direction=direction)
            entries.append((animation, direction, repo_root / relative))
    return entries


def _frames(sheet: Image.Image) -> list[Image.Image]:
    height = sheet.height
    count = sheet.width // height
    return [sheet.crop((index * height, 0, (index + 1) * height, height)) for index in range(count)]


def _mask(frame: Image.Image, threshold: int) -> Image.Image:
    return frame.getchannel("A").point(lambda a: 255 if a >= threshold else 0)


def _hue_histogram(frame: Image.Image, mask: Image.Image, bins: int) -> list[float]:
    hue = frame.convert("RGB").convert("HSV").getchannel(0)
    raw = hue.histogram(mask=mask)
    folded = [0.0] * bins
    for index, count in enumerate(raw):
        folded[index * bins // 256] += count
    total = sum(folded) or 1.0
    return [value / total for value in folded]


def _intersection(first: list[float], second: list[float]) -> float:
    return sum(min(a, b) for a, b in zip(first, second))


def _tone(frame: Image.Image, mask: Image.Image) -> tuple[float, float]:
    hsv = frame.convert("RGB").convert("HSV")
    means = ImageStat.Stat(hsv, mask).mean
    return means[2] / 255.0, means[1] / 255.0


def _edge_energy(frame: Image.Image, mask: Image.Image, gameplay_height: int) -> float:
    """Interior edge energy at gameplay size.  The silhouette boundary is excluded:
    even a featureless blob has a strong outline, but a readable unit needs internal
    form separation (head/torso/weapon) that survives the downscale."""
    scale = gameplay_height / frame.height
    size = (max(1, round(frame.width * scale)), gameplay_height)
    small = frame.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    small_mask = mask.resize(size, Image.Resampling.NEAREST)
    interior = small_mask.filter(ImageFilter.MinFilter(5))
    target = interior if ImageStat.Stat(interior).sum[0] > 0 else small_mask
    if ImageStat.Stat(target).sum[0] == 0:
        return 0.0
    edges = small.convert("L").filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges, target).mean[0]


def critique_unit(config_path: Path, repo_root: Path, report_path: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    unit = str(config["id"])
    thresholds = dict(DEFAULTS)
    thresholds.update(config.get("critic") or {})

    failures: list[dict[str, Any]] = []
    sheets_report: list[dict[str, Any]] = []
    reference_histogram: list[float] | None = None

    for animation, direction, path in _sheet_paths(config, Path(repo_root)):
        if not path.is_file():
            failures.append({"sheet": str(path), "check": "exists", "detail": "sheet file is missing"})
            continue
        with Image.open(path) as source:
            sheet = source.convert("RGBA")
        frames = _frames(sheet)
        values, saturations, edges = [], [], []
        histograms: list[list[float]] = []
        for index, frame in enumerate(frames):
            mask = _mask(frame, int(thresholds["alpha_threshold"]))
            if ImageStat.Stat(mask).sum[0] == 0:
                continue
            value, saturation = _tone(frame, mask)
            values.append(value)
            saturations.append(saturation)
            edges.append(_edge_energy(frame, mask, int(thresholds["gameplay_height"])))
            histograms.append(_hue_histogram(frame, mask, int(thresholds["hue_bins"])))
        if not values:
            failures.append({"sheet": str(path), "check": "content", "detail": "no visible frames"})
            continue

        value_mean = sum(values) / len(values)
        saturation_mean = sum(saturations) / len(saturations)
        edge_min = min(edges)
        coherence = min(
            (_intersection(histograms[i], histograms[i + 1]) for i in range(len(histograms) - 1)),
            default=1.0,
        )
        if animation == "idle" and direction == "south" and reference_histogram is None:
            reference_histogram = histograms[0]
        identity = _intersection(reference_histogram, histograms[0]) if reference_histogram else 1.0

        entry = {
            "animation": animation,
            "direction": direction,
            "sheet": str(path),
            "value_mean": round(value_mean, 4),
            "saturation_mean": round(saturation_mean, 4),
            "edge_min": round(edge_min, 2),
            "hue_coherence_min": round(coherence, 4),
            "identity_vs_idle_south": round(identity, 4),
        }
        sheets_report.append(entry)

        if not thresholds["min_value"] <= value_mean <= thresholds["max_value"]:
            failures.append({**entry, "check": "tone-value", "detail": f"masked brightness {value_mean:.3f} outside [{thresholds['min_value']}, {thresholds['max_value']}]"})
        if saturation_mean > thresholds["max_saturation"]:
            failures.append({**entry, "check": "tone-saturation", "detail": f"saturation {saturation_mean:.3f} above {thresholds['max_saturation']} (palette not grim/muted)"})
        if edge_min < thresholds["min_edge"]:
            failures.append({**entry, "check": "readability", "detail": f"edge energy {edge_min:.2f} below {thresholds['min_edge']} at {thresholds['gameplay_height']}px"})
        if coherence < thresholds["min_hue_coherence"]:
            failures.append({**entry, "check": "frame-coherence", "detail": f"hue coherence {coherence:.3f} below {thresholds['min_hue_coherence']} (identity flickers between frames)"})
        if identity < thresholds["min_identity"]:
            failures.append({**entry, "check": "identity", "detail": f"palette overlap with idle/south {identity:.3f} below {thresholds['min_identity']} (not the same character)"})

    report = {
        "schema_version": 1,
        "stage": "critic",
        "unit": unit,
        "passed": not failures,
        "thresholds": thresholds,
        "rubric": "docs/asset-forge-rubric.md",
        "sheets": sheets_report,
        "failures": failures,
        "completed_utc": utc_now(),
    }
    write_json(Path(report_path), report)
    return report
