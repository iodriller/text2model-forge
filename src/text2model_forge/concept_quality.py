"""Deterministic D1 quality gates that run before an image reaches a VLM.

These checks intentionally measure only properties pixels can prove: usable
alpha extraction, a complete centered silhouette, dynamic range, edge detail,
and resolution. Semantic identity remains the VLM and human gate's job.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat
from pydantic import Field

from .schemas import StrictModel


class ConceptQualityAssessment(StrictModel):
    score: float = Field(ge=0, le=1)
    hard_requirements_satisfied: bool
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def assess_concept_image(
    source: Path,
    alpha_source: Path | None,
    *,
    minimum_score: float,
) -> ConceptQualityAssessment:
    """Score a concept without claiming to understand its semantic content."""

    reasons: list[str] = []
    with Image.open(source).convert("RGB") as image:
        width, height = image.size
        grayscale = image.convert("L")
        variance = float(ImageStat.Stat(grayscale).var[0])
        edge_mean = float(ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES)).mean[0])

    nonblank = variance > 1.0
    if not nonblank:
        reasons.append("image is effectively blank")

    target_area = 768 * 1024
    resolution_score = _clamp(math.sqrt((width * height) / target_area))
    contrast_score = _clamp(math.sqrt(max(0.0, variance)) / 48.0)
    edge_score = _clamp(edge_mean / 18.0)

    meaningful_alpha = False
    foreground_fraction = 1.0
    border_touch_fraction = 1.0
    center_offset = 1.0
    layout_score = 0.0
    if alpha_source is None or not alpha_source.is_file():
        reasons.append("no safe edge-connected background mask was produced")
    else:
        with Image.open(alpha_source).convert("RGBA") as rgba:
            alpha = rgba.getchannel("A")
            mask = alpha.point(lambda value: 255 if value >= 128 else 0)
            histogram = mask.histogram()
            foreground_pixels = int(histogram[255])
            total_pixels = max(1, rgba.width * rgba.height)
            foreground_fraction = foreground_pixels / total_pixels
            bbox = mask.getbbox()
            if bbox is not None and foreground_pixels:
                x0, y0, x1, y1 = bbox
                center_x = (x0 + x1) / 2 / rgba.width
                center_y = (y0 + y1) / 2 / rgba.height
                center_offset = math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)
                border = list(mask.crop((0, 0, rgba.width, 1)).tobytes())
                border += list(mask.crop((0, rgba.height - 1, rgba.width, rgba.height)).tobytes())
                border += list(mask.crop((0, 1, 1, max(2, rgba.height - 1))).tobytes())
                border += list(
                    mask.crop((rgba.width - 1, 1, rgba.width, max(2, rgba.height - 1))).tobytes()
                )
                border_touch_fraction = sum(value > 0 for value in border) / max(1, len(border))
            meaningful_alpha = 0.05 <= foreground_fraction <= 0.90

        size_score = _clamp(1.0 - abs(foreground_fraction - 0.42) / 0.42)
        centered_score = _clamp(1.0 - center_offset / 0.35)
        isolation_score = _clamp(1.0 - border_touch_fraction / 0.08)
        layout_score = 0.45 * size_score + 0.35 * centered_score + 0.20 * isolation_score
        if not meaningful_alpha:
            reasons.append(
                f"foreground occupies {foreground_fraction:.1%}; expected an isolated full asset between 5% and 90%"
            )
        if border_touch_fraction > 0.08:
            reasons.append("foreground touches too much of the image border and may be cropped")
        if center_offset > 0.35:
            reasons.append("foreground is too far from the frame center")

    score = _clamp(
        0.15 * resolution_score
        + 0.20 * contrast_score
        + 0.20 * edge_score
        + 0.30 * layout_score
        + 0.15 * float(meaningful_alpha)
    )
    if score < minimum_score:
        reasons.append(f"deterministic quality score {score:.3f} is below required {minimum_score:.3f}")
    hard_passed = bool(
        nonblank
        and meaningful_alpha
        and foreground_fraction <= 0.90
        and border_touch_fraction <= 0.08
        and center_offset <= 0.35
        and score >= minimum_score
    )
    return ConceptQualityAssessment(
        score=round(score, 4),
        hard_requirements_satisfied=hard_passed,
        reasons=reasons,
        metrics={
            "quality_score": round(score, 4),
            "quality_gate_passed": hard_passed,
            "meaningful_alpha": meaningful_alpha,
            "foreground_fraction": round(foreground_fraction, 4),
            "border_touch_fraction": round(border_touch_fraction, 4),
            "center_offset": round(center_offset, 4),
            "contrast_score": round(contrast_score, 4),
            "edge_score": round(edge_score, 4),
            "resolution_score": round(resolution_score, 4),
        },
    )
