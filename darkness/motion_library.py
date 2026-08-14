"""A named, license-checked catalog of donor motion clips for D7 retargeting.

adapters/retarget_humanoid_motion.py already retargets one donor animation
file onto a generated rig; this module is the addressable-by-name layer on
top of it that darkness/settings.py's [stages.D7].donor_motion_id points at.
It does not run Blender or touch a mesh -- it only resolves an id to the
clip's local file and enforces that every catalog entry carries real
provenance before it can be selected.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from .schemas import StrictModel


class MotionClip(StrictModel):
    clip_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    compatible_anatomy_family: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    author: str = Field(min_length=1)
    license: str = Field(min_length=1)
    license_requires_attribution: bool
    # Where the actual animation file lives once downloaded -- relative to
    # the catalog file's own directory, or absolute. Not committed to git if
    # it points inside a gitignored workspace; the catalog only records
    # provenance, never the binary itself.
    local_path: str | None = None


class MotionLibrary(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    clips: list[MotionClip] = Field(default_factory=list)


def load_motion_library(path: str | Path) -> MotionLibrary:
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    return MotionLibrary.model_validate(data)


def find_motion_clip(library: MotionLibrary, clip_id: str) -> MotionClip | None:
    return next((clip for clip in library.clips if clip.clip_id == clip_id), None)


def resolve_donor_motion_path(library: MotionLibrary, clip_id: str, *, catalog_dir: Path) -> Path:
    """Resolve a configured donor_motion_id to the clip's local animation
    file, ready to pass as --source to adapters/retarget_humanoid_motion.py.

    Raises ValueError (not silently falling back) if the clip is unknown or
    has no local file recorded yet -- a stage should never guess at a
    donor motion.
    """
    clip = find_motion_clip(library, clip_id)
    if clip is None:
        raise ValueError(f"unknown donor_motion_id: {clip_id}")
    if not clip.local_path:
        raise ValueError(
            f"motion clip '{clip_id}' has no local_path recorded yet; "
            "download it and update the catalog before selecting it at D7"
        )
    local = Path(clip.local_path)
    return local if local.is_absolute() else (catalog_dir / local)
