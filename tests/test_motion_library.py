from pathlib import Path

import pytest
from pydantic import ValidationError

from text2model_forge.motion_library import (
    MotionClip,
    MotionLibrary,
    find_motion_clip,
    load_motion_library,
    resolve_donor_motion_path,
)


def _clip(**overrides) -> MotionClip:
    fields = dict(
        clip_id="walk_v1",
        display_name="Walk Cycle",
        description="A CC0 walk cycle.",
        compatible_anatomy_family="short_biped",
        source_url="https://example.com/walk",
        author="Example Author",
        license="CC0-1.0",
        license_requires_attribution=False,
    )
    fields.update(overrides)
    return MotionClip(**fields)


def test_motion_clip_requires_real_provenance_fields():
    with pytest.raises(ValidationError):
        MotionClip(
            clip_id="walk_v1",
            display_name="Walk Cycle",
            description="Missing license fields.",
            compatible_anatomy_family="short_biped",
            source_url="https://example.com/walk",
            author="Example Author",
            # license and license_requires_attribution omitted
        )


def test_find_motion_clip_returns_none_for_unknown_id():
    library = MotionLibrary(clips=[_clip()])
    assert find_motion_clip(library, "does_not_exist") is None
    assert find_motion_clip(library, "walk_v1") is not None


def test_resolve_donor_motion_path_rejects_unknown_clip_id(tmp_path: Path):
    library = MotionLibrary(clips=[_clip()])
    with pytest.raises(ValueError, match="unknown donor_motion_id"):
        resolve_donor_motion_path(library, "nope", catalog_dir=tmp_path)


def test_resolve_donor_motion_path_rejects_a_clip_with_no_local_file_yet(tmp_path: Path):
    library = MotionLibrary(clips=[_clip(local_path=None)])
    with pytest.raises(ValueError, match="no local_path recorded"):
        resolve_donor_motion_path(library, "walk_v1", catalog_dir=tmp_path)


def test_resolve_donor_motion_path_joins_relative_paths_against_the_catalog_directory(tmp_path: Path):
    library = MotionLibrary(clips=[_clip(local_path="clips/walk.glb")])
    resolved = resolve_donor_motion_path(library, "walk_v1", catalog_dir=tmp_path)
    assert resolved == tmp_path / "clips" / "walk.glb"


def test_resolve_donor_motion_path_keeps_an_absolute_local_path_as_is(tmp_path: Path):
    absolute = tmp_path / "elsewhere" / "walk.glb"
    library = MotionLibrary(clips=[_clip(local_path=str(absolute))])
    resolved = resolve_donor_motion_path(library, "walk_v1", catalog_dir=tmp_path / "catalog")
    assert resolved == absolute


def test_load_motion_library_reads_the_bundled_example_catalog():
    example = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "motion_library"
        / "catalog.example.json"
    )
    library = load_motion_library(example)
    assert library.clips[0].clip_id == "example_walk_cycle"
    # the example is explicitly a template with no downloaded file yet
    assert library.clips[0].local_path is None
