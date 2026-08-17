"""Tests for the deterministic dHash-based patch-duplication detector.

Synthetic PIL images only, so this is fast, offline, and exercises the exact
mechanism the real bug demonstrated: a patch that already appears elsewhere
in the image must be found, and a patch that genuinely does not appear
elsewhere must not produce a false positive.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from text2model_forge.duplicate_detection import (
    difference_hash,
    find_similar_regions,
    hamming_distance,
    patch_already_present_elsewhere,
)


def _canvas(size: tuple[int, int] = (400, 500), color: str = "#7a9e6e") -> Image.Image:
    return Image.new("RGB", size, color)


def _disc(canvas: Image.Image, center: tuple[int, int], radius: int, color: str) -> Image.Image:
    result = canvas.copy()
    draw = ImageDraw.Draw(result)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#222222", width=3)
    return result


def test_identical_hash_for_identical_crops():
    image = _disc(_canvas(), (100, 100), 60, "#b0b0b0")
    left = image.crop((20, 20, 180, 180))
    right = image.crop((20, 20, 180, 180))
    assert difference_hash(left) == difference_hash(right)
    assert hamming_distance(difference_hash(left), difference_hash(right)) == 0


def test_a_disc_pasted_twice_is_found_by_its_own_template():
    """The exact mechanism of the real bug: the same prop rendered at two
    locations in one image. Using one location as the template must find
    the other."""
    canvas = _canvas()
    canvas = _disc(canvas, (90, 350), 55, "#a8a8a8")   # the "pre-existing" shield
    canvas = _disc(canvas, (310, 350), 55, "#a8a8a8")  # the "freshly repaired" shield

    template_box = (255, 295, 365, 405)
    template = canvas.crop(template_box)

    matches = find_similar_regions(canvas, template, exclude_box=template_box)
    assert matches, "the second disc must find the first as a near-duplicate"
    best_box, similarity = matches[0]
    # the best match should land on (or very near) the first disc's location
    assert abs(((best_box[0] + best_box[2]) / 2) - 90) < 40
    assert similarity > 0.8


def test_a_patch_that_genuinely_does_not_appear_elsewhere_is_not_flagged():
    canvas = _canvas()
    canvas = _disc(canvas, (300, 350), 55, "#a8a8a8")  # only one disc, elsewhere is plain background
    template_box = (245, 295, 355, 405)
    template = canvas.crop(template_box)

    matches = find_similar_regions(canvas, template, exclude_box=template_box)
    assert matches == [], "a uniform background must not be mistaken for the template"


def test_exclude_box_removes_only_the_region_it_covers():
    """The real use: exclude the box a repair patch was generated for/pasted
    into, so the search only reports OTHER occurrences. Two discs, one
    excluded box covering each in turn."""
    canvas = _canvas()
    canvas = _disc(canvas, (90, 350), 55, "#a8a8a8")
    canvas = _disc(canvas, (310, 350), 55, "#a8a8a8")
    template_box = (255, 295, 365, 405)
    template = canvas.crop(template_box)

    # excluding the template's own box: the other disc is still found
    found_other = find_similar_regions(canvas, template, exclude_box=template_box)
    assert found_other

    # excluding a box wide enough to cover BOTH discs: nothing is left to find
    found_none = find_similar_regions(canvas, template, exclude_box=(0, 260, 400, 440))
    assert found_none == []


def test_color_shift_between_two_renders_of_the_same_prop_still_matches():
    """dHash encodes gradient direction, not absolute brightness, so it
    should tolerate the kind of color/lighting drift two separate diffusion
    passes produce for 'the same kind of prop' -- exactly what happened
    between the base render's stray shield and the repair pass's shield."""
    canvas = _canvas()
    canvas = _disc(canvas, (90, 350), 55, "#9c9c9c")
    canvas = _disc(canvas, (310, 350), 55, "#b4a888")  # warmer/lighter tone, same shape
    template_box = (255, 295, 365, 405)
    template = canvas.crop(template_box)
    matches = find_similar_regions(canvas, template, exclude_box=template_box)
    assert matches, "a modest color shift between two renders of the same prop must still match"


def test_patch_already_present_elsewhere_end_to_end(tmp_path):
    canvas = _canvas()
    canvas = _disc(canvas, (90, 350), 55, "#a8a8a8")
    canvas = _disc(canvas, (310, 350), 55, "#a8a8a8")
    image_path = tmp_path / "image.png"
    canvas.save(image_path)
    template_path = tmp_path / "template.png"
    canvas.crop((255, 295, 365, 405)).save(template_path)

    found, similarity = patch_already_present_elsewhere(
        image_path, template_path, exclude_box=(255, 295, 365, 405)
    )
    assert found is True
    assert similarity > 0.8

    # a template that is not present anywhere else
    lone_canvas = _disc(_canvas(), (300, 350), 55, "#a8a8a8")
    lone_path = tmp_path / "lone.png"
    lone_canvas.save(lone_path)
    lone_template_path = tmp_path / "lone_template.png"
    lone_canvas.crop((245, 295, 355, 405)).save(lone_template_path)
    found, similarity = patch_already_present_elsewhere(
        lone_path, lone_template_path, exclude_box=(245, 295, 355, 405)
    )
    assert found is False
    assert similarity == 0.0


def test_template_resize_matches_destination_scale_not_generation_scale(tmp_path):
    """An inpaint patch is often generated at a different resolution than the
    pixel box it will be pasted into. Searching at the generation size can
    miss a same-content match that only lines up once resized to the
    destination scale. Both discs fill ~90% of their own frame, so resizing
    the template to the destination's pixel size is the only thing that
    needs to make them line up -- proportion is deliberately held constant."""
    canvas = _canvas((600, 600))
    canvas = _disc(canvas, (150, 450), 36, "#a8a8a8")  # disc fills ~90% of an 80x80 window
    canvas_path = tmp_path / "base.png"
    canvas.save(canvas_path)

    # a "freshly generated" patch at a much higher resolution, same 90%-fill
    # proportion, as an inpaint workflow's own upscaled detail pass would be
    big_template = Image.new("RGB", (400, 400), "#7a9e6e")
    draw = ImageDraw.Draw(big_template)
    draw.ellipse((20, 20, 380, 380), fill="#a8a8a8", outline="#222222", width=12)
    template_path = tmp_path / "generated.png"
    big_template.save(template_path)

    # searching at the template's own 400x400 generation size must NOT find
    # the much smaller 80x80-scale disc in the base image
    not_found, _ = patch_already_present_elsewhere(
        canvas_path, template_path, exclude_box=(400, 400, 480, 480)
    )
    assert not_found is False

    # resizing to the destination's actual pixel size (80x80) finds it
    found, _similarity = patch_already_present_elsewhere(
        canvas_path, template_path, exclude_box=(400, 400, 480, 480), template_resize=(80, 80)
    )
    assert found is True
