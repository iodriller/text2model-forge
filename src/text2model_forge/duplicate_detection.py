"""Deterministic, model-free detection of a patch reappearing elsewhere in an image.

Why this exists
----------------
D1's deferred-shield repair generates a small patch and pastes it into the
image (`_composite_inpaint_crop` in studio_pipeline.py). Observed on a real
run: the base SDXL render ignored its own "the other hand is empty, no
shield" instruction and rendered a shield-shaped object anyway, on the side
opposite the deferred socket. The repair pass then pasted a second,
correctly-placed shield without ever checking whether one already existed --
producing two shields in one image.

A vision-language model could be asked "is there already a shield here", but
this check needs to run before every paste, for free, with no risk of a
timeout or a wrong VLM answer. Pixel similarity is the right tool for "does
this specific patch already appear somewhere in this specific image" --
`spec_conformance.py` is the semantic layer ("does this represent a
shield at all"); this module is the free, always-on first pass.

Method: a difference hash (dHash), not an average hash. dHash encodes the
sign of the gradient between adjacent pixels rather than an absolute
brightness level, so it survives the color/lighting drift a diffusion model
introduces between two renders of "the same kind of prop" far better than a
raw pixel or average-brightness comparison would. Pure PIL; no new
dependency, nothing that can time out.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


Box = tuple[int, int, int, int]


def difference_hash(image: Image.Image, *, hash_size: int = 8) -> int:
    """A dHash: one bit per pixel, set when that pixel is brighter than its
    right neighbor. Robust to uniform brightness/color shifts because it
    encodes local gradient direction, not an absolute value."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    # get_flattened_data() is Pillow's replacement for the deprecated
    # getdata(); fall back for any older Pillow this project ends up
    # installed against.
    pixels = list(
        resized.get_flattened_data() if hasattr(resized, "get_flattened_data") else resized.getdata()
    )
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | (1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _boxes_overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def find_similar_regions(
    image: Image.Image,
    template: Image.Image,
    *,
    exclude_box: Box | None = None,
    step_fraction: float = 0.06,
    hash_size: int = 8,
    max_hamming_fraction: float = 0.16,
) -> list[tuple[Box, float]]:
    """Slide a window the size of `template` across `image` and return every
    location whose dHash is close to the template's, sorted best-first.

    `exclude_box` skips the region the template was itself generated for (or
    pasted into), so the template is never compared against itself. Returns
    (box, similarity) pairs with similarity in [0, 1], 1.0 being identical
    hashes; `max_hamming_fraction` (of hash_size**2 bits) sets the cutoff --
    0.16 of 64 bits is ~10 bits, chosen loosely (stylized 2D renders of the
    "same kind of prop" vary more than photographs of a literal duplicate,
    so this is tuned to catch gross repetition, not fine-grained plagiarism).
    """
    template_hash = difference_hash(template, hash_size=hash_size)
    total_bits = hash_size * hash_size
    max_hamming = max(1, round(total_bits * max_hamming_fraction))

    width, height = image.size
    template_width, template_height = template.size
    step_x = max(1, round(template_width * step_fraction))
    step_y = max(1, round(template_height * step_fraction))

    matches: list[tuple[Box, float]] = []
    y = 0
    while y + template_height <= height:
        x = 0
        while x + template_width <= width:
            box = (x, y, x + template_width, y + template_height)
            if exclude_box is None or not _boxes_overlap(box, exclude_box):
                window = image.crop(box)
                distance = hamming_distance(template_hash, difference_hash(window, hash_size=hash_size))
                if distance <= max_hamming:
                    matches.append((box, 1.0 - distance / total_bits))
            x += step_x
        y += step_y
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


def patch_already_present_elsewhere(
    image_path: Path,
    template_path: Path,
    *,
    exclude_box: Box | None = None,
    template_resize: tuple[int, int] | None = None,
) -> tuple[bool, float]:
    """Convenience wrapper: does `template_path` (typically a freshly
    generated repair patch) closely resemble something already present in
    `image_path` outside `exclude_box`? Returns (found, best_similarity).

    `template_resize`, if given, resizes the template before searching. An
    inpaint workflow commonly generates its patch at an upscaled "detail
    work" resolution distinct from the pixel size it will actually occupy
    once composited back in (see `_prepare_inpaint_crop` in
    studio_pipeline.py); the sliding search compares whole-template-sized
    windows, so searching at the patch's *generation* resolution instead of
    its *destination* size would compare mismatched physical scales. Pass
    the destination crop box's pixel size here to search at the scale the
    patch will actually occupy.
    """
    with Image.open(image_path) as image, Image.open(template_path) as template:
        template_rgb = template.convert("RGB")
        if template_resize is not None:
            template_rgb = template_rgb.resize(template_resize, Image.LANCZOS)
        matches = find_similar_regions(image.convert("RGB"), template_rgb, exclude_box=exclude_box)
    return (bool(matches), matches[0][1] if matches else 0.0)
