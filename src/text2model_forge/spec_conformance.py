"""Verify a rendered candidate against D0's own equipment contract.

Why this exists
----------------
Every gate downstream of D1 compares a candidate against the *previous
stage's own output* (D2's review compares mesh <-> concept, D3 compares
candidate <-> source). Nothing re-checks a candidate against the one thing
that actually matters: `spec.equipment`, compiled once at D0. So a defect
introduced at D1 -- once it exists -- becomes each later stage's ground
truth, and every later "faithfulness" check correctly certifies a mesh that
faithfully reproduces a mistake.

Observed on a real 8 GB run: D1's own free-form critic reviewed a candidate
with two shields (a stray one the base render produced despite an explicit
"this hand is empty" instruction, plus a second pasted in by the deferred-
shield repair pass, which never checked whether one already existed --
see duplicate_detection.py for that half of the fix) and wrote:
"has correct equipment placement" at confidence 1.0. That is not a prompt-
wording bug in the critic; it is the documented VLM failure mode -- see
StudioQwen.visual_presence's docstring for the citations. An open-ended
"does this look right" question about a whole scene is unreliable for
counting, and the fix the literature actually supports is decomposing the
question into many small, spatially-scoped binary ones, not asking the same
open question more carefully.

Method
------
For each declared equipment item, crop the half of the image its side
occupies (mirrored: a character's right side renders on the viewer's LEFT
half of a front-facing image) and ask one bounded yes/no question: is the
described item visible here. That is the "present at expected side" check,
run for every item.

A second question -- is this SAME category of item ALSO present on the
OPPOSITE side, i.e. duplicated -- is run only for items with
`grip == "forearm_strap"` (shields and similar body-strapped attachments).
It is deliberately NOT run for hand-held weapons: a weapon the prompt asks
to be "extended away from the body" routinely crosses the image midline (the
real render this module was built against has the axe's head cross well
into the opposite half), which would make a naive half-image duplicate
check fire on every correctly-posed weapon. A forearm-strapped item stays
close to the body by construction, so the same check is safe there and is
exactly the class of item the observed bug involved.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Protocol

from PIL import Image
from pydantic import Field

from .schemas import StrictModel
from .studio_models import StudioAssetSpec


Box = tuple[float, float, float, float]


class VisualPresenceProvider(Protocol):
    def visual_presence(self, image_path: Path, question: str, *, max_tokens: int = 40) -> bool: ...


class EquipmentConformanceItem(StrictModel):
    equipment_id: str
    present_at_expected_side: bool
    # None (not False) means the duplicate check was intentionally not run
    # for this item -- see the grip == "forearm_strap" scoping above. A
    # reader must not treat "not checked" as "checked and clean".
    duplicate_on_opposite_side: bool | None = None


class EquipmentConformanceReport(StrictModel):
    items: list[EquipmentConformanceItem] = Field(default_factory=list)
    conforms: bool
    violations: list[str] = Field(default_factory=list)


def _side_box(side: str) -> Box | None:
    """Normalized (x0,y0,x1,y1) for the half of a front-facing render a
    character's own side occupies. Mirrored: a front-facing image shows the
    character's right side on the viewer's left half. `side == "center"` has
    no reliable half-image discriminator and is intentionally unhandled
    (returns None; callers skip the side-scoped checks for it)."""
    if side == "right":
        return (0.04, 0.04, 0.50, 0.96)
    if side == "left":
        return (0.50, 0.04, 0.96, 0.96)
    return None


def crop_to_temp(image_path: Path, box_normalized: Box) -> Path:
    """A normalized-box crop written to a temp file the caller must unlink.

    Public (not module-private) because studio_pipeline.py's D1
    shield-repair guard needs the exact same crop-then-ask shape this module
    uses internally, and duplicating the crop math in two places is how a
    box-coordinate convention drifts out of sync between them.
    """
    with Image.open(image_path) as image:
        width, height = image.size
        x0, y0, x1, y1 = box_normalized
        pixel_box = (
            max(0, int(x0 * width)),
            max(0, int(y0 * height)),
            min(width, int(x1 * width)),
            min(height, int(y1 * height)),
        )
        cropped = image.convert("RGB").crop(pixel_box)
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            cropped.save(handle.name)
        finally:
            handle.close()
        return Path(handle.name)


def presence_check(
    qwen: VisualPresenceProvider, image_path: Path, box_normalized: Box, question: str
) -> bool:
    """Crop once, ask one bounded yes/no question, always clean up the crop --
    the single-box version of what check_equipment_conformance does per item,
    exposed so a caller with just one region to ask about (D1's shield-repair
    guard) does not need to build a whole StudioAssetSpec to use it."""
    crop = crop_to_temp(image_path, box_normalized)
    try:
        return qwen.visual_presence(crop, question)
    finally:
        crop.unlink(missing_ok=True)


def check_equipment_conformance(
    qwen: VisualPresenceProvider, image_path: Path, spec: StudioAssetSpec
) -> EquipmentConformanceReport:
    """Run the localized presence/duplication checks for every equipment item.

    Returns conforms=True (no items, no violations) for anything that is not
    a character/creature or declares no equipment -- there is nothing this
    module can check for a prop or a static asset.
    """
    if spec.asset_kind not in {"character", "creature"} or not spec.equipment:
        return EquipmentConformanceReport(conforms=True)

    items: list[EquipmentConformanceItem] = []
    violations: list[str] = []
    made: list[Path] = []
    try:
        for equipment in spec.equipment:
            expected_box = _side_box(equipment.side)
            if expected_box is None:
                items.append(
                    EquipmentConformanceItem(
                        equipment_id=equipment.equipment_id, present_at_expected_side=True
                    )
                )
                continue

            expected_crop = crop_to_temp(image_path, expected_box)
            made.append(expected_crop)
            # Deliberately short and generic (the item's category noun, not
            # its full spec description): a live check against the real
            # duplicate-shield run showed the 4B reviewer answers a plain
            # "Does this image show a shield?" correctly on a crop it gets
            # WRONG when asked the verbose, spec-worded version of the same
            # question -- consistent with "Let Me Speak Freely?" (EMNLP
            # 2024), which found format/length constraints degrade small-
            # model accuracy. Keep every question here short and generic.
            present = qwen.visual_presence(
                expected_crop, f"Does this image clearly show a {equipment.category}?"
            )

            duplicate: bool | None = None
            if equipment.grip == "forearm_strap":
                opposite_side = "left" if equipment.side == "right" else "right"
                opposite_box = _side_box(opposite_side)
                if opposite_box is not None:
                    opposite_crop = crop_to_temp(image_path, opposite_box)
                    made.append(opposite_crop)
                    duplicate = qwen.visual_presence(
                        opposite_crop,
                        f"Does this image show a second, separate {equipment.category}?",
                    )

            items.append(
                EquipmentConformanceItem(
                    equipment_id=equipment.equipment_id,
                    present_at_expected_side=present,
                    duplicate_on_opposite_side=duplicate,
                )
            )
            if not present:
                violations.append(
                    f"{equipment.equipment_id}: not visibly present on its {equipment.side} side"
                )
            if duplicate:
                violations.append(
                    f"{equipment.equipment_id}: a second {equipment.category} appears on the opposite "
                    f"side, which the contract does not call for -- likely duplicated equipment"
                )
    finally:
        for crop in made:
            crop.unlink(missing_ok=True)

    return EquipmentConformanceReport(items=items, conforms=not violations, violations=violations)
