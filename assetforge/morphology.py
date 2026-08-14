"""Creature morphology contracts and deterministic build-report validation.

AI concept and texture tools are optional inputs.  Production anatomy is described
by a versioned profile, authored once in Blender, and verified before sprite render.
This keeps a goblin's face, an ogre's proportions, or a dragon's silhouette stable
across actions, directions, and future rebuilds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .core import ForgeError, sha256_file, sha256_json


SUPPORTED_FAMILIES = {
    "biped_humanoid_v1": "production",
    "winged_quadruped_v1": "contract_only",
}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForgeError(f"Could not load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ForgeError(f"{label.capitalize()} must contain a JSON object: {path}")
    return value


def _number(value: object, label: str, failures: list[str], minimum: float, maximum: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        failures.append(f"{label} must be numeric")
    elif not minimum <= float(value) <= maximum:
        failures.append(f"{label} must be in {minimum}..{maximum}")


def validate_profile(profile: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    if profile.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    if not str(profile.get("id", "")).strip():
        failures.append("id is required")
    family_id = str(profile.get("family", ""))
    if family_id not in SUPPORTED_FAMILIES:
        failures.append(f"unsupported creature family {family_id!r}")

    contract_value = profile.get("family_contract")
    contract_path = (repo_root / str(contract_value)).resolve() if contract_value else None
    contract: dict[str, Any] = {}
    if contract_path is None or not contract_path.is_file():
        failures.append("family_contract must name an existing tracked JSON file")
    else:
        contract = _read_object(contract_path, "family contract")
        if contract.get("id") != family_id:
            failures.append("family_contract id does not match profile.family")

    _number(profile.get("scale"), "scale", failures, 0.2, 4.0)
    posture = profile.get("posture")
    if not isinstance(posture, dict):
        failures.append("posture must be an object")
    else:
        if posture.get("space") != "bone_local_pose":
            failures.append("posture.space must be 'bone_local_pose'")
        if posture.get("apply_to_every_action") is not True:
            failures.append("posture.apply_to_every_action must be true")
        corrections = posture.get("bones_degrees_xyz")
        if not isinstance(corrections, dict) or not corrections:
            failures.append("posture.bones_degrees_xyz must contain at least one bone")
        else:
            for bone, angles in corrections.items():
                if not isinstance(angles, list) or len(angles) != 3:
                    failures.append(f"posture correction for {bone!r} must be [x,y,z]")
                    continue
                for axis, angle in zip("xyz", angles):
                    _number(angle, f"posture {bone}.{axis}", failures, -90.0, 90.0)

    head = profile.get("head")
    modules = head.get("modules") if isinstance(head, dict) else None
    if not isinstance(modules, dict) or not modules:
        failures.append("head.modules must contain authored anatomy modules")
    else:
        allowed = set(contract.get("supported_anatomy_modules", []))
        unknown = sorted(set(modules) - allowed)
        if unknown:
            failures.append("family does not support anatomy modules: " + ", ".join(unknown))

    silhouette = profile.get("silhouette_contract")
    if not isinstance(silhouette, dict):
        failures.append("silhouette_contract must be an object")
    else:
        landmarks = silhouette.get("required_landmarks")
        if not isinstance(landmarks, list) or not landmarks or not all(isinstance(v, str) and v for v in landmarks):
            failures.append("silhouette_contract.required_landmarks must be a non-empty string list")
        proof_views = set(silhouette.get("proof_views", []))
        required_views = set(contract.get("required_proof_views", []))
        if not required_views.issubset(proof_views):
            failures.append("silhouette proof_views do not cover the family contract")

    return contract, failures


def validate_morphology_build(profile_path: Path, report_path: Path, master_path: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    profile_path = profile_path.resolve()
    report_path = report_path.resolve()
    master_path = master_path.resolve()
    profile = _read_object(profile_path, "morphology profile")
    contract, failures = validate_profile(profile, repo_root)
    if contract and contract.get("implementation_status") != "production":
        failures.append(
            f"creature family {profile.get('family')!r} is not production-enabled "
            f"(status: {contract.get('implementation_status')})"
        )
    report = _read_object(report_path, "morphology build report") if report_path.is_file() else {}
    if not report:
        failures.append("morphology build report is missing")
    else:
        if report.get("profile_id") != profile.get("id"):
            failures.append("build report belongs to a different morphology profile")
        if report.get("profile_sha256") != sha256_file(profile_path):
            failures.append("build report profile hash is stale")
        if report.get("family") != profile.get("family"):
            failures.append("build report family does not match profile")
        if report.get("passed") is not True:
            failures.extend(str(value) for value in report.get("failures", ["builder reported failure"]))
        created = set(report.get("created_parts", []))
        required = set(profile.get("silhouette_contract", {}).get("required_landmarks", []))
        missing = sorted(required - created)
        if missing:
            failures.append("missing required anatomy landmarks: " + ", ".join(missing))
        expected_actions = set(profile.get("silhouette_contract", {}).get("proof_actions", []))
        posture_actions = set(report.get("posture_actions", []))
        if not expected_actions.issubset(posture_actions):
            failures.append("posture was not baked into every proof action")
    if not master_path.is_file() or master_path.suffix.lower() != ".blend":
        failures.append("morphology master must be an existing .blend file")

    result = {
        "schema_version": 1,
        "passed": not failures,
        "profile": str(profile_path),
        "profile_id": profile.get("id"),
        "profile_sha256": sha256_file(profile_path),
        "profile_contract_sha256": sha256_json(contract) if contract else None,
        "family": profile.get("family"),
        "family_status": contract.get("implementation_status"),
        "master": str(master_path),
        "master_sha256": sha256_file(master_path) if master_path.is_file() else None,
        "failures": failures,
    }
    if failures:
        raise ForgeError("Morphology validation failed:\n- " + "\n- ".join(failures))
    return result


def _fit_subject(image: Image.Image, size: tuple[int, int], head_only: bool = False) -> Image.Image:
    source = image.convert("RGBA")
    bounds = source.getchannel("A").getbbox()
    if bounds:
        if head_only:
            left, top, right, bottom = bounds
            head_bottom = min(bottom, top + max(12, round((bottom - top) * 0.38)))
            center = (left + right) // 2
            half = max((head_bottom - top), (right - left) // 2) + 8
            bounds = (max(0, center - half), max(0, top - 8), min(source.width, center + half), min(source.height, head_bottom + 8))
        source = source.crop(bounds)
    source.thumbnail(size, Image.Resampling.LANCZOS)
    output = Image.new("RGBA", size, (0, 0, 0, 0))
    output.alpha_composite(source, ((size[0] - source.width) // 2, (size[1] - source.height) // 2))
    return output


def create_morphology_proof(config_path: Path, profile_path: Path, frames_root: Path, output: Path) -> dict[str, Any]:
    """Create a fast geometry-first board before texture baking or full sheet work."""
    config = _read_object(config_path.resolve(), "character config")
    profile = _read_object(profile_path.resolve(), "morphology profile")
    frames_root = frames_root.resolve() / str(config["id"])

    def frame(action: str, direction: str, index: int) -> Image.Image:
        path = frames_root / action / direction / f"{index:02d}.png"
        if not path.is_file():
            raise ForgeError(f"Morphology proof frame is missing: {path}")
        with Image.open(path) as value:
            return value.convert("RGBA")

    idle_south = frame("idle", "south", 0)
    idle_east = frame("idle", "east", 0)
    attack = config.get("animations", {}).get("attack", {})
    phases = attack.get("phases", {})
    phase_frames = [
        ("anticipation", int(phases.get("anticipation_frame", 0))),
        ("contact", int(phases.get("contact_frame", max(0, int(attack.get("frames", 2)) // 2)))),
        ("recovery", int(phases.get("recovery_frame", max(0, int(attack.get("frames", 1)) - 1)))),
    ]

    width, height = 1400, 900
    board = Image.new("RGB", (width, height), (12, 18, 22))
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, width, 82), fill=(29, 38, 44))
    draw.text((28, 18), f"{config.get('display_name', config['id'])} | Creature DNA geometry proof", fill=(241, 220, 180))
    draw.text((28, 48), "Review anatomy and motion before AI paint, texture bake, full sheets, approval, or Unity publish", fill=(166, 181, 187))

    def panel(image: Image.Image, box: tuple[int, int, int, int], label: str, *, head_only: bool = False, silhouette: bool = False) -> None:
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(box, radius=9, fill=(22, 30, 35), outline=(64, 78, 83), width=1)
        fitted = _fit_subject(image, (x1 - x0 - 20, y1 - y0 - 46), head_only=head_only)
        if silhouette:
            alpha = fitted.getchannel("A")
            fitted = Image.new("RGBA", fitted.size, (46, 69, 57, 0))
            fitted.putalpha(alpha)
        board.paste(fitted, (x0 + 10, y0 + 8), fitted)
        draw.text((x0 + 10, y1 - 28), label, fill=(202, 210, 210))

    panel(idle_south, (28, 106, 300, 430), "full body / south")
    panel(idle_east, (322, 106, 594, 430), "posture + snout / east")
    panel(idle_south, (616, 106, 888, 430), "face anatomy / south", head_only=True)
    panel(idle_east, (910, 106, 1182, 430), "face anatomy / east", head_only=True)
    panel(idle_east, (1204, 106, 1372, 430), "pure silhouette", silhouette=True)

    for index, (label, frame_index) in enumerate(phase_frames):
        panel(frame("attack", "south", frame_index), (28 + index * 294, 456, 300 + index * 294, 780), f"attack {label} / f{frame_index}")

    draw.rounded_rectangle((910, 456, 1372, 780), radius=9, fill=(20, 28, 32), outline=(64, 78, 83))
    draw.text((930, 474), "Gameplay-scale read", fill=(224, 207, 176))
    for index, (label, source) in enumerate((("idle", idle_south), ("contact", frame("attack", "south", phase_frames[1][1])))):
        fitted = _fit_subject(source, (128, 128))
        x = 944 + index * 200
        board.paste(fitted, (x, 526), fitted)
        draw.text((x + 42, 666), label, fill=(194, 202, 203))
    required = profile.get("silhouette_contract", {}).get("required_landmarks", [])
    draw.text((930, 716), f"{profile.get('family')} | {len(required)} anatomy landmarks", fill=(166, 180, 184))
    draw.text((28, 824), "GATE: face, ears, snout, jaw, tusks, hunch, weapon path, and contact pose must all read without texture.", fill=(218, 154, 112))
    draw.text((28, 852), "This proof is intentionally unpainted. Surface diffusion cannot rescue failed anatomy or motion.", fill=(166, 181, 187))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, "PNG")
    return {
        "schema_version": 1,
        "profile_id": profile.get("id"),
        "profile_sha256": sha256_file(profile_path.resolve()),
        "proof": str(output),
        "geometry_only": True,
        "published": False,
    }
