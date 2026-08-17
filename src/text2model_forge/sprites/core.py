from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from text2model_forge.paths import resource_root


SCHEMA_VERSION = 1
SLUG = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
ASSET_KINDS = {"character", "creature", "building", "prop", "vfx", "icon", "background"}
APPROVAL_STAGES = (
    "brief",
    "concept",
    "turnaround",
    "master",
    "actions",
    "sheets",
    "engine_capture",
    "license",
)
ASSET_DIRECTORIES = (
    "concepts/candidates",
    "concepts/approved",
    "master",
    "renders",
    "sheets",
    "approvals",
    "jobs",
    "reports",
    "exports",
)


class ForgeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_slug(value: str, label: str = "id") -> str:
    if not SLUG.fullmatch(value):
        raise ForgeError(f"{label} must be snake_case and start with a letter: {value!r}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ForgeError(f"Required JSON file was not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ForgeError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ForgeError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any], overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ForgeError(f"Refusing to overwrite existing file: {path}")
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ForgeError(f"Artifact was not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_root() -> Path:
    return resource_root()


def project_path(workspace: Path) -> Path:
    return workspace.resolve() / "forge-project.json"


def load_project(workspace: Path) -> dict[str, Any]:
    project = read_json(project_path(workspace))
    if project.get("schema_version") != SCHEMA_VERSION:
        raise ForgeError(f"Unsupported project schema: {project.get('schema_version')}")
    return project


def register_model(workspace: Path, model_id: str, model_file: Path, reviewer: str) -> dict[str, Any]:
    project = load_project(workspace)
    registry_path = workspace.resolve() / project["model_registry"]
    registry = read_json(registry_path)
    model = next((item for item in registry.get("models", []) if item.get("id") == model_id), None)
    if model is None:
        raise ForgeError(f"Model id is not declared in the project registry: {model_id}")
    model_file = model_file.resolve()
    if not model_file.is_file():
        raise ForgeError(f"Model file was not found: {model_file}")
    model["installed_path"] = str(model_file)
    model["installed_sha256"] = sha256_file(model_file)
    model["reviewed_by"] = reviewer
    model["reviewed_utc"] = utc_now()
    write_json(registry_path, registry)
    return model


def init_workspace(
    workspace: Path,
    project_id: str,
    display_name: str,
    engine: str = "unity",
    engine_root: str | None = None,
    style_id: str = "chunky_heroic_dark_fantasy",
) -> dict[str, Any]:
    ensure_slug(project_id, "project id")
    ensure_slug(style_id, "style id")
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = project_path(root)
    if destination.exists():
        raise ForgeError(f"Text2Model Forge workspace already exists: {destination}")

    project = {
        "schema_version": SCHEMA_VERSION,
        "product": "Text2Model Forge Studio",
        "project_id": project_id,
        "display_name": display_name,
        "created_utc": utc_now(),
        "engine": {"adapter": engine, "root": engine_root},
        "hardware_profile": "rtx3080-laptop-8gb",
        "style_profile": "style/style-profile.json",
        "model_registry": "models/model-registry.json",
    }
    style = {
        "schema_version": SCHEMA_VERSION,
        "id": style_id,
        "display_name": "Original Chunky Heroic Dark Fantasy",
        "positive": (
            "original chunky heroic dark fantasy game art, broad readable silhouette, clear equipment, "
            "tarnished metal, worn leather, muted cool palette, deep readable shadows, restrained warm accent, "
            "painterly cel-shaded material, mobile-game clarity, consistent three-quarter orthographic camera"
        ),
        "negative": (
            "copyrighted character, existing game faction, logo, text, watermark, photoreal, anime screenshot, "
            "chibi, cute toy, pixel art, flat side-view puppet, cropped feet, duplicated equipment, extra limbs, "
            "inconsistent costume, inconsistent lighting, busy background"
        ),
        "directions": ["north", "south", "east", "west"],
        "palette": {
            "defender": ["#263B4A", "#77858C", "#3B2418", "#D75A2B"],
            "enemy": ["#39452A", "#4B2923", "#25282B", "#C94A2F"],
            "environment": ["#11191D", "#2A3334", "#514638", "#E1652F"],
        },
        "forbidden_references": [
            "copyrighted characters",
            "copied armor or faction symbols",
            "third-party game UI",
        ],
    }
    models = read_json(package_root() / "presets" / "model-registry.template.json")
    hardware = read_json(package_root() / "presets" / "rtx3080-8gb.profile.json")
    write_json(destination, project, overwrite=False)
    write_json(root / "style" / "style-profile.json", style, overwrite=False)
    write_json(root / "models" / "model-registry.json", models, overwrite=False)
    write_json(root / "hardware" / f"{project['hardware_profile']}.profile.json", hardware, overwrite=False)
    (root / "assets").mkdir(exist_ok=True)
    return project


def asset_root(workspace: Path, asset_id: str) -> Path:
    ensure_slug(asset_id, "asset id")
    return workspace.resolve() / "assets" / asset_id


def create_asset(
    workspace: Path,
    asset_id: str,
    display_name: str,
    kind: str,
    role: str,
    faction: str = "neutral",
    primary_action: str = "attack",
) -> dict[str, Any]:
    project = load_project(workspace)
    ensure_slug(asset_id, "asset id")
    if kind not in ASSET_KINDS:
        raise ForgeError(f"Unsupported asset kind {kind!r}; expected one of {sorted(ASSET_KINDS)}")
    if primary_action not in {"attack", "shoot", "cast", "none"}:
        raise ForgeError("primary action must be attack, shoot, cast, or none")
    root = asset_root(workspace, asset_id)
    manifest = root / "asset.json"
    if manifest.exists():
        raise ForgeError(f"Asset already exists: {asset_id}")
    for relative in ASSET_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)

    animated = kind in {"character", "creature", "vfx"}
    actions = ["idle", "walk"] if kind in {"character", "creature"} else []
    if primary_action != "none":
        actions.append(primary_action)
    if kind in {"character", "creature"}:
        actions.extend(["hit", "death"])
    asset = {
        "schema_version": SCHEMA_VERSION,
        "project_id": project["project_id"],
        "asset_id": asset_id,
        "display_name": display_name,
        "kind": kind,
        "role": role,
        "faction": faction,
        "created_utc": utc_now(),
        "brief": {
            "silhouette": "",
            "equipment": [],
            "materials": [],
            "palette_notes": "",
            "gameplay_read": role,
            "must_have": [],
            "must_not_have": [],
        },
        "production": {
            "animated": animated,
            "directions": ["north", "south", "east", "west"],
            "required_actions": actions,
            "cell_size": [256, 256],
            "target_height_fraction": 0.68,
        },
        "license": {"status": "original-planned", "inputs": []},
    }
    write_json(manifest, asset, overwrite=False)
    return asset


def load_asset(workspace: Path, asset_id: str) -> dict[str, Any]:
    return read_json(asset_root(workspace, asset_id) / "asset.json")


def update_asset_brief(workspace: Path, asset_id: str, brief: dict[str, Any]) -> dict[str, Any]:
    asset = load_asset(workspace, asset_id)
    allowed_text = {"silhouette", "palette_notes", "gameplay_read"}
    allowed_lists = {"equipment", "materials", "must_have", "must_not_have"}
    current = dict(asset.get("brief", {}))
    for key in allowed_text:
        if key in brief:
            current[key] = str(brief[key]).strip()
    for key in allowed_lists:
        if key in brief:
            value = brief[key]
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ForgeError(f"Brief field {key} must be a list of strings")
            current[key] = [item.strip() for item in value if item.strip()]
    asset["brief"] = current
    asset["brief_updated_utc"] = utc_now()
    write_json(asset_root(workspace, asset_id) / "asset.json", asset)
    prompt_pack = asset_root(workspace, asset_id) / "prompt-pack.json"
    if prompt_pack.exists():
        prompt_pack.unlink()
    return asset


def build_prompt_pack(workspace: Path, asset_id: str) -> dict[str, Any]:
    project = load_project(workspace)
    asset = load_asset(workspace, asset_id)
    style = read_json(workspace.resolve() / project["style_profile"])
    brief = asset["brief"]
    equipment = ", ".join(brief.get("equipment", [])) or "equipment defined by the role"
    materials = ", ".join(brief.get("materials", [])) or "weathered practical fantasy materials"
    must_have = ", ".join(brief.get("must_have", [])) or "clear feet and readable hands"
    must_not = ", ".join(brief.get("must_not_have", []))
    identity = (
        f"{asset['display_name']}, {asset['role']}, faction {asset['faction']}; "
        f"equipment: {equipment}; materials: {materials}; silhouette: {brief.get('silhouette') or 'broad and role-readable'}; "
        f"must have: {must_have}"
    )
    negative = style["negative"] + (f", {must_not}" if must_not else "")
    common = f"{identity}. {style['positive']}. Neutral studio background, entire body visible, feet on one baseline."
    pack = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": asset_id,
        "style_id": style["id"],
        "generated_utc": utc_now(),
        "negative": negative,
        "prompts": {
            "concept": common + " Single full-body production concept, neutral stance.",
            "turnaround": common + (
                " Four-view turnaround in one image: north/back, south/front, east, west; same identity, exact armor, "
                "equipment, proportions, colors, and scale in every view; equal spacing; orthographic presentation."
            ),
            "texture": common + " Flat material reference with clean color separation and no baked background.",
            "mesh_seed": common + " Symmetric A-pose, unobstructed limbs, minimal perspective, no effects or ground shadow.",
        },
    }
    write_json(asset_root(workspace, asset_id) / "prompt-pack.json", pack)
    return pack


def record_job(workspace: Path, asset_id: str, job: dict[str, Any]) -> Path:
    root = asset_root(workspace, asset_id) / "jobs"
    job = dict(job)
    job.setdefault("schema_version", SCHEMA_VERSION)
    job.setdefault("created_utc", utc_now())
    identity = sha256_json(job)[:12]
    path = root / f"{job['created_utc'].replace(':', '').replace('-', '')}-{identity}.json"
    write_json(path, job, overwrite=False)
    return path


def approve_artifact(
    workspace: Path,
    asset_id: str,
    stage: str,
    artifact: Path,
    reviewer: str,
    notes: str = "",
) -> Path:
    if stage not in APPROVAL_STAGES:
        raise ForgeError(f"Unknown approval stage {stage!r}; expected one of {APPROVAL_STAGES}")
    artifact_path = artifact.resolve()
    digest = sha256_file(artifact_path)
    approval = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": asset_id,
        "stage": stage,
        "artifact": str(artifact_path),
        "artifact_sha256": digest,
        "reviewer": reviewer,
        "notes": notes,
        "approved_utc": utc_now(),
    }
    name = f"{stage}-{approval['approved_utc'].replace(':', '').replace('-', '')}-{digest[:12]}.json"
    destination = asset_root(workspace, asset_id) / "approvals" / name
    write_json(destination, approval, overwrite=False)
    return destination


def approval_status(workspace: Path, asset_id: str) -> dict[str, Any]:
    directory = asset_root(workspace, asset_id) / "approvals"
    results = {stage: {"approved": False, "reason": "missing"} for stage in APPROVAL_STAGES}
    for path in sorted(directory.glob("*.json"), reverse=True):
        approval = read_json(path)
        stage = approval.get("stage")
        if stage not in results or results[stage]["approved"]:
            continue
        artifact = Path(approval.get("artifact", ""))
        if not artifact.is_file():
            results[stage] = {"approved": False, "reason": "artifact-missing", "approval": str(path)}
            continue
        current = sha256_file(artifact)
        if current != approval.get("artifact_sha256"):
            results[stage] = {"approved": False, "reason": "artifact-changed", "approval": str(path)}
            continue
        results[stage] = {"approved": True, "reason": "hash-valid", "approval": str(path)}
    return results


def project_status(workspace: Path) -> dict[str, Any]:
    project = load_project(workspace)
    assets = []
    for manifest in sorted((workspace.resolve() / "assets").glob("*/asset.json")):
        asset = read_json(manifest)
        approvals = approval_status(workspace, asset["asset_id"])
        assets.append({
            "asset_id": asset["asset_id"],
            "display_name": asset["display_name"],
            "kind": asset["kind"],
            "approved_stages": [stage for stage, value in approvals.items() if value["approved"]],
            "next_stage": next((stage for stage in APPROVAL_STAGES if not approvals[stage]["approved"]), None),
        })
    return {"project_id": project["project_id"], "assets": assets}
