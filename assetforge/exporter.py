from __future__ import annotations

import shutil
from pathlib import Path

from .core import ForgeError, approval_status, asset_root, load_asset, load_project, sha256_file, utc_now, write_json


def export_asset(
    workspace: Path,
    asset_id: str,
    destination: Path,
    adapter: str = "generic",
    require_approvals: bool = True,
) -> dict:
    project = load_project(workspace)
    asset = load_asset(workspace, asset_id)
    approvals = approval_status(workspace, asset_id)
    if require_approvals:
        missing = [stage for stage in ("sheets", "license") if not approvals[stage]["approved"]]
        if missing:
            raise ForgeError(f"Export blocked by missing or stale approvals: {missing}")
    sheets = sorted((asset_root(workspace, asset_id) / "sheets").glob("*.png"))
    if not sheets:
        raise ForgeError(f"No PNG sheets found for {asset_id}")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source in sheets:
        target = destination / source.name
        source_hash = sha256_file(source)
        if target.exists() and sha256_file(target) != source_hash:
            raise ForgeError(f"Refusing to overwrite different engine asset: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        outputs.append({"path": str(target), "sha256": source_hash})
    manifest = {
        "schema_version": 1,
        "product": "VettedMesh Studio",
        "project_id": project["project_id"],
        "asset_id": asset_id,
        "kind": asset["kind"],
        "adapter": adapter,
        "exported_utc": utc_now(),
        "files": outputs,
        "approvals": {stage: value for stage, value in approvals.items() if value["approved"]},
    }
    write_json(destination / "assetforge-export.json", manifest)
    return manifest
