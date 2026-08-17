from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import asset_root, load_asset, read_json, sha256_file, utc_now, write_json


def build_compliance_report(workspace: Path, asset_id: str, report_path: Path) -> dict[str, Any]:
    asset = load_asset(workspace, asset_id)
    failures: list[str] = []
    warnings: list[str] = []
    verified_jobs = []
    for job_path in sorted((asset_root(workspace, asset_id) / "jobs").glob("*.json")):
        job = read_json(job_path)
        engine = job.get("engine")
        if engine == "ComfyUI":
            model = job.get("model", {})
            if not model.get("installed_sha256"):
                failures.append(f"ComfyUI job has no registered checkpoint hash: {job_path.name}")
            if not model.get("source_url") or not model.get("license"):
                failures.append(f"ComfyUI job has incomplete model license provenance: {job_path.name}")
        if engine == "TripoSR" and not job.get("model_weights"):
            failures.append(f"TripoSR job has no discovered model-weight hash: {job_path.name}")
        output_results = []
        for output in job.get("outputs", []):
            path = Path(output.get("path", ""))
            if not path.is_file():
                failures.append(f"Job output is missing: {path}")
                output_results.append({"path": str(path), "valid": False, "reason": "missing"})
                continue
            current = sha256_file(path)
            valid = current == output.get("sha256")
            if not valid:
                failures.append(f"Job output hash changed: {path}")
            output_results.append({"path": str(path), "valid": valid, "sha256": current})
        verified_jobs.append({"path": str(job_path), "engine": engine, "outputs": output_results})

    license_info = asset.get("license", {})
    if license_info.get("status") not in {"original-confirmed", "mixed-reviewed"}:
        failures.append("Asset license status must be original-confirmed or mixed-reviewed")
    for index, source in enumerate(license_info.get("inputs", [])):
        missing = [key for key in ("name", "source_url", "author", "license", "commercial_use") if not source.get(key)]
        if missing:
            failures.append(f"License input {index} is missing fields: {missing}")
        if str(source.get("commercial_use", "")).lower() != "yes":
            failures.append(f"License input {index} is not confirmed for commercial use")
    if not verified_jobs:
        warnings.append("No generative or mesh-seed jobs were recorded; this is valid for fully manual original work")

    report = {
        "schema_version": 1,
        "asset_id": asset_id,
        "checked_utc": utc_now(),
        "passed": not failures,
        "license_status": license_info.get("status"),
        "verified_jobs": verified_jobs,
        "failures": failures,
        "warnings": warnings,
    }
    write_json(report_path.resolve(), report)
    return report
