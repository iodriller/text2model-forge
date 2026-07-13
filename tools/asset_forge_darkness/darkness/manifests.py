"""Strict worker-manifest discovery and machine preflight."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import httpx

from .schemas import WorkerManifest


class WorkerManifestError(ValueError):
    pass


def manifest_root() -> Path:
    return Path(__file__).resolve().parents[1] / "workers"


def load_manifests(root: str | Path | None = None) -> dict[str, WorkerManifest]:
    directory = Path(root) if root else manifest_root()
    result: dict[str, WorkerManifest] = {}
    for path in sorted(directory.glob("*.json")):
        manifest = WorkerManifest.model_validate_json(path.read_text(encoding="utf-8"))
        worker_id = manifest.capability.worker_id
        if worker_id in result:
            raise WorkerManifestError(f"duplicate worker_id '{worker_id}'")
        result[worker_id] = manifest
    return result


def _resolve_executable(candidate: str) -> str | None:
    expanded = os.path.expandvars(os.path.expanduser(candidate))
    path = Path(expanded)
    if path.is_file():
        return str(path.resolve())
    return shutil.which(candidate)


def preflight(
    manifest: WorkerManifest,
    *,
    command_prefix: list[str] | None = None,
    timeout_seconds: float = 2,
) -> dict:
    candidates = [command_prefix[0]] if command_prefix else manifest.executable_candidates
    executable = next((resolved for candidate in candidates if (resolved := _resolve_executable(candidate))), None)
    health_reachable: bool | None = None
    health_error: str | None = None
    if manifest.health_url:
        try:
            response = httpx.get(manifest.health_url, timeout=timeout_seconds)
            health_reachable = response.status_code < 500
            if not health_reachable:
                health_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            health_reachable = False
            health_error = str(exc)
    required_executable = manifest.adapter == "subprocess"
    required_health = manifest.adapter in {"http", "localdeploy"}
    ready = manifest.lifecycle in {"available", "research_ready"}
    if required_executable:
        ready = ready and executable is not None
    if required_health:
        ready = ready and health_reachable is True
    return {
        "worker_id": manifest.capability.worker_id,
        "declared_lifecycle": manifest.lifecycle,
        "ready": ready,
        "executable": executable,
        "health_reachable": health_reachable,
        "health_error": health_error,
        "blockers": manifest.blockers,
    }
