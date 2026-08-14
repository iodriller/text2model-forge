"""Machine-local worker bindings, deliberately separate from portable manifests."""
from __future__ import annotations

import tomllib
from pathlib import Path

from .schemas import DarknessLocalConfig, WorkerBinding


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.local.toml"


def load_local_config(path: str | Path | None = None) -> DarknessLocalConfig | None:
    target = Path(path) if path else default_config_path()
    if not target.exists():
        return None
    with target.open("rb") as stream:
        data = tomllib.load(stream)
    return DarknessLocalConfig.model_validate(data)


def worker_binding(config: DarknessLocalConfig | None, worker_id: str) -> WorkerBinding | None:
    return config.workers.get(worker_id) if config else None
