"""Machine-local worker bindings, deliberately separate from portable manifests."""
from __future__ import annotations

from pathlib import Path

from .schemas import DarknessLocalConfig, WorkerBinding


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.local.json"


def load_local_config(path: str | Path | None = None) -> DarknessLocalConfig | None:
    target = Path(path) if path else default_config_path()
    if not target.exists():
        return None
    return DarknessLocalConfig.model_validate_json(target.read_text(encoding="utf-8"))


def worker_binding(config: DarknessLocalConfig | None, worker_id: str) -> WorkerBinding | None:
    return config.workers.get(worker_id) if config else None
