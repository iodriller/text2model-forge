"""Machine-local worker bindings, deliberately separate from portable manifests."""
from __future__ import annotations

import tomllib
import os
from pathlib import Path
from text2model_forge.paths import source_checkout_root

from .schemas import Text2ModelLocalConfig, WorkerBinding


def default_config_path() -> Path:
    override = os.environ.get("TEXT2MODEL_FORGE_CONFIG", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    source = source_checkout_root()
    if source is not None:
        return source / "config.local.toml"
    return Path.cwd() / "config.local.toml"


def load_local_config(path: str | Path | None = None) -> Text2ModelLocalConfig | None:
    target = Path(path) if path else default_config_path()
    if not target.exists():
        return None
    with target.open("rb") as stream:
        data = tomllib.load(stream)
    return Text2ModelLocalConfig.model_validate(data)


def worker_binding(config: Text2ModelLocalConfig | None, worker_id: str) -> WorkerBinding | None:
    return config.workers.get(worker_id) if config else None
