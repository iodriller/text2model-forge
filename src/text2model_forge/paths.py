"""Locate Text2Model Forge's non-Python runtime resources.

An editable/source checkout keeps workers, adapters, Blender scripts, and
templates under ``resources/``. A wheel installs the same tree under the
environment's ``share/text2model-forge`` directory. Keeping this decision in
one module prevents each stage from inventing a different path rule.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


_MARKERS = ("workers", "adapters", "registry")


def _is_resource_root(path: Path) -> bool:
    return all((path / marker).is_dir() for marker in _MARKERS)


def source_checkout_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if _is_resource_root(candidate / "resources") else None


def resource_root() -> Path:
    override = os.environ.get("TEXT2MODEL_FORGE_RESOURCE_ROOT", "").strip()
    candidates = [Path(override).expanduser()] if override else []
    source = source_checkout_root()
    if source is not None:
        candidates.append(source / "resources")
    candidates.extend(
        [
            Path(sys.prefix) / "share" / "text2model-forge",
            Path(sys.base_prefix) / "share" / "text2model-forge",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if _is_resource_root(resolved):
            return resolved
    searched = ", ".join(str(item) for item in candidates) or "no candidate paths"
    raise FileNotFoundError(
        "Text2Model Forge runtime resources were not found. Install the full wheel or "
        f"set TEXT2MODEL_FORGE_RESOURCE_ROOT. Searched: {searched}"
    )


def source_revision() -> str | None:
    """Return the exact source commit when it can be proved locally.

    Source checkouts are queried directly. Packagers may inject the immutable
    revision through ``TEXT2MODEL_FORGE_SOURCE_REVISION``; arbitrary working-directory
    repositories are deliberately ignored.
    """
    injected = os.environ.get("TEXT2MODEL_FORGE_SOURCE_REVISION", "").strip().lower()
    if injected:
        return injected if re.fullmatch(r"[0-9a-f]{40}", injected) else None
    checkout = source_checkout_root()
    if checkout is None or not (checkout / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip().lower()
    return revision if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", revision) else None
