"""Layered configuration resolver.

Five layers, lowest precedence first: profiles/base.toml (every default,
documented), profiles/<name>.toml (a named overlay, e.g. "simple" or
"advanced"), config.local.toml (this machine's worker bindings and
workspace root -- see darkness.config), an explicit run override dict
(what a caller passes when creating a run), and a stage-level override
dict (a per-attempt retry/edit correction -- see StudioStore.decide).

This module only merges plain data; it does not know about StudioRun or
any other pydantic model. Call studio_overrides() to get just the subset
of resolved values that map onto known StudioRun constructor fields.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def bundled_profiles_dir() -> Path:
    """Default profiles shipped inside the installed package (package data),
    always present regardless of how asset-forge was installed."""
    return Path(__file__).resolve().parent / "profiles"


def profiles_dir(repo_root: Path | None = None) -> Path:
    """Where to read profiles/*.toml from. If a project working directory has
    its own profiles/base.toml (e.g. a consuming project like EmberDefense
    that keeps project-specific profiles), that wins; otherwise fall back to
    the profiles bundled with the installed package."""
    if repo_root is not None:
        return repo_root / "profiles"
    cwd_profiles = Path.cwd() / "profiles"
    if (cwd_profiles / "base.toml").is_file():
        return cwd_profiles
    return bundled_profiles_dir()


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


class ResolvedSettings(BaseModel):
    """A merged configuration plus, for every leaf key, which layer set it."""

    values: dict[str, Any]
    origin: dict[str, str]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.values
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def flat(self) -> dict[str, Any]:
        return _flatten(self.values)


def resolve_settings(
    *,
    profile: str = "simple",
    repo_root: Path | None = None,
    machine_path: str | Path | None = None,
    run_overrides: dict[str, Any] | None = None,
    stage_overrides: dict[str, Any] | None = None,
) -> ResolvedSettings:
    """Resolve the layered configuration for one run.

    Layer order (later wins): base.toml, profiles/<profile>.toml,
    config.local.toml's [studio_defaults] table if present, run_overrides,
    stage_overrides.

    profiles/base.toml is read from ./profiles (relative to repo_root, or
    the current working directory if repo_root is not given) when present,
    falling back to the profiles bundled with the installed package -- see
    profiles_dir(). config.local.toml is read from the same working
    directory by default; it is normal for it not to exist.
    """
    profiles = profiles_dir(repo_root)

    base_path = profiles / "base.toml"
    merged: dict[str, Any] = _load_toml(base_path)
    origin: dict[str, str] = {key: "base" for key in _flatten(merged)}

    if profile != "base":
        profile_path = profiles / f"{profile}.toml"
        if profile_path.is_file():
            overlay = _load_toml(profile_path)
            merged = _deep_merge(merged, overlay)
            for key in _flatten(overlay):
                origin[key] = f"profile:{profile}"

    machine_target = (
        Path(machine_path)
        if machine_path
        else (repo_root or Path.cwd()) / "config.local.toml"
    )
    if machine_target.is_file():
        machine_data = _load_toml(machine_target)
        overlay = machine_data.get("studio_defaults")
        if isinstance(overlay, dict):
            merged = _deep_merge(merged, {"studio": overlay})
            for key in _flatten({"studio": overlay}):
                origin[key] = "machine"

    if run_overrides:
        merged = _deep_merge(merged, run_overrides)
        for key in _flatten(run_overrides):
            origin[key] = "run"

    if stage_overrides:
        merged = _deep_merge(merged, stage_overrides)
        for key in _flatten(stage_overrides):
            origin[key] = "stage"

    return ResolvedSettings(values=merged, origin=origin)


# StudioRun constructor fields this resolver is allowed to populate. Keeping
# this list explicit (rather than passing the whole [studio] table through)
# means an unrecognized key in profiles/*.toml fails loudly instead of being
# silently accepted by a future, differently-shaped StudioRun.
_STUDIO_RUN_FIELDS = (
    "model",
    "comfy_url",
    "localdeploy_url",
    "concept_backend",
    "checkpoint",
    "style_lora",
    "style_lora_strength",
    "style_lora_trigger",
    "prop_lora",
    "prop_lora_strength",
)


_OPTIONAL_STRING_FIELDS = ("style_lora", "prop_lora")


def studio_overrides(resolved: ResolvedSettings) -> dict[str, Any]:
    """Extract the subset of resolved [studio] values StudioRun accepts.

    TOML has no null literal, so an optional string field (style_lora,
    prop_lora) is written as "" in profiles/*.toml to mean "not set"; that
    is translated to None here, matching StudioRun's own field default.
    """
    studio = resolved.values.get("studio", {})
    if not isinstance(studio, dict):
        return {}
    result = {key: studio[key] for key in _STUDIO_RUN_FIELDS if key in studio}
    for key in _OPTIONAL_STRING_FIELDS:
        if result.get(key) == "":
            result[key] = None
    return result
