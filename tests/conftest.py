"""Test-suite-wide isolation from machine-local state.

The one thing in here matters more than it looks. `resolve_settings()` and
`load_local_config()` both fall back to the repo-root `config.local.toml`
when no explicit path is given, and that file is gitignored, machine-
specific, and *present on any machine actually set up to run the pipeline*.

So before this fixture existed the suite was only green on a checkout with
no machine config. Creating a perfectly ordinary `config.local.toml` --
exactly what `docs/free-local-8gb-setup.md` tells a user to do -- turned 22
tests red, because assertions like

    assert resolved.get("studio.checkpoint") == "dreamshaper_xl_v2_turbo.safetensors"

were silently reading whatever checkpoint that developer happened to have
installed. The tests were not wrong; they were unisolated.

Redirecting only the *default* path keeps the tests that deliberately
exercise machine config working, because those pass an explicit
`machine_path=` / `path=` and never consult the default.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_machine_config(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> None:
    """Point the default config.local.toml lookup at a path that cannot exist.

    Both module-level names are patched, not just the source one:
    darkness.settings does `from .config import default_config_path`, so it
    holds its own reference and patching only darkness.config would leave the
    settings resolver still reading the real file.
    """
    missing = tmp_path_factory.mktemp("no-machine-config") / "config.local.toml"

    def _absent() -> Path:
        return missing

    monkeypatch.setattr("darkness.config.default_config_path", _absent)
    monkeypatch.setattr("darkness.settings.default_config_path", _absent, raising=False)
    monkeypatch.setattr("darkness.settings._default_machine_path", _absent, raising=False)
