from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CONSUMER = re.compile(r"ember[\s_-]*defense", re.IGNORECASE)
PUBLIC_SUFFIXES = {".py", ".toml", ".json", ".md", ".yml", ".yaml", ".cs"}


def test_public_source_has_no_consuming_project_reference() -> None:
    offenders: list[str] = []
    excluded_roots = {".git", ".venv", "runtime", "workspaces", "tests"}
    excluded_files = {"AGENTS.md", "CLAUDE.md"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in PUBLIC_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if relative.name in excluded_files or any(part in excluded_roots for part in relative.parts):
            continue
        if FORBIDDEN_CONSUMER.search(path.read_text(encoding="utf-8")):
            offenders.append(relative.as_posix())
    assert offenders == []


def test_default_product_configuration_has_no_third_party_art_style() -> None:
    defaults = (ROOT / "darkness" / "profiles" / "base.toml").read_text(encoding="utf-8")
    models = (ROOT / "darkness" / "studio_models.py").read_text(encoding="utf-8")
    combined = f"{defaults}\n{models}".lower()
    assert "warcraft" not in combined


def test_public_project_identity_and_legal_files_are_present() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "vettedmesh"' in metadata
    assert 'license = "Apache-2.0"' in metadata
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md"):
        assert (ROOT / name).is_file(), name


def test_public_studio_surface_uses_vettedmesh_identity() -> None:
    source = (ROOT / "darkness" / "studio_web.py").read_text(encoding="utf-8")
    assert "VettedMesh Studio" in source
    assert "Darkness Studio" not in source


def test_machine_local_motion_catalog_is_not_tracked_as_a_default() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/motion_library/catalog.json" in ignore
    assert (ROOT / "motion_library/catalog.example.json").is_file()
