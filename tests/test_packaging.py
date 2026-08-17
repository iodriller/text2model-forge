import json
from pathlib import Path

import pytest

from text2model_forge.packaging import CANONICAL_REQUIRED_FILES, PackagePolicyError, build_delivery_package


def _fixture(tmp_path: Path, *, qualification_status: str = "partial"):
    source = tmp_path / "source"
    source.mkdir()
    for relative in CANONICAL_REQUIRED_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.startswith("skinning_"):
            path.write_text('{"hard_failures": []}', encoding="utf-8")
        else:
            path.write_bytes(b"fixture")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "schema_version": 1,
                        "candidate_id": "fixture",
                        "display_name": "Fixture",
                        "capability": "test",
                        "source_url": "project://fixture",
                        "license_id": "Project-Owned",
                        "priority": "open_source_first",
                        "research_allowed": True,
                        "release_eligibility": "permitted",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "fixture",
                "worker_id": "fixture.worker",
                "source_revision": "sha256:" + "a" * 64,
                "code_license": "Project-Owned",
                "weights_license": "not_applicable",
                "tested_at": "2026-07-12T00:00:00Z",
                "host_gpu": "CPU",
                "status": qualification_status,
            }
        ),
        encoding="utf-8",
    )
    return source, registry, qualification


def test_research_package_records_release_blocker_and_hashes(tmp_path: Path) -> None:
    source, registry, qualification = _fixture(tmp_path)
    manifest = build_delivery_package(
        package_id="fixture.v1",
        candidate_id="fixture",
        source_root=source,
        output_root=tmp_path / "package",
        registry_path=registry,
        qualification_path=qualification,
    )
    assert manifest.eligibility == "research_only"
    assert manifest.blockers == ["runtime qualification is partial"]
    assert len(manifest.files) == len(CANONICAL_REQUIRED_FILES)


def test_release_package_fails_closed_on_partial_qualification(tmp_path: Path) -> None:
    source, registry, qualification = _fixture(tmp_path)
    with pytest.raises(PackagePolicyError, match="runtime qualification is partial"):
        build_delivery_package(
            package_id="fixture.v1",
            candidate_id="fixture",
            source_root=source,
            output_root=tmp_path / "package",
            registry_path=registry,
            qualification_path=qualification,
            mode="release",
        )
