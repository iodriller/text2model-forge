from __future__ import annotations

import json
from pathlib import Path

from darkness.mesh_evidence import build_mesh_evidence


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mesh"


def test_build_mesh_evidence_writes_reviewable_station_three_bundle(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    result = build_mesh_evidence(FIXTURES / "five_defects.obj", output)

    assert result["accepted"] is True
    assert result["after_diagnoses"] == []
    assert (output / "repaired.obj").is_file()
    assert json.loads((output / "after_health.json").read_text(encoding="utf-8"))["boundary_edges"] == 0
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Repair branch accepted: **true**" in report
    assert "This report is deterministic evidence only" in report


def test_build_mesh_evidence_never_overwrites_an_existing_bundle(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    build_mesh_evidence(FIXTURES / "five_defects.obj", output)

    try:
        build_mesh_evidence(FIXTURES / "five_defects.obj", output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("evidence output must be immutable")
