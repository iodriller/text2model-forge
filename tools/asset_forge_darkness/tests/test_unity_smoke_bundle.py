import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "adapters/build_unity_smoke_bundle.py"))["build"]
INGEST = runpy.run_path(str(ROOT / "adapters/ingest_unity_smoke_result.py"))["ingest"]
ARCHIVE = runpy.run_path(str(ROOT / "adapters/archive_unity_smoke_bundle.py"))["archive"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    package.mkdir()
    master = tmp_path / "candidate.blend"
    master.write_bytes(b"editable-darkness-master")
    actions = []
    for clip in ("idle", "walk", "attack", "death"):
        for direction in ("north", "south", "east", "west"):
            sheet = package / f"{clip}_{direction}.png"
            sheet.write_bytes(f"{clip}/{direction}".encode())
            actions.append(
                {
                    "name": clip,
                    "direction": direction,
                    "frames": 1,
                    "fps": 10.0,
                    "loop": clip in ("idle", "walk"),
                    "sheet": sheet.name,
                    "sha256": _sha256(sheet),
                }
            )
    manifest = {
        "schema_version": 1,
        "asset_id": "fixture_goblin",
        "status": "human_review_candidate",
        "automatic_gate_passed": True,
        "human_approval_required": True,
        "human_approved": False,
        "cell_width": 1,
        "cell_height": 1,
        "pixels_per_unit": 1.0,
        "pivot_x": 0.5,
        "pivot_y": 0.18,
        "source_master": str(master),
        "source_master_sha256": _sha256(master),
        "actions": actions,
    }
    (package / "candidate_unit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package


def test_standalone_unity_bundle_builds_and_ingests_hash_bound_result(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    built = BUILD(_candidate(tmp_path), bundle, ROOT / "unity_smoke_template")
    assert built["bundle_kind"] == "darkness_standalone_unity_smoke"
    assert built["directional_actions"] == 16
    assert built["expected_decoded_sprites"] == 16
    assert built["unity_version_family"] == "6000.5"
    assert built["unity_runtime_mutable_files"] == [
        "UnitySmokeProject/ProjectSettings/ProjectVersion.txt"
    ]
    assert (bundle / "run_unity_smoke.ps1").is_file()
    assert (bundle / "open_unity_review.ps1").is_file()
    assert (bundle / "review.html").is_file()
    review = (bundle / "review.html").read_text(encoding="utf-8")
    assert "Auto-tour all motions" in review
    assert "idle_south.png" in review
    portable = json.loads((bundle / "candidate/candidate_unit_manifest.json").read_text(encoding="utf-8"))
    assert portable["source_master"].startswith("source/")
    first = ARCHIVE(bundle, tmp_path / "bundle.zip")
    second = ARCHIVE(bundle, tmp_path / "bundle-second.zip")
    assert first["archive_sha256"] == second["archive_sha256"]
    with zipfile.ZipFile(tmp_path / "bundle.zip") as zipped:
        assert "bundle_manifest.json" in zipped.namelist()
        assert "UnitySmokeProject/Assets/Editor/DarknessCandidateValidator.cs" in zipped.namelist()
        assert "UnitySmokeProject/Assets/Editor/DarknessGoblinReviewWindow.cs" in zipped.namelist()

    result = bundle / "result"
    result.mkdir()
    capture = result / "unity_candidate_capture.png"
    capture.write_bytes(b"unity-rendered-capture")
    bundle_manifest = bundle / "bundle_manifest.json"
    report = {
        "passed": True,
        "asset_id": built["asset_id"],
        "project_kind": "darkness_standalone_unity_smoke",
        "unity_version": "6000.5.3f1",
        "candidate_manifest_sha256": built["candidate_manifest_sha256"],
        "bundle_manifest_sha256": _sha256(bundle_manifest),
        "directional_actions": 16,
        "decoded_sprites": 16,
        "animation_clips": 16,
        "source_master_hash_verified": True,
        "live_game_assets_modified": False,
        "capture_sha256": _sha256(capture),
        "human_approval_required": True,
        "human_approved": False,
    }
    (result / "unity_candidate_validation.json").write_text(json.dumps(report), encoding="utf-8")
    # Unity may rewrite its own editor-version metadata while importing the
    # generated smoke project. Candidate assets and validator sources remain
    # hash-bound; this one declared file is runtime metadata, not asset input.
    (bundle / "UnitySmokeProject/ProjectSettings/ProjectVersion.txt").write_text(
        "m_EditorVersion: runtime-rewritten\n", encoding="utf-8"
    )
    ingested = INGEST(bundle, result, tmp_path / "ingested")
    assert ingested["passed"] is True
    assert ingested["live_game_assets_modified"] is False
    assert ingested["human_approved"] is False


def test_unity_result_ingest_rejects_changed_bundle_file(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    BUILD(_candidate(tmp_path), bundle, ROOT / "unity_smoke_template")
    (bundle / "candidate/idle_north.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bundle file changed"):
        INGEST(bundle, bundle / "result", tmp_path / "ingested")
