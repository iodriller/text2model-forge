from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from text2model_forge.sprites.cli import build_parser, execute
from text2model_forge.sprites.core import ForgeError, sha256_file
from text2model_forge.sprites.production import package_production_unit, validate_production_config


DIRECTIONS = ["north", "south", "east", "west"]
ACTIONS = ["idle", "walk", "attack", "hit", "death"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_sheet(path: Path, color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((10, 6, 20, 28), fill=color)
    draw.rectangle((42, 8, 55, 28), fill=color)
    sheet.save(path)


def production_fixture(root: Path) -> tuple[Path, Path]:
    master = root / "master" / "unit.blend"
    master.parent.mkdir(parents=True)
    master.write_bytes(b"owned blender fixture")
    animations = {}
    for action in ACTIONS:
        animations[action] = {
            "actions": [action],
            "frames": 2,
            "fps": 10,
            "loop": action in ("idle", "walk"),
        }
        if action == "attack":
            animations[action]["phases"] = {"contact_frame": 1}
        for direction_index, direction in enumerate(DIRECTIONS):
            make_sheet(
                root / "sheets" / f"unit_{action}_{direction}_sheet.png",
                (110 + direction_index * 20, 70, 45, 255),
            )
    config = {
        "id": "unit",
        "display_name": "Test Unit",
        "source_kind": "owned_blender_master",
        "master_license": "project-original",
        "source": "master/unit.blend",
        "cell_size": [32, 32],
        "directions": DIRECTIONS,
        "default_facing": "south",
        "animations": animations,
        "output_path_pattern": "sheets/unit_{animation}_{direction}_sheet.png",
    }
    config_path = root / "unit.character.json"
    write_json(config_path, config)
    qa_path = root / "qa.json"
    qa_sheets = []
    for action in ACTIONS:
        for direction in DIRECTIONS:
            sheet = root / "sheets" / f"unit_{action}_{direction}_sheet.png"
            qa_sheets.append({
                "path": sheet.relative_to(root).as_posix(),
                "sha256": sha256_file(sheet),
            })
    write_json(qa_path, {"asset_id": "unit", "passed": True, "sheets": qa_sheets})
    return config_path, qa_path


def test_production_gate_rejects_diffusion_and_missing_directions(tmp_path: Path) -> None:
    config = {
        "source_kind": "diffusion_frame_sequence",
        "source": "missing.png",
        "directions": ["east"],
        "animations": {},
    }
    failures = validate_production_config(config, tmp_path)
    assert any("owned_blender_master" in failure for failure in failures)
    assert any("north, south, east, and west" in failure for failure in failures)
    assert any("missing required action" in failure for failure in failures)


def test_package_builds_hash_locked_manifest_and_review_reels(tmp_path: Path) -> None:
    config_path, qa_path = production_fixture(tmp_path)
    result = package_production_unit(
        config_path,
        tmp_path,
        qa_path,
        tmp_path / "unity-unit",
        tmp_path / "review",
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["production"]["eligible"] is True
    assert manifest["production"]["identity_lock"] == "single_master_sha256"
    assert len(manifest["production"]["master_sha256"]) == 64
    assert len(manifest["actions"]) == len(ACTIONS) * len(DIRECTIONS)
    assert Path(result["review"]["all_actions"]).is_file()
    assert Path(result["review"]["gameplay_transition"]).is_file()
    assert Path(result["review"]["acceptance_board"]).is_file()


def test_package_rejects_sheet_changed_after_qa(tmp_path: Path) -> None:
    config_path, qa_path = production_fixture(tmp_path)
    make_sheet(tmp_path / "sheets" / "unit_idle_south_sheet.png", (220, 20, 20, 255))
    with pytest.raises(ForgeError, match="sheet changed after QA"):
        package_production_unit(
            config_path,
            tmp_path,
            qa_path,
            tmp_path / "unity-unit",
            tmp_path / "review",
        )


def test_diffusion_create_unit_cannot_copy_to_unity(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "create-unit",
        "--workspace", str(tmp_path),
        "--asset-id", "unit",
        "--to-unity", str(tmp_path / "Unity"),
    ])
    with pytest.raises(ForgeError, match="motion prototypes"):
        execute(args)


def test_enabled_overpaint_cannot_package_without_passing_provenance(tmp_path: Path) -> None:
    config_path, qa_path = production_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["depth_pass"] = True
    config["equipment_objects"] = ["Sword*"]
    config["overpaint"] = {
        "enabled": True,
        "control": "depth",
        "anchor": True,
        "protect_equipment": True,
        "fail_closed": True,
    }
    write_json(config_path, config)
    with pytest.raises(ForgeError, match="no provenance report"):
        package_production_unit(
            config_path, tmp_path, qa_path, tmp_path / "unity-unit", tmp_path / "review"
        )


def test_protected_overpaint_is_hash_locked_in_manifest(tmp_path: Path) -> None:
    config_path, qa_path = production_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["depth_pass"] = True
    config["equipment_objects"] = ["Sword*"]
    config["overpaint"] = {
        "enabled": True,
        "control": "depth",
        "anchor": True,
        "protect_equipment": True,
        "fail_closed": True,
    }
    write_json(config_path, config)
    write_json(
        tmp_path / "artifacts" / "text2model-forge-work" / "unit" / "frames" / "unit-overpaint.json",
        {
            "schema_version": 3,
            "unit": "unit",
            "passed": True,
            "depth_control": True,
            "protected_equipment": True,
            "transactional_promotion": True,
            "checkpoint_sha256": "a" * 64,
            "controlnet_sha256": "b" * 64,
            "anchor_sha256": "c" * 64,
            "protected_frames": len(ACTIONS) * len(DIRECTIONS) * 2,
            "equipment_integrity_failures": [],
        },
    )
    result = package_production_unit(
        config_path, tmp_path, qa_path, tmp_path / "unity-unit", tmp_path / "review"
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["production"]["identity_lock"] == "single_master_sha256+protected_equipment"
    assert len(manifest["production"]["overpaint_report_sha256"]) == 64
