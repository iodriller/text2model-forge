import json
from pathlib import Path

import pytest

from assetforge.core import ForgeError, sha256_file
from PIL import Image

from assetforge.morphology import create_morphology_proof, validate_morphology_build, validate_profile


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_example_profile_satisfies_biped_family_contract():
    root = _repo_root()
    profile = json.loads((root / "tools/asset_forge/creatures/example_biped.morphology.json").read_text(encoding="utf-8"))
    contract, failures = validate_profile(profile, root)
    assert failures == []
    assert contract["id"] == "biped_humanoid_v1"
    assert contract["implementation_status"] == "production"


def test_morphology_build_requires_every_landmark_and_current_profile_hash(tmp_path):
    root = _repo_root()
    source_profile = root / "tools/asset_forge/creatures/example_biped.morphology.json"
    profile = json.loads(source_profile.read_text(encoding="utf-8"))
    profile["family_contract"] = str((root / profile["family_contract"]).resolve())
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    master = tmp_path / "example.blend"
    master.write_bytes(b"blend-fixture")
    report = {
        "profile_id": profile["id"],
        "profile_sha256": sha256_file(profile_path),
        "family": profile["family"],
        "passed": True,
        "failures": [],
        "created_parts": profile["silhouette_contract"]["required_landmarks"],
        "posture_actions": profile["silhouette_contract"]["proof_actions"],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_morphology_build(profile_path, report_path, master, root)["passed"] is True

    report["created_parts"] = report["created_parts"][:-1]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ForgeError, match="missing required anatomy landmarks"):
        validate_morphology_build(profile_path, report_path, master, root)


def test_morphology_proof_composes_required_views_and_action_phases(tmp_path):
    root = _repo_root()
    config = json.loads((root / "tools/asset_forge/characters/example_biped.character.json").read_text(encoding="utf-8"))
    config["animations"]["attack"]["phases"] = {
        "anticipation_frame": 0,
        "contact_frame": 1,
        "recovery_frame": 2,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    frames = tmp_path / "frames" / config["id"]
    for action, directions, count in (("idle", ("south", "east"), 1), ("attack", ("south",), 3)):
        for direction in directions:
            folder = frames / action / direction
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                for x in range(20, 44):
                    for y in range(8, 58):
                        image.putpixel((x, y), (80, 130, 70, 255))
                image.save(folder / f"{index:02d}.png")
    output = tmp_path / "proof.png"
    result = create_morphology_proof(
        config_path,
        root / "tools/asset_forge/creatures/example_biped.morphology.json",
        tmp_path / "frames",
        output,
    )
    assert result["geometry_only"] is True
    assert output.is_file()
    with Image.open(output) as proof:
        assert proof.size == (1400, 900)


def test_contract_only_winged_family_cannot_validate_a_production_build(tmp_path):
    root = _repo_root()
    source = json.loads((root / "tools/asset_forge/creatures/example_biped.morphology.json").read_text(encoding="utf-8"))
    source["family"] = "winged_quadruped_v1"
    source["family_contract"] = str(
        (root / "tools/asset_forge/creatures/families/winged_quadruped.family.json").resolve()
    )
    profile_path = tmp_path / "winged.json"
    profile_path.write_text(json.dumps(source), encoding="utf-8")
    master = tmp_path / "winged.blend"
    master.write_bytes(b"not-a-production-winged-creature")
    report = {
        "profile_id": source["id"],
        "profile_sha256": sha256_file(profile_path),
        "family": source["family"],
        "passed": True,
        "failures": [],
        "created_parts": source["silhouette_contract"]["required_landmarks"],
        "posture_actions": source["silhouette_contract"]["proof_actions"],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ForgeError, match="not production-enabled"):
        validate_morphology_build(profile_path, report_path, master, root)
