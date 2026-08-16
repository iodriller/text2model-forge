from pathlib import Path
import runpy

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_sprite_candidate_scripts_compile() -> None:
    for relative in (
        "adapters/package_motion_sprites.py",
        "adapters/build_motion_evidence.py",
        "adapters/retarget_humanoid_motion.py",
    ):
        path = ROOT / relative
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_sprite_render_config_contains_current_four_clip_checkpoint() -> None:
    import json

    config = json.loads((ROOT / "configs/short_biped_sprite_render.json").read_text(encoding="utf-8"))
    assert set(config["animations"]) == {"idle", "walk", "attack", "death"}
    assert config["directions"] == ["north", "south", "east", "west"]
    assert config["auto_frame"] is True
    assert config["auto_frame_margin_fraction"] >= 0.08


def test_sword_attack_candidate_resolves_physical_side_and_builds_digit_grip_rig() -> None:
    source = (ROOT / "adapters/retarget_humanoid_motion.py").read_text(encoding="utf-8")
    assert '"fixture.weapon.short_biped.v1"' in source
    assert '"articulated_digit_grip_v1"' in source
    assert '"source_weapon_bone": "hand_r"' in source
    assert '"bone": "hand_l"' in source
    assert '"wrist_l": ("hand_r",)' in source
    assert "_source_motion_hand_analysis" in source
    assert "source_motion_dominance_plus_rest_x_alignment_v1" in source
    assert "_build_articulated_grip" in source
    assert "articulated_digit_landmark_grip_v1" in source
    assert '"grip_socket_bone": "grip_socket_right"' in source
    assert "distal_components" in source
    assert "detected_digit_branches" in source
    assert "generated_deform_bones" in source
    assert "base_landmark" in source
    assert "joint_landmark" in source
    assert "tip_landmark" in source
    assert "maximum_weight_sum_delta" in source
    assert "closest_surface_error" in source
    assert "deep_penetrating_vertices" in source
    assert "collapsed_polygons" in source
    assert "body_relative_dimensions" in source
    assert "maximum_grip_error" in source


def test_public_hand_sockets_are_anatomical_despite_legacy_x_side_bone_suffixes() -> None:
    canonical = (ROOT / "adapters/canonical_short_biped_worker.py").read_text(encoding="utf-8")
    blender = (ROOT / "adapters/blender_worker.py").read_text(encoding="utf-8")
    retarget = (ROOT / "adapters/retarget_humanoid_motion.py").read_text(encoding="utf-8")
    assert '"hand_right": {"joint": "hand_l"' in canonical
    assert '"hand_left": {"joint": "hand_r"' in canonical
    assert '"hand_right": "hand_l"' in blender
    assert '"socket": "hand_right"' in retarget
    assert '"bone": "hand_l"' in retarget


def test_qwen_retarget_review_requires_dedicated_grip_closeups() -> None:
    adapter = (ROOT / "adapters/review_retargeted_motion.py").read_text(encoding="utf-8")
    reviewer = (ROOT / "darkness/retarget_review.py").read_text(encoding="utf-8")
    assert 'glob("grip_attack_*_front.png")' in adapter
    assert "missing dedicated grip close-up evidence" in adapter
    assert '"--previous-retarget"' in adapter
    assert "previous iteration grip close-up" in adapter
    assert "fingers visibly close around the handle" in reviewer
    assert "rest-X" in reviewer


def test_sprite_packager_rejects_visible_but_effectively_black_frames(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "adapters/package_motion_sprites.py"))
    frames = tmp_path / "frames" / "darkness_short_biped_candidate"
    for action, (count, _loop) in namespace["ACTIONS"].items():
        for direction in namespace["DIRECTIONS"]:
            folder = frames / action / direction
            folder.mkdir(parents=True)
            for index in range(count):
                image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                image.paste((8, 10, 12, 255), (16, 12, 48, 56))
                image.save(folder / f"{index:02d}.png")
    master = tmp_path / "surface.blend"
    master.write_bytes(b"fixture")
    with pytest.raises(RuntimeError, match="visible_surface_too_dark"):
        namespace["main"](
            [
                "--frames-root",
                str(tmp_path / "frames"),
                "--output-directory",
                str(tmp_path / "package"),
                "--master",
                str(master),
            ]
        )
