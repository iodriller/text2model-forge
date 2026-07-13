from pathlib import Path


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


def test_sword_attack_candidate_declares_a_body_relative_rigid_club_socket() -> None:
    source = (ROOT / "adapters/retarget_humanoid_motion.py").read_text(encoding="utf-8")
    assert '"darkness.club.short_biped.v1"' in source
    assert '"rigid_socket"' in source
    assert '"hand_r"' in source
    assert "body_relative_dimensions" in source
    assert "maximum_grip_error" in source
