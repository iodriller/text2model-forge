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
