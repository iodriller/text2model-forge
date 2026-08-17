from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_motion_review_script_compiles_and_requires_full_clip_set() -> None:
    source = (ROOT / "resources" / "adapters" / "review_short_biped_motion.py").read_text(encoding="utf-8")
    compile(source, "review_short_biped_motion.py", "exec")
    for clip in ("idle", "walk", "attack", "hit", "death"):
        assert f'"{clip}"' in source
    assert "all_motion_front_keyposes.png" in source
    assert "LocalDeployMotionReviewer" in source
