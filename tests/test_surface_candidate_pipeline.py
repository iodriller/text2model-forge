from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


def test_surface_scripts_compile_and_declare_persistent_master_contract() -> None:
    for relative in (
        "adapters/bake_darkness_surface.py",
        "adapters/prepare_semantic_surface_baseline.py",
        "adapters/project_painted_vertex_surface.py",
        "adapters/review_surface_master.py",
        "darkness/surface_review.py",
    ):
        path = ROOT / relative
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    source = (ROOT / "adapters/bake_darkness_surface.py").read_text(encoding="utf-8")
    assert "darkness_surface_master.blend" in source
    assert "maximum_alpha_mismatch_pixels" in source
    assert "rejected_black_projection_atlases" in source
    assert "workflow_run_signature" in source


def test_darkness_surface_runtime_config_keeps_paint_on_one_master(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "adapters/bake_darkness_surface.py"))
    master = tmp_path / "input.blend"
    baseline = tmp_path / "surface" / "semantic_surface_baseline.blend"
    baked = tmp_path / "surface" / "darkness_surface_master.blend"
    args = type(
        "Args",
        (),
        {
            "master": master,
            "checkpoint": "dreamshaper_xl_v2_turbo.safetensors",
            "seed": 42,
        },
    )()
    spec = {
        "asset_id": "test_footman",
        "title": "Test Footman",
        "description": "original stylized armored footman",
        "creative_direction": "chunky mobile-readable shapes",
        "asset_kind": "character",
        "behavior": "deformable_animated",
        "materials": ["worn steel", "blue cloth", "leather"],
        "equipment": [{"description": "one right-hand arming sword"}],
        "animations": ["idle"],
        "locked_features": ["right-hand sword"],
        "negative_constraints": ["no copied insignia"],
        "dimensions_m": [1.0, 1.8, 0.8],
    }
    config = namespace["_config"](args, baked.parent, baseline, baked, spec)
    assert config["source"] == str(baseline.resolve())
    assert config["texture_master"]["baked_source"] == str(baked)
    assert config["texture_master"]["work_root"] == str(baked.parent / "work")
    assert config["texture_master"]["icons_output"] == str(baked.parent / "icons")
    assert config["texture_master"]["views"] >= 6
    assert config["overpaint"]["enabled"] is False
    assert "right-hand arming sword" in config["overpaint"]["prompt"]
    assert config["id"] == "darkness_test_footman_surface"


def test_semantic_baseline_derives_generic_materials_from_asset_spec() -> None:
    source = (ROOT / "adapters/prepare_semantic_surface_baseline.py").read_text(encoding="utf-8")
    for semantic in ("skin", "cloth", "leather", "wood", "iron", "stone"):
        assert f'"{semantic}"' in source
    assert "material_face_counts" in source
    assert "asset_spec_sha256" in source
    assert "surface baseline requires DarknessClub" not in source


def test_qwen_surface_prompt_carries_before_current_numbers_and_stops_iteration() -> None:
    source = (ROOT / "darkness/surface_review.py").read_text(encoding="utf-8")
    assert "previous_current_numeric_history" in source
    assert "do not justify endless iteration" in source
    assert "one bounded revision" in source
    assert "adjust_club_contrast" in source
