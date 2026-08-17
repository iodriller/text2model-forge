import json
import subprocess
import sys
from pathlib import Path

import pytest

from text2model_forge.settings import quality_overrides, resolve_settings, studio_overrides
from text2model_forge.studio_models import new_studio_run


def test_base_profile_matches_documented_studio_defaults():
    resolved = resolve_settings(profile="base")
    assert resolved.get("studio.checkpoint") == "dreamshaper_xl_v2_turbo.safetensors"
    assert resolved.get("studio.model") == "qwen3_6_27b"
    assert resolved.origin["studio.checkpoint"] == "base"


def test_simple_profile_overlays_without_losing_base_values():
    resolved = resolve_settings(profile="simple")
    # simple.toml only sets [asset].quality and [optimizer].enabled; everything
    # else must still come from base.toml.
    assert resolved.get("asset.quality") == "standard"
    assert resolved.origin["asset.quality"] == "profile:simple"
    assert resolved.get("studio.checkpoint") == "dreamshaper_xl_v2_turbo.safetensors"
    assert resolved.origin["studio.checkpoint"] == "base"


def test_advanced_profile_overrides_stage_parameters():
    resolved = resolve_settings(profile="advanced")
    assert resolved.get("stages.D2.texture_size") == 4096
    assert resolved.origin["stages.D2.texture_size"] == "profile:advanced"
    # A key advanced.toml does not touch still resolves from base.
    assert resolved.get("stages.D2.worker_id") == "trellis2.4b"
    assert resolved.origin["stages.D2.worker_id"] == "base"


def test_base_profile_names_the_worker_d2_actually_calls():
    """Regression: base.toml claimed [stages.D2].worker_id was
    canonical.short_biped (the deterministic procedural fallback used by
    `text2model_forge run-worker`), while _run_d2 has always called trellis2.4b.
    The documented default must name the worker the stage really uses."""
    assert resolve_settings(profile="base").get("stages.D2.worker_id") == "trellis2.4b"


def test_adapter_defaults_are_separate_from_live_stage_config():
    """Values that only mirror a standalone script's argparse defaults live
    under [adapter_defaults], not [stages.*], so nobody mistakes them for
    settings the pipeline reads."""
    resolved = resolve_settings(profile="base")
    assert resolved.get("adapter_defaults.retopology.target_quads") == 20000
    assert resolved.get("stages.D2b") is None  # D2b is not a pipeline stage at all
    assert resolved.get("stages.D6") is None  # D6 adopts D4 output; runs no worker
    assert resolved.get("stages.D10") is None  # D10 adopts D9 output; runs no worker


def test_unknown_profile_falls_back_to_base_only():
    resolved = resolve_settings(profile="does-not-exist")
    assert resolved.get("studio.model") == "qwen3_6_27b"
    assert resolved.origin["studio.model"] == "base"


def test_run_overrides_win_over_profile():
    resolved = resolve_settings(profile="simple", run_overrides={"studio": {"checkpoint": "custom.safetensors"}})
    assert resolved.get("studio.checkpoint") == "custom.safetensors"
    assert resolved.origin["studio.checkpoint"] == "run"
    # sibling values under the same table are untouched by the partial override
    assert resolved.get("studio.model") == "qwen3_6_27b"


def test_stage_overrides_win_over_run_overrides():
    resolved = resolve_settings(
        profile="simple",
        run_overrides={"studio": {"checkpoint": "run-level.safetensors"}},
        stage_overrides={"studio": {"checkpoint": "stage-level.safetensors"}},
    )
    assert resolved.get("studio.checkpoint") == "stage-level.safetensors"
    assert resolved.origin["studio.checkpoint"] == "stage"


def test_machine_local_studio_defaults_override_profile(tmp_path):
    machine = tmp_path / "config.local.toml"
    machine.write_text('[studio_defaults]\nmodel = "machine-pinned-model"\n', encoding="utf-8")
    resolved = resolve_settings(profile="simple", machine_path=machine)
    assert resolved.get("studio.model") == "machine-pinned-model"
    assert resolved.origin["studio.model"] == "machine"


def test_missing_base_toml_in_an_explicit_repo_root_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_settings(profile="base", repo_root=tmp_path)


def test_studio_defaults_are_read_from_the_same_file_as_worker_bindings():
    """Regression: resolve_settings() defaulted to CWD/config.local.toml while
    text2model_forge.config read worker bindings from the package root, so the
    documented 'copy machine.example.toml to config.local.toml' workflow
    silently dropped [studio_defaults] unless CWD happened to match."""
    from text2model_forge.config import default_config_path
    from text2model_forge.settings import _default_machine_path

    assert _default_machine_path() == default_config_path()


def test_project_profiles_directory_wins_over_bundled_defaults(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "base.toml").write_text('[studio]\nmodel = "project-base-model"\n', encoding="utf-8")
    resolved = resolve_settings(profile="base", repo_root=tmp_path)
    assert resolved.get("studio.model") == "project-base-model"


def test_studio_overrides_extracts_only_known_studio_run_fields():
    resolved = resolve_settings(profile="simple")
    overrides = studio_overrides(resolved)
    assert overrides["checkpoint"] == "dreamshaper_xl_v2_turbo.safetensors"
    assert "asset_kind" not in overrides  # [asset] table is not a StudioRun field
    assert "quality" not in overrides  # the tier NAME is not a field; its resolved values are


def test_quality_overrides_resolves_asset_quality_to_its_tier_section():
    standard = resolve_settings(profile="simple")  # simple.toml pins quality="standard"
    assert quality_overrides(standard) == {
        "concept_steps": 30,
        "concept_cfg": 6.0,
        "concept_candidates": 3,
        "concept_width": 768,
        "concept_height": 1024,
        "concept_min_quality_score": 0.4,
        "vae_tiling": True,
    }

    high = resolve_settings(profile="advanced")  # advanced.toml pins quality="high"
    assert quality_overrides(high) == {
        "concept_steps": 45,
        "concept_cfg": 7.0,
        "concept_candidates": 6,
        "concept_width": 768,
        "concept_height": 1024,
        "concept_min_quality_score": 0.48,
        "vae_tiling": True,
    }


def test_quality_overrides_excludes_sprite_views_since_it_is_not_a_studio_run_field():
    resolved = resolve_settings(profile="advanced")
    assert "sprite_views" not in quality_overrides(resolved)


def test_quality_overrides_is_empty_for_an_unknown_tier():
    resolved = resolve_settings(
        profile="simple", run_overrides={"asset": {"quality": "ultra-does-not-exist"}}
    )
    assert quality_overrides(resolved) == {}


def test_studio_overrides_includes_the_resolved_quality_tiers_values():
    resolved = resolve_settings(profile="advanced")
    overrides = studio_overrides(resolved)
    assert overrides["concept_steps"] == 45
    assert overrides["concept_cfg"] == 7.0


def test_standard_quality_matches_studio_runs_own_field_defaults_exactly():
    """The whole point of pinning standard to concept_workflow()'s real
    defaults in base.toml: a run built from the resolved base/simple
    overrides must be pixel-for-pixel identical to one built with none."""
    from text2model_forge.studio_models import new_studio_run

    default_run = new_studio_run("run.a", "a run with absolutely no overrides")
    resolved = resolve_settings(profile="simple")
    resolved_run = new_studio_run("run.b", "a run built from resolved simple-profile overrides", studio_overrides(resolved))
    assert resolved_run.concept_steps == default_run.concept_steps == 30
    assert resolved_run.concept_cfg == default_run.concept_cfg == 6.0


def test_studio_overrides_translates_empty_string_to_none_for_optional_fields():
    resolved = resolve_settings(profile="simple", run_overrides={"studio": {"prop_lora": ""}})
    overrides = studio_overrides(resolved)
    assert overrides["prop_lora"] is None


def test_new_studio_run_applies_resolved_overrides_and_keeps_normal_default_identical():
    default_run = new_studio_run("run.default", "a default run with no overrides at all")
    resolved = resolve_settings(profile="base")
    resolved_run = new_studio_run(
        "run.resolved", "a run built from the base-profile resolved overrides", studio_overrides(resolved)
    )
    # base.toml mirrors StudioRun's own field defaults, so applying it changes nothing observable.
    assert resolved_run.checkpoint == default_run.checkpoint
    assert resolved_run.model == default_run.model
    assert resolved_run.style_lora == default_run.style_lora
    assert resolved_run.prop_lora == default_run.prop_lora


def test_new_studio_run_with_custom_override_actually_changes_the_field():
    run = new_studio_run("run.custom", "a run overridden to use a different checkpoint entirely", {"checkpoint": "other.safetensors"})
    assert run.checkpoint == "other.safetensors"


def test_cli_config_show_prints_value_and_origin_for_every_key(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "text2model_forge", "config", "show", "--profile", "advanced",
            # A subprocess cannot see conftest's monkeypatch, so it would read
            # the developer's real config.local.toml and report "set_by:
            # machine" for anything that file overrides. Point it at a path
            # that cannot exist so the assertions below stay about base and
            # profile layers only.
            "--machine-path", str(tmp_path / "absent.toml"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["stages.D2.texture_size"] == {"value": 4096, "set_by": "profile:advanced"}
    assert report["studio.model"] == {"value": "qwen3_6_27b", "set_by": "base"}


def test_cli_config_show_values_only_is_plain_merged_json(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "text2model_forge", "config", "show", "--profile", "simple",
            "--values-only", "--machine-path", str(tmp_path / "absent.toml"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    values = json.loads(result.stdout)
    assert values["studio"]["checkpoint"] == "dreamshaper_xl_v2_turbo.safetensors"
    assert values["asset"]["quality"] == "standard"
