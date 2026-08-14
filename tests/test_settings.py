import json
import subprocess
import sys
from pathlib import Path

import pytest

from darkness.settings import resolve_settings, studio_overrides
from darkness.studio_models import new_studio_run


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
    assert resolved.get("stages.D2b.target_quads") == 40000
    assert resolved.origin["stages.D2b.target_quads"] == "profile:advanced"
    # A key advanced.toml does not touch still resolves from base.
    assert resolved.get("stages.D2.worker_id") == "canonical.short_biped"
    assert resolved.origin["stages.D2.worker_id"] == "base"


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
    assert "quality" not in overrides


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


def test_cli_config_show_prints_value_and_origin_for_every_key():
    result = subprocess.run(
        [sys.executable, "-m", "darkness", "config", "show", "--profile", "advanced"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["stages.D2b.target_quads"] == {"value": 40000, "set_by": "profile:advanced"}
    assert report["studio.model"] == {"value": "qwen3_6_27b", "set_by": "base"}


def test_cli_config_show_values_only_is_plain_merged_json():
    result = subprocess.run(
        [sys.executable, "-m", "darkness", "config", "show", "--profile", "simple", "--values-only"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    values = json.loads(result.stdout)
    assert values["studio"]["checkpoint"] == "dreamshaper_xl_v2_turbo.safetensors"
    assert values["asset"]["quality"] == "standard"
