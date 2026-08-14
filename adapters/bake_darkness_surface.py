"""Bake one ComfyUI-assisted painted master for every downstream Darkness render."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageStat

from assetforge.bake import _run_blender as run_bake_blender
from assetforge.bake import bake_texture_master


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument(
        "--asset-spec",
        "--character-spec",
        dest="asset_spec",
        type=Path,
        help="Darkness Studio asset_spec.json used for material semantics and paint prompts.",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--checkpoint", default="dreamshaper_xl_v2_turbo.safetensors")
    parser.add_argument("--seed", type=int, default=7132026)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_asset_spec(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "asset_id": "legacy_goblin",
            "title": "Legacy Goblin Surface Candidate",
            "description": "original stylized dark fantasy goblin raider",
            "creative_direction": "broad readable mobile game shapes",
            "asset_kind": "character",
            "behavior": "deformable_animated",
            "materials": ["olive skin", "charcoal cloth", "worn leather", "dark wood", "iron"],
            "equipment": [{"description": "rough wooden club with dark iron bands"}],
            "animations": ["idle"],
            "locked_features": ["same exact geometry and pose"],
            "negative_constraints": ["changed silhouette", "extra limbs", "text", "watermark"],
            "dimensions_m": [1.0, 1.4, 0.8],
        }
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("asset_id"):
        raise ValueError("asset spec must be a Studio asset specification object")
    return value


def _config(
    args: argparse.Namespace,
    output: Path,
    baseline: Path,
    baked: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    work = output / "work"
    asset_id = str(spec.get("asset_id", "darkness_asset"))
    materials = ", ".join(str(item) for item in spec.get("materials", []))
    equipment = ", ".join(
        str(item.get("description", "")) for item in spec.get("equipment", []) if isinstance(item, dict)
    )
    positive = ", ".join(
        item
        for item in (
            str(spec.get("description", "original game asset")),
            str(spec.get("creative_direction", "stylized readable production asset")),
            materials,
            equipment,
            ", ".join(str(item) for item in spec.get("locked_features", [])),
            "same exact geometry and pose in every view, coherent hand-authored PBR surface",
        )
        if item
    )
    negative = ", ".join(
        [str(item) for item in spec.get("negative_constraints", [])]
        + [
            "changed geometry",
            "changed silhouette",
            "missing component",
            "extra component",
            "text",
            "watermark",
            "baked directional lighting",
        ]
    )
    dimensions = [float(item) for item in spec.get("dimensions_m", [1.0, 1.8, 1.0])]
    view_scale = max(dimensions) * 1.75
    animated = bool(spec.get("animations")) and spec.get("behavior") == "deformable_animated"
    config = {
        "schema_version": 1,
        "id": f"darkness_{asset_id}_surface",
        "display_name": f"{spec.get('title', asset_id)} Surface Candidate",
        "source": str(baseline.resolve()),
        "render_size": [768, 768],
        "orthographic_scale": view_scale,
        "animations": {"idle": {"actions": ["idle"], "sample_frames": [1]}},
        "overpaint": {
            "enabled": False,
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "steps": 7,
            "cfg": 2.5,
            "prompt": positive,
            "negative": negative,
            "control_strength": 0.72,
        },
        "texture_master": {
            "enabled": True,
            "baked_source": str(baked),
            "views": 6,
            "view_size": 512,
            "view_scale": view_scale,
            "view_elevation": 14.0,
            "view_target_height": 0.0,
            "bake_resolution": 1024,
            "grid_cell": 512,
            "grid_columns": 3,
            "denoise": 0.44,
            "control_strength": 0.72,
            "work_root": str(work),
            "icons_output": str(output / "icons"),
        },
    }
    if animated:
        config["animation_object"] = "DarknessShortBipedRig"
        config["texture_master"].update({"pose_action": "idle", "pose_frame": 1})
    return config


def _run_blender(command: list[str], *, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    if Path(command[0]).name.lower() == "blender-launcher.exe":
        def literal(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        script = (
            f"$p=Start-Process -FilePath {literal(command[0])} "
            f"-ArgumentList {literal(subprocess.list2cmdline(command[1:]))} "
            "-WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        error_path = work / "semantic_surface_error.txt"
        detail = error_path.read_text(encoding="utf-8") if error_path.is_file() else ""
        raise RuntimeError(
            "semantic surface preparation failed: "
            + (detail or (completed.stdout + "\n" + completed.stderr)[-4000:])
        )


def _difference(first: Image.Image, second: Image.Image) -> float:
    left = first.convert("RGB").resize(second.size, Image.Resampling.LANCZOS)
    means = ImageStat.Stat(ImageChops.difference(left, second.convert("RGB"))).mean
    return sum(means) / len(means)


def _review_board(
    output: Path,
    records: list[dict[str, Any]],
    *,
    before_folder: Path,
    after_folder: Path,
) -> tuple[Path, dict[str, Any]]:
    pairs: list[tuple[Image.Image, Image.Image]] = []
    deltas: list[float] = []
    alpha_mismatches: list[int] = []
    for record in records[:4]:
        index = int(record["index"])
        with Image.open(before_folder / f"view_{index:02d}.png") as source:
            before = source.convert("RGBA")
        with Image.open(after_folder / f"view_{index:02d}.png") as source:
            after = source.convert("RGBA")
        pairs.append((before, after))
        deltas.append(_difference(before, after))
        alpha = ImageChops.difference(before.getchannel("A"), after.getchannel("A"))
        alpha_mismatches.append(sum(1 for value in alpha.get_flattened_data() if value != 0))

    cell = 384
    header = 64
    canvas = Image.new("RGB", (cell * 2, header + cell * len(pairs)), "#0d1118")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 14), "BEFORE: deterministic 3D", fill="white")
    draw.text((cell + 14, 14), "AFTER: one projected surface master", fill="white")
    for row, (before, after) in enumerate(pairs):
        y = header + row * cell
        for column, image in enumerate((before, after)):
            flat = Image.new("RGBA", image.size, (18, 22, 29, 255))
            flat.alpha_composite(image)
            canvas.paste(flat.convert("RGB").resize((cell, cell)), (column * cell, y))
    path = output / "surface_review.png"
    canvas.save(path)
    return path, {
        "mean_absolute_paint_delta": round(sum(deltas) / max(len(deltas), 1), 4),
        "per_view_mean_absolute_paint_delta": [round(value, 4) for value in deltas],
        "maximum_alpha_mismatch_pixels": max(alpha_mismatches, default=0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    master = args.master.resolve()
    blender = args.blender.resolve()
    output = args.output_directory.resolve()
    spec = _load_asset_spec(args.asset_spec)
    prior_board_hash = None
    prior_report_path = output / "surface_validation.json"
    if prior_report_path.is_file():
        try:
            prior_board_hash = json.loads(prior_report_path.read_text(encoding="utf-8")).get(
                "review_board_sha256"
            )
        except (OSError, json.JSONDecodeError):
            pass
    if not master.is_file():
        raise FileNotFoundError(master)
    if not blender.is_file():
        raise FileNotFoundError(blender)
    output.mkdir(parents=True, exist_ok=True)
    baseline = output / "semantic_surface_baseline.blend"
    baseline_report = output / "semantic_surface_baseline.json"
    regenerate_baseline = True
    if baseline.is_file() and baseline_report.is_file():
        prior_baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
        expected_spec_hash = _sha256(args.asset_spec.resolve()) if args.asset_spec is not None else None
        regenerate_baseline = (
            prior_baseline.get("source_sha256") != _sha256(master)
            or prior_baseline.get("asset_spec_sha256") != expected_spec_hash
        )
    if regenerate_baseline:
        baseline_command = [
            str(blender),
            "--background",
            str(master),
            "--python-exit-code",
            "1",
            "--python",
            str(repo / "adapters/prepare_semantic_surface_baseline.py"),
            "--",
            "--source",
            str(master),
            "--output",
            str(baseline),
            "--report",
            str(baseline_report),
        ]
        if args.asset_spec is not None:
            baseline_command.extend(["--asset-spec", str(args.asset_spec.resolve())])
        _run_blender(
            baseline_command,
            work=output,
        )
        for stale_anchor in (output / "anchors").glob("darkness_*_surface.anchor*.png"):
            stale_anchor.unlink()
    baked = output / "darkness_surface_master.blend"
    config_path = output / "surface_config.json"
    config_path.write_text(
        json.dumps(_config(args, output, baseline, baked, spec), indent=2) + "\n", encoding="utf-8"
    )

    provenance_path = output / "surface_provenance.json"
    painted_metadata = output / "work" / "views.json"
    provenance = None
    if not args.force and provenance_path.is_file() and painted_metadata.is_file():
        candidate = json.loads(provenance_path.read_text(encoding="utf-8"))
        metadata_candidate = json.loads(painted_metadata.read_text(encoding="utf-8"))
        painted_complete = all(
            (output / "work" / "views" / f"painted_{int(record['index']):02d}.png").is_file()
            for record in metadata_candidate.get("views", [])
        )
        if (
            painted_complete
            and candidate.get("master_sha256") == _sha256(baseline)
            and candidate.get("checkpoint") == args.checkpoint
        ):
            provenance = candidate
    if provenance is None:
        provenance = bake_texture_master(
            config_path,
            repo,
            str(blender),
            base_url=args.comfy_url,
            timeout_seconds=args.timeout_seconds,
            force=args.force,
        )
    projection_report_path = output / "painted_vertex_projection.json"
    _run_blender(
        [
            str(blender),
            "--background",
            str(baseline),
            "--python-exit-code",
            "1",
            "--python",
            str(repo / "adapters/project_painted_vertex_surface.py"),
            "--",
            "--metadata",
            str(output / "work" / "views.json"),
            "--views",
            str(output / "work" / "views"),
            "--output",
            str(baked),
            "--report",
            str(projection_report_path),
        ],
        work=output,
    )
    projection_report = json.loads(projection_report_path.read_text(encoding="utf-8"))
    provenance["final_surface_method"] = projection_report["method"]
    provenance["projection_report"] = str(projection_report_path)
    provenance["baked_sha256"] = _sha256(baked)
    _write_json(output / "surface_provenance.json", provenance)
    metadata = json.loads((output / "work" / "views.json").read_text(encoding="utf-8"))
    verification = output / "verification"
    run_bake_blender(
        str(blender),
        baked,
        "render-views",
        config_path,
        verification,
        repo,
    )
    board, image_metrics = _review_board(
        output,
        list(metadata["views"]),
        before_folder=output / "work" / "views",
        after_folder=verification / "views",
    )
    manifest = json.loads((output / "work" / "bake-manifest.json").read_text(encoding="utf-8"))

    atlas_records: list[dict[str, Any]] = []
    failures: list[str] = []
    for record in manifest.get("objects", []):
        atlas = Path(str(record["atlas"])).resolve()
        if not atlas.is_file():
            failures.append(f"missing_atlas:{record.get('object')}")
            continue
        with Image.open(atlas).convert("RGB") as atlas_image:
            atlas_luminance = sum(ImageStat.Stat(atlas_image).mean) / 3.0
        atlas_records.append(
            {
                "object": record["object"],
                "path": str(atlas),
                "resolution": int(record["resolution"]),
                "sha256": _sha256(atlas),
                "mean_luminance_8bit": round(atlas_luminance, 4),
                "production_eligible": atlas_luminance >= 2.0,
            }
        )
    objects = {str(record["object"]) for record in atlas_records}
    rejected_atlases = [record["object"] for record in atlas_records if not record["production_eligible"]]
    if image_metrics["mean_absolute_paint_delta"] < 2.0:
        failures.append("paint_did_not_materially_change_surface")
    if image_metrics["maximum_alpha_mismatch_pixels"] != 0:
        failures.append("paint_changed_geometry_silhouette")
    if not baked.is_file():
        failures.append("baked_master_missing")
    if projection_report.get("automatic_gate_passed") is not True:
        failures.append("painted_vertex_projection_failed")
    if not provenance.get("checkpoint_sha256"):
        failures.append("checkpoint_hash_not_registered")
    if not provenance.get("controlnet_sha256"):
        failures.append("controlnet_hash_not_registered")

    report = {
        "schema_version": 1,
        "stage": "darkness_surface_master",
        "source_master": str(master),
        "source_master_sha256": _sha256(master),
        "semantic_baseline": str(baseline),
        "semantic_baseline_sha256": _sha256(baseline),
        "surface_master": str(baked),
        "surface_master_sha256": _sha256(baked) if baked.is_file() else None,
        "review_board": str(board),
        "review_board_sha256": _sha256(board),
        "views": len(metadata["views"]),
        "atlases": atlas_records,
        "rejected_black_projection_atlases": rejected_atlases,
        "surface_storage": "persistent_corner_color_plus_stable_uv",
        "projection_method": projection_report["method"],
        "projection_metrics": projection_report["objects"],
        "checkpoint": provenance.get("checkpoint"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "controlnet": provenance.get("controlnet"),
        "controlnet_sha256": provenance.get("controlnet_sha256"),
        "seed": provenance.get("seed"),
        "workflow_run_signature": provenance.get("run_signature"),
        "image_metrics": image_metrics,
        "hard_failures": failures,
        "automatic_gate_passed": not failures,
        "human_approval_required": True,
        "human_approved": False,
    }
    _write_json(output / "surface_validation.json", report)
    if prior_board_hash != report["review_board_sha256"]:
        for stale_review in (
            output / "qwen_surface_review.json",
            output / "qwen_surface_mediator.json",
            output / "human_review.md",
        ):
            if stale_review.is_file():
                stale_review.unlink()
    if failures:
        raise RuntimeError("surface gate failed: " + ", ".join(failures))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    run(_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
