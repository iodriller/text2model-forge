"""Resume the Darkness motion-library -> evidence -> sprite -> Unity candidate chain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


STAGES = (
    "retarget",
    "retarget_evidence",
    "retarget_qwen_review",
    "surface_bake",
    "surface_qwen_review",
    "sprite_render",
    "sprite_package",
    "sprite_qwen_review",
    "unity_smoke_bundle",
    "unity_candidate_validation",
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--motion-source", type=Path, required=True)
    parser.add_argument(
        "--character-spec",
        type=Path,
        help="Darkness Studio character_spec.json for equipment, naming, and surface semantics.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--blender",
        type=Path,
        help="Blender executable; required only when a retarget or sprite-render stage must run.",
    )
    parser.add_argument("--unity", type=Path)
    parser.add_argument(
        "--unity-result",
        type=Path,
        help="Result directory returned by the standalone Unity smoke computer.",
    )
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--model", default="qwen3_6_27b")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--surface-checkpoint", default="dreamshaper_xl_v2_turbo.safetensors")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--stop-after",
        choices=STAGES,
        help="Return after this resumable stage so an external control plane can enforce a human gate.",
    )
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


def _require_executable(path: Path | None, stage: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{stage} requires its executable dependency: {path}")
    return path


def _run(command: list[str], *, cwd: Path, timeout: int, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command}\n"
            f"stdout={completed.stdout[-4000:]}\nstderr={completed.stderr[-4000:]}"
        )


def _launch_and_wait(
    command: list[str],
    *,
    cwd: Path,
    success: Path,
    failure: Path | None,
    timeout: int,
) -> None:
    if Path(command[0]).name.lower() == "blender-launcher.exe":
        def literal(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        script = (
            f"$p=Start-Process -FilePath {literal(command[0])} "
            f"-ArgumentList {literal(subprocess.list2cmdline(command[1:]))} "
            "-WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode"
        )
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ]
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if success.is_file():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return
        if failure is not None and failure.is_file():
            raise RuntimeError(failure.read_text(encoding="utf-8"))
        if process.poll() not in (None, 0):
            raise RuntimeError(f"background worker exited with code {process.returncode}: {command}")
        time.sleep(0.5)
    if process.poll() is None:
        process.kill()
    raise TimeoutError(f"timed out waiting for {success}")


def _status_path(root: Path) -> Path:
    return root / "pipeline_status.json"


def _approved_automatic_candidate_manifest(
    path: Path, *, expected_master_sha256: str | None = None
) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema_version") == 1
        and manifest.get("automatic_gate_passed") is True
        and manifest.get("quality_gate_version") == 2
        and manifest.get("hard_failures") == []
        and manifest.get("human_approval_required") is True
        and manifest.get("human_approved") is False
        and (
            expected_master_sha256 is None
            or manifest.get("source_master_sha256") == expected_master_sha256
        )
    )


def _bundle_is_current(bundle_manifest: Path, candidate_manifest: Path) -> bool:
    if not bundle_manifest.is_file() or not candidate_manifest.is_file():
        return False
    try:
        bundle = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bundle.get("candidate_manifest_sha256") == _sha256(candidate_manifest)


def _unity_result_is_current(report_path: Path, bundle_manifest: Path) -> bool:
    if not report_path.is_file() or not bundle_manifest.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return report.get("bundle_manifest_sha256") == _sha256(bundle_manifest)


def _archive_partial(path: Path) -> None:
    """Preserve an interrupted stage and give the deterministic retry an empty directory."""
    if not path.exists() or not any(path.iterdir()):
        return
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    candidate = path.with_name(path.name + f".partial-{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(path.name + f".partial-{timestamp}-{suffix}")
        suffix += 1
    path.rename(candidate)


def _record(status: dict[str, object], root: Path, stage: str, state: str, detail: str) -> None:
    status["stages"][stage] = {"state": state, "detail": detail}
    status["current_stage"] = stage
    _write_json(_status_path(root), status)


def _stop_after(args: argparse.Namespace, status: dict[str, object], root: Path, stage: str) -> bool:
    if args.stop_after != stage:
        return False
    status["outcome"] = f"stopped_after_{stage}"
    status["human_review"] = str(_write_review_index(root, unity_state="ready"))
    _write_json(_status_path(root), status)
    return True


def _write_review_index(root: Path, *, unity_state: str) -> Path:
    path = root / "human_review.md"
    interactive_review = (
        "- [Open interactive motion review](unity_human_review/review.html); the matching Unity project opens with "
        "`unity_human_review/open_unity_review.ps1`."
        if (root / "unity_human_review/review.html").is_file()
        else "- [Open interactive motion review](unity_smoke_bundle/review.html); the matching Unity project opens "
        "with `unity_smoke_bundle/open_unity_review.ps1`."
    )
    path.write_text(
        "\n".join(
            [
                "# Darkness painted motion candidate — human review",
                "",
                "This is a non-promoting review checkpoint. It does not replace the live Unity goblin.",
                "",
                "## Motion",
                "",
                "- [Four-motion key-pose sheet](retarget/human_review/all_motion_front_keyposes.png)",
                "- [Motion critic and mediator summary](retarget/human_review/human_review.md)",
                "- [Editable retargeted Blender master](retarget/quaternius_retargeted_candidate.blend)",
                "- [GLB with idle, walk, attack, and death](retarget/quaternius_retargeted_candidate.glb)",
                "",
                "## Persistent painted surface",
                "",
                "- [Before/current multi-view paint board](surface/surface_review.png)",
                "- [Surface critic and mediator summary](surface/human_review.md)",
                "- [Surface numeric/provenance gate](surface/surface_validation.json)",
                "- [Editable painted Blender master](surface/darkness_surface_master.blend)",
                "",
                "## Directional sprites",
                "",
                "- [Gameplay sprite proof](sprites/package/sprite_review.png)",
                "- [Sprite critic and mediator summary](sprites/package/human_review.md)",
                "- [Hash-bound candidate manifest](sprites/package/candidate_unit_manifest.json)",
                "",
                "## Unity",
                "",
                (
                    "- Unity candidate validation is complete: [capture](unity/unity_candidate_capture.png) and "
                    "[report](unity/unity_candidate_validation.json)."
                    if unity_state == "complete"
                    else "- Standalone Unity smoke bundle is ready: [instructions](unity_smoke_bundle/README.md), "
                    "[transfer ZIP](unity_smoke_bundle.zip), and [ZIP hash](unity_smoke_bundle.zip.json). "
                    "Run it on the licensed Unity computer, return its `result` folder, then resume with "
                    "`--unity-result <result-folder>`. No game project import is required."
                ),
                interactive_review,
                "",
                "## Human decision",
                "",
                "Approve or reject the painted identity, motion direction, and gameplay readability. Human approval "
                "remains false until an explicit decision is recorded.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def run(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo_root.resolve()
    target = args.target.resolve()
    motion_source = args.motion_source.resolve()
    character_spec = args.character_spec.resolve() if args.character_spec is not None else None
    blender = args.blender.resolve() if args.blender is not None else None
    root = args.output_root.resolve()
    for path in (repo, target, motion_source, character_spec):
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
    root.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "schema_version": 1,
        "pipeline": "darkness_motion_surface_candidate_v2",
        "target": str(target),
        "target_sha256": _sha256(target),
        "motion_source": str(motion_source),
        "motion_source_sha256": _sha256(motion_source),
        "character_spec": str(character_spec) if character_spec else None,
        "character_spec_sha256": _sha256(character_spec) if character_spec else None,
        "stages": {name: {"state": "pending", "detail": ""} for name in STAGES},
        "current_stage": "retarget",
        "human_approval_required": True,
        "human_approved": False,
    }
    existing = _status_path(root)
    if existing.is_file():
        prior = json.loads(existing.read_text(encoding="utf-8"))
        if prior.get("target_sha256") != status["target_sha256"] or prior.get(
            "motion_source_sha256"
        ) != status["motion_source_sha256"]:
            raise RuntimeError("resume refused: target or motion-source hash changed")
        if prior.get("character_spec_sha256") != status["character_spec_sha256"]:
            raise RuntimeError("resume refused: character specification hash changed")
        status = prior
        status["pipeline"] = "darkness_motion_surface_candidate_v2"
        stages = status.setdefault("stages", {})
        for name in STAGES:
            stages.setdefault(name, {"state": "pending", "detail": ""})

    adapters = repo / "adapters"
    retarget = root / "retarget"
    retarget_report = retarget / "retarget_validation.json"
    retarget_glb = retarget / "quaternius_retargeted_candidate.glb"
    retarget_error = retarget / "retarget_error.txt"
    if not retarget_glb.is_file():
        blender = _require_executable(blender, "retarget")
        _archive_partial(retarget)
        _record(status, root, "retarget", "running", "Baking four Quaternius actions onto the Darkness rig.")
        retarget_command = [
                str(blender),
                "--background",
                "--factory-startup",
                "--offline-mode",
                "--python-exit-code",
                "23",
                "--python",
                str(adapters / "retarget_humanoid_motion.py"),
                "--",
                "--target",
                str(target),
                "--source",
                str(motion_source),
                "--output-directory",
                str(retarget),
                "--render-size",
                "320",
            ]
        if character_spec is not None:
            retarget_command.extend(["--character-spec", str(character_spec)])
        _launch_and_wait(
            retarget_command,
            cwd=repo,
            success=retarget_glb,
            failure=retarget_error,
            timeout=args.timeout_seconds,
        )
    _record(status, root, "retarget", "complete", str(retarget_report))

    evidence = retarget / "human_review"
    master_sheet = evidence / "all_motion_front_keyposes.png"
    if not master_sheet.is_file():
        _archive_partial(evidence)
        _record(status, root, "retarget_evidence", "running", "Building human-readable key-pose sheets.")
        _run(
            [
                sys.executable,
                str(adapters / "build_motion_evidence.py"),
                "--input-directory",
                str(retarget),
                "--output-directory",
                str(evidence),
                "--clips",
                "idle",
                "walk",
                "attack",
                "death",
                "--title",
                "Quaternius CC0 retarget on Darkness goblin",
            ],
            cwd=repo,
            timeout=120,
        )
    _record(status, root, "retarget_evidence", "complete", str(master_sheet))

    retarget_mediator = evidence / "qwen_retarget_mediator.json"
    if not retarget_mediator.is_file():
        _record(status, root, "retarget_qwen_review", "running", "Running critic and independent mediator once.")
        _run(
            [
                sys.executable,
                str(adapters / "review_retargeted_motion.py"),
                "--input-directory",
                str(retarget),
                "--evidence-directory",
                str(evidence),
                "--model",
                args.model,
            ],
            cwd=repo,
            timeout=args.timeout_seconds,
        )
    _record(status, root, "retarget_qwen_review", "complete", str(retarget_mediator))
    if _stop_after(args, status, root, "retarget_qwen_review"):
        return status

    surface = root / "surface"
    surface_master = surface / "darkness_surface_master.blend"
    surface_report = surface / "surface_validation.json"
    if not surface_master.is_file() or not surface_report.is_file():
        blender = _require_executable(blender, "surface_bake")
        _archive_partial(surface)
        _record(
            status,
            root,
            "surface_bake",
            "running",
            "Painting canonical depth-locked views and projecting them once into a persistent UV master.",
        )
        try:
            surface_command = [
                    sys.executable,
                    str(adapters / "bake_darkness_surface.py"),
                    "--master",
                    str(retarget / "quaternius_retargeted_candidate.blend"),
                    "--output-directory",
                    str(surface),
                    "--blender",
                    str(blender),
                    "--repo-root",
                    str(repo),
                    "--comfy-url",
                    args.comfy_url,
                    "--checkpoint",
                    args.surface_checkpoint,
                    "--timeout-seconds",
                    str(args.timeout_seconds),
                ]
            if character_spec is not None:
                surface_command.extend(["--character-spec", str(character_spec)])
            _run(
                surface_command,
                cwd=repo,
                timeout=max(args.timeout_seconds, 1800),
            )
        except RuntimeError as error:
            _record(status, root, "surface_bake", "waiting_for_dependency", str(error)[-1600:])
            status["outcome"] = "waiting_for_surface_painter"
            _write_json(_status_path(root), status)
            return status
    surface_validation = json.loads(surface_report.read_text(encoding="utf-8"))
    if surface_validation.get("automatic_gate_passed") is not True:
        raise RuntimeError(f"surface master failed automatic gate: {surface_report}")
    surface_master_sha256 = _sha256(surface_master)
    if surface_validation.get("surface_master_sha256") != surface_master_sha256:
        raise RuntimeError("surface master hash no longer matches its validation report")
    _record(status, root, "surface_bake", "complete", str(surface_report))

    surface_mediator = surface / "qwen_surface_mediator.json"
    if not surface_mediator.is_file():
        _record(
            status,
            root,
            "surface_qwen_review",
            "running",
            "Giving Qwen labeled before/current images, numeric deltas, operation history, and one referee pass.",
        )
        _run(
            [
                sys.executable,
                str(adapters / "review_surface_master.py"),
                "--surface-directory",
                str(surface),
                "--model",
                args.model,
            ],
            cwd=repo,
            timeout=args.timeout_seconds,
        )
    _record(status, root, "surface_qwen_review", "complete", str(surface_mediator))
    if _stop_after(args, status, root, "surface_qwen_review"):
        return status

    frames = root / "sprites/frames"
    sprite_last = frames / "darkness_short_biped_candidate/death/west/09.png"
    sprite_render_record = root / "sprites/render_contract.json"
    render_is_current = False
    if sprite_render_record.is_file() and sprite_last.is_file():
        try:
            render_is_current = (
                json.loads(sprite_render_record.read_text(encoding="utf-8")).get("surface_master_sha256")
                == surface_master_sha256
            )
        except (OSError, json.JSONDecodeError):
            render_is_current = False
    if not render_is_current:
        _archive_partial(root / "sprites")
        frames = root / "sprites/frames"
        sprite_last = frames / "darkness_short_biped_candidate/death/west/09.png"
        blender = _require_executable(blender, "sprite_render")
        _record(
            status,
            root,
            "sprite_render",
            "running",
            "Rendering four actions in four directions from the hash-bound painted master.",
        )
        _launch_and_wait(
            [
                str(blender),
                str(surface_master),
                "--background",
                "--offline-mode",
                "--python-exit-code",
                "23",
                "--python",
                str(repo / "blender/render_sprites.py"),
                "--",
                "--config",
                str(repo / "configs/short_biped_sprite_render.json"),
                "--repo-root",
                str(repo),
                "--frames-root",
                str(frames),
                "--source",
                str(surface_master),
                "--skip-aux-passes",
            ],
            cwd=repo,
            success=sprite_last,
            failure=None,
            timeout=args.timeout_seconds,
        )
        _write_json(
            sprite_render_record,
            {
                "schema_version": 1,
                "surface_master": str(surface_master),
                "surface_master_sha256": surface_master_sha256,
                "final_frame": str(sprite_last),
                "final_frame_sha256": _sha256(sprite_last),
            },
        )
    _record(status, root, "sprite_render", "complete", str(frames))

    package = root / "sprites/package"
    manifest = package / "candidate_unit_manifest.json"
    if not _approved_automatic_candidate_manifest(
        manifest, expected_master_sha256=surface_master_sha256
    ):
        _archive_partial(package)
        _record(status, root, "sprite_package", "running", "Packaging sheets with alpha/edge/hash gates.")
        _run(
            [
                sys.executable,
                str(adapters / "package_motion_sprites.py"),
                "--frames-root",
                str(frames),
                "--source-id",
                "darkness_short_biped_candidate",
                "--output-directory",
                str(package),
                "--master",
                str(surface_master),
            ],
            cwd=repo,
            timeout=120,
        )
    _record(status, root, "sprite_package", "complete", str(manifest))

    sprite_mediator = package / "qwen_sprite_mediator.json"
    if not sprite_mediator.is_file():
        _record(status, root, "sprite_qwen_review", "running", "Running sprite critic and mediator once.")
        _run(
            [
                sys.executable,
                str(adapters / "review_motion_sprites.py"),
                "--package",
                str(package),
                "--retarget-report",
                str(retarget_report),
                "--model",
                args.model,
            ],
            cwd=repo,
            timeout=args.timeout_seconds,
        )
    _record(status, root, "sprite_qwen_review", "complete", str(sprite_mediator))
    if _stop_after(args, status, root, "sprite_qwen_review"):
        return status
    smoke_bundle = root / "unity_smoke_bundle"
    smoke_manifest = smoke_bundle / "bundle_manifest.json"
    if not _bundle_is_current(smoke_manifest, manifest):
        _archive_partial(smoke_bundle)
        _record(status, root, "unity_smoke_bundle", "running", "Building portable standalone Unity smoke project.")
        _run(
            [
                sys.executable,
                str(adapters / "build_unity_smoke_bundle.py"),
                "--package",
                str(package),
                "--output-directory",
                str(smoke_bundle),
            ],
            cwd=repo,
            timeout=180,
        )
    smoke_archive = root / "unity_smoke_bundle.zip"
    smoke_archive_record = root / "unity_smoke_bundle.zip.json"
    archive_current = False
    if smoke_archive.is_file() and smoke_archive_record.is_file():
        archive_record = json.loads(smoke_archive_record.read_text(encoding="utf-8"))
        archive_current = (
            archive_record.get("bundle_manifest_sha256") == _sha256(smoke_manifest)
            and archive_record.get("archive_sha256") == _sha256(smoke_archive)
        )
    if not archive_current:
        _run(
            [
                sys.executable,
                str(adapters / "archive_unity_smoke_bundle.py"),
                "--bundle",
                str(smoke_bundle),
                "--output",
                str(smoke_archive),
            ],
            cwd=repo,
            timeout=180,
        )
    status["unity_smoke_archive"] = str(smoke_archive)
    status["unity_smoke_archive_sha256"] = _sha256(smoke_archive)
    _record(status, root, "unity_smoke_bundle", "complete", str(smoke_archive_record))

    unity = args.unity.resolve() if args.unity is not None else None
    returned_result = args.unity_result.resolve() if args.unity_result is not None else smoke_bundle / "result"
    returned_report = returned_result / "unity_candidate_validation.json"
    unity_output = root / "unity"
    unity_report = unity_output / "unity_candidate_validation.json"
    if unity_report.is_file() and not _unity_result_is_current(unity_report, smoke_manifest):
        _archive_partial(unity_output)
    if (
        not unity_report.is_file()
        and returned_report.is_file()
        and _unity_result_is_current(returned_report, smoke_manifest)
    ):
        _archive_partial(unity_output)
        _record(status, root, "unity_candidate_validation", "ingesting", "Verifying returned standalone Unity proof.")
        _run(
            [
                sys.executable,
                str(adapters / "ingest_unity_smoke_result.py"),
                "--bundle",
                str(smoke_bundle),
                "--result",
                str(returned_result),
                "--output-directory",
                str(unity_output),
            ],
            cwd=repo,
            timeout=120,
        )
    if not unity_report.is_file() and unity is not None and unity.is_file():
        result_output = smoke_bundle / "result"
        result_output.mkdir(parents=True, exist_ok=True)
        environment = dict(__import__("os").environ)
        environment["DARKNESS_CANDIDATE_PACKAGE"] = str(smoke_bundle / "candidate")
        environment["DARKNESS_CANDIDATE_OUTPUT"] = str(result_output)
        environment["DARKNESS_BUNDLE_MANIFEST"] = str(smoke_manifest)
        _record(status, root, "unity_candidate_validation", "running", "Running non-promoting Unity batch proof.")
        _run(
            [
                str(unity),
                "-quit",
                "-batchmode",
                "-projectPath",
                str(smoke_bundle / "UnitySmokeProject"),
                "-executeMethod",
                "Darkness.EditorTools.DarknessCandidateValidator.ValidateFromBatch",
                "-logFile",
                str(result_output / "unity.log"),
            ],
            cwd=repo,
            timeout=args.timeout_seconds,
            environment=environment,
        )
        _archive_partial(unity_output)
        _run(
            [
                sys.executable,
                str(adapters / "ingest_unity_smoke_result.py"),
                "--bundle",
                str(smoke_bundle),
                "--result",
                str(result_output),
                "--output-directory",
                str(unity_output),
            ],
            cwd=repo,
            timeout=120,
        )
    if not unity_report.is_file():
        _record(
            status,
            root,
            "unity_candidate_validation",
            "ready_for_external_execution",
            str(smoke_bundle / "README.md"),
        )
        status["outcome"] = "ready_for_external_unity_smoke"
        status["human_review"] = str(_write_review_index(root, unity_state="ready"))
        _write_json(_status_path(root), status)
        return status
    _record(status, root, "unity_candidate_validation", "complete", str(unity_report))
    status["outcome"] = "waiting_for_human_approval"
    status["human_review"] = str(_write_review_index(root, unity_state="complete"))
    _write_json(_status_path(root), status)
    return status


def main(argv: list[str] | None = None) -> int:
    status = run(_arguments(argv))
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
