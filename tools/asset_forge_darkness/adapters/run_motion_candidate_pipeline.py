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
    "sprite_render",
    "sprite_package",
    "sprite_qwen_review",
    "unity_smoke_bundle",
    "unity_candidate_validation",
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--motion-source", type=Path, required=True)
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
    parser.add_argument("--timeout-seconds", type=int, default=900)
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


def _approved_automatic_candidate_manifest(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("automatic_gate_passed") is True
        and manifest.get("hard_failures") == []
        and manifest.get("human_approval_required") is True
        and manifest.get("human_approved") is False
    )


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
                "# Darkness motion candidate — human review",
                "",
                "This is a non-promoting review checkpoint. It does not replace the live Unity goblin.",
                "",
                "## Motion",
                "",
                "- [Four-motion key-pose sheet](retarget/human_review/all_motion_front_keyposes.png)",
                "- [Motion critic and mediator summary](retarget/human_review/human_review.md)",
                "- [Editable retargeted Blender master](retarget/quaternius_retargeted_goblin.blend)",
                "- [GLB with idle, walk, attack, and death](retarget/quaternius_retargeted_goblin.glb)",
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
                    "`--unity-result <result-folder>`. No EmberDefense project import is required."
                ),
                interactive_review,
                "",
                "## Human decision",
                "",
                "Approve or reject the motion direction and sprite readability. Human approval remains false until an "
                "explicit decision is recorded.",
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
    blender = args.blender.resolve() if args.blender is not None else None
    root = args.output_root.resolve()
    for path in (repo, target, motion_source):
        if not path.exists():
            raise FileNotFoundError(path)
    root.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "schema_version": 1,
        "pipeline": "darkness_motion_candidate_v1",
        "target": str(target),
        "target_sha256": _sha256(target),
        "motion_source": str(motion_source),
        "motion_source_sha256": _sha256(motion_source),
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
        status = prior
        stages = status.setdefault("stages", {})
        for name in STAGES:
            stages.setdefault(name, {"state": "pending", "detail": ""})

    adapters = repo / "tools/asset_forge_darkness/adapters"
    retarget = root / "retarget"
    retarget_report = retarget / "retarget_validation.json"
    retarget_glb = retarget / "quaternius_retargeted_goblin.glb"
    retarget_error = retarget / "retarget_error.txt"
    if not retarget_glb.is_file():
        blender = _require_executable(blender, "retarget")
        _archive_partial(retarget)
        _record(status, root, "retarget", "running", "Baking four Quaternius actions onto the Darkness rig.")
        _launch_and_wait(
            [
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
            ],
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

    frames = root / "sprites/frames"
    sprite_last = frames / "darkness_short_biped_candidate/death/west/09.png"
    if not sprite_last.is_file():
        blender = _require_executable(blender, "sprite_render")
        _record(status, root, "sprite_render", "running", "Rendering four actions in four directions.")
        _launch_and_wait(
            [
                str(blender),
                str(retarget / "quaternius_retargeted_goblin.blend"),
                "--background",
                "--offline-mode",
                "--python-exit-code",
                "23",
                "--python",
                str(repo / "tools/asset_forge/blender/render_sprites.py"),
                "--",
                "--config",
                str(repo / "tools/asset_forge_darkness/configs/short_biped_sprite_render.json"),
                "--repo-root",
                str(repo),
                "--frames-root",
                str(frames),
                "--source",
                str(retarget / "quaternius_retargeted_goblin.blend"),
                "--skip-aux-passes",
            ],
            cwd=repo,
            success=sprite_last,
            failure=None,
            timeout=args.timeout_seconds,
        )
    _record(status, root, "sprite_render", "complete", str(frames))

    package = root / "sprites/package"
    manifest = package / "candidate_unit_manifest.json"
    if not _approved_automatic_candidate_manifest(manifest):
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
                str(retarget / "quaternius_retargeted_goblin.blend"),
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
    smoke_bundle = root / "unity_smoke_bundle"
    smoke_manifest = smoke_bundle / "bundle_manifest.json"
    if not smoke_manifest.is_file():
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
    if not unity_report.is_file() and returned_report.is_file():
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
