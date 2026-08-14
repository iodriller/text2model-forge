"""Build key-pose sheets and ask LocalDeploy Qwen for a structured motion review."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darkness.motion_review import LocalDeployMotionReviewer  # noqa: E402


CLIPS = ("idle", "walk", "attack", "hit", "death")
VIEWS = ("front", "left")
FRAME_PATTERN = re.compile(r"^motion_(idle|walk|attack|hit|death)_(\d{3})_(front|left)\.png$")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="qwen3_6_27b")
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _discover(input_directory: Path) -> dict[tuple[str, str], list[tuple[int, Path]]]:
    discovered: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for path in input_directory.glob("motion_*.png"):
        match = FRAME_PATTERN.match(path.name)
        if match is None:
            continue
        clip, frame, view = match.groups()
        discovered.setdefault((clip, view), []).append((int(frame), path))
    for key, frames in discovered.items():
        frames.sort()
    missing = [f"{clip}/{view}" for clip in CLIPS for view in VIEWS if not discovered.get((clip, view))]
    if missing:
        raise FileNotFoundError("missing motion evidence strips: " + ", ".join(missing))
    return discovered


def _strip(
    frames: list[tuple[int, Path]],
    output: Path,
    *,
    clip: str,
    view: str,
    thumbnail: int = 320,
) -> Path:
    header = 44
    canvas = Image.new("RGB", (len(frames) * thumbnail, thumbnail + header), "#15181d")
    draw = ImageDraw.Draw(canvas)
    for index, (frame, path) in enumerate(frames):
        with Image.open(path).convert("RGB") as image:
            image.thumbnail((thumbnail, thumbnail))
            x = index * thumbnail + (thumbnail - image.width) // 2
            canvas.paste(image, (x, header))
        draw.text((index * thumbnail + 10, 14), f"{clip.upper()}  frame {frame}  {view}", fill="white")
    canvas.save(output, format="PNG")
    return output


def _master_sheet(
    strips: list[tuple[str, Path]],
    output: Path,
    *,
    width: int = 1400,
) -> Path:
    rendered: list[tuple[str, Image.Image]] = []
    for clip, path in strips:
        with Image.open(path).convert("RGB") as source:
            ratio = width / source.width
            rendered.append((clip, source.resize((width, max(1, int(source.height * ratio))))))
    label_width = 100
    height = sum(image.height for _, image in rendered)
    canvas = Image.new("RGB", (width + label_width, height), "#0d0f12")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for clip, image in rendered:
        draw.text((14, y + 18), clip.upper(), fill="white")
        canvas.paste(image, (label_width, y))
        y += image.height
    canvas.save(output, format="PNG")
    return output


def _image_content(path: Path) -> dict[str, object]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _numeric_summary(report: dict[str, object]) -> dict[str, object]:
    clip_reports = report["clip_reports"]
    return {
        "motion_gate_passed": report["motion_gate_passed"],
        "hard_failures": report["hard_failures"],
        "clips": {
            clip: {
                "collapsed_faces": clip_reports[clip]["collapsed_faces"],
                "severely_compressed_faces": clip_reports[clip]["severely_compressed_faces"],
                "maximum_ground_penetration": clip_reports[clip]["maximum_ground_penetration"],
                "loop_seam_error": clip_reports[clip]["loop_seam_error"],
                "inactive_required_joints": clip_reports[clip]["inactive_required_joints"],
                "joint_excursion_degrees": clip_reports[clip]["joint_excursion_degrees"],
            }
            for clip in CLIPS
        },
        "walk_contact": report["walk_contact"],
        "walk_swing": report["walk_swing"],
        "attack_function": report["attack_function"],
        "hit_function": report["hit_function"],
        "death_function": report["death_function"],
        "ground": report["ground"],
    }


def run_review(args: argparse.Namespace) -> dict[str, object]:
    input_directory = args.input_directory.resolve()
    output_directory = args.output_directory.resolve()
    if not input_directory.is_dir():
        raise FileNotFoundError(input_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"motion review output is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = input_directory / "motion_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    discovered = _discover(input_directory)
    strips: dict[str, dict[str, str]] = {}
    front_strips: list[tuple[str, Path]] = []
    for clip in CLIPS:
        strips[clip] = {}
        for view in VIEWS:
            path = _strip(
                discovered[(clip, view)],
                output_directory / f"{clip}_{view}_keyposes.png",
                clip=clip,
                view=view,
            )
            strips[clip][view] = str(path)
            if view == "front":
                front_strips.append((clip, path))
    master = _master_sheet(front_strips, output_directory / "all_motion_front_keyposes.png")
    numeric = _numeric_summary(report)
    reviewer = LocalDeployMotionReviewer(model=args.model)
    decision = reviewer.review(
        numeric_summary=numeric,
        image_content=[
            {"type": "text", "text": "Image 1: all front-view key-pose strips."},
            _image_content(master),
            {"type": "text", "text": "Image 2: enlarged walk front key poses."},
            _image_content(Path(strips["walk"]["front"])),
            {"type": "text", "text": "Image 3: enlarged attack front key poses."},
            _image_content(Path(strips["attack"]["front"])),
            {"type": "text", "text": "Image 4: enlarged hit front key poses."},
            _image_content(Path(strips["hit"]["front"])),
        ],
    )
    decision_path = output_directory / "qwen_motion_review.json"
    decision_path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")
    mediator = reviewer.mediate(
        numeric_summary=numeric,
        critic=decision,
        image_content=[
            {"type": "text", "text": "Blinded mediator evidence: full front-view key-pose sheet."},
            _image_content(master),
        ],
    )
    mediator_path = output_directory / "qwen_motion_mediator.json"
    mediator_path.write_text(mediator.model_dump_json(indent=2), encoding="utf-8")
    rows = []
    for clip in CLIPS:
        clip_report = report["clip_reports"][clip]
        clip_decision = getattr(decision, clip)
        issues = "; ".join(clip_decision.issues) or "none reported"
        rows.append(
            f"| {clip} | {clip_decision.readability} | {clip_decision.critical_joint_verdict} | "
            f"{clip_report['collapsed_faces']} | {clip_report['severely_compressed_faces']} | {issues} |"
        )
    review_path = output_directory / "human_motion_review.md"
    review_path.write_text(
        "\n".join(
            [
                "# Darkness short-biped motion review",
                "",
                f"- Deterministic motion gate: `{report['motion_gate_passed']}`",
                f"- Qwen overall: `{decision.overall}`",
                f"- Qwen confidence: `{decision.confidence}`",
                f"- Mediator corrected overall: `{mediator.corrected_overall}`",
                f"- Mediator recommends more iteration: `{mediator.recommend_more_iteration}`",
                "- Status: human motion-quality approval required.",
                "",
                "| Clip | Readability | Critical joints | Collapsed | Severe compression | Qwen issues |",
                "|---|---|---|---:|---:|---|",
                *rows,
                "",
                "## Qwen broad strategy analysis",
                "",
                decision.unconstrained_strategy_analysis,
                "",
                "## Requested new capability",
                "",
                decision.requested_new_capability or "None.",
                "",
                "## Independent mediator",
                "",
                mediator.reason,
                "",
                "Unsupported or overstated critic claims:",
                "",
                *(
                    [f"- {claim}" for claim in mediator.unsupported_or_overstated_claims]
                    or ["- None."]
                ),
                "",
                "## Human decision",
                "",
                "Which clips are acceptable as the procedural baseline, and which should be replaced first by an "
                "open-source retargeted motion?",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "input_directory": str(input_directory),
        "model": args.model,
        "master_sheet": str(master),
        "strips": strips,
        "numeric_summary": numeric,
        "qwen_decision": str(decision_path),
        "qwen_mediator": str(mediator_path),
        "human_review": str(review_path),
        "human_approval_required": True,
        "human_approved": False,
    }
    _write_json(output_directory / "motion_review.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    result = run_review(_arguments(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
