"""Send a retarget checkpoint's images and exact metrics to Qwen critic/referee."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darkness.retarget_review import LocalDeployRetargetReviewer  # noqa: E402


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument(
        "--previous-retarget",
        type=Path,
        help="Optional prior retarget root; replaces the attack side strip with a labeled previous grip image.",
    )
    parser.add_argument("--model", default="qwen3_6_27b")
    return parser.parse_args(argv)


def _image(path: Path) -> dict[str, object]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _summary(report: dict[str, object]) -> dict[str, object]:
    clips = report["clips"]
    return {
        "source_library": report["source"]["name"],
        "source_license": report["source"]["license"],
        "retarget_gate_passed": report["retarget_gate_passed"],
        "finite_critical_joint_transforms": report["finite_critical_joint_transforms"],
        "body_height": report["body_height"],
        "walk_swing_ranges": report["walk_swing_ranges"],
        "attack_maximum_hand_travel": report["attack_maximum_hand_travel"],
        "death_hips_descent": report["death_hips_descent"],
        "death_head_descent": report["death_head_descent"],
        "equipment": report.get("equipment"),
        "clips": {
            name: {
                "source_action": clips[name]["source_action"],
                "frame_start": clips[name]["frame_start"],
                "frame_end": clips[name]["frame_end"],
                "collapsed_faces": clips[name]["collapsed_faces"],
                "severely_compressed_faces": clips[name]["severely_compressed_faces"],
                "critical_joint_excursion_degrees": clips[name][
                    "critical_joint_excursion_degrees"
                ],
            }
            for name in ("idle", "walk", "attack", "death")
        },
        "history": [
            "raw absolute-pose transfer left arms open and made Punch_Cross poorly readable",
            "idle-relative neutral calibration corrected idle/walk arm posture",
            "Punch_Cross was replaced by Sword_Attack after direct human-visible evidence showed weak readability",
            "human review rejected the empty-handed Sword_Attack because a weapon-authored clip needs equipment",
            "club iteration 1 proved the rigid hand socket but its short round silhouette read as a mace/spoon",
            "club iteration 2 improved the taper but exposed a pre-deformation body-height measurement mismatch",
            "club iteration 3 attached to equal-suffix hand_r, but human review identified that the source and target side conventions are mirrored and the open hand did not grasp the handle",
            "club iteration 4 fixed the physical side but its radial squeeze made a closed blob rather than a convincing grip, so human review rejected it before downstream packaging",
            "club iteration 5 preserved the goblin hand morphology by curling two detected claw branches separately; Qwen and its referee accepted the wrap, but close visual review found the 2.2%-of-body shaft too thick for the palm opening",
            "club iteration 6 kept the accepted branch curl and reduced the handle radius to 1.8% of body height; comparative Qwen/referee review accepted the clearer shaft clearance",
            "club iteration 7 corrected the public anatomical hand_right socket, but human review rejected its mesh-only curl because it still had no explicit digit landmarks/bones and the shaft missed the actual claw opening",
            "current iteration 8 derives base/joint/tip landmarks from both claw branches, adds four local deform bones plus a dedicated grip socket, places the shaft through the detected opening, and solves both digit chains around it",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    root = args.input_directory.resolve()
    evidence = args.evidence_directory.resolve()
    report = json.loads((root / "retarget_validation.json").read_text(encoding="utf-8"))
    numeric = _summary(report)
    master = evidence / "all_motion_front_keyposes.png"
    attack_side = evidence / "attack_left_keyposes.png"
    grip_front = sorted(root.glob("grip_attack_*_front.png"))
    grip_side = sorted(root.glob("grip_attack_*_right.png"))
    if not grip_front or not grip_side:
        raise FileNotFoundError("retarget output is missing dedicated grip close-up evidence")
    previous_grip: Path | None = None
    if args.previous_retarget is not None:
        previous = sorted(args.previous_retarget.resolve().glob("grip_attack_*_front.png"))
        if not previous:
            raise FileNotFoundError("previous retarget is missing grip close-up evidence")
        previous_grip = previous[0]
    reviewer = LocalDeployRetargetReviewer(model=args.model)
    comparison_content = (
        [
            {
                "type": "text",
                "text": "Image 2: previous iteration grip close-up; use it only for direct comparison.",
            },
            _image(previous_grip),
        ]
        if previous_grip is not None
        else [
            {"type": "text", "text": "Image 2: current attack side-view key poses."},
            _image(attack_side),
        ]
    )
    critic = reviewer.review(
        numeric_history=numeric,
        image_content=[
            {"type": "text", "text": "Image 1: fixed-camera front key poses for all four clips."},
            _image(master),
            *comparison_content,
            {
                "type": "text",
                "text": "Image 3: current front close-up of the weapon hand around the handle.",
            },
            _image(grip_front[0]),
            {
                "type": "text",
                "text": "Image 4: dedicated side close-up of the same deterministic grip corrective.",
            },
            _image(grip_side[0]),
        ],
    )
    critic_path = evidence / "qwen_retarget_review.json"
    critic_path.write_text(critic.model_dump_json(indent=2), encoding="utf-8")
    mediator = reviewer.mediate(
        numeric_history=numeric,
        critic=critic,
        image_content=[
            {"type": "text", "text": "Blinded mediator image: fixed-camera front key poses."},
            _image(master),
            *(
                [
                    {"type": "text", "text": "Blinded mediator image: previous grip close-up."},
                    _image(previous_grip),
                ]
                if previous_grip is not None
                else []
            ),
            {
                "type": "text",
                "text": "Blinded mediator image: current weapon-hand grip close-up.",
            },
            _image(grip_front[0]),
        ],
    )
    mediator_path = evidence / "qwen_retarget_mediator.json"
    mediator_path.write_text(mediator.model_dump_json(indent=2), encoding="utf-8")
    review_path = evidence / "human_review.md"
    review_path.write_text(
        "\n".join(
            [
                "# Quaternius retarget checkpoint",
                "",
                f"- Deterministic retarget gate: `{report['retarget_gate_passed']}`",
                f"- Qwen critic: `{critic.overall}` ({critic.confidence})",
                f"- Independent mediator: `{mediator.corrected_overall}`",
                f"- More iteration recommended: `{mediator.recommend_more_iteration}`",
                f"- Source weapon hand: `{report['equipment']['source_hand_analysis']['selected_source_weapon_bone']}`",
                f"- Resolved target bone: `{report['equipment']['bone']}`",
                f"- Grip corrective gate: `{report['equipment']['grip_corrective']['automatic_gate_passed']}`",
                f"- [Front grip close-up](../{grip_front[0].name})",
                f"- [Side grip close-up](../{grip_side[0].name})",
                "- Human approval remains required.",
                "",
                "| Clip | Readability | Critical limbs | Observations |",
                "|---|---|---|---|",
                *[
                    f"| {name} | {getattr(critic, name).readability} | "
                    f"{getattr(critic, name).critical_limb_verdict} | "
                    f"{'; '.join(getattr(critic, name).observations) or 'none'} |"
                    for name in ("idle", "walk", "attack", "death")
                ],
                "",
                "## Broad critic strategy",
                "",
                critic.unconstrained_strategy_analysis,
                "",
                "## Mediator",
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
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "critic": str(critic_path),
                "mediator": str(mediator_path),
                "human_review": str(review_path),
                "corrected_overall": mediator.corrected_overall,
                "recommend_more_iteration": mediator.recommend_more_iteration,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
