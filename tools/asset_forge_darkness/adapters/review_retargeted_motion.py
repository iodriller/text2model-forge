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
            "current iteration 3 uses final vertex height and targets 52% body-relative club reach",
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
    walk_side = evidence / "walk_left_keyposes.png"
    death_side = evidence / "death_left_keyposes.png"
    reviewer = LocalDeployRetargetReviewer(model=args.model)
    critic = reviewer.review(
        numeric_history=numeric,
        image_content=[
            {"type": "text", "text": "Image 1: fixed-camera front key poses for all four clips."},
            _image(master),
            {"type": "text", "text": "Image 2: attack side-view key poses."},
            _image(attack_side),
            {"type": "text", "text": "Image 3: walk side-view key poses."},
            _image(walk_side),
            {"type": "text", "text": "Image 4: death side-view key poses."},
            _image(death_side),
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
