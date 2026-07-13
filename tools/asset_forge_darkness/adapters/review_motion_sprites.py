"""Run one Qwen critic/referee pass on a candidate sprite package."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darkness.sprite_review import LocalDeploySpriteReviewer  # noqa: E402


def _image(path: Path) -> dict[str, object]:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--retarget-report", type=Path)
    parser.add_argument("--model", default="qwen3_6_27b")
    args = parser.parse_args(argv)
    package = args.package.resolve()
    manifest = json.loads((package / "candidate_unit_manifest.json").read_text(encoding="utf-8"))
    summary = {
        "automatic_gate_passed": manifest["automatic_gate_passed"],
        "hard_failures": manifest["hard_failures"],
        "cell_size": [manifest["cell_width"], manifest["cell_height"]],
        "actions": manifest["actions"],
        "framing_history": [
            "global 4.4 camera made idle/walk/attack too small",
            "2.8 camera made those three clips readable but cropped the far-travel death",
            "the larger club exposed edge clipping in a fixed 2.8 active-motion camera",
            "current package measures evaluated body-plus-equipment bounds across every sampled frame and direction, "
            "then uses one stable auto-frame scale per clip with 10% margin; death retains a 4.4 minimum",
        ],
    }
    if args.retarget_report:
        retarget_report = json.loads(args.retarget_report.resolve().read_text(encoding="utf-8"))
        summary["equipment"] = retarget_report.get("equipment")
        summary["equipment_history"] = [
            "empty-handed Sword_Attack failed human semantic review",
            "club iteration 1 passed its socket gate but looked too small and round",
            "club iteration 2 improved the taper but exposed a pre-deformation body-height mismatch",
            "current iteration 3 uses the final validated vertex height and targets 52% body-relative reach",
        ]
    sheet = package / manifest["review_sheet"]
    reviewer = LocalDeploySpriteReviewer(model=args.model)
    critic = reviewer.review(
        numeric_summary=summary,
        image_content=[{"type": "text", "text": "Image 1: south/east strips for all four clips."}, _image(sheet)],
    )
    critic_path = package / "qwen_sprite_review.json"
    critic_path.write_text(critic.model_dump_json(indent=2), encoding="utf-8")
    mediator = reviewer.mediate(
        numeric_summary=summary,
        critic=critic,
        image_content=[{"type": "text", "text": "Blinded mediator image: same complete sprite sheet."}, _image(sheet)],
    )
    mediator_path = package / "qwen_sprite_mediator.json"
    mediator_path.write_text(mediator.model_dump_json(indent=2), encoding="utf-8")
    review_path = package / "human_review.md"
    review_path.write_text(
        "\n".join(
            [
                "# Darkness sprite candidate review",
                "",
                f"- Automatic package gate: `{manifest['automatic_gate_passed']}`",
                f"- Qwen critic: `{critic.overall}` ({critic.confidence})",
                f"- Mediator: `{mediator.corrected_overall}`",
                f"- More iteration recommended: `{mediator.recommend_more_iteration}`",
                "- Unity candidate validation and human approval remain required.",
                "",
                "| Clip | Readability | Scale/silhouette | Observations |",
                "|---|---|---|---|",
                *[
                    f"| {name} | {getattr(critic, name).readability} | "
                    f"{getattr(critic, name).scale_and_silhouette} | "
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
                "Unsupported or overstated claims:",
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
