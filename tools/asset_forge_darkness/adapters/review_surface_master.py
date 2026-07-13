"""Give Qwen before/current surface evidence, then publish a concise human checkpoint."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from darkness.surface_review import LocalDeploySurfaceReviewer  # noqa: E402


def _image(path: Path) -> dict[str, object]:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")},
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-directory", type=Path, required=True)
    parser.add_argument("--model", default="qwen3_6_27b")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    surface = args.surface_directory.resolve()
    report = json.loads((surface / "surface_validation.json").read_text(encoding="utf-8"))
    board = surface / "surface_review.png"
    numeric_history = {
        "previous": {
            "master_sha256": report["source_master_sha256"],
            "surface_state": "unpainted deterministic retarget master",
        },
        "operation": {
            "name": "depth_locked_multiview_projection_bake",
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "controlnet": report["controlnet"],
            "controlnet_sha256": report["controlnet_sha256"],
            "seed": report["seed"],
        },
        "current": {
            "master_sha256": report["surface_master_sha256"],
            "views": report["views"],
            "atlas_count": len(report["atlases"]),
            "atlas_resolutions": [item["resolution"] for item in report["atlases"]],
            "rejected_black_projection_atlases": report.get("rejected_black_projection_atlases", []),
            "surface_storage": report.get("surface_storage"),
            "projection_method": report.get("projection_method"),
            "projection_metrics": report.get("projection_metrics"),
            **report["image_metrics"],
            "hard_failures": report["hard_failures"],
            "automatic_gate_passed": report["automatic_gate_passed"],
        },
        "iteration": 1,
        "maximum_useful_iterations": 3,
    }
    reviewer = LocalDeploySurfaceReviewer(model=args.model)
    try:
        critic = reviewer.review(
            numeric_history=numeric_history,
            image_content=[
                {"type": "text", "text": "Labeled before/current multi-view surface board."},
                _image(board),
            ],
        )
        critic_path = surface / "qwen_surface_review.json"
        critic_path.write_text(critic.model_dump_json(indent=2), encoding="utf-8")
        mediator = reviewer.mediate(
            numeric_history=numeric_history,
            critic=critic,
            image_content=[
                {"type": "text", "text": "Blinded referee board: same before/current evidence."},
                _image(board),
            ],
        )
        mediator_path = surface / "qwen_surface_mediator.json"
        mediator_path.write_text(mediator.model_dump_json(indent=2), encoding="utf-8")
        critic_summary = f"`{critic.overall}` ({critic.confidence})"
        mediator_summary = f"`{mediator.corrected_overall}`"
        more = mediator.recommend_more_iteration
        defects = critic.visible_defects or ["None reported."]
        strategy = critic.unconstrained_strategy_analysis
        mediator_reason = mediator.reason
    except Exception as error:  # Qwen is advisory and must not destroy a passing surface bake.
        fallback = {
            "schema_version": 1,
            "review_mode": "deterministic_fallback",
            "error": str(error)[:1500],
            "corrected_overall": "uncertain",
            "recommend_more_iteration": False,
            "human_review_required": True,
        }
        mediator_path = surface / "qwen_surface_mediator.json"
        mediator_path.write_text(json.dumps(fallback, indent=2), encoding="utf-8")
        critic_summary = "unavailable; deterministic fallback used"
        mediator_summary = "`uncertain`"
        more = False
        defects = ["Qwen was unavailable; inspect the board directly."]
        strategy = "No AI strategy was accepted because the advisory service was unavailable."
        mediator_reason = "The numeric surface gate passed; human review remains authoritative."

    review = surface / "human_review.md"
    review.write_text(
        "\n".join(
            [
                "# Darkness painted surface — human review",
                "",
                f"- [Before/current surface board]({board.name})",
                f"- Automatic UV/atlas/alpha/hash gate: `{report['automatic_gate_passed']}`",
                f"- Qwen critic: {critic_summary}",
                f"- Independent mediator: {mediator_summary}",
                f"- More paint iteration recommended: `{more}`",
                "- This one persistent `.blend` master feeds idle, walk, attack, death, and every direction.",
                "",
                "## Visible issues to judge",
                "",
                *[f"- {value}" for value in defects],
                "",
                "## Broad critic strategy",
                "",
                strategy,
                "",
                "## Referee",
                "",
                mediator_reason,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"mediator": str(mediator_path), "human_review": str(review)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
