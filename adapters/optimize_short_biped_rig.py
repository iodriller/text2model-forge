"""Run a bounded Blender -> evidence -> LocalDeploy Qwen rig-optimization episode."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image, ImageChops, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darkness.canonical_optimizer import (  # noqa: E402
    AdjustLandmarkPairParameters,
    CanonicalRigOptimizerDecision,
    RedistributeJointPairWeightsParameters,
    apply_landmark_pair_adjustment,
    canonical_optimizer_operations,
)
from darkness.config import load_local_config, worker_binding  # noqa: E402
from darkness.external_worker import SubprocessWorkerAdapter  # noqa: E402
from darkness.hashing import sha256_file  # noqa: E402
from darkness.manifests import load_manifests  # noqa: E402
from darkness.optimizer import LocalDeployOptimizer  # noqa: E402
from darkness.schemas import (  # noqa: E402
    ArtifactLineage,
    ArtifactRecord,
    AssetStage,
    EvidenceBundle,
    EvidenceItem,
    ExternalWorkerRequest,
)
from darkness.workers import WorkerManager  # noqa: E402


GUARDED_METRICS = (
    "hard_failure_count",
    "shoulder_collapsed_faces",
    "shoulder_severely_compressed_faces",
    "leg_collapsed_faces",
    "leg_severely_compressed_faces",
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--run-id", default="blender.triposg.rig.optimize.qwen.v1")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--model", default="qwen3_6_27b")
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_summary(metrics: dict[str, object] | None) -> str:
    if metrics is None:
        return "no prior metrics"
    return (
        f"hard={metrics['hard_failure_count']}  "
        f"shoulder collapsed/severe={metrics['shoulder_collapsed_faces']}/"
        f"{metrics['shoulder_severely_compressed_faces']}  "
        f"leg collapsed/severe={metrics['leg_collapsed_faces']}/"
        f"{metrics['leg_severely_compressed_faces']}"
    )


def _paired_image(
    left_path: Path,
    right_path: Path,
    output: Path,
    *,
    view: str,
    left_label: str,
    right_label: str,
    left_metrics: dict[str, object] | None,
    right_metrics: dict[str, object] | None,
) -> Path:
    with Image.open(left_path).convert("RGB") as left, Image.open(right_path).convert("RGB") as right:
        width = left.width + right.width
        header = 62
        canvas = Image.new("RGB", (width, max(left.height, right.height) + header), "#161616")
        canvas.paste(left, (0, header))
        canvas.paste(right, (left.width, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"{view}: {left_label}", fill="white")
        draw.text((10, 30), _metric_summary(left_metrics), fill="#b9c2cc")
        draw.text((left.width + 10, 8), f"{view}: {right_label}", fill="white")
        draw.text((left.width + 10, 30), _metric_summary(right_metrics), fill="#b9c2cc")
        canvas.save(output, format="PNG")
    return output


def _image_content(path: Path) -> dict[str, object]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _probe_metrics(directory: Path) -> dict[str, float | int | bool | str]:
    deformation = json.loads((directory / "deformation_report.json").read_text(encoding="utf-8"))
    shoulder = next(item for item in deformation["poses"] if item["pose"] == "shoulder_elbow_stress")
    leg = next(item for item in deformation["poses"] if item["pose"] == "hip_knee_stress")
    return {
        "hard_failure_count": len(deformation["hard_failures"]),
        "shoulder_collapsed_faces": int(shoulder["collapsed_faces"]),
        "shoulder_severely_compressed_faces": int(shoulder["severely_compressed_faces"]),
        "leg_collapsed_faces": int(leg["collapsed_faces"]),
        "leg_severely_compressed_faces": int(leg["severely_compressed_faces"]),
        "automatic_deformation_gate_passed": bool(deformation["gate_passed"]),
    }


def _metric_deltas(
    current: dict[str, object],
    previous: dict[str, object] | None,
) -> dict[str, int | None]:
    return {
        name: None if previous is None else int(current[name]) - int(previous[name])
        for name in GUARDED_METRICS
    }


def _image_delta(previous: Path, current: Path) -> dict[str, float]:
    with Image.open(previous).convert("RGB") as left, Image.open(current).convert("RGB") as right:
        if left.size != right.size:
            raise ValueError("comparison renders must have identical dimensions")
        difference = ImageChops.difference(left, right)
        pixels = list(difference.get_flattened_data())
        return {
            "changed_pixel_fraction": sum(1 for pixel in pixels if max(pixel) > 2) / len(pixels),
            "mean_absolute_channel_delta": (
                sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3 * 255)
            ),
        }


def _numeric_regression(candidate: dict[str, object], accepted: dict[str, object]) -> bool:
    return any(int(candidate[name]) > int(accepted[name]) for name in GUARDED_METRICS)


def _decision_inconsistency(
    decision: Any,
    *,
    metrics: dict[str, object],
    previous_metrics: dict[str, object] | None,
    numeric_regression: bool,
    metric_deltas: dict[str, int | None],
    image_deltas: dict[str, float | None],
    operation_under_test: dict[str, object] | None,
    seen_signatures: set[str],
    failed_strategies: set[str],
) -> str | None:
    unresolved = int(metrics["shoulder_severely_compressed_faces"]) + int(
        metrics["leg_severely_compressed_faces"]
    )
    if decision.goal_satisfied and unresolved:
        return f"goal_satisfied cannot be true while {unresolved} severely compressed faces remain"
    if previous_metrics is None:
        if decision.comparison is not None and decision.comparison.preferred not in {"uncertain", "tie"}:
            return "the baseline has no previous accepted checkpoint to prefer"
        return None
    if numeric_regression and decision.comparison is not None and decision.comparison.preferred == "current":
        return "the current attempt cannot be preferred after a hard numeric regression"
    maximum_changed_fraction = max(
        float(image_deltas["front_changed_pixel_fraction"] or 0.0),
        float(image_deltas["left_changed_pixel_fraction"] or 0.0),
    )
    metrics_unchanged = all(delta == 0 for delta in metric_deltas.values())
    if (
        metrics_unchanged
        and maximum_changed_fraction < 0.002
        and decision.comparison is not None
        and decision.comparison.preferred == "current"
    ):
        return (
            "current cannot be preferred as an improvement when every guarded metric is unchanged and fewer than "
            "0.2% of fixed-camera pixels changed; use tie/previous and propose a different bounded cause"
        )
    for proposal in decision.proposals:
        signature = _proposal_signature(proposal.operation_id, proposal.parameters)
        if signature in seen_signatures:
            return "the proposed operation signature was already attempted; choose a different bounded cause"
        strategy = _proposal_strategy(proposal.operation_id, proposal.parameters)
        if strategy in failed_strategies:
            return "the proposed causal operation strategy already failed; changing only its magnitude is not allowed"
        current_failed = numeric_regression or (
            metrics_unchanged and maximum_changed_fraction < 0.002
        )
        if current_failed and operation_under_test is not None:
            current_strategy = _proposal_strategy(
                str(operation_under_test["operation_id"]),
                dict(operation_under_test["parameters"]),
            )
            if strategy == current_strategy:
                return "the operation under test failed; the next proposal must change causal strategy"
    return None


def _source_record(source: Path) -> ArtifactRecord:
    digest = sha256_file(source)
    artifact_id = "topology.triposg.editable.v1"
    return ArtifactRecord(
        artifact_id=artifact_id,
        sha256=digest,
        size_bytes=source.stat().st_size,
        media_type="application/x-blender",
        stage=AssetStage.topology,
        blob_path=f"smoke/{source.name}",
        created_at=datetime.now(timezone.utc),
        lineage=ArtifactLineage(
            artifact_id=artifact_id,
            artifact_sha256=digest,
            stage=AssetStage.topology.value,
            source_license_ids=["Project-Owned"],
        ),
        metadata={"source_path": str(source)},
    )


def _proposal_signature(operation_id: str, parameters: dict[str, Any]) -> str:
    return json.dumps(
        {"operation_id": operation_id, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
    )


def _proposal_strategy(operation_id: str, parameters: dict[str, Any]) -> str:
    if operation_id == "rig.adjust_landmark_pair":
        delta = float(parameters["delta_fraction"])
        causal_parameters = {
            "landmark_pair": parameters["landmark_pair"],
            "axis": parameters["axis"],
            "direction": "positive" if delta > 0 else "negative",
        }
    elif operation_id == "skin.redistribute_joint_pair_weights":
        causal_parameters = {
            "joint_pair": parameters["joint_pair"],
            "direction": parameters["direction"],
        }
    else:
        causal_parameters = parameters
    return json.dumps(
        {"operation_id": operation_id, "causal_parameters": causal_parameters},
        sort_keys=True,
        separators=(",", ":"),
    )


def _evidence_diagnostics(
    *,
    iteration: int,
    previous: dict[str, object] | None,
    current: dict[str, object],
    deltas: dict[str, int | None],
    numeric_verdict: str,
    operation_under_test: dict[str, object] | None,
    landmark_adjustments: dict[str, list[float]],
    weight_adjustments: list[dict[str, object]],
    history: list[dict[str, object]],
    worker_seconds: float,
    image_deltas: dict[str, float | None],
) -> dict[str, float | int | bool | str | None]:
    compact_history = []
    for item in history:
        source_proposal = item.get("source_proposal")
        proposal = (
            None
            if source_proposal is None
            else {
                "operation_id": source_proposal["operation_id"],
                "parameters": source_proposal["parameters"],
            }
        )
        comparison = item.get("qwen_comparison")
        compact_history.append(
            {
                "iteration": item["iteration"],
                "operation_under_test": proposal,
                "current_metrics": {
                    name: item["metrics"][name]
                    for name in GUARDED_METRICS
                },
                "metric_deltas": item["metric_deltas"],
                "numeric_verdict": item["numeric_verdict"],
                "accepted": item["accepted"],
                "qwen_comparison": (
                    None
                    if comparison is None
                    else {
                        "preferred": comparison["preferred"],
                        "visual_delta": comparison["visual_delta"],
                    }
                ),
                "top_qwen_cause": (
                    item.get("qwen_root_causes", [{}])[0].get("cause")
                    if item.get("qwen_root_causes")
                    else None
                ),
            }
        )
    diagnostics: dict[str, float | int | bool | str | None] = {
        "iteration": iteration,
        "worker_seconds": round(worker_seconds, 3),
        "current_numeric_verdict": numeric_verdict,
        "operation_under_test": json.dumps(operation_under_test, sort_keys=True),
        "attempted_landmark_adjustments": json.dumps(landmark_adjustments, sort_keys=True),
        "attempted_weight_adjustments": json.dumps(weight_adjustments, sort_keys=True),
        "iteration_history": json.dumps(compact_history, sort_keys=True, separators=(",", ":")),
        **image_deltas,
    }
    for name in GUARDED_METRICS:
        diagnostics[f"previous_{name}"] = None if previous is None else int(previous[name])
        diagnostics[f"current_{name}"] = int(current[name])
        diagnostics[f"delta_{name}"] = deltas[name]
    return diagnostics


def _write_human_review(
    output_root: Path,
    *,
    source: Path,
    model: str,
    baseline_dir: Path,
    accepted_dir: Path,
    baseline_metrics: dict[str, object],
    accepted_metrics: dict[str, object],
    accepted_landmarks: dict[str, list[float]],
    accepted_weights: list[dict[str, object]],
    history: list[dict[str, object]],
    stop_reason: str,
) -> dict[str, str]:
    board_paths: dict[str, str] = {}
    for view in ("front", "left"):
        board = _paired_image(
            baseline_dir / f"rig_shoulder_stress_{view}.png",
            accepted_dir / f"rig_shoulder_stress_{view}.png",
            output_root / f"human_review_{view}.png",
            view=view.upper(),
            left_label="BASELINE STRESS",
            right_label="BEST ACCEPTED STRESS",
            left_metrics=baseline_metrics,
            right_metrics=accepted_metrics,
        )
        board_paths[view] = str(board)

    rows = []
    for item in history:
        proposal = item.get("source_proposal") or {"operation_id": "baseline", "parameters": {}}
        comparison = item.get("qwen_comparison") or {}
        rows.append(
            "| {iteration} | `{operation}` | `{parameters}` | `{deltas}` | {verdict} | {preferred} | {result} |".format(
                iteration=item["iteration"],
                operation=proposal.get("operation_id", "unknown"),
                parameters=json.dumps(proposal.get("parameters", {}), sort_keys=True),
                deltas=json.dumps(item["metric_deltas"], sort_keys=True),
                verdict=item["numeric_verdict"],
                preferred=comparison.get("preferred", "not supplied"),
                result="accepted" if item["accepted"] else f"rolled back: {item['acceptance_reason']}",
            )
        )
    observations = []
    for item in history:
        for observation in item.get("qwen_observations", []):
            observations.append(
                f"- Iteration {item['iteration']}: {observation['region']} — {observation['issue']} "
                f"(severity {observation['severity']:.2f})"
            )
    remaining = []
    if int(accepted_metrics["hard_failure_count"]):
        remaining.append(f"- {accepted_metrics['hard_failure_count']} hard deformation failures remain.")
    if int(accepted_metrics["shoulder_severely_compressed_faces"]):
        remaining.append(
            f"- Shoulder stress still has {accepted_metrics['shoulder_severely_compressed_faces']} "
            "severely compressed faces."
        )
    if int(accepted_metrics["leg_severely_compressed_faces"]):
        remaining.append(
            f"- Leg stress still has {accepted_metrics['leg_severely_compressed_faces']} "
            "severely compressed faces."
        )
    if not remaining:
        remaining.append("- Numeric stress gates found no remaining collapsed or severely compressed faces.")

    review = "\n".join(
        [
            "# Darkness rig optimizer interim review",
            "",
            f"- Source: `{source}`",
            f"- Qwen model/profile: `{model}`",
            f"- Stop reason: `{stop_reason}`",
            f"- Attempts completed: {len(history)}",
            "- Status: human review required; this is not a production approval.",
            "",
            "## Baseline and best accepted numbers",
            "",
            f"- Baseline: `{json.dumps(baseline_metrics, sort_keys=True)}`",
            f"- Best accepted: `{json.dumps(accepted_metrics, sort_keys=True)}`",
            f"- Accepted landmark changes: `{json.dumps(accepted_landmarks, sort_keys=True)}`",
            f"- Accepted weight changes: `{json.dumps(accepted_weights, sort_keys=True)}`",
            "",
            "## Attempt history",
            "",
            "| Iteration | Operation under test | Parameters | Numeric deltas | Numeric verdict | Qwen preferred | Result |",
            "|---:|---|---|---|---|---|---|",
            *rows,
            "",
            "## Qwen diagnoses",
            "",
            *(observations or ["- Qwen supplied no structured observations."]),
            "",
            "## Remaining defects",
            "",
            *remaining,
            "",
            "## Human decision",
            "",
            "Do you approve the best accepted checkpoint as the canonical-rig direction, or should Darkness "
            "resume a calibrated episode (up to ten total attempts) focused on the remaining defect?",
            "",
        ]
    )
    review_path = output_root / "human_review.md"
    review_path.write_text(review, encoding="utf-8")
    board_paths["report"] = str(review_path)
    return board_paths


def run_episode(args: argparse.Namespace) -> dict[str, object]:
    if not 1 <= args.iterations <= 10:
        raise ValueError("iterations must be between one and ten")
    if not 128 <= args.render_size <= 2048:
        raise ValueError("render-size must be between 128 and 2048")
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_root = args.output_directory.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"optimization output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_local_config()
    if config is None:
        raise RuntimeError("Darkness config.local.toml is required")
    workspace = Path(config.workspace_root).resolve()
    if output_root != workspace and workspace not in output_root.parents:
        raise ValueError("optimization output directory must be inside the Darkness workspace")
    binding = worker_binding(config, "blender")
    if binding is None:
        raise RuntimeError("Blender has no configured Darkness worker binding")
    manifest = load_manifests()["blender"]
    manager = WorkerManager(workspace, allowed_roots=[workspace])
    adapter = SubprocessWorkerAdapter(
        manager,
        manifest,
        binding.command_prefix,
        environment=binding.environment,
    )
    source_record = _source_record(source)
    operations, parameter_models = canonical_optimizer_operations()
    rig_operations = [item for item in operations if AssetStage.rig in item.stages]
    rig_parameter_models = {
        item.operation_id: parameter_models[item.operation_id] for item in rig_operations
    }
    optimizer = LocalDeployOptimizer(model=args.model)

    accepted_landmarks: dict[str, list[float]] = {}
    accepted_weights: list[dict[str, object]] = []
    accepted_metrics: dict[str, object] | None = None
    accepted_dir: Path | None = None
    baseline_metrics: dict[str, object] | None = None
    baseline_dir: Path | None = None
    candidate_landmarks: dict[str, list[float]] = {}
    candidate_weights: list[dict[str, object]] = []
    candidate_source_proposal: dict[str, object] | None = None
    seen_signatures: set[str] = set()
    failed_strategies: set[str] = set()
    history: list[dict[str, object]] = []
    stop_reason = "iteration_budget_exhausted"

    for iteration in range(args.iterations):
        iteration_dir = output_root / f"iteration-{iteration:02d}"
        job_id = f"{args.run_id}.iteration.{iteration:02d}"
        previous_metrics = accepted_metrics
        previous_dir = accepted_dir
        request = ExternalWorkerRequest(
            job_id=job_id,
            run_id=args.run_id,
            operation_id="blender.propose_short_biped_rig",
            stage=AssetStage.rig,
            inputs=[source_record],
            input_paths={source_record.artifact_id: str(source)},
            parameters={
                "render_size": args.render_size,
                "maximum_material_change_fraction": 0.02,
                "landmark_adjustments": candidate_landmarks,
                "weight_adjustments": candidate_weights,
            },
            output_directory=str(iteration_dir),
        )
        started = time.monotonic()
        adapter.execute(request, timeout_seconds=600)
        worker_seconds = time.monotonic() - started
        metrics = _probe_metrics(iteration_dir)
        deltas = _metric_deltas(metrics, previous_metrics)
        numeric_regression = previous_metrics is not None and _numeric_regression(metrics, previous_metrics)
        numeric_verdict = (
            "baseline"
            if previous_metrics is None
            else "rejected_regression"
            if numeric_regression
            else "not_worse_than_previous_accepted"
        )
        image_deltas: dict[str, float | None] = {
            "front_changed_pixel_fraction": None,
            "front_mean_absolute_channel_delta": None,
            "left_changed_pixel_fraction": None,
            "left_mean_absolute_channel_delta": None,
        }
        if previous_dir is not None:
            front_delta = _image_delta(
                previous_dir / "rig_shoulder_stress_front.png",
                iteration_dir / "rig_shoulder_stress_front.png",
            )
            left_delta = _image_delta(
                previous_dir / "rig_shoulder_stress_left.png",
                iteration_dir / "rig_shoulder_stress_left.png",
            )
            image_deltas = {
                "front_changed_pixel_fraction": front_delta["changed_pixel_fraction"],
                "front_mean_absolute_channel_delta": front_delta["mean_absolute_channel_delta"],
                "left_changed_pixel_fraction": left_delta["changed_pixel_fraction"],
                "left_mean_absolute_channel_delta": left_delta["mean_absolute_channel_delta"],
            }

        if previous_dir is None:
            left_front = iteration_dir / "rig_neutral_front.png"
            left_left = iteration_dir / "rig_neutral_left.png"
            left_label = "BASELINE NEUTRAL"
            right_label = "BASELINE STRESS"
            image_context = "The first panel establishes the baseline: neutral beside the current stress pose."
            comparison_goal = (
                "This is the baseline diagnostic, not a previous-versus-current trial. There is no previous accepted "
                "result: do not claim that a clean previous result exists or that the baseline regressed. If you emit "
                "comparison, use preferred=uncertain and visual_delta=0. "
            )
        else:
            left_front = previous_dir / "rig_shoulder_stress_front.png"
            left_left = previous_dir / "rig_shoulder_stress_left.png"
            left_label = "PREVIOUS ACCEPTED STRESS"
            right_label = "CURRENT ATTEMPT STRESS"
            image_context = (
                "Compare previous accepted (left) with current attempted (right). The exact numeric deltas and "
                "operation under test are supplied in the evidence."
            )
            comparison_goal = (
                "This is a true previous-versus-current trial. Use the supplied render pixel deltas to calibrate "
                "whether a visual difference actually exists. "
            )
        front = _paired_image(
            left_front,
            iteration_dir / "rig_shoulder_stress_front.png",
            iteration_dir / "qwen_evolution_front.png",
            view="FRONT",
            left_label=left_label,
            right_label=right_label,
            left_metrics=previous_metrics,
            right_metrics=metrics,
        )
        side = _paired_image(
            left_left,
            iteration_dir / "rig_shoulder_stress_left.png",
            iteration_dir / "qwen_evolution_left.png",
            view="LEFT",
            left_label=left_label,
            right_label=right_label,
            left_metrics=previous_metrics,
            right_metrics=metrics,
        )
        evidence = EvidenceBundle(
            evidence_id=f"rig.triposg.shoulders.iteration.{iteration:02d}",
            stage=AssetStage.rig,
            goal=(
                comparison_goal
                + "Evaluate the current short-biped shoulder/armpit and elbow deformation against the previous "
                "accepted checkpoint. Explicitly compare the images and numeric values. The hard numeric verdict "
                "cannot be overridden. Diagnose why the operation under test helped or failed. If another attempt "
                "is useful, propose exactly one smallest bounded operation at stage rig: either move one bilateral "
                "landmark pair or redistribute weights at one bilateral joint pair. Change the causal strategy after "
                "a failed operation; never repeat a rejected signature or a smaller version of the same failed move. "
                "Preserve identity, topology, rest coordinates, and symmetry. This is optimization, not production approval."
                " Set goal_satisfied=false while any shoulder or leg severely-compressed-face count is nonzero. "
                "Stability without measured or visible improvement is a tie, not a preferred current result."
            ),
            items=[
                EvidenceItem(artifact_id=f"rig.iteration.{iteration:02d}.front", role="previous_vs_current_front"),
                EvidenceItem(artifact_id=f"rig.iteration.{iteration:02d}.left", role="previous_vs_current_left"),
            ],
            numeric_diagnostics=_evidence_diagnostics(
                iteration=iteration,
                previous=previous_metrics,
                current=metrics,
                deltas=deltas,
                numeric_verdict=numeric_verdict,
                operation_under_test=candidate_source_proposal,
                landmark_adjustments=candidate_landmarks,
                weight_adjustments=candidate_weights,
                history=history,
                worker_seconds=worker_seconds,
                image_deltas=image_deltas,
            ),
            locked_features=[
                "original_triposg_identity",
                "bilateral_symmetry",
                "rest_vertex_coordinates",
                "rest_topology",
            ],
        )
        evidence_path = iteration_dir / "optimizer_evidence.json"
        evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
        decision_started = time.monotonic()
        try:
            decision = optimizer.diagnose(
                evidence,
                rig_operations,
                image_content=[
                    {"type": "text", "text": image_context + " Image 1 is the front view."},
                    _image_content(front),
                    {"type": "text", "text": "Image 2 is the left view of the same previous/current comparison."},
                    _image_content(side),
                ],
                parameter_models=rig_parameter_models,
                decision_model=CanonicalRigOptimizerDecision,
                decision_validator=lambda item: _decision_inconsistency(
                    item,
                    metrics=metrics,
                    previous_metrics=previous_metrics,
                    numeric_regression=numeric_regression,
                    metric_deltas=deltas,
                    image_deltas=image_deltas,
                    operation_under_test=candidate_source_proposal,
                    seen_signatures=seen_signatures,
                    failed_strategies=failed_strategies,
                ),
            )
        except Exception as exc:
            optimizer_seconds = time.monotonic() - decision_started
            failure_path = iteration_dir / "optimizer_failure.json"
            _write_json(
                failure_path,
                {
                    "error": str(exc),
                    "action": "stopped_after_automatic_semantic_retry",
                    "human_review_required": True,
                },
            )
            accepted = previous_metrics is None
            record = {
                "iteration": iteration,
                "source_proposal": candidate_source_proposal,
                "attempted_landmark_adjustments": candidate_landmarks,
                "attempted_weight_adjustments": candidate_weights,
                "previous_accepted_metrics": previous_metrics,
                "metrics": metrics,
                "metric_deltas": deltas,
                "image_deltas": image_deltas,
                "numeric_verdict": numeric_verdict,
                "worker_seconds": round(worker_seconds, 3),
                "optimizer_seconds": round(optimizer_seconds, 3),
                "optimizer_error": str(exc),
                "evidence_path": str(evidence_path),
                "qwen_comparison": None,
                "qwen_observations": [],
                "qwen_root_causes": [],
                "accepted": accepted,
                "acceptance_reason": (
                    "baseline_checkpoint_despite_optimizer_error"
                    if accepted
                    else "optimizer_response_invalid_after_retry"
                ),
            }
            history.append(record)
            if accepted:
                accepted_landmarks = {
                    name: list(values) for name, values in candidate_landmarks.items()
                }
                accepted_weights = [dict(item) for item in candidate_weights]
                accepted_metrics = metrics
                accepted_dir = iteration_dir
                baseline_metrics = metrics
                baseline_dir = iteration_dir
            stop_reason = "optimizer_invalid_after_semantic_retry_human_review_required"
            break
        optimizer_seconds = time.monotonic() - decision_started
        decision_path = iteration_dir / "optimizer_decision.json"
        decision_path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")

        visual_regression = bool(
            previous_metrics is not None
            and decision.comparison is not None
            and decision.comparison.preferred == "previous"
            and decision.confidence >= 0.55
        )
        numeric_improvement = any(
            delta is not None and delta < 0
            for delta in deltas.values()
        )
        visual_improvement = bool(
            previous_metrics is not None
            and decision.comparison is not None
            and decision.comparison.preferred == "current"
            and decision.comparison.visual_delta > 0
            and decision.confidence >= 0.55
            and max(
                float(image_deltas["front_changed_pixel_fraction"] or 0.0),
                float(image_deltas["left_changed_pixel_fraction"] or 0.0),
            )
            >= 0.002
        )
        if previous_metrics is None:
            accepted = True
            acceptance_reason = "baseline_checkpoint"
        elif numeric_regression:
            accepted = False
            acceptance_reason = "hard_numeric_regression"
        elif visual_regression:
            accepted = False
            acceptance_reason = "qwen_preferred_previous_with_sufficient_confidence"
        elif not numeric_improvement and not visual_improvement:
            accepted = False
            acceptance_reason = "no_proven_numeric_or_visual_improvement"
        else:
            accepted = True
            acceptance_reason = "proven_numeric_or_visual_improvement_without_regression"

        record: dict[str, object] = {
            "iteration": iteration,
            "source_proposal": candidate_source_proposal,
            "attempted_landmark_adjustments": candidate_landmarks,
            "attempted_weight_adjustments": candidate_weights,
            "previous_accepted_metrics": previous_metrics,
            "metrics": metrics,
            "metric_deltas": deltas,
            "image_deltas": image_deltas,
            "numeric_verdict": numeric_verdict,
            "worker_seconds": round(worker_seconds, 3),
            "optimizer_seconds": round(optimizer_seconds, 3),
            "visual_score": decision.visual_score,
            "technical_score": decision.technical_score,
            "goal_satisfied": decision.goal_satisfied,
            "request_human_review": decision.request_human_review,
            "qwen_comparison": None if decision.comparison is None else decision.comparison.model_dump(mode="json"),
            "qwen_observations": [item.model_dump(mode="json") for item in decision.observations],
            "qwen_root_causes": [item.model_dump(mode="json") for item in decision.root_causes],
            "decision_path": str(decision_path),
            "evidence_path": str(evidence_path),
            "accepted": accepted,
            "acceptance_reason": acceptance_reason,
        }
        history.append(record)

        if accepted:
            accepted_landmarks = {name: list(values) for name, values in candidate_landmarks.items()}
            accepted_weights = [dict(item) for item in candidate_weights]
            accepted_metrics = metrics
            accepted_dir = iteration_dir
            if baseline_dir is None:
                baseline_dir = iteration_dir
                baseline_metrics = metrics
        elif candidate_source_proposal is not None:
            failed_strategies.add(
                _proposal_strategy(
                    str(candidate_source_proposal["operation_id"]),
                    dict(candidate_source_proposal["parameters"]),
                )
            )

        if accepted and decision.goal_satisfied:
            stop_reason = "optimizer_goal_satisfied_human_review_required"
            break
        if decision.request_human_review and not decision.proposals:
            stop_reason = "optimizer_requested_human_review"
            break

        proposals = [
            item
            for item in decision.proposals
            if item.operation_id in {"rig.adjust_landmark_pair", "skin.redistribute_joint_pair_weights"}
        ]
        if not proposals:
            stop_reason = "optimizer_proposed_no_applicable_fix"
            break
        proposal = proposals[0]
        signature = _proposal_signature(proposal.operation_id, proposal.parameters)
        if signature in seen_signatures:
            record["rejected_next_proposal"] = "repeated_operation_signature"
            stop_reason = "repeated_proposal_signature_human_review_required"
            break
        seen_signatures.add(signature)
        next_proposal = {
            "operation_id": proposal.operation_id,
            "parameters": proposal.parameters,
            "signature": signature,
            "strategy": _proposal_strategy(proposal.operation_id, proposal.parameters),
        }
        if iteration + 1 >= args.iterations:
            record["proposal_not_applied"] = next_proposal
            stop_reason = "iteration_budget_exhausted_human_review_required"
            break

        candidate_landmarks = {name: list(values) for name, values in accepted_landmarks.items()}
        candidate_weights = [dict(item) for item in accepted_weights]
        if proposal.operation_id == "rig.adjust_landmark_pair":
            parameters = AdjustLandmarkPairParameters.model_validate(proposal.parameters)
            candidate_landmarks = apply_landmark_pair_adjustment(accepted_landmarks, parameters)
        else:
            parameters = RedistributeJointPairWeightsParameters.model_validate(proposal.parameters)
            if len(candidate_weights) >= 4:
                record["proposal_not_applied"] = next_proposal
                stop_reason = "weight_adjustment_budget_exhausted_human_review_required"
                break
            candidate_weights.append(parameters.model_dump(mode="json"))
        candidate_source_proposal = next_proposal
        record["applied_next_proposal"] = next_proposal

    if baseline_dir is None or baseline_metrics is None or accepted_dir is None or accepted_metrics is None:
        raise RuntimeError("optimizer episode did not establish an accepted baseline")
    review_paths = _write_human_review(
        output_root,
        source=source,
        model=args.model,
        baseline_dir=baseline_dir,
        accepted_dir=accepted_dir,
        baseline_metrics=baseline_metrics,
        accepted_metrics=accepted_metrics,
        accepted_landmarks=accepted_landmarks,
        accepted_weights=accepted_weights,
        history=history,
        stop_reason=stop_reason,
    )
    result = {
        "schema_version": 2,
        "run_id": args.run_id,
        "model": args.model,
        "source": str(source),
        "source_sha256": source_record.sha256,
        "iteration_budget": args.iterations,
        "iterations_completed": len(history),
        "stop_reason": stop_reason,
        "baseline_metrics": baseline_metrics,
        "accepted_landmark_adjustments": accepted_landmarks,
        "accepted_weight_adjustments": accepted_weights,
        "accepted_metrics": accepted_metrics,
        "human_approval_required": True,
        "human_approved": False,
        "human_review": review_paths,
        "history": history,
    }
    _write_json(output_root / "optimization_episode.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    result = run_episode(_arguments(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
