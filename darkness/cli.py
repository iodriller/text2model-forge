"""Command-line entry point for local, resumable Darkness runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import DarknessCompiler
from .config import load_local_config, worker_binding
from .external_worker import SubprocessWorkerAdapter
from .fake_worker import build_fake_registry
from .glb_components import build_component_audit
from .manifests import load_manifests, preflight
from .mesh_evidence import build_mesh_evidence
from .packaging import build_delivery_package
from .regression import evaluate_candidate
from .settings import resolve_settings
from .schemas import (
    AssetBrief,
    AssetComponent,
    AssetStage,
    DestinationContract,
    MetricResult,
    ExternalWorkerRequest,
)
from .workers import WorkerManager


def _demo_brief() -> AssetBrief:
    return AssetBrief(
        asset_id="darkness_goblin_demo",
        asset_type="character",
        title="Darkness Goblin Demo",
        description="Original short-biped stylized fantasy goblin used to prove compiler orchestration.",
        style="chunky heroic dark fantasy with ember accents",
        anatomy_family="short_biped",
        height_m=1.25,
        components=[
            AssetComponent(
                component_id="body",
                semantic_type="deformable_body",
                rig_policy="skinned",
            ),
            AssetComponent(
                component_id="weapon",
                semantic_type="rigid_equipment",
                rig_policy="socket",
            ),
        ],
        required_animations=["idle", "walk", "attack", "hit", "death"],
        destination=DestinationContract(delivery=["blend", "glb", "sprite_sheet"]),
        locked_features=["original_identity", "short_biped_proportions"],
    )


def run_demo(workspace: Path, run_id: str) -> dict:
    compiler = DarknessCompiler(workspace, build_fake_registry())
    first = compiler.create_run(run_id, _demo_brief())
    evaluation = evaluate_candidate(
        evaluation_id="eval.brief.v1",
        candidate_artifact_id=first.artifact_id,
        baseline_artifact_id=None,
        metrics=[MetricResult(name="schema_valid", value=True, passed=True, hard_failure=True)],
        technical_score=1.0,
    )
    compiler.record_evaluation(run_id, evaluation)
    compiler.approve(run_id, artifact_id=first.artifact_id, approved_by="fixture-human")
    previous_id = first.artifact_id

    for target_stage in list(AssetStage)[1:]:
        compiler.advance(run_id)
        artifact_id = f"darkness_goblin_demo.{target_stage.name}.v1"
        created = compiler.execute_candidate(
            run_id,
            operation_id="fake.advance",
            artifact_id=artifact_id,
            input_artifact_ids=[previous_id],
            parameters={
                "target_stage": target_stage.value,
                "annotations": {"fixture": True, "real_worker_required": target_stage is not AssetStage.brief},
            },
        )
        evaluation = evaluate_candidate(
            evaluation_id=f"eval.{target_stage.name}.v1",
            candidate_artifact_id=created.artifact_id,
            baseline_artifact_id=previous_id,
            metrics=[
                MetricResult(name="artifact_integrity", value=True, passed=True, hard_failure=True),
                MetricResult(name="lineage_complete", value=True, passed=True, hard_failure=True),
            ],
            technical_score=1.0,
        )
        compiler.record_evaluation(run_id, evaluation)
        compiler.approve(run_id, artifact_id=created.artifact_id, approved_by="fixture-human")
        previous_id = created.artifact_id

    compiler.complete(run_id)
    return compiler.log(run_id).state().model_dump(mode="json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="darkness", description="Asset Forge Darkness compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the deterministic D0-D10 orchestration fixture")
    demo.add_argument("--workspace", type=Path, required=True)
    demo.add_argument("--run-id", default="darkness.demo.v1")

    state = subparsers.add_parser("state", help="print reduced state for a run")
    state.add_argument("--workspace", type=Path, required=True)
    state.add_argument("--run-id", required=True)

    verify = subparsers.add_parser("verify", help="verify event chain and every artifact digest")
    verify.add_argument("--workspace", type=Path, required=True)
    verify.add_argument("--run-id", required=True)

    subparsers.add_parser("workers", help="print strict worker manifests and live preflight state")

    doctor = subparsers.add_parser(
        "doctor",
        help="check this machine's hardware and every cross-stage assumption before a run",
    )
    doctor.add_argument("--profile", default="simple")
    doctor.add_argument(
        "--deep",
        action="store_true",
        help="also launch Blender to verify its render engine and the donor motion's bone names",
    )
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.add_argument(
        "--asset-height-m",
        type=float,
        default=1.8,
        help="height of the asset you intend to build; voxel sizing depends on it",
    )

    studio = subparsers.add_parser(
        "studio",
        help="launch the standalone description-to-character browser control plane, "
        "or drive it headlessly with a subcommand",
    )
    studio.add_argument("--workspace", type=Path)
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=int, default=8766)
    studio.add_argument("--open-browser", action="store_true")
    studio.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="permit a non-loopback bind (intended for a container whose published port is host-loopback only)",
    )
    # No subcommand (just `darkness studio --workspace ... [--open-browser]`)
    # keeps launching the browser control plane exactly as before -- these
    # subcommands are additive, not a replacement, so existing invocations in
    # README.md and machine setup docs keep working unchanged.
    studio_sub = studio.add_subparsers(dest="studio_command")

    studio_list = studio_sub.add_parser("list", help="list studio runs as JSON")
    studio_list.add_argument("--workspace", type=Path)
    studio_list.add_argument(
        "--include-archived", action="store_true", help="also list runs hidden from the dashboard"
    )

    studio_show = studio_sub.add_parser("show", help="print one run's full persisted state as JSON")
    studio_show.add_argument("--workspace", type=Path)
    studio_show.add_argument("--run-id", required=True)

    studio_decide = studio_sub.add_parser(
        "decide",
        help="record one human decision at a stage gate and, by default, drive the pipeline "
        "to its next stopping point",
    )
    studio_decide.add_argument("--workspace", type=Path)
    studio_decide.add_argument("--run-id", required=True)
    studio_decide.add_argument("--stage-id", required=True)
    studio_decide.add_argument(
        "--decision", required=True, choices=("approve", "reject", "retry", "edit", "skip", "rollback")
    )
    studio_decide.add_argument("--comment", default="")
    studio_decide.add_argument("--selected-evidence-id", default=None)
    studio_decide.add_argument(
        "--overrides", default=None, help='JSON object, for example \'{"seed": 42}\''
    )
    studio_decide.add_argument("--target-stage-id", default=None)
    studio_decide.add_argument(
        "--no-resume",
        action="store_true",
        help="record the decision only; do not drive the pipeline afterward",
    )

    worker = subparsers.add_parser("run-worker", help="run one configured worker through the validated file contract")
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--timeout", type=float, default=1800)

    mesh_check = subparsers.add_parser(
        "mesh-check",
        help="diagnose and safely repair a triangular OBJ, then write station-3 evidence",
    )
    mesh_check.add_argument("--input", type=Path, required=True)
    mesh_check.add_argument("--output", type=Path, required=True)
    mesh_check.add_argument("--weld-tolerance", type=float, default=1e-9)
    mesh_check.add_argument("--minimum-component-faces", type=int, default=2)

    component_audit = subparsers.add_parser(
        "glb-component-audit",
        help="emit non-destructive GLB component cleanup branches and turntables",
    )
    component_audit.add_argument("--input", type=Path, required=True)
    component_audit.add_argument("--output", type=Path, required=True)
    component_audit.add_argument("--minimum-structural-faces", type=int, default=100)
    component_audit.add_argument("--minimum-structural-fraction", type=float, default=0.001)

    config_cmd = subparsers.add_parser("config", help="inspect the resolved layered configuration")
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser(
        "show", help="print the fully resolved configuration and which layer set each value"
    )
    config_show.add_argument("--profile", default="simple")
    config_show.add_argument("--machine-path", type=Path, default=None)
    config_show.add_argument(
        "--values-only", action="store_true", help="print only the merged values, without per-key origin"
    )

    package = subparsers.add_parser("package", help="build a digest-pinned research or release delivery")
    package.add_argument("--package-id", required=True)
    package.add_argument("--candidate-id", required=True)
    package.add_argument("--source", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--qualification", type=Path, required=True)
    package.add_argument("--mode", choices=("research", "release"), default="research")
    return parser


def _studio_workspace(args: argparse.Namespace) -> Path:
    config = load_local_config()
    workspace = args.workspace or (Path(config.workspace_root) if config else None)
    if workspace is None:
        raise SystemExit("--workspace is required when config.local.toml is missing")
    return workspace


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "studio":
        studio_command = getattr(args, "studio_command", None)
        if studio_command is None:
            from .studio_web import serve

            serve(
                _studio_workspace(args),
                host=args.host,
                port=args.port,
                open_browser=args.open_browser,
                allow_non_loopback=args.allow_non_loopback,
            )
            return 0
        from .studio_cli import decide, list_runs, show_run
        from .studio_store import StudioStore

        store = StudioStore(_studio_workspace(args))
        if studio_command == "list":
            print(json.dumps(list_runs(store, include_archived=args.include_archived), indent=2))
            return 0
        if studio_command == "show":
            print(json.dumps(show_run(store, args.run_id), indent=2))
            return 0
        if studio_command == "decide":
            overrides = None
            if args.overrides:
                try:
                    overrides = json.loads(args.overrides)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"--overrides must be valid JSON: {exc.msg}") from exc
                if not isinstance(overrides, dict):
                    raise SystemExit('--overrides must be a JSON object, for example {"seed": 42}')
            result = decide(
                store,
                args.run_id,
                args.stage_id,
                args.decision,
                args.comment,
                args.selected_evidence_id,
                overrides=overrides,
                target_stage_id=args.target_stage_id,
                resume=not args.no_resume,
            )
            print(json.dumps(result, indent=2))
            return 0
    if args.command == "doctor":
        from .preflight import run_preflight
        from .hardware import recommend_stack

        hardware, checks = run_preflight(
            profile=args.profile, deep=args.deep, asset_height_m=args.asset_height_m
        )
        recommendation = recommend_stack(hardware, asset_height_m=args.asset_height_m)
        if args.json:
            print(
                json.dumps(
                    {
                        "hardware": hardware.model_dump(mode="json"),
                        "recommendation": recommendation.model_dump(mode="json"),
                        "checks": [check.model_dump(mode="json") for check in checks],
                    },
                    indent=2,
                )
            )
        else:
            gpu = hardware.primary
            print("HARDWARE")
            print(f"  source        {hardware.source}")
            print(f"  gpu           {gpu.name if gpu else 'none detected'}")
            if gpu:
                print(f"  vram          {gpu.vram_total_gb} GB total, {gpu.vram_free_gb} GB free")
            print(f"  system ram    {hardware.system_ram_gb} GB")
            print()
            print(f"RECOMMENDED STACK  (profile {recommendation.profile})")
            print(f"  reviewer      {recommendation.reviewer_size}-class vision model")
            print(f"  spec_strategy {recommendation.spec_strategy}")
            print(f"  vram_handoff  {recommendation.vram_handoff}")
            print(f"  voxel_fraction {recommendation.voxel_fraction}")
            for reason in recommendation.reasons:
                print(f"    - {reason}")
            for warning in recommendation.warnings:
                print(f"    ! {warning}")
            print()
            print("PREFLIGHT")
            for check in checks:
                print(f"  [{check.status.upper():4}] {check.name}")
                if check.detail:
                    print(f"         {check.detail}")
                if check.remedy and check.status in {"fail", "warn"}:
                    print(f"      fix: {check.remedy}")
        failures = [check for check in checks if check.status == "fail"]
        return 1 if failures else 0
    if args.command == "workers":
        config = load_local_config()
        result = {}
        for worker_id, manifest in load_manifests().items():
            binding = worker_binding(config, worker_id)
            result[worker_id] = preflight(
                manifest,
                command_prefix=binding.command_prefix if binding else None,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-worker":
        config = load_local_config()
        manifests = load_manifests()
        if args.worker_id not in manifests:
            raise SystemExit(f"unknown worker_id: {args.worker_id}")
        binding = worker_binding(config, args.worker_id)
        if binding is None:
            raise SystemExit(f"worker '{args.worker_id}' has no local binding")
        # Accept PowerShell-authored JSON as well as canonical UTF-8. Windows
        # PowerShell commonly emits an UTF-8 BOM, which JSON parsers otherwise
        # report as an invalid first character.
        request = ExternalWorkerRequest.model_validate_json(args.request.read_text(encoding="utf-8-sig"))
        manager = WorkerManager(config.workspace_root, allowed_roots=[config.workspace_root])
        adapter = SubprocessWorkerAdapter(
            manager,
            manifests[args.worker_id],
            binding.command_prefix,
            environment=binding.environment,
        )
        response = adapter.execute(request, timeout_seconds=args.timeout)
        print(response.model_dump_json(indent=2))
        return 0
    if args.command == "config":
        resolved = resolve_settings(profile=args.profile, machine_path=args.machine_path)
        if args.values_only:
            print(json.dumps(resolved.values, indent=2, sort_keys=True))
        else:
            report = {
                key: {"value": value, "set_by": resolved.origin.get(key, "base")}
                for key, value in sorted(resolved.flat().items())
            }
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "package":
        registry_path = Path(__file__).resolve().parents[1] / "registry" / "candidates.json"
        manifest = build_delivery_package(
            package_id=args.package_id,
            candidate_id=args.candidate_id,
            source_root=args.source,
            output_root=args.output,
            registry_path=registry_path,
            qualification_path=args.qualification,
            mode=args.mode,
        )
        print(manifest.model_dump_json(indent=2))
        return 0
    if args.command == "mesh-check":
        result = build_mesh_evidence(
            args.input,
            args.output,
            tolerance=args.weld_tolerance,
            minimum_component_faces=args.minimum_component_faces,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "glb-component-audit":
        result = build_component_audit(
            args.input,
            args.output,
            minimum_structural_faces=args.minimum_structural_faces,
            minimum_structural_fraction=args.minimum_structural_fraction,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    compiler = DarknessCompiler(args.workspace, build_fake_registry())
    if args.command == "demo":
        result = run_demo(args.workspace, args.run_id)
    elif args.command == "state":
        result = compiler.log(args.run_id).state().model_dump(mode="json")
    else:
        events = compiler.log(args.run_id).read(verify=True)
        records = compiler.artifacts.all_records()
        for artifact_id in records:
            compiler.artifacts.read_bytes(artifact_id, verify=True)
        result = {"valid": True, "events": len(events), "artifacts": len(records)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
