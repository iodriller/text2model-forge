from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .animate import DEFAULT_CHECKPOINT, DEFAULT_CONTROLNET, generate_animation
from .comfy import ComfyClient, build_sdxl_workflow, generate
from .blender_worker import audit_master
from .pose import load_pack, materialize_pack, render_action_frames
from .sheets import pack_sheets
from .core import (
    APPROVAL_STAGES,
    ASSET_KINDS,
    ForgeError,
    approval_status,
    approve_artifact,
    asset_root,
    build_prompt_pack,
    create_asset,
    init_workspace,
    load_asset,
    load_project,
    package_root,
    project_status,
    read_json,
    register_model,
    write_json,
)
from .compliance import build_compliance_report
from .mesh import run_triposr
from .exporter import export_asset
from .qa import validate_sheets
from .production import package_production_unit


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def run_probe(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}
    output = (result.stdout or result.stderr).strip()
    return {"ok": result.returncode == 0, "exit_code": result.returncode, "output": output}


def doctor(base_url: str) -> dict[str, object]:
    blender_candidates = [
        shutil.which("blender"),
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    ]
    blender = next((value for value in blender_candidates if value and Path(value).is_file()), None)
    report: dict[str, object] = {
        "python": {"ok": True, "version": sys.version.split()[0], "executable": sys.executable},
        "nvidia": run_probe(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "blender": run_probe([blender, "--version"]) if blender else {"ok": False, "error": "not found"},
        "comfyui": {"ok": False, "base_url": base_url},
    }
    try:
        stats = ComfyClient(base_url, timeout=3).system_stats()
        report["comfyui"] = {"ok": True, "base_url": base_url, "system_stats": stats}
    except ForgeError as error:
        report["comfyui"] = {"ok": False, "base_url": base_url, "error": str(error)}
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assetforge", description="Asset Forge Studio local production CLI")
    parser.add_argument("--version", action="version", version="Asset Forge Studio 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create a portable Asset Forge workspace")
    init.add_argument("--workspace", type=Path, required=True)
    init.add_argument("--project-id", required=True)
    init.add_argument("--name", required=True)
    init.add_argument("--engine", default="unity")
    init.add_argument("--engine-root")

    new = sub.add_parser("new", help="create a structured asset brief")
    new.add_argument("--workspace", type=Path, required=True)
    new.add_argument("--asset-id", required=True)
    new.add_argument("--name", required=True)
    new.add_argument("--kind", choices=sorted(ASSET_KINDS), required=True)
    new.add_argument("--role", required=True)
    new.add_argument("--faction", default="neutral")
    new.add_argument("--primary-action", choices=["attack", "shoot", "cast", "none"], default="attack")

    prompt = sub.add_parser("prompt", help="compile the style profile and brief into reproducible prompts")
    prompt.add_argument("--workspace", type=Path, required=True)
    prompt.add_argument("--asset-id", required=True)

    workflow = sub.add_parser("workflow", help="write a core-node ComfyUI API workflow and provenance sidecar")
    workflow.add_argument("--workspace", type=Path, required=True)
    workflow.add_argument("--asset-id", required=True)
    workflow.add_argument("--mode", choices=["concept", "turnaround", "texture", "mesh_seed"], default="concept")
    workflow.add_argument("--checkpoint", default="sd_xl_base_1.0.safetensors")
    workflow.add_argument("--seed", type=int, required=True)
    workflow.add_argument("--output", type=Path)

    run = sub.add_parser("generate", help="submit a workflow to local ComfyUI and record outputs/provenance")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--asset-id", required=True)
    run.add_argument("--mode", choices=["concept", "turnaround", "texture", "mesh_seed"], default="concept")
    run.add_argument("--checkpoint", default="sd_xl_base_1.0.safetensors")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    run.add_argument("--timeout", type=float, default=900)

    unit = sub.add_parser("create-unit", help="create a diffusion motion prototype for review (never production eligible)")
    unit.add_argument("--workspace", type=Path, required=True)
    unit.add_argument("--asset-id", required=True)
    unit.add_argument("--name", help="required only when the asset does not exist yet")
    unit.add_argument("--kind", choices=sorted(ASSET_KINDS), default="character")
    unit.add_argument("--role", help="required only when the asset does not exist yet")
    unit.add_argument("--faction", default="neutral")
    unit.add_argument("--primary-action", choices=["attack", "shoot", "cast", "none"], default="attack")
    unit.add_argument("--reference", type=Path, help="identity image; defaults to the latest approved concept")
    unit.add_argument("--actions", help="comma-separated; defaults to the asset's required actions")
    unit.add_argument("--seed", type=int, default=1001)
    unit.add_argument("--denoise", type=float, default=0.68)
    unit.add_argument("--control-strength", type=float, default=0.85)
    unit.add_argument("--pack", default="humanoid_side_east")
    unit.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    unit.add_argument("--timeout", type=float, default=600)
    unit.add_argument("--to-unity", type=Path, help=argparse.SUPPRESS)

    poses = sub.add_parser("poses", help="materialize the editable pose pack and render control-image previews")
    poses.add_argument("--workspace", type=Path, required=True)
    poses.add_argument("--pack", default="humanoid_side_east")
    poses.add_argument("--preview", action="store_true", help="render every action's control images for review")

    animate = sub.add_parser("animate", help="generate pose-controlled motion-board frames (prototype only)")
    animate.add_argument("--workspace", type=Path, required=True)
    animate.add_argument("--asset-id", required=True)
    animate.add_argument("--actions", required=True, help="comma-separated, e.g. idle,walk,attack")
    animate.add_argument("--reference", type=Path, required=True, help="approved concept image that locks identity")
    animate.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    animate.add_argument("--controlnet", default=DEFAULT_CONTROLNET)
    animate.add_argument("--pack", default="humanoid_side_east")
    animate.add_argument("--seed", type=int, default=1001)
    animate.add_argument("--denoise", type=float, default=0.6)
    animate.add_argument("--control-strength", type=float, default=0.85)
    animate.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    animate.add_argument("--timeout", type=float, default=600)

    pack = sub.add_parser("pack-sheets", help="pack generated frames into baseline-locked sheets and run QA")
    pack.add_argument("--workspace", type=Path, required=True)
    pack.add_argument("--asset-id", required=True)
    pack.add_argument("--actions", help="comma-separated; defaults to every action with rendered frames")
    pack.add_argument("--pack", default="humanoid_side_east")

    approve = sub.add_parser("approve", help="create an immutable hash-bound stage approval")
    approve.add_argument("--workspace", type=Path, required=True)
    approve.add_argument("--asset-id", required=True)
    approve.add_argument("--stage", choices=APPROVAL_STAGES, required=True)
    approve.add_argument("--artifact", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--notes", default="")

    mesh = sub.add_parser("mesh-seed", help="create a provenance-tracked TripoSR mesh seed from an approved concept")
    mesh.add_argument("--workspace", type=Path, required=True)
    mesh.add_argument("--asset-id", required=True)
    mesh.add_argument("--input", type=Path, required=True)
    mesh.add_argument("--runtime-root", type=Path)
    mesh.add_argument("--texture-resolution", type=int, choices=[512, 1024, 2048], default=1024)
    mesh.add_argument("--allow-unapproved", action="store_true")

    audit = sub.add_parser("audit-master", help="audit a Blender master, materials, textures, scale, polygons, and actions")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--blender")

    morphology = sub.add_parser(
        "validate-morphology",
        help="validate a Creature DNA profile and its hash-bound Blender build report",
    )
    morphology.add_argument("--profile", type=Path, required=True)
    morphology.add_argument("--build-report", type=Path, required=True)
    morphology.add_argument("--master", type=Path, required=True)
    morphology.add_argument("--repo-root", type=Path, default=Path.cwd())

    morphology_proof = sub.add_parser(
        "morphology-proof",
        help="compose a geometry-first creature review board from smoke-rendered frames",
    )
    morphology_proof.add_argument("--config", type=Path, required=True)
    morphology_proof.add_argument("--profile", type=Path, required=True)
    morphology_proof.add_argument("--frames-root", type=Path, required=True)
    morphology_proof.add_argument("--output", type=Path, required=True)

    overpaint = sub.add_parser("overpaint", help="repaint rendered frames with SDXL img2img; alpha stays 3D ground truth")
    overpaint.add_argument("--config", type=Path, required=True, help="character config with an 'overpaint' block")
    overpaint.add_argument("--frames-root", type=Path, required=True)
    overpaint.add_argument("--limit", type=int, help="process only the first N frames (tuning runs)")
    overpaint.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    overpaint.add_argument("--timeout", type=float, default=300)

    repair = sub.add_parser(
        "repair-overpaint",
        help="restore protected master pixels in historical/partial overpaint frames (review-only)",
    )
    repair.add_argument("--config", type=Path, required=True)
    repair.add_argument("--frames-root", type=Path, required=True)

    bake = sub.add_parser("bake-master", help="paint canonical views once and bake them into the master's textures")
    bake.add_argument("--config", type=Path, required=True, help="character config with a 'texture_master' block")
    bake.add_argument("--repo-root", type=Path, default=Path.cwd())
    bake.add_argument("--blender", required=True)
    bake.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    bake.add_argument("--timeout", type=float, default=600)
    bake.add_argument("--force", action="store_true", help="rebake even when the run signature matches")

    critic = sub.add_parser("critique", help="run the deterministic art-direction gates on packed production sheets")
    critic.add_argument("--config", type=Path, required=True)
    critic.add_argument("--repo-root", type=Path, default=Path.cwd())
    critic.add_argument("--report", type=Path, required=True)

    deform = sub.add_parser(
        "check-deformation",
        help="check 2D silhouette continuity, area, and baseline drift in raw action renders",
    )
    deform.add_argument("--config", type=Path, required=True)
    deform.add_argument("--repo-root", type=Path, default=Path.cwd())
    deform.add_argument("--frames-root", type=Path, required=True)
    deform.add_argument("--report", type=Path, required=True)

    genesis_spec = sub.add_parser(
        "validate-genesis-spec",
        help="validate a from-scratch semantic anatomy specification",
    )
    genesis_spec.add_argument("--spec", type=Path, required=True)

    genesis_target = sub.add_parser(
        "validate-genesis-target",
        help="verify a fused anatomy target while keeping it blocked from production",
    )
    genesis_target.add_argument("--spec", type=Path, required=True)
    genesis_target.add_argument("--target", type=Path, required=True)
    genesis_target.add_argument("--build-report", type=Path, required=True)

    family_contract = sub.add_parser(
        "validate-family-contract",
        help="validate a reusable canonical topology/deformation family contract",
    )
    family_contract.add_argument("--contract", type=Path, required=True)

    canonical_master = sub.add_parser(
        "validate-canonical-master",
        help="gate a fitted, skinned, action-bearing canonical family master",
    )
    canonical_master.add_argument("--contract", type=Path, required=True)
    canonical_master.add_argument("--master", type=Path, required=True)
    canonical_master.add_argument("--build-report", type=Path, required=True)

    canonical_review = sub.add_parser(
        "canonical-review",
        help="compose representative canonical-master renders and gate metrics",
    )
    canonical_review.add_argument("--frames-root", type=Path, required=True)
    canonical_review.add_argument("--build-report", type=Path, required=True)
    canonical_review.add_argument("--output", type=Path, required=True)

    qa = sub.add_parser("qa-sheets", help="run commercial sheet QA and create a review contact sheet")
    qa.add_argument("--config", type=Path, required=True)
    qa.add_argument("--repo-root", type=Path, default=Path.cwd())
    qa.add_argument("--report", type=Path, required=True)
    qa.add_argument("--contact-sheet", type=Path)

    package = sub.add_parser("package-production", help="gate, review, and package one owned Blender unit for Unity")
    package.add_argument("--config", type=Path, required=True)
    package.add_argument("--repo-root", type=Path, default=Path.cwd())
    package.add_argument("--qa-report", type=Path, required=True)
    package.add_argument("--destination", type=Path, required=True)
    package.add_argument("--review-root", type=Path, required=True)

    export = sub.add_parser("export", help="export approved sheets through a generic or engine-specific adapter")
    export.add_argument("--workspace", type=Path, required=True)
    export.add_argument("--asset-id", required=True)
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--adapter", default="generic")
    export.add_argument("--allow-unapproved", action="store_true")

    status = sub.add_parser("status", help="show derived project and stage status")
    status.add_argument("--workspace", type=Path, required=True)

    check = sub.add_parser("doctor", help="check GPU, Blender, Python, and local ComfyUI")
    check.add_argument("--comfy-url", default="http://127.0.0.1:8188")

    register = sub.add_parser("register-model", help="hash and register the exact local model used for production")
    register.add_argument("--workspace", type=Path, required=True)
    register.add_argument("--model-id", required=True)
    register.add_argument("--file", type=Path, required=True)
    register.add_argument("--reviewer", required=True)

    compliance = sub.add_parser("compliance", help="verify model/job/output/license provenance for an asset")
    compliance.add_argument("--workspace", type=Path, required=True)
    compliance.add_argument("--asset-id", required=True)
    compliance.add_argument("--report", type=Path, required=True)

    serve = sub.add_parser("serve", help="start the loopback-only Asset Forge Studio UI")
    serve.add_argument("--workspace", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def execute(args: argparse.Namespace) -> object:
    if args.command == "init":
        return init_workspace(args.workspace, args.project_id, args.name, args.engine, args.engine_root)
    if args.command == "new":
        return create_asset(args.workspace, args.asset_id, args.name, args.kind, args.role, args.faction, args.primary_action)
    if args.command == "prompt":
        return build_prompt_pack(args.workspace, args.asset_id)
    if args.command == "workflow":
        workflow, provenance = build_sdxl_workflow(args.workspace, args.asset_id, args.mode, args.checkpoint, args.seed)
        output = args.output or (asset_root(args.workspace, args.asset_id) / "jobs" / f"workflow-{args.mode}-{args.seed}.json")
        write_json(output, workflow)
        write_json(output.with_suffix(".provenance.json"), provenance)
        return {"workflow": str(output.resolve()), "provenance": str(output.with_suffix('.provenance.json').resolve())}
    if args.command == "generate":
        outputs = generate(args.workspace, args.asset_id, args.mode, args.checkpoint, args.seed, args.comfy_url, args.timeout)
        return {"outputs": [str(path.resolve()) for path in outputs]}
    if args.command == "create-unit":
        if args.to_unity:
            raise ForgeError(
                "Diffusion animation frames are motion prototypes and cannot be copied into Unity as final art. "
                "Build one owned Blender master, run qa-sheets, then use package-production."
            )
        try:
            asset = load_asset(args.workspace, args.asset_id)
        except ForgeError:
            if not (args.name and args.role):
                raise ForgeError(
                    f"Asset {args.asset_id!r} does not exist; pass --name and --role to create it"
                ) from None
            asset = create_asset(
                args.workspace, args.asset_id, args.name, args.kind, args.role,
                args.faction, args.primary_action,
            )
        build_prompt_pack(args.workspace, args.asset_id)
        reference = args.reference
        if reference is None:
            status = approval_status(args.workspace, args.asset_id).get("concept", {})
            if status.get("approved"):
                reference = Path(read_json(Path(status["approval"]))["artifact"])
        if reference is None:
            raise ForgeError(
                "No identity reference: pass --reference, or generate concepts "
                "(assetforge generate --mode concept) and approve one (assetforge approve --stage concept)"
            )
        if args.actions:
            actions = [item.strip() for item in args.actions.split(",") if item.strip()]
        else:
            actions = list(asset.get("production", {}).get("required_actions", [])) or ["idle", "walk", "attack", "hit", "death"]
        animation = generate_animation(
            args.workspace, args.asset_id, actions,
            reference=reference, pack_id=args.pack, seed=args.seed,
            denoise=args.denoise, control_strength=args.control_strength,
            base_url=args.comfy_url, timeout_seconds=args.timeout,
        )
        report = pack_sheets(args.workspace, args.asset_id, actions, args.pack)
        if not report["passed"]:
            raise ForgeError("Sheet QA failed; failures: " + "; ".join(report["failures"][:8]))
        payload = {
            "asset_id": args.asset_id,
            "actions": sorted(animation["actions"]),
            "sheet_qa_passed": True,
            "production_eligible": False,
            "production_blocker": "diffusion_frame_sequence_has_no_single_master_identity_lock",
            "unit_manifest": report["unit_manifest"],
            "gameplay_preview": report["gameplay_preview"],
        }
        return payload
    if args.command == "poses":
        path = materialize_pack(args.workspace, args.pack)
        result = {"pose_pack": str(path)}
        if args.preview:
            pack = load_pack(args.workspace, args.pack)
            preview_root = args.workspace.resolve() / "poses" / "preview" / args.pack
            for action_name in pack.get("actions", {}):
                folder = preview_root / action_name
                folder.mkdir(parents=True, exist_ok=True)
                for index, image in enumerate(render_action_frames(pack, action_name)):
                    image.save(folder / f"{index:02d}.png", "PNG")
            result["preview"] = str(preview_root)
        return result
    if args.command == "animate":
        return generate_animation(
            args.workspace,
            args.asset_id,
            [item.strip() for item in args.actions.split(",") if item.strip()],
            checkpoint=args.checkpoint,
            controlnet=args.controlnet,
            reference=args.reference,
            pack_id=args.pack,
            seed=args.seed,
            denoise=args.denoise,
            control_strength=args.control_strength,
            base_url=args.comfy_url,
            timeout_seconds=args.timeout,
        )
    if args.command == "pack-sheets":
        actions = [item.strip() for item in args.actions.split(",") if item.strip()] if args.actions else None
        report = pack_sheets(args.workspace, args.asset_id, actions, args.pack)
        if not report["passed"]:
            raise ForgeError("Sheet QA failed; failures: " + "; ".join(report["failures"][:8]))
        return report
    if args.command == "approve":
        path = approve_artifact(args.workspace, args.asset_id, args.stage, args.artifact, args.reviewer, args.notes)
        return {"approval": str(path.resolve())}
    if args.command == "mesh-seed":
        meshes = run_triposr(
            args.workspace,
            args.asset_id,
            args.input,
            args.runtime_root,
            args.texture_resolution,
            allow_unapproved=args.allow_unapproved,
        )
        return {"mesh_seeds": [str(path.resolve()) for path in meshes]}
    if args.command == "audit-master":
        return audit_master(args.config, args.report, args.blender)
    if args.command == "validate-morphology":
        from .morphology import validate_morphology_build
        return validate_morphology_build(
            args.profile, args.build_report, args.master, args.repo_root,
        )
    if args.command == "morphology-proof":
        from .morphology import create_morphology_proof
        return create_morphology_proof(
            args.config, args.profile, args.frames_root, args.output,
        )
    if args.command == "overpaint":
        from .overpaint import overpaint_frames
        return overpaint_frames(args.config, args.frames_root, args.limit, args.comfy_url, args.timeout)
    if args.command == "repair-overpaint":
        from .overpaint import repair_protected_frames
        return repair_protected_frames(args.config, args.frames_root)
    if args.command == "bake-master":
        from .bake import bake_texture_master
        return bake_texture_master(args.config, args.repo_root, args.blender, args.comfy_url, args.timeout, args.force)
    if args.command == "critique":
        from .critic import critique_unit
        report = critique_unit(args.config, args.repo_root.resolve(), args.report)
        if not report["passed"]:
            raise ForgeError("Art-direction critique failed; inspect " + str(args.report.resolve()))
        return report
    if args.command == "check-deformation":
        from .deformation import critique_deformation
        report = critique_deformation(args.config, args.repo_root.resolve(), args.frames_root, args.report)
        if not report["passed"]:
            raise ForgeError("Render-space deformation preflight failed (silhouette split, area jump, or baseline drift); inspect " + str(args.report.resolve()))
        return report
    if args.command == "validate-genesis-spec":
        from .genesis import validate_genesis_spec
        return validate_genesis_spec(args.spec)
    if args.command == "validate-genesis-target":
        from .genesis import validate_anatomy_target
        return validate_anatomy_target(args.spec, args.target, args.build_report)
    if args.command == "validate-family-contract":
        from .families import validate_family_contract
        return validate_family_contract(args.contract)
    if args.command == "validate-canonical-master":
        from .families import validate_canonical_master
        return validate_canonical_master(args.contract, args.master, args.build_report)
    if args.command == "canonical-review":
        from .canonical_review import compose_canonical_review
        return compose_canonical_review(args.frames_root, args.build_report, args.output)
    if args.command == "qa-sheets":
        report = validate_sheets(args.config, args.repo_root.resolve(), args.report, args.contact_sheet)
        if not report["passed"]:
            raise ForgeError("Sheet QA failed; inspect " + str(args.report.resolve()))
        return report
    if args.command == "package-production":
        return package_production_unit(
            args.config,
            args.repo_root,
            args.qa_report,
            args.destination,
            args.review_root,
        )
    if args.command == "export":
        return export_asset(args.workspace, args.asset_id, args.destination, args.adapter, not args.allow_unapproved)
    if args.command == "status":
        return project_status(args.workspace)
    if args.command == "doctor":
        return doctor(args.comfy_url)
    if args.command == "register-model":
        return register_model(args.workspace, args.model_id, args.file, args.reviewer)
    if args.command == "compliance":
        report = build_compliance_report(args.workspace, args.asset_id, args.report)
        if not report["passed"]:
            raise ForgeError("Compliance failed; inspect " + str(args.report.resolve()))
        return report
    if args.command == "serve":
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ForgeError("Studio binds to loopback only; remote hosts are intentionally rejected")
        from .studio import serve
        serve(args.workspace, args.host, args.port)
        return {"stopped": True}
    raise ForgeError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        result = execute(args)
        if result is not None:
            print_json(result)
        return 0
    except ForgeError as error:
        print(f"Asset Forge error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Asset Forge interrupted", file=sys.stderr)
        return 130
