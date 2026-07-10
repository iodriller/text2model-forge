from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .comfy import ComfyClient, build_sdxl_workflow, generate
from .blender_worker import audit_master
from .core import (
    APPROVAL_STAGES,
    ASSET_KINDS,
    ForgeError,
    approve_artifact,
    asset_root,
    build_prompt_pack,
    create_asset,
    init_workspace,
    load_project,
    package_root,
    project_status,
    register_model,
    write_json,
)
from .compliance import build_compliance_report
from .mesh import run_triposr
from .exporter import export_asset
from .qa import validate_sheets


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

    qa = sub.add_parser("qa-sheets", help="run commercial sheet QA and create a review contact sheet")
    qa.add_argument("--config", type=Path, required=True)
    qa.add_argument("--repo-root", type=Path, default=Path.cwd())
    qa.add_argument("--report", type=Path, required=True)
    qa.add_argument("--contact-sheet", type=Path)

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
    if args.command == "qa-sheets":
        report = validate_sheets(args.config, args.repo_root.resolve(), args.report, args.contact_sheet)
        if not report["passed"]:
            raise ForgeError("Sheet QA failed; inspect " + str(args.report.resolve()))
        return report
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
