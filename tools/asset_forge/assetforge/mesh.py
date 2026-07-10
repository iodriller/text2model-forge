from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .core import ForgeError, approval_status, asset_root, package_root, read_json, record_job, sha256_file, utc_now, write_json


def require_approved_concept(workspace: Path, asset_id: str, input_image: Path) -> None:
    status = approval_status(workspace, asset_id)["concept"]
    if not status["approved"]:
        raise ForgeError("Mesh seeding requires a hash-valid concept approval")
    approval = read_json(Path(status["approval"]))
    if Path(approval["artifact"]).resolve() != input_image.resolve():
        raise ForgeError("The mesh-seed input is not the artifact covered by the current concept approval")


def run_triposr(
    workspace: Path,
    asset_id: str,
    input_image: Path,
    runtime_root: Path | None = None,
    texture_resolution: int = 1024,
    timeout_seconds: float = 1800,
    allow_unapproved: bool = False,
) -> list[Path]:
    input_image = input_image.resolve()
    if not input_image.is_file():
        raise ForgeError(f"Mesh-seed image was not found: {input_image}")
    if not allow_unapproved:
        require_approved_concept(workspace, asset_id, input_image)
    if texture_resolution not in {512, 1024, 2048}:
        raise ForgeError("TripoSR texture resolution must be 512, 1024, or 2048")

    runtime = (runtime_root or (package_root() / "runtime")).resolve()
    tripo = runtime / "triposr"
    python = tripo / ".venv" / "Scripts" / "python.exe"
    runner = tripo / "run.py"
    if not python.is_file() or not runner.is_file():
        raise ForgeError("TripoSR runtime is missing; run setup_local_ai.ps1 install-triposr")

    destination = asset_root(workspace, asset_id) / "master" / "mesh-seed" / input_image.stem
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["HF_HOME"] = str(runtime / "triposr-models" / "huggingface")
    environment["TORCH_HOME"] = str(runtime / "triposr-models" / "torch")
    command = [
        str(python),
        str(runner),
        str(input_image),
        "--output-dir",
        str(destination),
        "--bake-texture",
        "--texture-resolution",
        str(texture_resolution),
    ]
    try:
        process = subprocess.run(
            command,
            cwd=tripo,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ForgeError(f"TripoSR failed to launch: {error}") from error
    if process.returncode != 0:
        log = destination / "triposr-error.log"
        log.write_text((process.stdout or "") + "\n" + (process.stderr or ""), encoding="utf-8")
        raise ForgeError(f"TripoSR failed with exit code {process.returncode}; see {log}")

    outputs = sorted(path for path in destination.rglob("*") if path.suffix.lower() in {".obj", ".glb", ".gltf", ".png"})
    meshes = [path for path in outputs if path.suffix.lower() in {".obj", ".glb", ".gltf"}]
    if not meshes:
        raise ForgeError(f"TripoSR completed without a mesh under {destination}")
    weights = sorted((runtime / "triposr-models").rglob("model.ckpt"))
    provenance = {
        "schema_version": 1,
        "engine": "TripoSR",
        "source_url": "https://github.com/VAST-AI-Research/TripoSR",
        "license": "MIT",
        "asset_id": asset_id,
        "created_utc": utc_now(),
        "input": {"path": str(input_image), "sha256": sha256_file(input_image)},
        "texture_resolution": texture_resolution,
        "model_weights": [{"path": str(path), "sha256": sha256_file(path)} for path in weights],
        "outputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in outputs],
        "note": "Mesh seed only; retopology, material cleanup, rigging, and master approval are required.",
    }
    record_job(workspace, asset_id, provenance)
    write_json(destination / "provenance.json", provenance)
    return meshes
