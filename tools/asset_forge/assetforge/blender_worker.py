from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .core import ForgeError, package_root, read_json


def find_blender(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        shutil.which("blender"),
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    ]
    for value in candidates:
        if value and Path(value).is_file():
            return Path(value).resolve()
    raise ForgeError("Blender was not found; install Blender 4.5 LTS or newer")


def audit_master(config_path: Path, report_path: Path, blender: str | None = None, timeout_seconds: float = 300) -> dict:
    config_path = config_path.resolve()
    config = read_json(config_path)
    source_value = Path(config["source"])
    repo_root = package_root().parents[1]
    source = source_value if source_value.is_absolute() else repo_root / source_value
    source = source.resolve()
    if source.suffix.lower() != ".blend" or not source.is_file():
        raise ForgeError(f"Master audit currently requires an existing .blend source: {source}")
    executable = find_blender(blender)
    script = package_root() / "blender" / "audit_master.py"
    command = [
        str(executable),
        "--background",
        str(source),
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        "--config",
        str(config_path),
        "--report",
        str(report_path.resolve()),
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ForgeError(f"Blender master audit failed to launch: {error}") from error
    if process.returncode != 0:
        details = process.stderr.strip() or process.stdout.strip()
        raise ForgeError(f"Blender master audit failed with exit code {process.returncode}: {details[-2000:]}")
    return read_json(report_path.resolve())
