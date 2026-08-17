"""Windows-side launcher that runs a Linux worker script inside WSL2.

Bridges Text2Model's native-Windows SubprocessWorkerAdapter contract
(Windows --request/--response file paths) to a worker script that must run
inside WSL2 because its dependencies (compiled CUDA kernels such as
flash-attn or spatial-sparse-attention) have no Windows build. Reusable
across every WSL2-hosted worker; per-model behavior lives entirely in the
Linux-side script named by --wsl-script.

Path handling: workspace_root (and therefore every job's request/response
file and output_directory) is always a native Windows drive path that WSL2's
default DrvFS mount also exposes at /mnt/<drive>/... . This launcher
translates --request/--response and the output_directory/input_paths fields
embedded in the request JSON to that form before invoking the Linux script,
then translates any /mnt/<drive>/... paths found in outputs[].path of the
resulting response.json back to native Windows form, so the rest of Text2Model
never has to know a worker ran across the WSL2 boundary.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

_WSL_MOUNT_RE = re.compile(r"^/mnt/([a-z])/(.*)$")


class WslPathError(ValueError):
    pass


def windows_path_to_wsl(path: str) -> str:
    """Convert an absolute Windows drive path to its default WSL2 DrvFS path.

    Only the `<drive>:\\...` / `<drive>:/...` form is supported; Text2Model
    never produces UNC or relative paths here, so anything else is rejected
    rather than guessed.
    """
    resolved = str(Path(path).resolve())
    if len(resolved) < 3 or resolved[1] != ":":
        raise WslPathError(f"expected an absolute drive path, got: {path!r}")
    drive = resolved[0].lower()
    rest = resolved[2:].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"


def wsl_path_to_windows(path: str) -> str:
    """Convert a /mnt/<drive>/... DrvFS path back to a native Windows path.

    Paths that are not in /mnt/<drive>/ form are returned unchanged, since a
    worker may legitimately report a Linux-only path (e.g. a log inside its
    own venv) that Text2Model is not meant to dereference.
    """
    match = _WSL_MOUNT_RE.match(path)
    if not match:
        return path
    drive, rest = match.groups()
    return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"


def _translate_request(original: dict) -> dict:
    translated = dict(original)
    if "output_directory" in translated:
        translated["output_directory"] = windows_path_to_wsl(translated["output_directory"])
    if "input_paths" in translated:
        translated["input_paths"] = {
            key: windows_path_to_wsl(value) for key, value in translated["input_paths"].items()
        }
    return translated


def _translate_response_paths_in_place(response_path: Path) -> None:
    if not response_path.exists():
        return
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    for output in payload.get("outputs", []):
        if "path" in output:
            output["path"] = wsl_path_to_windows(output["path"])
    response_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--distro", required=True, help="WSL distro name, e.g. Ubuntu-24.04")
    parser.add_argument("--wsl-python", required=True, help="absolute Linux-side interpreter path")
    parser.add_argument("--wsl-script", required=True, help="absolute Linux-side worker script path")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable forwarded into the WSL process (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    original_request = json.loads(args.request.read_text(encoding="utf-8-sig"))
    translated_request = _translate_request(original_request)
    translated_request_path = args.request.with_suffix(".wsl.json")
    translated_request_path.write_text(
        json.dumps(translated_request, indent=2, sort_keys=True), encoding="utf-8"
    )

    wsl_request = windows_path_to_wsl(str(translated_request_path))
    wsl_response = windows_path_to_wsl(str(args.response))

    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item!r}")
    command = [
        "wsl.exe",
        "-d",
        args.distro,
        "--exec",
        "env",
        *args.env,
        args.wsl_python,
        args.wsl_script,
        "--request",
        wsl_request,
        "--response",
        wsl_response,
    ]
    # --exec bypasses the distro's login shell entirely, so argv is passed
    # through literally with no extra quoting/interpretation on either side.
    # If this process is killed (Windows TerminateProcess), the wsl.exe child
    # and the Linux-side process it spawned are not guaranteed to be reaped;
    # cross-boundary cancellation stays a known gap until the GPU scheduler
    # owns WSL process trees directly (tracked in resources/workers/trellis2.json).
    completed = subprocess.run(command)
    _translate_response_paths_in_place(args.response)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
