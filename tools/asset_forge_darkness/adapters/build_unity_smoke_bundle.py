"""Build a portable, non-promoting Unity smoke project for a Darkness sprite candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


UNITY_VERSION = "6000.5.1f1"
UNITY_REVISION = "0d9463e84828"
EXPECTED_ACTIONS = {f"{clip}/{direction}" for clip in ("idle", "walk", "attack", "death") for direction in ("north", "south", "east", "west")}


def _arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=root / "unity_smoke_template")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_package_file(package: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError(f"candidate file must be a package-local filename: {name!r}")
    path = package / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_candidate(package: Path) -> tuple[dict[str, object], Path]:
    manifest_path = package / "candidate_unit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "human_review_candidate"
        or manifest.get("automatic_gate_passed") is not True
        or manifest.get("human_approval_required") is not True
        or manifest.get("human_approved") is not False
        or not isinstance(manifest.get("asset_id"), str)
    ):
        raise ValueError("candidate manifest is not an unapproved, automatic-gate-passing review candidate")
    actions = manifest.get("actions")
    if not isinstance(actions, list) or len(actions) != 16:
        raise ValueError("candidate must contain exactly sixteen directional actions")
    observed = {f"{action.get('name')}/{action.get('direction')}" for action in actions}
    if observed != EXPECTED_ACTIONS:
        raise ValueError("candidate action/direction set is incomplete or duplicated")
    for action in actions:
        sheet = _safe_package_file(package, str(action.get("sheet", "")))
        if _sha256(sheet).lower() != str(action.get("sha256", "")).lower():
            raise ValueError(f"candidate sheet hash mismatch: {sheet.name}")
    master = Path(str(manifest.get("source_master", ""))).resolve()
    if not master.is_file() or _sha256(master).lower() != str(manifest.get("source_master_sha256", "")).lower():
        raise ValueError("candidate source-master hash mismatch")
    return manifest, master


def _runner_text() -> str:
    return r'''param(
    [string]$Unity
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Version = "6000.5.1f1"
if (-not $Unity) {
    $Candidates = @(
        "C:\Program Files\Unity\Hub\Editor\$Version\Editor\Unity.exe",
        "C:\UnityLocal\$Version\Editor\Unity.exe"
    )
    $Unity = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Unity -or -not (Test-Path -LiteralPath $Unity)) {
    throw "Unity $Version was not found. Pass -Unity <path-to-Unity.exe>."
}

$Project = Join-Path $Root "UnitySmokeProject"
$Package = Join-Path $Root "candidate"
$Result = Join-Path $Root "result"
New-Item -ItemType Directory -Path $Result -Force | Out-Null
Remove-Item -LiteralPath (Join-Path $Result "unity_candidate_validation.json") -Force -ErrorAction SilentlyContinue
$env:DARKNESS_CANDIDATE_PACKAGE = $Package
$env:DARKNESS_CANDIDATE_OUTPUT = $Result
$env:DARKNESS_BUNDLE_MANIFEST = Join-Path $Root "bundle_manifest.json"

$Arguments = @(
    "-quit",
    "-batchmode",
    "-projectPath", ('"' + $Project + '"'),
    "-executeMethod", "Darkness.EditorTools.DarknessCandidateValidator.ValidateFromBatch",
    "-logFile", ('"' + (Join-Path $Result "unity.log") + '"')
)
$Process = Start-Process -FilePath $Unity -ArgumentList $Arguments -PassThru -Wait -WindowStyle Hidden
if ($Process.ExitCode -ne 0) {
    throw "Unity smoke validation failed with exit code $($Process.ExitCode). See result\unity.log."
}
$Report = Join-Path $Result "unity_candidate_validation.json"
if (-not (Test-Path -LiteralPath $Report)) {
    throw "Unity exited without producing result\unity_candidate_validation.json."
}
Write-Output "Darkness Unity smoke validation passed. Return the result folder to the Darkness pipeline."
'''


def _readme_text() -> str:
    return f"""# Darkness standalone Unity smoke check

This bundle checks the candidate in an isolated Unity project. It does not import anything into EmberDefense or modify
another game project.

1. Copy this complete folder to the Unity computer.
2. Ensure Unity `{UNITY_VERSION}` is licensed on that computer.
3. Run `powershell -ExecutionPolicy Bypass -File .\\run_unity_smoke.ps1`.
4. Copy the generated `result` folder back with this bundle.
5. Resume the Darkness pipeline; it will verify every returned hash before accepting the Unity proof.

The candidate remains non-promoting and requires human approval even after the smoke check passes.
"""


def build(package: Path, output: Path, template: Path) -> dict[str, object]:
    package = package.resolve()
    output = output.resolve()
    template = template.resolve()
    if not package.is_dir() or not template.is_dir():
        raise FileNotFoundError(package if not package.is_dir() else template)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest, master = _validate_candidate(package)

    candidate = output / "candidate"
    project = output / "UnitySmokeProject"
    candidate.mkdir()
    shutil.copytree(template, project)
    for source in package.iterdir():
        if source.is_file() and source.name != "candidate_unit_manifest.json":
            shutil.copy2(source, candidate / source.name)
    source_dir = candidate / "source"
    source_dir.mkdir()
    portable_master = source_dir / master.name
    shutil.copy2(master, portable_master)
    portable_manifest = dict(manifest)
    portable_manifest["source_master"] = f"source/{master.name}"
    portable_manifest_path = candidate / "candidate_unit_manifest.json"
    _write_json(portable_manifest_path, portable_manifest)

    (output / "run_unity_smoke.ps1").write_text(_runner_text(), encoding="utf-8")
    (output / "README.md").write_text(_readme_text(), encoding="utf-8")
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    bundle = {
        "schema_version": 1,
        "bundle_kind": "darkness_standalone_unity_smoke",
        "asset_id": manifest["asset_id"],
        "unity_version": UNITY_VERSION,
        "unity_revision": UNITY_REVISION,
        "entrypoint": "run_unity_smoke.ps1",
        "project": "UnitySmokeProject",
        "candidate_manifest": "candidate/candidate_unit_manifest.json",
        "candidate_manifest_sha256": _sha256(portable_manifest_path),
        "source_master_sha256": manifest["source_master_sha256"],
        "directional_actions": 16,
        "expected_decoded_sprites": sum(int(action["frames"]) for action in manifest["actions"]),
        "human_approval_required": True,
        "human_approved": False,
        "files": files,
    }
    _write_json(output / "bundle_manifest.json", bundle)
    return bundle


def main() -> int:
    args = _arguments()
    bundle = build(args.package, args.output_directory, args.template)
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
