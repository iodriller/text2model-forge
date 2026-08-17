"""Verify and ingest a standalone Text2Model Unity smoke result without trusting the other computer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ALLOWED_UNITY_RUNTIME_MUTABLE_FILES = {
    "UnitySmokeProject/ProjectSettings/ProjectVersion.txt",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def ingest(bundle_root: Path, result_root: Path, output: Path) -> dict[str, object]:
    bundle_root = bundle_root.resolve()
    result_root = result_root.resolve()
    output = output.resolve()
    bundle_path = bundle_root / "bundle_manifest.json"
    bundle = _load(bundle_path)
    if (
        bundle.get("schema_version") != 1
        or bundle.get("bundle_kind") != "text2model_standalone_unity_smoke"
        or bundle.get("human_approval_required") is not True
        or bundle.get("human_approved") is not False
    ):
        raise ValueError("invalid or promoting Unity smoke bundle")
    mutable_files = {str(path) for path in bundle.get("unity_runtime_mutable_files", [])}
    unsupported_mutable_files = mutable_files - ALLOWED_UNITY_RUNTIME_MUTABLE_FILES
    if unsupported_mutable_files:
        raise ValueError(f"unsupported Unity runtime-mutable bundle files: {sorted(unsupported_mutable_files)}")
    for entry in bundle.get("files", []):
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe bundle path: {relative}")
        path = bundle_root / relative
        if not path.is_file():
            raise ValueError(f"Unity smoke bundle file changed: {relative.as_posix()}")
        if relative.as_posix() in mutable_files:
            continue
        if _sha256(path).lower() != str(entry.get("sha256", "")).lower():
            raise ValueError(f"Unity smoke bundle file changed: {relative.as_posix()}")

    report_path = result_root / "unity_candidate_validation.json"
    capture_path = result_root / "unity_candidate_capture.png"
    report = _load(report_path)
    required = {
        "passed": True,
        "asset_id": bundle.get("asset_id"),
        "project_kind": "text2model_standalone_unity_smoke",
        "candidate_manifest_sha256": bundle.get("candidate_manifest_sha256"),
        "directional_actions": 16,
        "decoded_sprites": bundle.get("expected_decoded_sprites"),
        "animation_clips": 16,
        "source_master_hash_verified": True,
        "live_game_assets_modified": False,
        "human_approval_required": True,
        "human_approved": False,
    }
    mismatches = {key: {"expected": expected, "actual": report.get(key)} for key, expected in required.items() if report.get(key) != expected}
    expected_unity_family = str(
        bundle.get("unity_version_family")
        or ".".join(str(bundle.get("unity_version", "")).split(".")[:2])
    )
    actual_unity_version = str(report.get("unity_version", ""))
    if not expected_unity_family or not actual_unity_version.startswith(expected_unity_family + "."):
        mismatches["unity_version_family"] = {
            "expected": expected_unity_family,
            "actual": actual_unity_version,
        }
    if mismatches:
        raise ValueError(f"Unity smoke report contract mismatch: {mismatches}")
    if report.get("bundle_manifest_sha256") != _sha256(bundle_path):
        raise ValueError("Unity smoke report was not produced from this bundle manifest")
    if not capture_path.is_file() or report.get("capture_sha256") != _sha256(capture_path):
        raise ValueError("Unity smoke capture is missing or its hash does not match the report")

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, output / report_path.name)
    shutil.copy2(capture_path, output / capture_path.name)
    log = result_root / "unity.log"
    if log.is_file():
        shutil.copy2(log, output / log.name)
    validation = {
        "schema_version": 1,
        "passed": True,
        "asset_id": bundle["asset_id"],
        "bundle_manifest_sha256": _sha256(bundle_path),
        "unity_report_sha256": _sha256(output / report_path.name),
        "unity_capture_sha256": _sha256(output / capture_path.name),
        "unity_version": report["unity_version"],
        "live_game_assets_modified": False,
        "human_approval_required": True,
        "human_approved": False,
    }
    (output / "unity_ingest_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def main() -> int:
    args = _arguments()
    bundle = args.bundle.resolve()
    result = args.result.resolve() if args.result is not None else bundle / "result"
    validation = ingest(bundle, result, args.output_directory)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
