"""Fail-closed delivery packaging for validated Text2Model artifacts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .hashing import sha256_file
from .schemas import (
    CandidateRegistry,
    DeliveryPackageManifest,
    PackageFile,
    WorkerRuntimeQualification,
)


CANONICAL_REQUIRED_FILES = {
    "canonical_short_biped_v1.glb",
    "canonical_short_biped_v1_lod1.glb",
    "canonical_short_biped_v1_lod2.glb",
    "rig_contract.json",
    "physics_contract.json",
    "fit_contract.json",
    "skinning_lod0.json",
    "skinning_lod1.json",
    "skinning_lod2.json",
    "sprites/sprite_report.json",
}


class PackagePolicyError(RuntimeError):
    pass


def build_delivery_package(
    *,
    package_id: str,
    candidate_id: str,
    source_root: str | Path,
    output_root: str | Path,
    registry_path: str | Path,
    qualification_path: str | Path,
    mode: str = "research",
) -> DeliveryPackageManifest:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output package trees must be independent")
    registry = CandidateRegistry.model_validate_json(Path(registry_path).read_text(encoding="utf-8-sig"))
    candidates = {item.candidate_id: item for item in registry.candidates}
    if candidate_id not in candidates:
        raise PackagePolicyError(f"candidate is absent from registry: {candidate_id}")
    candidate = candidates[candidate_id]
    qualification = WorkerRuntimeQualification.model_validate_json(
        Path(qualification_path).read_text(encoding="utf-8-sig")
    )
    if qualification.candidate_id != candidate_id:
        raise PackagePolicyError("qualification candidate_id mismatch")
    missing = sorted(item for item in CANONICAL_REQUIRED_FILES if not (source / item).is_file())
    if missing:
        raise PackagePolicyError(f"required package files are missing: {', '.join(missing)}")
    blockers: list[str] = []
    if candidate.release_eligibility.value != "permitted":
        blockers.append(f"registry release eligibility is {candidate.release_eligibility.value}")
    if qualification.status != "passed":
        blockers.append(f"runtime qualification is {qualification.status}")
    for report_name in ("skinning_lod0.json", "skinning_lod1.json", "skinning_lod2.json"):
        report = json.loads((source / report_name).read_text(encoding="utf-8-sig"))
        if report.get("hard_failures"):
            blockers.append(f"{report_name} has hard skinning failures")
    if mode not in {"research", "release"}:
        raise ValueError("mode must be research or release")
    if mode == "release" and blockers:
        raise PackagePolicyError("release package blocked: " + "; ".join(blockers))
    if output.exists():
        raise FileExistsError(f"package output already exists: {output}")
    output.mkdir(parents=True)
    files: list[PackageFile] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        files.append(
            PackageFile(
                relative_path=relative.as_posix(),
                sha256=sha256_file(destination),
                size_bytes=destination.stat().st_size,
            )
        )
    manifest = DeliveryPackageManifest(
        package_id=package_id,
        candidate_id=candidate_id,
        mode=mode,
        eligibility="release_ready" if mode == "release" else "research_only",
        code_license=qualification.code_license,
        weights_license=qualification.weights_license,
        source_revision=qualification.source_revision,
        weights_revision=qualification.weights_revision,
        qualification_status=qualification.status,
        files=files,
        blockers=blockers,
    )
    (output / "text2model-package.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
