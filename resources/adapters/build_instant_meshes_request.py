"""Build a strict deterministic Instant Meshes D2b request."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from text2model_forge.hashing import sha256_file
from text2model_forge.schemas import ArtifactLineage, ArtifactRecord, AssetStage, ExternalWorkerRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-id", default="instant.meshes.smoke.run.v1")
    parser.add_argument("--artifact-id", default="geometry.instant.meshes.input.v1")
    parser.add_argument("--field-faces", type=int, default=12500)
    parser.add_argument("--maximum-output-faces", type=int, default=75000)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--crease-degrees", type=float)
    parser.add_argument("--intrinsic", action="store_true")
    parser.add_argument("--align-boundaries", action="store_true")
    args = parser.parse_args(argv)

    source = args.input.resolve()
    if source.suffix.lower() != ".obj" or not source.is_file():
        raise SystemExit(f"input must be an existing OBJ file: {source}")
    digest = sha256_file(source)
    lineage = ArtifactLineage(
        artifact_id=args.artifact_id,
        artifact_sha256=digest,
        stage=AssetStage.geometry.value,
        source_license_ids=["Project-Owned"],
    )
    record = ArtifactRecord(
        artifact_id=args.artifact_id,
        sha256=digest,
        size_bytes=source.stat().st_size,
        media_type="model/obj",
        stage=AssetStage.geometry,
        blob_path=f"instant-meshes-input/{source.name}",
        created_at=datetime.now(timezone.utc),
        lineage=lineage,
        metadata={"source_path": str(source)},
    )
    request = ExternalWorkerRequest(
        job_id=args.job_id,
        run_id=args.run_id,
        operation_id="retopology.instant_meshes",
        stage=AssetStage.topology,
        inputs=[record],
        input_paths={args.artifact_id: str(source)},
        parameters={
            "field_faces": args.field_faces,
            "maximum_output_faces": args.maximum_output_faces,
            "smooth_iterations": args.smooth_iterations,
            "crease_degrees": args.crease_degrees,
            "intrinsic": args.intrinsic,
            "align_boundaries": args.align_boundaries,
        },
        output_directory=str(args.output_directory.resolve()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
