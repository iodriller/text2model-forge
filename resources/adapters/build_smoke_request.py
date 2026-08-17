"""Build a schema-valid ExternalWorkerRequest for an ad hoc worker smoke test.

Used to drive `text2model_forge run-worker` against a single input image without
running the full compiler/run machinery. Not part of any run's lineage; the
resulting request/response belong to the local smoke-test workspace only.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from text2model_forge.hashing import sha256_file
from text2model_forge.schemas import (
    ArtifactLineage,
    ArtifactRecord,
    AssetStage,
    ExternalWorkerRequest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-id", default="smoke.run.v1")
    parser.add_argument("--operation-id", default="geometry.generate_from_rgba")
    parser.add_argument("--artifact-id", default="concept.smoke.v1")
    parser.add_argument("--out", type=Path, required=True, help="where to write request.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise SystemExit(f"input image does not exist: {image_path}")
    digest = sha256_file(image_path)

    lineage = ArtifactLineage(
        artifact_id=args.artifact_id,
        artifact_sha256=digest,
        stage=AssetStage.concept.value,
        producer_candidate_ids=[],
        parent_artifact_ids=[],
        source_license_ids=["Project-Owned"],
        source_license_status="cleared",
    )
    record = ArtifactRecord(
        artifact_id=args.artifact_id,
        sha256=digest,
        size_bytes=image_path.stat().st_size,
        media_type="image/png",
        stage=AssetStage.concept,
        blob_path=f"smoke/{image_path.name}",
        created_at=datetime.now(timezone.utc),
        lineage=lineage,
        metadata={"source_path": str(image_path)},
    )
    request = ExternalWorkerRequest(
        job_id=args.job_id,
        run_id=args.run_id,
        operation_id=args.operation_id,
        stage=AssetStage.geometry,
        inputs=[record],
        input_paths={args.artifact_id: str(image_path)},
        parameters={"seed": args.seed},
        output_directory=str(args.output_directory.resolve()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
