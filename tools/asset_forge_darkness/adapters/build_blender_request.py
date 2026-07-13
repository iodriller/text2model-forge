"""Build a strict ad hoc Blender analysis/repair request for a local mesh candidate."""
from __future__ import annotations

import argparse
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darkness.hashing import sha256_file
from darkness.schemas import ArtifactLineage, ArtifactRecord, AssetStage, ExternalWorkerRequest


MEDIA_TYPES = {
    ".blend": "application/x-blender",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".obj": "model/obj",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-id", default="blender.smoke.run.v1")
    parser.add_argument("--artifact-id", default="geometry.blender.input.v1")
    parser.add_argument("--operation-id", choices=("blender.analyze", "blender.repair", "blender.export"), default="blender.analyze")
    parser.add_argument("--component-policy", choices=("none", "keep_largest"), default="none")
    parser.add_argument("--weld-distance", type=float, default=0.0)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--maximum-material-change-fraction", type=float, default=0.02)
    args = parser.parse_args(argv)

    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"input does not exist: {source}")
    media_type = MEDIA_TYPES.get(source.suffix.lower()) or mimetypes.guess_type(source)[0]
    if media_type is None:
        raise SystemExit(f"unsupported input extension: {source.suffix}")
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
        media_type=media_type,
        stage=AssetStage.geometry,
        blob_path=f"smoke/{source.name}",
        created_at=datetime.now(timezone.utc),
        lineage=lineage,
        metadata={"source_path": str(source)},
    )
    request = ExternalWorkerRequest(
        job_id=args.job_id,
        run_id=args.run_id,
        operation_id=args.operation_id,
        stage=AssetStage.geometry,
        inputs=[record],
        input_paths={args.artifact_id: str(source)},
        parameters={
            "component_policy": args.component_policy,
            "weld_distance": args.weld_distance,
            "render_size": args.render_size,
            "maximum_material_change_fraction": args.maximum_material_change_fraction,
        },
        output_directory=str(args.output_directory.resolve()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
