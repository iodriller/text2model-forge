"""Record a hash-bound human approval for an external Text2Model evidence artifact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from text2model_forge.config import load_local_config  # noqa: E402
from text2model_forge.hashing import sha256_file  # noqa: E402
from text2model_forge.schemas import ApprovalRecord, AssetStage  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--stage", choices=[item.value for item in AssetStage], required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--locked-feature", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_local_config()
    if config is None:
        raise SystemExit("Text2Model config.local.toml is required")
    workspace = Path(config.workspace_root).resolve()
    artifact = args.artifact.resolve()
    output = args.out.resolve()
    if not artifact.is_file():
        raise SystemExit(f"approval artifact does not exist: {artifact}")
    if artifact != workspace and workspace not in artifact.parents:
        raise SystemExit("approval artifact must be inside the Text2Model workspace")
    if output != workspace and workspace not in output.parents:
        raise SystemExit("approval record must be inside the Text2Model workspace")
    record = ApprovalRecord(
        approval_id=args.approval_id,
        artifact_id=args.artifact_id,
        stage=AssetStage(args.stage),
        artifact_sha256=sha256_file(artifact),
        approved_by=args.approved_by,
        approved_at=datetime.now(timezone.utc),
        locked_features=args.locked_feature,
        notes=args.notes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
