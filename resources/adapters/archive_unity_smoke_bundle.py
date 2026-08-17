"""Create a deterministic transfer ZIP for a verified Text2Model Unity smoke bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive(bundle: Path, output: Path) -> dict[str, object]:
    bundle = bundle.resolve()
    output = output.resolve()
    manifest_path = bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_kind") != "text2model_standalone_unity_smoke":
        raise ValueError("not a Text2Model standalone Unity smoke bundle")
    members = [manifest_path]
    for entry in manifest.get("files", []):
        relative = PurePosixPath(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe bundle path: {relative}")
        path = bundle.joinpath(*relative.parts)
        if not path.is_file() or _sha256(path).lower() != str(entry.get("sha256", "")).lower():
            raise ValueError(f"bundle file changed before archive: {relative}")
        members.append(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for path in sorted(members, key=lambda item: item.relative_to(bundle).as_posix()):
            relative = path.relative_to(bundle).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            target.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(output)
    record = {
        "schema_version": 1,
        "archive": output.name,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": _sha256(output),
        "bundle_manifest_sha256": _sha256(manifest_path),
        "files": len(members),
        "human_approval_required": True,
        "human_approved": False,
    }
    record_path = output.with_suffix(output.suffix + ".json")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    args = _arguments()
    record = archive(args.bundle, args.output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
