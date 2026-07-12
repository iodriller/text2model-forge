"""Immutable content-addressed artifact storage."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_bytes
from .schemas import ArtifactLineage, ArtifactRecord, AssetStage


class ArtifactConflictError(RuntimeError):
    pass


class ArtifactIntegrityError(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.blobs = self.root / "blobs"
        self.records = self.root / "records"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.records.mkdir(parents=True, exist_ok=True)

    def _record_path(self, artifact_id: str) -> Path:
        # Pydantic validates IDs before this is called by public write methods.
        return self.records / f"{artifact_id}.json"

    def _blob_path(self, digest: str) -> Path:
        return self.blobs / digest[:2] / digest

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def record_bytes(
        self,
        data: bytes,
        *,
        artifact_id: str,
        media_type: str,
        stage: AssetStage,
        lineage: ArtifactLineage,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        digest = sha256_bytes(data)
        if lineage.artifact_id != artifact_id:
            raise ArtifactIntegrityError("lineage artifact_id does not match the record artifact_id")
        if lineage.artifact_sha256 != digest:
            raise ArtifactIntegrityError("lineage SHA-256 does not match artifact bytes")
        if lineage.stage != stage.value:
            raise ArtifactIntegrityError("lineage stage does not match the artifact stage")

        blob_path = self._blob_path(digest)
        relative_blob = blob_path.relative_to(self.root).as_posix()
        record = ArtifactRecord(
            artifact_id=artifact_id,
            sha256=digest,
            size_bytes=len(data),
            media_type=media_type,
            stage=stage,
            blob_path=relative_blob,
            created_at=datetime.now(timezone.utc),
            lineage=lineage,
            metadata=metadata or {},
        )
        record_path = self._record_path(artifact_id)
        if record_path.exists():
            existing = self.load_record(artifact_id)
            if existing.sha256 == digest and existing.lineage == lineage:
                return existing
            raise ArtifactConflictError(f"artifact_id '{artifact_id}' is immutable and already exists")

        if not blob_path.exists():
            self._atomic_write(blob_path, data)
        self._atomic_write(record_path, record.model_dump_json(indent=2).encode("utf-8"))
        return record

    def record_file(
        self,
        source: str | Path,
        **kwargs: Any,
    ) -> ArtifactRecord:
        return self.record_bytes(Path(source).read_bytes(), **kwargs)

    def load_record(self, artifact_id: str) -> ArtifactRecord:
        path = self._record_path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"unknown artifact '{artifact_id}'")
        return ArtifactRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def read_bytes(self, artifact_id: str, *, verify: bool = True) -> bytes:
        record = self.load_record(artifact_id)
        path = (self.root / record.blob_path).resolve()
        if self.root not in path.parents:
            raise ArtifactIntegrityError("artifact blob path escapes the store")
        data = path.read_bytes()
        if verify and (len(data) != record.size_bytes or sha256_bytes(data) != record.sha256):
            raise ArtifactIntegrityError(f"artifact '{artifact_id}' failed integrity verification")
        return data

    def all_records(self) -> dict[str, ArtifactRecord]:
        result: dict[str, ArtifactRecord] = {}
        for path in sorted(self.records.glob("*.json")):
            record = ArtifactRecord.model_validate_json(path.read_text(encoding="utf-8"))
            result[record.artifact_id] = record
        return result

    def all_lineage(self) -> dict[str, ArtifactLineage]:
        return {artifact_id: record.lineage for artifact_id, record in self.all_records().items()}
