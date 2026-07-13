from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from darkness.schemas import AssetStage, ExternalWorkerRequest


ROOT = Path(__file__).resolve().parents[1]


def test_instant_meshes_worker_source_compiles() -> None:
    source = (ROOT / "adapters" / "instant_meshes_retopology_worker.py").read_text(encoding="utf-8")
    compile(source, "instant_meshes_retopology_worker.py", "exec")
    assert '"--deterministic"' in source
    assert '"--threads", "1"' in source
    assert "automatic_promotion_gate_passed" in source


def test_instant_meshes_builder_emits_strict_d4_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    request_path = tmp_path / "request.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "adapters" / "build_instant_meshes_request.py"),
            "--input", str(source),
            "--output-directory", str(tmp_path / "output"),
            "--out", str(request_path),
            "--job-id", "instant.meshes.contract.v1",
            "--field-faces", "10000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    request = ExternalWorkerRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    assert request.stage is AssetStage.topology
    assert request.operation_id == "retopology.instant_meshes"
    assert request.parameters["field_faces"] == 10000
    assert request.parameters["intrinsic"] is False
