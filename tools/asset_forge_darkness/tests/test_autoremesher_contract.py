from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from darkness.polygon_obj import PolygonObj
from darkness.schemas import AssetStage, ExternalWorkerRequest


ROOT = Path(__file__).resolve().parents[1]


def test_polygon_obj_reports_closed_all_quad_cube() -> None:
    mesh = PolygonObj.parse(
        """
        v -1 -1 -1
        v 1 -1 -1
        v 1 1 -1
        v -1 1 -1
        v -1 -1 1
        v 1 -1 1
        v 1 1 1
        v -1 1 1
        f 1 4 3 2
        f 5 6 7 8
        f 1 2 6 5
        f 2 3 7 6
        f 3 4 8 7
        f 4 1 5 8
        """
    )
    report = mesh.analyze()
    assert report["quads"] == 6
    assert report["non_quad_faces"] == 0
    assert report["connected_components"] == 1
    assert report["boundary_edges"] == 0
    assert report["non_manifold_edges"] == 0


def test_polygon_obj_reports_open_non_quad_components() -> None:
    mesh = PolygonObj.parse(
        """
        v 0 0 0
        v 1 0 0
        v 0 1 0
        v 10 0 0
        v 11 0 0
        v 10 1 0
        f 1 2 3
        f 4 5 6
        """
    )
    report = mesh.analyze()
    assert report["triangles"] == 2
    assert report["non_quad_faces"] == 2
    assert report["connected_components"] == 2
    assert report["boundary_edges"] == 6


def test_retopology_request_builder_emits_strict_d4_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    request_path = tmp_path / "request.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "adapters" / "build_retopology_request.py"),
            "--input", str(source),
            "--output-directory", str(tmp_path / "output"),
            "--out", str(request_path),
            "--job-id", "autoremesher.contract.v1",
            "--target-quads", "12000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    request = ExternalWorkerRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    assert request.stage is AssetStage.topology
    assert request.operation_id == "retopology.autoremesher"
    assert request.parameters["target_quads"] == 12000


def test_autoremesher_worker_source_compiles() -> None:
    source = (ROOT / "adapters" / "autoremesher_worker.py").read_text(encoding="utf-8")
    compile(source, "autoremesher_worker.py", "exec")
    assert "source_overwritten" in source
    assert "automatic_promotion_gate_passed" in source
