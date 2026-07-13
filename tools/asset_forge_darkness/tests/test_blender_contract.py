from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from darkness.schemas import AssetStage, ExternalWorkerRequest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "mesh"


def test_blender_worker_script_compiles_without_importing_bpy() -> None:
    source = (ROOT / "adapters" / "blender_worker.py").read_text(encoding="utf-8")
    compile(source, "blender_worker.py", "exec")
    assert '"blender.repair"' in source
    assert '"blender.repair_retopology"' in source
    assert '"blender.propose_short_biped_rig"' in source
    assert '"blender.author_short_biped_motion"' in source
    assert "human_approval_required" in source
    assert "source_overwritten" in source
    assert "vertex_coordinates_unchanged" in source
    assert "automatic_rig_probe_gate_passed" in source
    assert "automatic_motion_gate_passed" in source
    assert "MOTION_CRITICAL_JOINTS" in source
    assert "walk_swing" in source
    assert "attack_function" in source
    assert "hit_function" in source
    assert "death_function" in source
    assert "_apply_weight_redistributions" in source
    assert 'parameters.get("weight_adjustments", [])' in source


def test_rig_optimizer_script_compiles_without_running_blender_or_qwen() -> None:
    source = (ROOT / "adapters" / "optimize_short_biped_rig.py").read_text(encoding="utf-8")
    compile(source, "optimize_short_biped_rig.py", "exec")
    assert "LocalDeployOptimizer" in source
    assert "apply_landmark_pair_adjustment" in source
    assert "numeric_regression" in source
    assert "default=6" in source
    assert "PREVIOUS ACCEPTED STRESS" in source
    assert "human_review.md" in source


def test_external_human_approval_script_compiles_and_binds_hash() -> None:
    source = (ROOT / "adapters" / "record_human_approval.py").read_text(encoding="utf-8")
    compile(source, "record_human_approval.py", "exec")
    assert "ApprovalRecord" in source
    assert "sha256_file" in source
    assert "Darkness workspace" in source


def test_blender_request_builder_emits_strict_geometry_contract(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "adapters" / "build_blender_request.py"),
            "--input",
            str(FIXTURES / "defective_tetrahedron.obj"),
            "--output-directory",
            str(output),
            "--out",
            str(request_path),
            "--job-id",
            "blender.contract.v1",
            "--operation-id",
            "blender.repair",
            "--component-policy",
            "keep_largest",
            "--render-size",
            "256",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    request = ExternalWorkerRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    assert request.stage is AssetStage.geometry
    assert request.operation_id == "blender.repair"
    assert request.parameters["component_policy"] == "keep_largest"
    assert request.parameters["render_size"] == 256


def test_blender_request_builder_emits_bounded_retopology_repair(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "adapters" / "build_blender_request.py"),
            "--input", str(FIXTURES / "defective_tetrahedron.obj"),
            "--output-directory", str(tmp_path / "output"),
            "--out", str(request_path),
            "--job-id", "blender.retopology.contract.v1",
            "--operation-id", "blender.repair_retopology",
            "--minimum-quad-fraction", "0.99",
            "--maximum-removed-faces", "16",
            "--maximum-created-faces", "16",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    request = ExternalWorkerRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    assert request.operation_id == "blender.repair_retopology"
    assert request.parameters["minimum_quad_fraction"] == 0.99
    assert request.parameters["maximum_removed_faces"] == 16
    assert request.parameters["maximum_created_faces"] == 16


def test_blender_request_builder_emits_short_biped_rig_probe(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    adjustments_path = tmp_path / "adjustments.json"
    adjustments_path.write_text(
        '{"shoulder_l":[-0.02,0.0,0.0],"shoulder_r":[0.02,0.0,0.0]}',
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        '[{"joint_pair":"shoulders","direction":"parent_to_child",'
        '"transfer_fraction":0.05,"radius_fraction":0.08}]',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "adapters" / "build_blender_request.py"),
            "--input", str(FIXTURES / "defective_tetrahedron.obj"),
            "--output-directory", str(tmp_path / "output"),
            "--out", str(request_path),
            "--job-id", "blender.rig.contract.v1",
            "--operation-id", "blender.propose_short_biped_rig",
            "--landmark-adjustments-file", str(adjustments_path),
            "--weight-adjustments-file", str(weights_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    request = ExternalWorkerRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    assert request.operation_id == "blender.propose_short_biped_rig"
    assert request.parameters["maximum_material_change_fraction"] == 0.02
    assert request.parameters["landmark_adjustments"]["shoulder_l"] == [-0.02, 0.0, 0.0]
    assert request.parameters["weight_adjustments"][0]["joint_pair"] == "shoulders"
    assert request.parameters["weight_adjustments"][0]["transfer_fraction"] == 0.05


def test_blender_request_builder_emits_short_biped_motion_authoring(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "adapters" / "build_blender_request.py"),
            "--input", str(FIXTURES / "defective_tetrahedron.obj"),
            "--output-directory", str(tmp_path / "output"),
            "--out", str(request_path),
            "--job-id", "blender.motion.contract.v1",
            "--operation-id", "blender.author_short_biped_motion",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    request = ExternalWorkerRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    assert request.operation_id == "blender.author_short_biped_motion"
