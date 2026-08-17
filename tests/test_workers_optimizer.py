from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import pytest
from pydantic import BaseModel, ConfigDict, Field

from text2model_forge.gpu import GpuLease, GpuLeaseBusy
from text2model_forge.localdeploy import LocalDeployStructuredClient
from text2model_forge.optimizer import LocalDeployOptimizer
from text2model_forge.schemas import (
    AssetStage,
    EvidenceBundle,
    EvidenceItem,
    OperationDefinition,
    WorkerJob,
)
from text2model_forge.workers import WorkerManager


def test_gpu_lease_is_exclusive_and_releasable(tmp_path) -> None:
    first = GpuLease(tmp_path, run_id="run.one", worker_id="worker.one")
    second = GpuLease(tmp_path, run_id="run.two", worker_id="worker.two")
    first.acquire()
    with pytest.raises(GpuLeaseBusy):
        second.acquire()
    first.release()
    second.acquire()
    second.release()
    assert not second.path.exists()


def test_worker_manager_captures_output_and_times_out_process_tree(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        pid = 424242

        def __init__(self, command, *, stdout, stderr, **kwargs):
            self.command = command
            self.returncode = None
            self.stdout = stdout
            self.stderr = stderr
            self.wait_count = 0
            stdout.write("worker-ok\n")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_count += 1
            if "timeout-fixture" in self.command and self.wait_count == 1:
                raise subprocess.TimeoutExpired(self.command, timeout)
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("text2model_forge.workers.subprocess.Popen", FakeProcess)
    monkeypatch.setattr(
        WorkerManager,
        "_terminate_tree",
        staticmethod(lambda process: process.kill()),
    )
    manager = WorkerManager(tmp_path)
    success = manager.run(
        WorkerJob(
            job_id="worker.success",
            run_id="run.one",
            worker_id="fixture.worker",
            command=[sys.executable, "-c", "print('worker-ok')"],
            cwd=str(tmp_path),
            timeout_seconds=10,
        )
    )
    assert success.return_code == 0
    with open(success.stdout_path, encoding="utf-8") as stdout:
        assert "worker-ok" in stdout.read()

    timed = manager.run(
        WorkerJob(
            job_id="worker.timeout",
            run_id="run.one",
            worker_id="fixture.worker",
            command=[sys.executable, "timeout-fixture"],
            cwd=str(tmp_path),
            timeout_seconds=0.2,
        )
    )
    assert timed.timed_out is True
    assert timed.return_code is not None


def test_worker_manager_cancels_active_job(tmp_path) -> None:
    if os.name == "nt":
        pytest.skip("real child termination requires an isolated Windows service/container boundary")
    manager = WorkerManager(tmp_path)
    job = WorkerJob(
        job_id="worker.cancel",
        run_id="run.one",
        worker_id="fixture.worker",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=str(tmp_path),
        timeout_seconds=10,
    )
    result = []
    thread = threading.Thread(target=lambda: result.append(manager.run(job)))
    thread.start()
    deadline = time.monotonic() + 3
    while not manager.cancel(job.job_id):
        if time.monotonic() >= deadline:
            pytest.fail("worker never became active")
        time.sleep(0.02)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result[0].cancelled is True


def test_optimizer_accepts_only_registered_operations() -> None:
    operation = OperationDefinition(
        operation_id="geometry.smooth",
        worker_id="blender",
        stages=[AssetStage.geometry],
        description="Smooth an allowlisted region.",
        output_media_type="model/gltf-binary",
    )
    evidence = EvidenceBundle(
        evidence_id="geometry.shoulder.v1",
        stage=AssetStage.geometry,
        goal="Remove the shoulder spike while preserving silhouette.",
        items=[EvidenceItem(artifact_id="render.front", role="front_render")],
    )

    def sender(payload):
        prompt = payload["messages"][0]["content"][0]["text"]
        assert "previous accepted and current attempted" in prompt
        assert "hard numeric verdict is authoritative" in prompt
        proposal = {
            "schema_version": 1,
            "goal_satisfied": False,
            "technical_score": 0.8,
            "observations": [{"region": "shoulder", "issue": "spike", "severity": 0.7}],
            "root_causes": [{"cause": "surface noise", "probability": 0.8}],
            "proposals": [
                {
                    "schema_version": 1,
                    "proposal_id": "proposal.shoulder.1",
                    "operation_id": "geometry.smooth",
                    "stage": AssetStage.geometry.value,
                    "input_artifact_ids": ["mesh.v1"],
                    "parameters": {"region": "shoulder"},
                    "rationale": "Localized deterministic repair.",
                }
            ],
            "preserve": ["silhouette"],
            "confidence": 0.8,
            "request_human_review": False,
            "comparison": {
                "preferred": "current",
                "visual_delta": 0.2,
                "reason": "The current shoulder silhouette is less pinched.",
            },
        }
        return {"choices": [{"message": {"content": json.dumps(proposal)}}]}

    client = LocalDeployStructuredClient(sender=sender)
    decision = LocalDeployOptimizer(client=client).diagnose(evidence, [operation])
    assert decision.proposals[0].operation_id == "geometry.smooth"
    assert decision.comparison is not None
    assert decision.comparison.preferred == "current"
    assert decision.comparison.visual_delta == 0.2

    def bad_sender(payload):
        bad = json.loads(sender(payload)["choices"][0]["message"]["content"])
        bad["proposals"][0]["operation_id"] = "arbitrary.execute_code"
        return {"choices": [{"message": {"content": json.dumps(bad)}}]}

    with pytest.raises(ValueError, match="unregistered"):
        LocalDeployOptimizer(client=LocalDeployStructuredClient(sender=bad_sender)).diagnose(
            evidence, [operation]
        )


def test_optimizer_rejects_out_of_bounds_registered_parameters() -> None:
    class BoundedParameters(BaseModel):
        model_config = ConfigDict(extra="forbid")
        delta_fraction: float = Field(ge=-0.025, le=0.025)

    operation = OperationDefinition(
        operation_id="rig.adjust_landmark_pair",
        worker_id="blender",
        stages=[AssetStage.rig],
        description="Move a landmark pair in a bounded trust region.",
        output_media_type="application/json",
        parameter_schema=BoundedParameters.model_json_schema(),
    )
    evidence = EvidenceBundle(
        evidence_id="rig.shoulders.v1",
        stage=AssetStage.rig,
        goal="Reduce shoulder compression.",
        items=[EvidenceItem(artifact_id="rig.shoulders.front", role="stress_render")],
    )

    def sender(payload):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "schema_version": 1,
                                "goal_satisfied": False,
                                "technical_score": 0.5,
                                "proposals": [
                                    {
                                        "schema_version": 1,
                                        "proposal_id": "proposal.shoulders.1",
                                        "operation_id": "rig.adjust_landmark_pair",
                                        "stage": AssetStage.rig.value,
                                        "input_artifact_ids": ["rig.shoulders.front"],
                                        "parameters": {"delta_fraction": 0.2},
                                        "rationale": "Too large on purpose.",
                                    }
                                ],
                                "confidence": 0.9,
                            }
                        )
                    }
                }
            ]
        }

    optimizer = LocalDeployOptimizer(client=LocalDeployStructuredClient(sender=sender))
    with pytest.raises(ValueError, match="invalid parameters"):
        optimizer.diagnose(
            evidence,
            [operation],
            parameter_models={"rig.adjust_landmark_pair": BoundedParameters},
        )


def test_optimizer_repairs_one_semantically_invalid_parameter_response() -> None:
    class JointParameters(BaseModel):
        model_config = ConfigDict(extra="forbid")
        joint_pair: str = Field(pattern=r"^(shoulders|knees)$")

    operation = OperationDefinition(
        operation_id="skin.redistribute_joint_pair_weights",
        worker_id="blender",
        stages=[AssetStage.rig],
        description="Transfer bounded joint weights.",
        output_media_type="application/json",
        parameter_schema=JointParameters.model_json_schema(),
    )
    evidence = EvidenceBundle(
        evidence_id="rig.weights.v1",
        stage=AssetStage.rig,
        goal="Reduce shoulder compression.",
        items=[EvidenceItem(artifact_id="rig.weights.front", role="stress_render")],
    )
    calls = 0

    def sender(payload):
        nonlocal calls
        calls += 1
        joint_pair = "legs" if calls == 1 else "shoulders"
        if calls == 2:
            correction = payload["messages"][0]["content"][-1]["text"]
            assert "preceding decision was rejected" in correction
            assert "joint_pair" in correction
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "schema_version": 1,
                                "goal_satisfied": False,
                                "technical_score": 0.6,
                                "proposals": [
                                    {
                                        "schema_version": 1,
                                        "proposal_id": "proposal.weights.1",
                                        "operation_id": operation.operation_id,
                                        "stage": AssetStage.rig.value,
                                        "input_artifact_ids": ["rig.weights.front"],
                                        "parameters": {"joint_pair": joint_pair},
                                        "rationale": "Bounded correction.",
                                    }
                                ],
                                "confidence": 0.7,
                            }
                        )
                    }
                }
            ]
        }

    optimizer = LocalDeployOptimizer(client=LocalDeployStructuredClient(sender=sender))
    decision = optimizer.diagnose(
        evidence,
        [operation],
        parameter_models={operation.operation_id: JointParameters},
    )
    assert calls == 2
    assert decision.proposals[0].parameters["joint_pair"] == "shoulders"
