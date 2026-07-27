"""Validated file contract for isolated research-worker wrappers."""
from __future__ import annotations

from pathlib import Path

from .schemas import ExternalWorkerRequest, ExternalWorkerResponse, WorkerJob, WorkerManifest
from .workers import WorkerManager


class ExternalWorkerContractError(RuntimeError):
    pass


class SubprocessWorkerAdapter:
    def __init__(
        self,
        manager: WorkerManager,
        manifest: WorkerManifest,
        command_prefix: list[str | Path],
        *,
        environment: dict[str, str] | None = None,
    ) -> None:
        if manifest.adapter != "subprocess":
            raise ValueError("manifest is not a subprocess worker")
        if not command_prefix:
            raise ValueError("command_prefix is required")
        self.manager = manager
        self.manifest = manifest
        self.command_prefix = [str(item) for item in command_prefix]
        self.environment = environment or {}

    def execute(self, request: ExternalWorkerRequest, *, timeout_seconds: float = 1800) -> ExternalWorkerResponse:
        job_root = self.manager.workspace / "jobs" / request.job_id
        job_root.mkdir(parents=True, exist_ok=False)
        request_path = job_root / "request.json"
        response_path = job_root / "response.json"
        request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        result = self.manager.run(
            WorkerJob(
                job_id=request.job_id,
                run_id=request.run_id,
                worker_id=self.manifest.capability.worker_id,
                command=[*self.command_prefix, "--request", str(request_path), "--response", str(response_path)],
                cwd=str(job_root),
                timeout_seconds=timeout_seconds,
                exclusive_gpu=self.manifest.capability.exclusive_gpu,
                environment=self.environment,
            )
        )
        if result.timed_out or result.cancelled or result.return_code != 0:
            raise ExternalWorkerContractError(
                f"worker failed: return_code={result.return_code} timed_out={result.timed_out} cancelled={result.cancelled}"
            )
        if not response_path.exists():
            raise ExternalWorkerContractError("worker exited successfully without response.json")
        response = ExternalWorkerResponse.model_validate_json(response_path.read_text(encoding="utf-8"))
        if response.job_id != request.job_id:
            raise ExternalWorkerContractError("worker response job_id mismatch")
        if response.status != "succeeded":
            raise ExternalWorkerContractError(response.error or f"worker status is {response.status}")
        output_root = Path(request.output_directory).resolve()
        for output in response.outputs:
            path = Path(output.path).resolve()
            if output_root != path and output_root not in path.parents:
                raise ExternalWorkerContractError("worker output escapes its declared output directory")
            if not path.is_file():
                raise ExternalWorkerContractError(f"declared worker output does not exist: {path}")
        return response
