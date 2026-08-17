"""Isolated subprocess worker execution with logs, timeout, cancellation, and GPU lease."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .gpu import GpuLease, GpuPeakSampler, admit_gpu_memory, gpu_memory_snapshot
from .schemas import WorkerJob, WorkerResult


class WorkerManager:
    def __init__(self, workspace: str | Path, *, allowed_roots: list[str | Path] | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.allowed_roots = [Path(item).resolve() for item in (allowed_roots or [self.workspace])]
        self.logs = self.workspace / "worker_logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, subprocess.Popen[str]] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            # `taskkill /T` can escape an embedding host's Windows job object and
            # disrupt the control plane. Real heavyweight workers are launched in
            # their own container/service boundary; locally, kill only the exact
            # process we created.
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            process = self._active.get(job_id)
            if process is None:
                return False
            self._cancelled.add(job_id)
        self._terminate_tree(process)
        return True

    def run(self, job: WorkerJob) -> WorkerResult:
        cwd = Path(job.cwd).resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(f"worker cwd does not exist: {cwd}")
        if not any(cwd == root or root in cwd.parents for root in self.allowed_roots):
            raise PermissionError(f"worker cwd is outside allowlisted roots: {cwd}")
        stdout_path = self.logs / f"{job.job_id}.stdout.log"
        stderr_path = self.logs / f"{job.job_id}.stderr.log"
        safe_keys = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "TMPDIR",
            "HOME",
            "USERPROFILE",
            "LANG",
        }
        environment = {key: value for key, value in os.environ.items() if key.upper() in safe_keys}
        environment.update(job.environment)
        if job.device_policy in {"gpu_compute_only", "strict_device_only"}:
            environment.update(
                {
                    "TEXT2MODEL_DEVICE_POLICY": job.device_policy,
                    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                }
            )
        lease = GpuLease(self.workspace / "scheduler", run_id=job.run_id, worker_id=job.worker_id)
        if job.exclusive_gpu:
            lease.acquire(timeout_seconds=job.timeout_seconds)
        gpu_before = None
        admission_free_gb = None
        sampler = GpuPeakSampler()
        if job.exclusive_gpu:
            try:
                gpu_before = admit_gpu_memory(
                    job.gpu_memory_gb,
                    safety_margin_gb=job.gpu_safety_margin_gb,
                    require_measurement=job.require_gpu_measurement,
                )
            except Exception:
                lease.release()
                raise
            admission_free_gb = ((gpu_before or {}).get("device") or {}).get("free_gb")
            sampler.start()
        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        timed_out = False
        creationflags = 0
        preexec_fn = os.setsid if os.name != "nt" else None
        process: subprocess.Popen[str] | None = None
        try:
            with self._lock:
                if job.job_id in self._active:
                    raise RuntimeError(f"duplicate active job_id '{job.job_id}'")
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    job.command,
                    cwd=cwd,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    shell=False,
                    creationflags=creationflags,
                    preexec_fn=preexec_fn,
                )
                with self._lock:
                    self._active[job.job_id] = process
                try:
                    process.wait(timeout=job.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._terminate_tree(process)
                    process.wait(timeout=10)
        finally:
            with self._lock:
                self._active.pop(job.job_id, None)
                cancelled = job.job_id in self._cancelled
                self._cancelled.discard(job.job_id)
            gpu_peak_used_gb = sampler.stop() if job.exclusive_gpu else None
            gpu_after = gpu_memory_snapshot() if job.exclusive_gpu else None
            lease.release()
        finished_at = datetime.now(timezone.utc)
        return WorkerResult(
            job_id=job.job_id,
            worker_id=job.worker_id,
            return_code=process.returncode if process else None,
            timed_out=timed_out,
            cancelled=cancelled,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - started,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            gpu_before=gpu_before,
            gpu_after=gpu_after,
            gpu_peak_used_gb=gpu_peak_used_gb,
            gpu_admission_required_gb=job.gpu_memory_gb,
            gpu_admission_free_gb=admission_free_gb,
        )
