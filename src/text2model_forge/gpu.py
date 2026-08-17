"""Cross-process exclusive GPU lease and optional LocalDeploy telemetry."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


class GpuLeaseBusy(RuntimeError):
    pass


class GpuMemoryAdmissionError(RuntimeError):
    """The requested GPU job cannot be proven to fit the live device."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class GpuLease:
    def __init__(self, root: str | Path, *, run_id: str, worker_id: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "gpu.lease.json"
        self.run_id = run_id
        self.worker_id = worker_id
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self, timeout_seconds: float = 0) -> None:
        deadline = time.monotonic() + timeout_seconds
        payload = {
            "schema_version": 1,
            "token": self.token,
            "pid": os.getpid(),
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                self.acquired = True
                return
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if not _pid_alive(int(existing.get("pid", -1))):
                    self.path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    raise GpuLeaseBusy(
                        f"GPU leased by run={existing.get('run_id')} worker={existing.get('worker_id')}"
                    )
                time.sleep(0.1)

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("token") == self.token:
            self.path.unlink(missing_ok=True)
        self.acquired = False

    def __enter__(self) -> "GpuLease":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def localdeploy_gpu_snapshot(base_url: str = "http://127.0.0.1:8000", timeout: float = 5) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/system/status", timeout=timeout)
        response.raise_for_status()
        hardware = response.json().get("hardware", {})
        return {"gpus": hardware.get("gpus", []), "served_models": response.json().get("served_models", [])}
    except (httpx.HTTPError, ValueError):
        return None


def gpu_memory_snapshot() -> dict[str, Any] | None:
    """Read live device memory without importing a heavyweight ML runtime.

    NVIDIA is the first qualified backend.  Other backends may supply the
    same fields through LocalDeploy later; returning ``None`` is deliberate
    because strict admission must never turn an unknown measurement into zero
    usage or unlimited capacity.
    """

    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            total_gb = float(parts[2]) / 1024
            used_gb = float(parts[3]) / 1024
            free_gb = float(parts[4]) / 1024
        except ValueError:
            continue
        devices.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "total_gb": round(total_gb, 3),
                "used_gb": round(used_gb, 3),
                "free_gb": round(free_gb, 3),
                "driver_version": parts[5],
                "source": "nvidia-smi",
            }
        )
    if not devices:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    selected = devices[0]
    if visible and visible not in {"-1", "none", "None"}:
        first = visible.split(",", 1)[0].strip()
        if first.isdigit():
            selected = next((item for item in devices if item["index"] == int(first)), selected)
    return {"device": selected, "devices": devices, "observed_at": datetime.now(timezone.utc).isoformat()}


def admit_gpu_memory(
    required_gb: float | None,
    *,
    safety_margin_gb: float = 0.75,
    require_measurement: bool = False,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fail before launch when a measured free-memory budget is insufficient."""

    observed = snapshot if snapshot is not None else gpu_memory_snapshot()
    if observed is None:
        if require_measurement:
            raise GpuMemoryAdmissionError(
                "GPU memory could not be measured; strict admission requires a supported telemetry backend"
            )
        return None
    device = observed.get("device") or {}
    free_gb = device.get("free_gb")
    if not isinstance(free_gb, (int, float)):
        if require_measurement:
            raise GpuMemoryAdmissionError("GPU telemetry did not report free VRAM")
        return observed
    if required_gb is not None:
        admitted_gb = float(required_gb) + float(safety_margin_gb)
        if free_gb < admitted_gb:
            raise GpuMemoryAdmissionError(
                f"GPU admission denied: job needs {required_gb:.2f} GiB plus "
                f"{safety_margin_gb:.2f} GiB safety headroom, but only {free_gb:.2f} GiB is free"
            )
    return observed


def wait_for_free_vram(
    required_gb: float,
    *,
    safety_margin_gb: float = 0.75,
    timeout_seconds: float = 45,
    poll_seconds: float = 0.5,
) -> dict[str, Any]:
    """Wait for an acknowledged unload to become visible in device telemetry."""

    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = gpu_memory_snapshot()
        try:
            admitted = admit_gpu_memory(
                required_gb,
                safety_margin_gb=safety_margin_gb,
                require_measurement=True,
                snapshot=last,
            )
            assert admitted is not None
            return admitted
        except GpuMemoryAdmissionError:
            time.sleep(max(0.05, min(poll_seconds, 2.0)))
    device = (last or {}).get("device") or {}
    raise GpuMemoryAdmissionError(
        f"GPU memory did not recover within {timeout_seconds:.0f}s after unload; "
        f"last free VRAM was {device.get('free_gb', 'unknown')} GiB"
    )


class GpuPeakSampler:
    """Sample whole-device usage while an exclusive worker owns the GPU."""

    def __init__(self, *, poll_seconds: float = 0.2) -> None:
        self.poll_seconds = poll_seconds
        self.peak_used_gb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            snapshot = gpu_memory_snapshot()
            used = ((snapshot or {}).get("device") or {}).get("used_gb")
            if isinstance(used, (int, float)):
                self.peak_used_gb = max(self.peak_used_gb or 0.0, float(used))
            self._stop.wait(self.poll_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, name="text2model-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_used_gb
