"""Cross-process exclusive GPU lease and optional LocalDeploy telemetry."""
from __future__ import annotations

import json
import os
import time
import uuid
import ctypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


class GpuLeaseBusy(RuntimeError):
    pass


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
