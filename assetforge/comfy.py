from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .core import ForgeError, asset_root, build_prompt_pack, load_project, read_json, record_job, sha256_file, sha256_json, utc_now, write_json


def _dimensions(profile: dict[str, Any], mode: str) -> tuple[int, int]:
    comfy = profile["comfyui"]
    if mode == "turnaround":
        width, height = int(comfy["turnaround_width"]), int(comfy["turnaround_height"])
    else:
        width, height = int(comfy["concept_width"]), int(comfy["concept_height"])
    if width % 64 or height % 64:
        raise ForgeError(f"ComfyUI dimensions must be divisible by 64: {width}x{height}")
    return width, height


def build_sdxl_workflow(
    workspace: Path,
    asset_id: str,
    mode: str,
    checkpoint: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in {"concept", "turnaround", "texture", "mesh_seed"}:
        raise ForgeError("mode must be concept, turnaround, texture, or mesh_seed")
    project = load_project(workspace)
    profile_path = workspace.resolve() / "hardware" / f"{project['hardware_profile']}.profile.json"
    profile = read_json(profile_path)
    prompt_pack_path = asset_root(workspace, asset_id) / "prompt-pack.json"
    prompt_pack = read_json(prompt_pack_path) if prompt_pack_path.exists() else build_prompt_pack(workspace, asset_id)
    registry = read_json(workspace.resolve() / project["model_registry"])
    model_record = next((item for item in registry.get("models", []) if item.get("filename") == checkpoint), None)
    width, height = _dimensions(profile, mode)
    settings = profile["comfyui"]
    prefix = f"AssetForge/{project['project_id']}/{asset_id}/{mode}"
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt_pack["prompts"][mode]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt_pack["negative"]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": int(seed),
                "steps": int(settings["steps"]),
                "cfg": float(settings["cfg"]),
                "sampler_name": settings["sampler"],
                "scheduler": settings["scheduler"],
                "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["6", 0]}},
    }
    provenance = {
        "schema_version": 1,
        "engine": "ComfyUI",
        "workflow_type": "sdxl-core-only",
        "workflow_sha256": sha256_json(workflow),
        "asset_id": asset_id,
        "mode": mode,
        "checkpoint": checkpoint,
        "model": {
            "id": model_record.get("id") if model_record else None,
            "source_url": model_record.get("source_url") if model_record else None,
            "license": model_record.get("license") if model_record else None,
            "installed_sha256": model_record.get("installed_sha256") if model_record else None,
            "registry_status": "hash-registered" if model_record and model_record.get("installed_sha256") else "unregistered",
        },
        "seed": int(seed),
        "width": width,
        "height": height,
        "steps": int(settings["steps"]),
        "cfg": float(settings["cfg"]),
        "sampler": settings["sampler"],
        "scheduler": settings["scheduler"],
        "positive_prompt": prompt_pack["prompts"][mode],
        "negative_prompt": prompt_pack["negative"],
        "created_utc": utc_now(),
    }
    return workflow, provenance


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ForgeError(f"ComfyUI request failed for {path}: {error}") from error
        if not isinstance(value, dict):
            raise ForgeError(f"ComfyUI returned an unexpected response for {path}")
        return value

    def system_stats(self) -> dict[str, Any]:
        return self._json("/system_stats")

    def upload_image(self, name: str, data: bytes, subfolder: str = "assetforge") -> str:
        boundary = "----AssetForge" + uuid.uuid4().hex
        parts = []
        for field, value in (("overwrite", "true"), ("type", "input"), ("subfolder", subfolder)):
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n".encode("utf-8")
            )
        parts.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                "Content-Type: image/png\r\n\r\n"
            ).encode("utf-8")
        )
        parts.append(data)
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)
        request = urllib.request.Request(
            self.base_url + "/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ForgeError(f"ComfyUI image upload failed for {name}: {error}") from error
        stored = value.get("name", name)
        stored_subfolder = value.get("subfolder", subfolder)
        return f"{stored_subfolder}/{stored}" if stored_subfolder else stored

    def checkpoints(self) -> list[str]:
        request = urllib.request.Request(self.base_url + "/models/checkpoints")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ForgeError(f"Could not list ComfyUI checkpoints: {error}") from error
        if not isinstance(value, list):
            raise ForgeError("ComfyUI checkpoint response was not a list")
        return [str(item) for item in value]

    def submit(self, workflow: dict[str, Any]) -> str:
        response = self._json("/prompt", {"prompt": workflow})
        if response.get("node_errors"):
            raise ForgeError(f"ComfyUI rejected workflow: {json.dumps(response['node_errors'], indent=2)}")
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise ForgeError(f"ComfyUI did not return prompt_id: {response}")
        return str(prompt_id)

    def wait(self, prompt_id: str, timeout_seconds: float = 600.0, poll_seconds: float = 1.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            history = self._json(f"/history/{urllib.parse.quote(prompt_id)}")
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ForgeError(f"ComfyUI job failed: {json.dumps(status, indent=2)}")
                if status.get("completed") is True:
                    return entry
            time.sleep(poll_seconds)
        raise ForgeError(f"Timed out waiting {timeout_seconds}s for ComfyUI job {prompt_id}")

    def download_outputs(self, history: dict[str, Any], destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        downloaded = []
        for node in history.get("outputs", {}).values():
            for image in node.get("images", []):
                query = urllib.parse.urlencode({
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                })
                target = destination / Path(image["filename"]).name
                try:
                    with urllib.request.urlopen(self.base_url + "/view?" + query, timeout=self.timeout) as response:
                        target.write_bytes(response.read())
                except (urllib.error.URLError, TimeoutError) as error:
                    raise ForgeError(f"Could not download ComfyUI output {image['filename']}: {error}") from error
                downloaded.append(target)
        if not downloaded:
            raise ForgeError("ComfyUI job completed without image outputs")
        return downloaded


def generate(
    workspace: Path,
    asset_id: str,
    mode: str,
    checkpoint: str,
    seed: int,
    base_url: str,
    timeout_seconds: float,
) -> list[Path]:
    workflow, provenance = build_sdxl_workflow(workspace, asset_id, mode, checkpoint, seed)
    client = ComfyClient(base_url)
    available = client.checkpoints()
    if checkpoint not in available:
        raise ForgeError(f"Checkpoint {checkpoint!r} is not installed. Available: {available}")
    prompt_id = client.submit(workflow)
    provenance["prompt_id"] = prompt_id
    provenance["base_url"] = base_url
    job_path = record_job(workspace, asset_id, provenance)
    history = client.wait(prompt_id, timeout_seconds=timeout_seconds)
    destination = asset_root(workspace, asset_id) / "concepts" / "candidates" / mode / str(seed)
    outputs = client.download_outputs(history, destination)
    provenance["completed_utc"] = utc_now()
    provenance["outputs"] = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in outputs]
    write_json(job_path, provenance)
    return outputs
