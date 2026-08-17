from __future__ import annotations

import json
from pathlib import Path

import pytest


ADAPTER = Path(__file__).resolve().parents[1] / "resources" / "adapters" / "wsl_launch.py"


def _load_adapter_namespace() -> dict:
    namespace = {"__name__": "wsl_launch_test"}
    exec(compile(ADAPTER.read_text(encoding="utf-8"), str(ADAPTER), "exec"), namespace)
    return namespace


def test_windows_path_to_wsl_translates_drive_paths(tmp_path) -> None:
    windows_path_to_wsl = _load_adapter_namespace()["windows_path_to_wsl"]
    target = tmp_path / "jobs" / "job.one" / "request.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")

    wsl_path = windows_path_to_wsl(str(target))
    drive = str(tmp_path.resolve())[0].lower()
    assert wsl_path.startswith(f"/mnt/{drive}/")
    assert wsl_path.endswith("jobs/job.one/request.json")
    assert "\\" not in wsl_path


def test_windows_path_to_wsl_rejects_unc_paths() -> None:
    namespace = _load_adapter_namespace()
    with pytest.raises(namespace["WslPathError"]):
        namespace["windows_path_to_wsl"](r"\\server\share\file.json")


def test_wsl_path_to_windows_round_trips(tmp_path) -> None:
    namespace = _load_adapter_namespace()
    target = tmp_path / "output" / "trellis2_candidate.glb"
    target.parent.mkdir(parents=True)
    target.write_text("glb", encoding="utf-8")

    wsl_path = namespace["windows_path_to_wsl"](str(target))
    back = namespace["wsl_path_to_windows"](wsl_path)
    assert Path(back).resolve() == target.resolve()


def test_wsl_path_to_windows_passes_through_non_mount_paths() -> None:
    wsl_path_to_windows = _load_adapter_namespace()["wsl_path_to_windows"]
    linux_only = "/home/user/.cache/trellis2/log.txt"
    assert wsl_path_to_windows(linux_only) == linux_only


def test_translate_request_rewrites_output_directory_and_input_paths(tmp_path) -> None:
    namespace = _load_adapter_namespace()
    output_dir = tmp_path / "jobs" / "job.one" / "output"
    output_dir.mkdir(parents=True)
    image = tmp_path / "concept.png"
    image.write_text("png", encoding="utf-8")

    original = {
        "job_id": "job.one",
        "output_directory": str(output_dir),
        "input_paths": {"concept.v1": str(image)},
    }
    translated = namespace["_translate_request"](original)

    assert translated["output_directory"].startswith("/mnt/")
    assert translated["input_paths"]["concept.v1"].startswith("/mnt/")
    assert translated["job_id"] == "job.one"
    assert original["output_directory"] == str(output_dir), "input dict must not be mutated"


def test_translate_response_paths_rewrites_outputs_back_to_windows(tmp_path) -> None:
    namespace = _load_adapter_namespace()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    glb = output_dir / "trellis2_candidate.glb"
    glb.write_text("glb", encoding="utf-8")

    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": "job.one",
                "status": "succeeded",
                "outputs": [
                    {
                        "path": namespace["windows_path_to_wsl"](str(glb)),
                        "media_type": "model/gltf-binary",
                        "role": "geometry_candidate",
                    }
                ],
                "diagnostics": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    namespace["_translate_response_paths_in_place"](response_path)

    payload = json.loads(response_path.read_text(encoding="utf-8"))
    assert Path(payload["outputs"][0]["path"]).resolve() == glb.resolve()


def test_translate_response_paths_is_a_noop_when_response_missing(tmp_path) -> None:
    namespace = _load_adapter_namespace()
    missing = tmp_path / "response.json"
    namespace["_translate_response_paths_in_place"](missing)
    assert not missing.exists()
