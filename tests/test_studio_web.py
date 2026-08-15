"""HTTP-level tests for the Studio browser control plane.

These drive a real loopback ThreadingHTTPServer rather than calling handler
methods directly, so routing, CSRF enforcement, form parsing, redirects, and
error rendering are all exercised as a browser would exercise them. Before
this file darkness/studio_web.py had zero test coverage.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from darkness.studio_pipeline import StudioCoordinator
from darkness.studio_web import build_server

from test_studio import DESCRIPTION, FakeComfy, FakeQwen, FakeWorkerExecutor, wait_for


@pytest.fixture
def studio(tmp_path: Path):
    """A running Studio server backed entirely by fakes, on an ephemeral port.

    The coordinator is built FROM the server's own store via the factory, not
    handed in pre-built on a store of its own: two StudioStore instances over
    the same directory hold independent locks, so the coordinator thread and
    the request handlers would race on run.json writes and list() could
    intermittently observe a run mid-replace.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = build_server(
            tmp_path,
            port=0,
            coordinator_factory=lambda store: StudioCoordinator(
                store,
                qwen_factory=lambda run: FakeQwen(),
                comfy_factory=lambda run: FakeComfy(),
                worker_executor=FakeWorkerExecutor(),
                executor=executor,
            ),
        )
        thread = threading.Thread(target=running.server.serve_forever, daemon=True)
        thread.start()
        try:
            yield running
        finally:
            running.server.shutdown()
            running.server.server_close()
            thread.join(timeout=5)


def _get(studio, path: str):
    try:
        with urllib.request.urlopen(studio.url + path, timeout=10) as response:
            return response.status, response.read().decode("utf-8"), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8"), dict(error.headers)


def _only_run_id(studio, timeout: float = 5) -> str:
    """The single run this test created. Polls rather than indexing list()[0]
    directly: store.list() skips any run.json it cannot parse, so indexing it
    the instant after a POST can raise IndexError if the coordinator thread
    happens to be mid-write."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runs = studio.store.list()
        if runs:
            return runs[0].run_id
        time.sleep(0.02)
    raise AssertionError("no run was created")


def _post(studio, path: str, fields: dict[str, str], *, follow: bool = False):
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(studio.url + path, data=data, method="POST")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def test_dashboard_and_new_asset_pages_render(studio):
    status, body, headers = _get(studio, "/")
    assert status == HTTPStatus.OK
    assert "Asset production runs" in body

    status, body, _ = _get(studio, "/new")
    assert status == HTTPStatus.OK
    assert "Describe one original asset" in body
    # The form carries the server's CSRF token and the profile selector.
    assert studio.csrf in body
    assert 'name=profile' in body
    assert "advanced" in body


def test_security_headers_are_set_on_every_page(studio):
    _, _, headers = _get(studio, "/")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"


def test_unknown_path_is_a_clean_404_not_a_traceback(studio):
    status, body = _post(studio, "/definitely-not-a-route", {"csrf": studio.csrf})
    assert status == HTTPStatus.NOT_FOUND
    assert "Traceback" not in body


def test_post_without_a_valid_csrf_token_is_refused(studio):
    status, body = _post(
        studio,
        "/runs",
        {"csrf": "not-the-real-token", "description": "x" * 40, "profile": "simple"},
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "invalid form token" in body
    # and no run was created
    assert studio.store.list() == []


def test_creating_a_run_requires_a_substantial_description(studio):
    status, body = _post(
        studio, "/runs", {"csrf": studio.csrf, "description": "too short", "profile": "simple"}
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "at least 20 characters" in body
    assert studio.store.list() == []


def test_creating_a_run_records_the_selected_profile(studio):
    status, _ = _post(
        studio,
        "/runs",
        {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "advanced"},
    )
    assert status in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}
    runs = studio.store.list()
    assert len(runs) == 1
    assert runs[0].profile == "advanced"
    # advanced.toml pins quality=high, which resolves to concept_steps=45
    assert runs[0].concept_steps == 45


def test_malformed_override_json_is_explained_not_dumped_as_a_traceback(studio):
    """The override guard added to the decision route is verified here for
    real rather than only by reading it."""
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    candidate = next(
        item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True
    )

    status, body = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "retry",
            "comment": "",
            "selected_evidence_id": candidate.evidence_id,
            "overrides": "{not valid json",
        },
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "must be valid JSON" in body
    assert "Traceback" not in body

    status, body = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "retry",
            "comment": "",
            "selected_evidence_id": candidate.evidence_id,
            "overrides": "[1, 2, 3]",
        },
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "must be a JSON object" in body

    # An out-of-range value is rejected by the shared validator, at the gate.
    status, body = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "retry",
            "comment": "",
            "selected_evidence_id": candidate.evidence_id,
            "overrides": '{"seed": -5}',
        },
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "non-negative whole number" in body

    # None of the three rejected attempts changed the stage.
    assert studio.store.load(run.run_id).stage("D1").state == "awaiting_review"


def test_approving_a_gate_through_the_web_form_advances_the_run(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    candidate = next(
        item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True
    )
    status, _ = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "approve",
            "comment": "Looks good.",
            "selected_evidence_id": candidate.evidence_id,
        },
    )
    assert status in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}
    decided = studio.store.load(run.run_id)
    assert decided.stage("D1").state == "approved"
    assert decided.stage("D1").human_decisions[-1].comment == "Looks good."


def test_run_page_and_json_api_expose_the_runs_state(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")

    status, body, _ = _get(studio, f"/run/{run.run_id}")
    assert status == HTTPStatus.OK
    assert run.run_id in body

    status, body, headers = _get(studio, f"/api/run/{run.run_id}")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("application/json")
    import json as _json

    payload = _json.loads(body)
    assert payload["run_id"] == run.run_id


def test_artifact_route_refuses_to_escape_the_run_directory(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    escape = urllib.parse.quote("../../../../etc/passwd", safe="")
    status, body, _ = _get(studio, f"/artifact/{run.run_id}/{escape}")
    assert status == HTTPStatus.NOT_FOUND
    assert "Traceback" not in body
    assert "passwd" not in body or "artifact not found" in body


def test_doctor_page_reports_dependency_state_without_crashing(studio):
    """The /doctor route probes ComfyUI and LocalDeploy, neither of which is
    running here -- it must report them as down rather than raise."""
    status, body, _ = _get(studio, "/doctor")
    assert status == HTTPStatus.OK
    assert "ComfyUI" in body or "comfy" in body.lower()
