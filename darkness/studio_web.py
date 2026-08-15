"""Loopback-only browser UI for Darkness Studio runs and human gates."""
from __future__ import annotations

import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
import urllib.request
import uuid
import webbrowser

from .config import load_local_config
from .settings import profiles_dir, resolve_settings, studio_overrides
from .studio_pipeline import StudioCoordinator
from .studio_store import StudioStore


STYLE = """
:root{color-scheme:dark;--bg:#0c1114;--panel:#182126;--panel2:#202c32;--line:#3a4a51;--text:#eee6d7;--muted:#a4afb0;--ember:#dc6837;--steel:#45728b;--ok:#62bd79;--bad:#e26d62;--wait:#d5a746}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% -20%,#28373c,#0c1114 46%);color:var(--text);font:15px/1.45 system-ui,sans-serif}header{position:sticky;top:0;z-index:4;padding:18px 28px;background:#0d1417ee;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:24px}header h1{font-size:20px;letter-spacing:.12em;margin:0}header a{color:#c8dce5;text-decoration:none}.wrap{max-width:1320px;margin:auto;padding:26px}h1,h2,h3{margin:0 0 12px}p{margin:8px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.card{background:linear-gradient(145deg,var(--panel),#141c20);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:0 12px 30px #0005}.hero{padding:28px}.muted{color:var(--muted)}.error{color:#ff9e95}.good{color:#a7e7b6}.warning{color:#f0cf79}.timeline{display:grid;grid-template-columns:repeat(11,minmax(82px,1fr));gap:6px;margin:18px 0;overflow:auto}.stage{min-height:88px;border:1px solid var(--line);border-radius:9px;padding:9px;background:#12191d}.stage strong{display:block}.stage small{display:block;color:var(--muted)}.stage.approved{border-color:#3d8050;background:#17281c}.stage.skipped{opacity:.56;border-style:dashed}.stage.running,.stage.queued{border-color:var(--steel)}.stage.awaiting_review{border-color:var(--wait);background:#292315}.stage.rejected,.stage.failed{border-color:var(--bad);background:#2a1918}.stage.blocked{border-color:#8a6e3a}.bar{height:5px;background:#0b1012;border-radius:4px;margin-top:10px;overflow:hidden}.bar span{display:block;height:100%;background:var(--ember)}label{display:block;color:var(--muted);margin:10px 0 5px}textarea,input{width:100%;padding:11px;border:1px solid var(--line);border-radius:7px;background:#0d1417;color:var(--text)}textarea{min-height:130px;resize:vertical}button,.button{display:inline-block;border:0;border-radius:7px;padding:10px 16px;margin:10px 8px 0 0;background:var(--steel);color:white;text-decoration:none;cursor:pointer}.primary{background:var(--ember)!important}.reject,.danger{background:#9d4039!important}.memory{background:#735c24!important}.secondary{background:#34454c!important}.evidence{position:relative}.evidence img{width:100%;max-height:620px;object-fit:contain;background:#0a0f11;border:1px solid var(--line);border-radius:8px}.choice{display:flex;gap:8px;align-items:center;margin:8px 0}.choice input{width:auto}.badge{display:inline-block;padding:3px 7px;border-radius:999px;background:#29373d;color:#cbd8dc;font-size:12px}.recommended{background:#49381c;color:#ffe09a}pre,code{background:#0b1012;border:1px solid #2b373c;border-radius:6px}pre{padding:12px;white-space:pre-wrap;overflow:auto}code{padding:2px 5px}.review ul{margin-top:5px}.events{max-height:330px;overflow:auto}.event{border-left:2px solid var(--line);padding:5px 0 5px 12px;margin:4px 0}.run-card h2 a{color:var(--text);text-decoration:none}.actions{display:flex;gap:8px;flex-wrap:wrap}.health{display:flex;gap:8px;flex-wrap:wrap}.health span{padding:5px 9px;border-radius:8px;background:#253239}.health .down{background:#4a2220}.console{border-color:#5a707a;background:linear-gradient(135deg,#1b2a31,#142025)}.console-head{display:flex;justify-content:space-between;gap:12px;align-items:start}.console-head h2{margin:0}.control-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:10px;margin-top:15px}.control-action{border:1px solid var(--line);border-radius:9px;padding:12px;background:#10191d}.control-action h3{font-size:14px;margin-bottom:5px}.control-action p{font-size:13px;min-height:39px}.status-line{margin:14px 0 0;padding:10px 12px;border-radius:8px;background:#10191d;border-left:3px solid var(--steel)}.status-line.stopping{border-left-color:var(--bad);background:#2b1b1a}.status-line.idle{border-left-color:var(--ok)}.prompt-meta{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.prompt-tools{display:flex;align-items:center;gap:2px;flex-wrap:wrap}.prompt-tools button{margin-top:10px}button:disabled{cursor:not-allowed;opacity:.5}@media(max-width:800px){.wrap{padding:15px}.timeline{grid-template-columns:repeat(11,100px)}header{padding:14px}.hero{padding:18px}}
"""


def _page(title: str, body: str, *, refresh: bool = False) -> bytes:
    meta = '<meta http-equiv="refresh" content="5">' if refresh else ""
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"{meta}<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        '<body><header><h1>DARKNESS STUDIO</h1><a href="/">Runs</a><a href="/new">New asset</a>'
        '<a href="/doctor">System</a></header><main class="wrap">'
        f"{body}</main></body></html>"
    ).encode("utf-8")


def _probe(url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200, f"HTTP {response.status}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _form(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length > 1_000_000:
        raise ValueError("form is too large")
    parsed = parse_qs(handler.rfile.read(length).decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def _slug() -> str:
    from datetime import datetime, timezone

    return "asset-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5]


def _artifact_url(run_id: str, relative_path: str) -> str:
    return f"/artifact/{quote(run_id)}/{quote(relative_path, safe='/')}"


def _timeline(run) -> str:
    return '<div class="timeline">' + "".join(
        (
            f'<div class="stage {item.state}"><strong>{item.stage_id}</strong><small>{html.escape(item.label)}</small>'
            f'<span class="badge">{html.escape(item.state.replace("_", " "))}</span>'
            f'<div class="bar"><span style="width:{item.progress * 100:.0f}%"></span></div></div>'
        )
        for item in run.stages
    ) + "</div>"


def _spec(run) -> str:
    if run.spec is None:
        return '<p class="muted">Qwen has not compiled the description yet.</p>'
    equipment = "".join(
        "<li>"
        f"<strong>{html.escape(item.equipment_id)}</strong> — {html.escape(item.category)}, "
        f"{html.escape(item.side)}, <code>{html.escape(item.socket)}</code>, {html.escape(item.grip)}"
        "</li>"
        for item in run.spec.equipment
    )
    components = "".join(
        "<li>"
        f"<strong>{html.escape(item.component_id)}</strong> — {html.escape(item.role)}, "
        f"{html.escape(item.motion)} via {html.escape(item.connection)}"
        "</li>"
        for item in run.spec.components
    )
    anatomy = html.escape(run.spec.anatomy_family or "not applicable")
    height = f"{run.spec.height_m:.2f} m high" if run.spec.height_m is not None else "height not applicable"
    dimensions = " × ".join(f"{item:g}" for item in run.spec.dimensions_m) + " m"
    return (
        f"<p>{html.escape(run.spec.creative_direction)}</p>"
        f"<p><span class=badge>{html.escape(run.spec.asset_kind)}</span> "
        f"<span class=badge>{html.escape(run.spec.behavior)}</span> "
        f"<span class=badge>{anatomy}</span> <span class=badge>{height}</span> "
        f"<span class=badge>{dimensions}</span></p>"
        f"<h3>Component contract</h3><ul>{components or '<li>Single continuous asset</li>'}</ul>"
        f"<h3>Equipment contract</h3><ul>{equipment or '<li>None</li>'}</ul>"
        f"<h3>Animations / states</h3><p>{html.escape(', '.join(run.spec.animations) or 'Static')}</p>"
        f"<h3>Locked features</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in run.spec.locked_features)}</ul>"
    )


def _qwen_review(stage) -> str:
    if not stage.qwen_reviews:
        return '<p class="muted">No Qwen review yet.</p>'
    review = stage.qwen_reviews[-1]
    return (
        '<div class="review">'
        f"<p>{html.escape(review.summary)}</p>"
        f"<p><span class=badge>confidence {review.confidence:.0%}</span> "
        f"<span class=badge>locked requirements {'pass' if review.hard_requirements_satisfied else 'fail'}</span> "
        f"<span class='badge recommended'>recommended {html.escape(review.recommended_evidence_id or 'none')}</span></p>"
        f"<h3>Strengths</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in review.strengths) or '<li>None recorded</li>'}</ul>"
        f"<h3>Issues</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in review.issues) or '<li>None recorded</li>'}</ul>"
        f"<h3>If rejected</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in review.recommended_changes) or '<li>Use the human comment.</li>'}</ul>"
        "</div>"
    )


def _evidence(run, stage) -> tuple[str, list[str]]:
    recommended = stage.qwen_reviews[-1].recommended_evidence_id if stage.qwen_reviews else None
    cards = []
    choices = []
    for item in stage.evidence:
        url = _artifact_url(run.run_id, item.relative_path)
        if item.media_type.startswith("image/"):
            current = item.metrics.get("iteration") in (None, stage.iteration)
            selectable = item.metrics.get("selectable") is True
            checked = " checked" if item.evidence_id == recommended else ""
            choice = (
                f'<label class=choice><input type=radio name=selected_evidence_id value="{html.escape(item.evidence_id)}"{checked}>'
                f'Select {html.escape(item.label)}</label>'
                if current and selectable and stage.state == "awaiting_review"
                else ""
            )
            if choice:
                choices.append(choice)
            cards.append(
                '<section class="card evidence">'
                f"<h3>{html.escape(item.label)} {'<span class=\"badge recommended\">Qwen pick</span>' if item.evidence_id == recommended else ''}</h3>"
                f'<a href="{url}" target=_blank><img loading=lazy src="{url}" alt="{html.escape(item.label)}"></a>'
                f"<pre>{html.escape(json.dumps(item.metrics, indent=2))}</pre>{choice}</section>"
            )
        else:
            cards.append(
                '<section class="card evidence">'
                f"<h3>{html.escape(item.label)}</h3><p><a class=button href=\"{url}\" target=_blank>Open evidence</a></p>"
                f"<pre>{html.escape(json.dumps(item.metrics, indent=2))}</pre></section>"
            )
    return '<div class="grid">' + "".join(cards) + "</div>", choices


def _decision_form(run, stage, csrf: str, choices: list[str]) -> str:
    if stage.state != "awaiting_review":
        return ""
    stage_index = next(i for i, item in enumerate(run.stages) if item.stage_id == stage.stage_id)
    rollback_targets = [
        item
        for item in run.stages[:stage_index]
        if item.state in {"approved", "skipped", "rejected", "failed"}
    ]
    rollback_options = "".join(
        f'<option value="{item.stage_id}">{html.escape(item.stage_id)} -- {html.escape(item.label)}</option>'
        for item in rollback_targets
    )
    rollback_block = (
        f'<label>Roll back to</label><select name=target_stage_id><option value="">(not a rollback)</option>{rollback_options}</select>'
        if rollback_targets
        else '<input type=hidden name=target_stage_id value="">'
    )
    return (
        '<section class="card"><h2>Your decision</h2><p>Approve a selected candidate, reject with a comment for '
        "Qwen's next attempt, retry the same stage fresh, edit with a concrete correction, skip this stage, or "
        'roll back to an earlier stage. Full history is preserved either way.</p>'
        f'<form method=post action="/run/{html.escape(run.run_id)}/decision">'
        f'<input type=hidden name=csrf value="{csrf}"><input type=hidden name=stage_id value="{stage.stage_id}">'
        + "".join(choices)
        + '<label>Comment</label><textarea name=comment placeholder="Required for reject and skip; optional otherwise."></textarea>'
        '<label>Overrides (JSON, optional -- used by retry and edit)</label>'
        '<textarea name=overrides placeholder=\'e.g. {"seed": 42}\'></textarea>'
        + rollback_block
        + '<div class=actions>'
        '<button class=primary name=decision value=approve type=submit>Approve and continue</button>'
        '<button class=reject name=decision value=reject type=submit>Reject and iterate</button>'
        '<button class=secondary name=decision value=retry type=submit>Retry this stage</button>'
        '<button class=secondary name=decision value=edit type=submit>Edit and retry</button>'
        '<button class=secondary name=decision value=skip type=submit>Skip this stage</button>'
        + (
            '<button class=danger name=decision value=rollback type=submit>Roll back</button>'
            if rollback_targets
            else ""
        )
        + '</div></form></section>'
    )


def _manual_qwen_image_form(run, stage, csrf: str, *, busy: bool) -> str:
    """Offer a direct D1 candidate without bypassing the stored production contract."""
    if stage.stage_id != "D1":
        return ""
    if busy or stage.state != "awaiting_review":
        return (
            '<section class="card console"><h2>Direct Qwen Image 2512 render</h2>'
            '<p class=muted>A direct prompt is available when this concept gate is waiting for review. '
            'Use the controls below to monitor or stop a running job.</p></section>'
        )
    return (
        '<section class="card console"><div class=console-head><div><h2>Direct Qwen Image 2512 render</h2>'
        '<p class=muted>One controlled candidate, returned to this same human-review gate.</p></div>'
        '<span class="badge recommended">ready</span></div>'
        '<p>Try a specific visual direction without restarting the asset. Your words are stored as evidence; '
        'Qwen expands them against the typed asset contract, renders one new candidate, then critiques it before '
        'returning it here for your decision.</p>'
        '<div class=prompt-meta><span class=badge>Qwen Image 2512</span><span class=badge>1104 &times; 1472</span>'
        '<span class=badge>50 sampling steps</span><span class=badge>Qwen contract rewrite + critic</span></div>'
        f'<form method=post action="/run/{html.escape(run.run_id)}/qwen-image">'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<label>Visual prompt</label><textarea name=prompt minlength=12 required '
        'placeholder="Example: premium hand-painted 3D fantasy footman with a solid opaque kite shield on his left forearm and a straight sword gripped in his right hand; neutral studio background, natural proportions."></textarea>'
        '<label>Optional deterministic seed</label><input type=number name=seed min=0 step=1 placeholder="Leave blank for a fresh random seed">'
        '<div class=prompt-tools><button class=primary type=submit>Render direct Qwen candidate</button>'
        '<button class=secondary type=reset>Clear prompt and seed</button></div></form></section>'
    )


def _studio_controls(run, stage, coordinator: StudioCoordinator, csrf: str, *, busy: bool) -> str:
    stopping = coordinator.stopping(run.run_id)
    if stopping:
        status = "Stop requested. Waiting for the active worker to reach a safe stop point."
        status_class = "stopping"
    elif busy:
        status = f"Running {stage.stage_id} at {stage.progress:.0%}. This page refreshes every five seconds."
        status_class = ""
    else:
        status = "Idle. You can render, review, resume a blocked stage, or free ComfyUI model memory."
        status_class = "idle"
    stop_action = (
        f'<form method=post action="/run/{html.escape(run.run_id)}/stop"><input type=hidden name=csrf value="{csrf}">'
        '<button class=danger type=submit>Stop current Studio work</button></form>'
        if busy and not stopping
        else '<button class=danger type=button disabled>No active Studio job</button>'
    )
    memory_action = (
        f'<form method=post action="/run/{html.escape(run.run_id)}/gpu-free"><input type=hidden name=csrf value="{csrf}">'
        '<button class=memory type=submit>Unload ComfyUI models + free VRAM</button></form>'
        if not busy
        else '<button class=memory type=button disabled>Free VRAM after the job stops</button>'
    )
    return (
        '<section class="card console"><div class=console-head><div><h2>Studio control console</h2>'
        '<p class=muted>Controls apply only to the local Darkness Studio job and local ComfyUI service.</p></div>'
        f'<span class=badge>{"stopping" if stopping else "running" if busy else "idle"}</span></div>'
        f'<div class="status-line {status_class}">{html.escape(status)}</div>'
        '<div class=control-grid><section class=control-action><h3>Current work</h3>'
        '<p>Interrupts the tracked ComfyUI workflow and preserves all prior evidence. Resume is always explicit.</p>'
        f'{stop_action}</section><section class=control-action><h3>GPU / model memory</h3>'
        '<p>Unloads ComfyUI models and releases its execution cache. The next render will take longer while models reload.</p>'
        f'{memory_action}</section><section class=control-action><h3>Prompt workspace</h3>'
        '<p>The direct-render form includes a local Clear prompt and seed button. It never deletes stored evidence.</p>'
        '<p class=muted>ComfyUI is a shared local service: avoid unrelated Comfy jobs while stopping a Studio render.</p>'
        '</section></div></section>'
    )


def _run_page(store: StudioStore, coordinator: StudioCoordinator, run_id: str, csrf: str) -> tuple[str, bool]:
    run = store.load(run_id)
    stage = run.stage(run.current_stage)
    evidence, choices = _evidence(run, stage)
    events = store.read_events(run_id)[-30:]
    event_html = "".join(
        f'<div class=event><strong>{html.escape(item["event_type"])}</strong> '
        f'<span class=muted>{html.escape(item["occurred_at"])}</span><br>'
        f'<small>{html.escape(json.dumps(item["payload"], default=str))}</small></div>'
        for item in reversed(events)
    )
    busy = coordinator.busy(run_id) or run.state == "running"
    actions = ""
    if run.state in {"failed", "blocked", "created"} and not coordinator.busy(run_id):
        actions = (
            f'<form method=post action="/run/{html.escape(run_id)}/resume"><input type=hidden name=csrf value="{csrf}">'
            '<button class=primary type=submit>Resume pipeline</button></form>'
        )
    error = f'<pre class=error>{html.escape(stage.error)}</pre>' if stage.error else ""
    body = (
        '<section class="card hero">'
        f"<h1>{html.escape(run.title)}</h1><p>{html.escape(run.description)}</p>"
        f'<p><span class=badge>{html.escape(run.run_id)}</span> <span class=badge>{html.escape(run.state)}</span> '
        f'<span class=badge>concept backend: {html.escape(run.concept_backend)}</span></p>'
        f"{_timeline(run)}<h2>{stage.stage_id} — {html.escape(stage.label)}</h2>"
        f"<p>{html.escape(stage.message)}</p>{error}{actions}</section>"
        '<div class=grid style="margin-top:16px"><section class=card><h2>Qwen production contract</h2>'
        f"{_spec(run)}</section><section class=card><h2>Qwen gate review</h2>{_qwen_review(stage)}</section></div>"
        f"{_studio_controls(run, stage, coordinator, csrf, busy=busy)}"
        f"<h2 style='margin-top:22px'>Evidence</h2>{evidence}"
        f"{_manual_qwen_image_form(run, stage, csrf, busy=busy)}"
        f"{_decision_form(run, stage, csrf, choices)}"
        f'<section class=card style="margin-top:16px"><h2>Run history</h2><div class=events>{event_html}</div></section>'
    )
    return body, busy


def _dashboard(store: StudioStore) -> str:
    runs = store.list()
    cards = "".join(
        '<section class="card run-card">'
        f'<h2><a href="/run/{quote(run.run_id)}">{html.escape(run.title)}</a></h2>'
        f'<p class=muted>{html.escape(run.run_id)} · {html.escape(run.state)} · {html.escape(run.current_stage)}</p>'
        f"<p>{html.escape(run.description[:240])}</p>"
        f'<a class=button href="/run/{quote(run.run_id)}">Open run</a></section>'
        for run in runs
    )
    if not cards:
        cards = '<section class=card><h2>No runs yet</h2><p>Describe any asset; Darkness compiles the production contract.</p><a class="button primary" href=/new>Create asset</a></section>'
    return '<section class="card hero"><h1>Asset production runs</h1><p class=muted>One description in. Qwen acts, critiques, and mediates bounded corrections; deterministic gates preserve evidence and explicit human decisions.</p></section><div class=grid style="margin-top:16px">' + cards + "</div>"


def _available_profiles() -> list[str]:
    directory = profiles_dir()
    if not directory.is_dir():
        return ["simple"]
    names = sorted(p.stem for p in directory.glob("*.toml") if p.stem != "base")
    return names or ["simple"]


def _new_form(csrf: str) -> str:
    profiles = _available_profiles()
    default = "simple" if "simple" in profiles else profiles[0]
    options = "".join(
        f'<option value="{html.escape(name)}"{" selected" if name == default else ""}>{html.escape(name)}</option>'
        for name in profiles
    )
    return (
        '<section class="card hero"><h1>Describe one original asset</h1>'
        '<p>This is the only production input. It may be a character, creature, door, wall, prop, environment, material, or VFX. Include handedness, moving pieces, and required states when they matter; Qwen compiles the rest.</p>'
        '<form method=post action=/runs>'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<label>Description</label><textarea name=description minlength=20 required '
        'placeholder="Examples: an original armored footman with a right-hand sword and left shield; or a worn stone gate with two hinged iron doors and open/close states..."></textarea>'
        f'<label>Configuration profile</label><select name=profile>{options}</select>'
        '<button class=primary type=submit>Compile asset and start</button></form></section>'
    )


def _doctor() -> str:
    localdeploy, localdeploy_detail = _probe("http://127.0.0.1:8000/v1/models")
    comfy, comfy_detail = _probe("http://127.0.0.1:8188/system_stats")
    config = load_local_config()
    configured = sorted(config.workers) if config else []
    return (
        '<section class="card hero"><h1>Local production system</h1><div class=health>'
        f'<span class="{"" if localdeploy else "down"}">Qwen / LocalDeploy: {html.escape(localdeploy_detail)}</span>'
        f'<span class="{"" if comfy else "down"}">ComfyUI: {html.escape(comfy_detail)}</span>'
        f'<span class="{"" if config else "down"}">Darkness config: {"loaded" if config else "missing"}</span>'
        '</div><h2 style="margin-top:18px">Configured deterministic workers</h2>'
        f'<pre>{html.escape(json.dumps(configured, indent=2))}</pre>'
        '<p class=muted>The browser is bound to loopback only. Qwen proposes structured decisions; it never executes code or edits artifacts.</p></section>'
    )


def serve(
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = False,
) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Darkness Studio may bind only to a loopback address")
    store = StudioStore(workspace)
    recovered = store.recover_interrupted_runs()
    if recovered:
        print(f"Darkness Studio recovered interrupted runs: {', '.join(recovered)}")
    coordinator = StudioCoordinator(store)
    csrf = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
            )
            self.end_headers()

        def page(self, title: str, body: str, status: HTTPStatus = HTTPStatus.OK, refresh: bool = False) -> None:
            payload = _page(title, body, refresh=refresh)
            self._headers(status, "text/html; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def redirect(self, path: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", path)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/":
                    self.page("Darkness Studio", _dashboard(store))
                elif path == "/new":
                    self.page("New asset", _new_form(csrf))
                elif path == "/doctor":
                    self.page("System", _doctor())
                elif path.startswith("/run/"):
                    run_id = unquote(path.split("/", 2)[2])
                    body, refresh = _run_page(store, coordinator, run_id, csrf)
                    self.page("Production run", body, refresh=refresh)
                elif path.startswith("/api/run/"):
                    run_id = unquote(path.split("/", 3)[3])
                    payload = store.load(run_id).model_dump_json(indent=2).encode("utf-8")
                    self._headers(HTTPStatus.OK, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path.startswith("/artifact/"):
                    parts = path.split("/", 3)
                    if len(parts) != 4:
                        raise FileNotFoundError("artifact path is incomplete")
                    run_id = unquote(parts[2])
                    relative = unquote(parts[3])
                    target = store.artifact_path(run_id, relative)
                    payload = target.read_bytes()
                    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    self._headers(HTTPStatus.OK, content_type, len(payload))
                    self.wfile.write(payload)
                else:
                    self.page("Not found", "<h1>Not found</h1>", HTTPStatus.NOT_FOUND)
            except (FileNotFoundError, ValueError) as exc:
                self.page("Not found", f'<section class=card><h1 class=error>Error</h1><pre>{html.escape(str(exc))}</pre></section>', HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                path = urlparse(self.path).path
                values = _form(self)
                if not secrets.compare_digest(values.get("csrf", ""), csrf):
                    raise ValueError("invalid form token")
                if path == "/runs":
                    description = values.get("description", "").strip()
                    if len(description) < 20:
                        raise ValueError("description must contain at least 20 characters")
                    profile = values.get("profile", "simple").strip() or "simple"
                    resolved = resolve_settings(profile=profile)
                    run_id = _slug()
                    store.create(
                        run_id,
                        description,
                        {**studio_overrides(resolved), "profile": profile},
                    )
                    coordinator.submit(run_id)
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/decision"):
                    run_id = unquote(path.split("/")[2])
                    overrides_raw = values.get("overrides", "").strip()
                    overrides = None
                    if overrides_raw:
                        try:
                            overrides = json.loads(overrides_raw)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Overrides must be valid JSON (for example {{\"seed\": 42}}): {exc.msg}"
                            ) from exc
                        if not isinstance(overrides, dict):
                            raise ValueError(
                                'Overrides must be a JSON object, for example {"seed": 42}.'
                            )
                    store.decide(
                        run_id,
                        values["stage_id"],
                        values["decision"],
                        values.get("comment", ""),
                        values.get("selected_evidence_id") or None,
                        overrides=overrides,
                        target_stage_id=values.get("target_stage_id") or None,
                    )
                    coordinator.submit(run_id)
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/qwen-image"):
                    run_id = unquote(path.split("/")[2])
                    prompt = values.get("prompt", "").strip()
                    raw_seed = values.get("seed", "").strip()
                    try:
                        seed = int(raw_seed) if raw_seed else None
                    except ValueError as exc:
                        raise ValueError("seed must be a whole non-negative number") from exc
                    if not coordinator.submit_manual_qwen_image(run_id, prompt, seed=seed):
                        raise ValueError("another Studio job is already running for this asset")
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/stop"):
                    run_id = unquote(path.split("/")[2])
                    accepted, message = coordinator.stop(run_id)
                    if not accepted:
                        raise ValueError(message)
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/gpu-free"):
                    run_id = unquote(path.split("/")[2])
                    released, message = coordinator.release_comfy_memory(run_id)
                    if not released:
                        raise ValueError(message)
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/resume"):
                    run_id = unquote(path.split("/")[2])
                    coordinator.submit(run_id)
                    self.redirect("/run/" + quote(run_id))
                else:
                    self.page("Not found", "<h1>Not found</h1>", HTTPStatus.NOT_FOUND)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                self.page(
                    "Darkness Studio error",
                    f'<section class=card><h1 class=error>Could not apply that action</h1><pre>{html.escape(str(exc))}</pre><a class=button href="/">Back to runs</a></section>',
                    HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"Darkness Studio {self.address_string()}: {format % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Darkness Studio: {url}")
    if open_browser:
        webbrowser.open(url + "/new")
    try:
        server.serve_forever()
    finally:
        coordinator.close()
        server.server_close()
