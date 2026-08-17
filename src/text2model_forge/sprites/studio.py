from __future__ import annotations

import html
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .comfy import build_sdxl_workflow, generate
from .core import ForgeError, approve_artifact, asset_root, build_prompt_pack, create_asset, load_asset, project_status, read_json, update_asset_brief, write_json


STYLE = """
:root{color-scheme:dark;--bg:#10161a;--panel:#1b2429;--line:#39474f;--text:#eee4d2;--muted:#9ca7a8;--ember:#d75a2b;--steel:#31556b;--ok:#62b870}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#0c1215,#182126);color:var(--text);font:15px system-ui,sans-serif}
header{padding:24px 32px;border-bottom:1px solid var(--line);background:#11191ddd;position:sticky;top:0;z-index:2}h1,h2,h3{margin:0 0 14px}h1{letter-spacing:.08em}.wrap{max-width:1180px;margin:auto;padding:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;box-shadow:0 12px 30px #0004}.stage{display:inline-block;padding:5px 8px;margin:3px;border-radius:8px;background:#263238;color:var(--muted)}.stage.ok{background:#25432e;color:#ccebd3}.stage.next{outline:1px solid var(--ember);color:#ffd7c8}label{display:block;color:var(--muted);margin:10px 0 5px}input,select,textarea{width:100%;padding:10px;border:1px solid var(--line);border-radius:7px;background:#10171b;color:var(--text)}textarea{min-height:90px}button,.button{display:inline-block;margin-top:14px;padding:10px 16px;border:0;border-radius:7px;background:var(--steel);color:white;text-decoration:none;cursor:pointer}button.primary{background:var(--ember)}code,pre{background:#0c1114;border:1px solid #2d383e;border-radius:7px}code{padding:2px 5px}pre{padding:12px;white-space:pre-wrap;overflow:auto}.muted{color:var(--muted)}.error{color:#ff9f8d}.oktext{color:var(--ok)}nav a{color:#c7dce7;margin-right:18px;text-decoration:none}
"""


def page(title: str, body: str) -> bytes:
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body><header><h1>TEXT2MODEL_FORGE STUDIO</h1><nav><a href="/">Dashboard</a><a href="/new">New asset</a><a href="/doctor">Setup</a></nav></header><main class="wrap">{body}</main></body></html>"""
    return document.encode("utf-8")


class StudioState:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="text2model-forge-gpu")
        self.jobs: dict[str, dict[str, object]] = {}
        self.lock = threading.Lock()

    def submit_generation(self, values: dict[str, str]) -> str:
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {"state": "queued", "asset_id": values["asset_id"], "mode": values["mode"]}

        def work() -> None:
            with self.lock:
                self.jobs[job_id]["state"] = "running"
            try:
                outputs = generate(
                    self.workspace,
                    values["asset_id"],
                    values["mode"],
                    values["checkpoint"],
                    int(values["seed"]),
                    values.get("comfy_url", "http://127.0.0.1:8188"),
                    900,
                )
                result: dict[str, object] = {"state": "complete", "outputs": [str(path) for path in outputs]}
            except Exception as error:  # captured for the local operator dashboard
                result = {"state": "failed", "error": str(error)}
            with self.lock:
                self.jobs[job_id].update(result)

        self.executor.submit(work)
        return job_id


def form_values(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    length = int(handler.headers.get("Content-Length", "0"))
    parsed = parse_qs(handler.rfile.read(length).decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def dashboard(state: StudioState) -> str:
    status = project_status(state.workspace)
    cards = []
    for asset in status["assets"]:
        approved = set(asset["approved_stages"])
        stages = "".join(
            f'<span class="stage {"ok" if stage in approved else "next" if stage == asset["next_stage"] else ""}">{html.escape(stage)}</span>'
            for stage in ("brief", "concept", "turnaround", "master", "actions", "sheets", "engine_capture", "license")
        )
        cards.append(f"""<section class="card"><h2>{html.escape(asset['display_name'])}</h2><p class="muted">{html.escape(asset['asset_id'])} · {html.escape(asset['kind'])}</p><div>{stages}</div><a class="button" href="/asset/{html.escape(asset['asset_id'])}">Open production card</a></section>""")
    empty = '<section class="card"><h2>No assets yet</h2><p class="muted">Create the first structured production brief.</p><a class="button" href="/new">New asset</a></section>'
    body = "".join(cards) if cards else empty
    return f'<h2>{html.escape(status["project_id"])}</h2><div class="grid">{body}</div>'


def new_asset_form() -> str:
    kinds = "".join(f'<option value="{kind}">{kind}</option>' for kind in ("character", "creature", "building", "prop", "vfx", "icon", "background"))
    return f"""<section class="card"><h2>New production asset</h2><form method="post" action="/assets"><label>Asset ID (snake_case)</label><input name="asset_id" required pattern="[a-z][a-z0-9_]*"><label>Display name</label><input name="name" required><label>Kind</label><select name="kind">{kinds}</select><label>Gameplay role</label><textarea name="role" required placeholder="What must the player understand at gameplay size?"></textarea><label>Faction / palette group</label><input name="faction" value="neutral"><label>Primary action</label><select name="primary_action"><option>attack</option><option>shoot</option><option>cast</option><option>none</option></select><button class="primary" type="submit">Create asset</button></form></section>"""


def asset_page(state: StudioState, asset_id: str) -> str:
    asset = load_asset(state.workspace, asset_id)
    prompt_path = asset_root(state.workspace, asset_id) / "prompt-pack.json"
    prompts = read_json(prompt_path) if prompt_path.exists() else None
    prompt_block = ""
    if prompts:
        prompt_block = "".join(f"<h3>{html.escape(name)}</h3><pre>{html.escape(value)}</pre>" for name, value in prompts["prompts"].items())
    else:
        prompt_block = '<p class="muted">No compiled prompt pack yet.</p>'
    brief = asset.get("brief", {})
    line_value = lambda name: html.escape("\n".join(brief.get(name, [])))
    text_value = lambda name: html.escape(str(brief.get(name, "")))
    return f"""<div class="grid"><section class="card"><h2>{html.escape(asset['display_name'])}</h2><p class="muted">{html.escape(asset['kind'])} · {html.escape(asset['role'])}</p><form method="post" action="/brief"><input type="hidden" name="asset_id" value="{html.escape(asset_id)}"><label>Silhouette</label><textarea name="silhouette">{text_value('silhouette')}</textarea><label>Equipment (one per line)</label><textarea name="equipment">{line_value('equipment')}</textarea><label>Materials (one per line)</label><textarea name="materials">{line_value('materials')}</textarea><label>Palette notes</label><textarea name="palette_notes">{text_value('palette_notes')}</textarea><label>Must have (one per line)</label><textarea name="must_have">{line_value('must_have')}</textarea><label>Must not have (one per line)</label><textarea name="must_not_have">{line_value('must_not_have')}</textarea><button type="submit">Save brief</button></form><form method="post" action="/prompt"><input type="hidden" name="asset_id" value="{html.escape(asset_id)}"><button type="submit">Compile prompt pack</button></form></section><section class="card"><h2>Local generation</h2><form method="post" action="/generate"><input type="hidden" name="asset_id" value="{html.escape(asset_id)}"><label>Mode</label><select name="mode"><option>concept</option><option>turnaround</option><option>texture</option><option>mesh_seed</option></select><label>Checkpoint filename</label><input name="checkpoint" value="sd_xl_base_1.0.safetensors" required><label>Seed</label><input name="seed" type="number" value="1001" required><label>ComfyUI URL</label><input name="comfy_url" value="http://127.0.0.1:8188" required><button class="primary" type="submit">Queue generation</button></form><p class="muted">One GPU job at a time on the 8 GB profile.</p><hr><h2>Approve artifact</h2><form method="post" action="/approve"><input type="hidden" name="asset_id" value="{html.escape(asset_id)}"><label>Stage</label><select name="stage">{''.join(f'<option>{stage}</option>' for stage in ('brief','concept','turnaround','master','actions','sheets','engine_capture','license'))}</select><label>Artifact path</label><input name="artifact" required><label>Reviewer</label><input name="reviewer" required><label>Notes</label><textarea name="notes"></textarea><button type="submit">Approve current hash</button></form></section></div><section class="card" style="margin-top:18px"><h2>Prompt pack</h2>{prompt_block}</section>"""


def job_page(state: StudioState, job_id: str) -> str:
    with state.lock:
        job = dict(state.jobs.get(job_id, {"state": "unknown", "error": "job not found"}))
    refresh = '<meta http-equiv="refresh" content="2">' if job.get("state") in {"queued", "running"} else ""
    return refresh + f'<section class="card"><h2>Generation job</h2><pre>{html.escape(json.dumps(job, indent=2))}</pre><a class="button" href="/">Dashboard</a></section>'


def serve(workspace: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    state = StudioState(workspace)
    load_project = project_status(state.workspace)  # validate before binding a port
    del load_project

    class Handler(BaseHTTPRequestHandler):
        def send_page(self, title: str, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = page(title, body)
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def redirect(self, path: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", path)
            self.end_headers()

        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/":
                    self.send_page("Text2Model Forge Studio", dashboard(state))
                elif path == "/new":
                    self.send_page("New asset", new_asset_form())
                elif path.startswith("/asset/"):
                    self.send_page("Asset", asset_page(state, path.split("/", 2)[2]))
                elif path.startswith("/job/"):
                    self.send_page("Job", job_page(state, path.split("/", 2)[2]))
                elif path == "/doctor":
                    self.send_page("Setup", '<section class="card"><h2>Local stack</h2><p>Run <code>.\\scripts\\forge.ps1 doctor</code>, then start ComfyUI on <code>127.0.0.1:8188</code>.</p></section>')
                else:
                    self.send_page("Not found", "<h2>Not found</h2>", HTTPStatus.NOT_FOUND)
            except ForgeError as error:
                self.send_page("Text2Model Forge error", f'<section class="card"><h2 class="error">Error</h2><pre>{html.escape(str(error))}</pre></section>', HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            try:
                path = urlparse(self.path).path
                values = form_values(self)
                if path == "/assets":
                    create_asset(state.workspace, values["asset_id"], values["name"], values["kind"], values["role"], values.get("faction", "neutral"), values.get("primary_action", "attack"))
                    self.redirect("/asset/" + values["asset_id"])
                elif path == "/prompt":
                    build_prompt_pack(state.workspace, values["asset_id"])
                    self.redirect("/asset/" + values["asset_id"])
                elif path == "/brief":
                    lines = lambda name: [item.strip() for item in values.get(name, "").splitlines() if item.strip()]
                    update_asset_brief(state.workspace, values["asset_id"], {
                        "silhouette": values.get("silhouette", ""),
                        "equipment": lines("equipment"),
                        "materials": lines("materials"),
                        "palette_notes": values.get("palette_notes", ""),
                        "must_have": lines("must_have"),
                        "must_not_have": lines("must_not_have"),
                    })
                    self.redirect("/asset/" + values["asset_id"])
                elif path == "/approve":
                    approve_artifact(state.workspace, values["asset_id"], values["stage"], Path(values["artifact"]), values["reviewer"], values.get("notes", ""))
                    self.redirect("/asset/" + values["asset_id"])
                elif path == "/generate":
                    job_id = state.submit_generation(values)
                    self.redirect("/job/" + job_id)
                else:
                    self.send_page("Not found", "<h2>Not found</h2>", HTTPStatus.NOT_FOUND)
            except (ForgeError, KeyError, ValueError) as error:
                self.send_page("Text2Model Forge error", f'<section class="card"><h2 class="error">Error</h2><pre>{html.escape(str(error))}</pre></section>', HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            print(f"Studio {self.address_string()}: {format % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Text2Model Forge Studio: http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        state.executor.shutdown(wait=False, cancel_futures=True)
        server.server_close()
