"""Local-first browser UI for Text2Model Forge Studio runs and human gates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
from typing import Any, NamedTuple
from urllib.parse import parse_qs, quote, unquote, urlparse
import urllib.request
import uuid
import webbrowser

from .config import load_local_config, worker_binding
from .hardware import detect_hardware, recommend_stack
from .manifests import load_manifests, preflight
from .settings import profiles_dir, resolve_settings, studio_overrides
from .studio_models import utc_now
from .studio_pipeline import StudioCoordinator
from .studio_store import StudioStore


STYLE = """
:root{color-scheme:dark;--bg:#0c1114;--panel:#182126;--panel2:#202c32;--line:#3a4a51;--text:#eee6d7;--muted:#a4afb0;--accent:#dc6837;--steel:#45728b;--ok:#62bd79;--bad:#e26d62;--wait:#d5a746}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% -20%,#28373c,#0c1114 46%);color:var(--text);font:15px/1.45 system-ui,sans-serif}header{position:sticky;top:0;z-index:4;padding:18px 28px;background:#0d1417ee;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:24px}header h1{font-size:20px;letter-spacing:.12em;margin:0}header a{color:#c8dce5;text-decoration:none}.wrap{max-width:1320px;margin:auto;padding:26px}h1,h2,h3{margin:0 0 12px}p{margin:8px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.card{background:linear-gradient(145deg,var(--panel),#141c20);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:0 12px 30px #0005}.hero{padding:28px}.muted{color:var(--muted)}.error{color:#ff9e95}.good{color:#a7e7b6}.warning{color:#f0cf79}.timeline{display:grid;grid-template-columns:repeat(auto-fit,minmax(88px,1fr));gap:6px;margin:18px 0;overflow:auto}.stage{min-height:88px;border:1px solid var(--line);border-radius:9px;padding:9px;background:#12191d}.stage strong{display:block}.stage small{display:block;color:var(--muted)}.stage.approved{border-color:#3d8050;background:#17281c}.stage.skipped{opacity:.56;border-style:dashed}.stage.running,.stage.queued{border-color:var(--steel)}.stage.awaiting_review{border-color:var(--wait);background:#292315}.stage.rejected,.stage.failed{border-color:var(--bad);background:#2a1918}.stage.blocked{border-color:#8a6e3a}.bar{height:5px;background:#0b1012;border-radius:4px;margin-top:10px;overflow:hidden}.bar span{display:block;height:100%;background:var(--accent)}label{display:block;color:var(--muted);margin:10px 0 5px}textarea,input{width:100%;padding:11px;border:1px solid var(--line);border-radius:7px;background:#0d1417;color:var(--text)}textarea{min-height:130px;resize:vertical}button,.button{display:inline-block;border:0;border-radius:7px;padding:10px 16px;margin:10px 8px 0 0;background:var(--steel);color:white;text-decoration:none;cursor:pointer}.primary{background:var(--accent)!important}.reject,.danger{background:#9d4039!important}.memory{background:#735c24!important}.secondary{background:#34454c!important}.evidence{position:relative}.evidence img{width:100%;max-height:620px;object-fit:contain;background:#0a0f11;border:1px solid var(--line);border-radius:8px}.choice{display:flex;gap:8px;align-items:center;margin:8px 0}.choice input{width:auto}.badge{display:inline-block;padding:3px 7px;border-radius:999px;background:#29373d;color:#cbd8dc;font-size:12px}.recommended{background:#49381c;color:#ffe09a}pre,code{background:#0b1012;border:1px solid #2b373c;border-radius:6px}pre{padding:12px;white-space:pre-wrap;overflow:auto}code{padding:2px 5px}.review ul{margin-top:5px}.events{max-height:330px;overflow:auto}.event{border-left:2px solid var(--line);padding:5px 0 5px 12px;margin:4px 0}.run-card h2 a{color:var(--text);text-decoration:none}.actions{display:flex;gap:8px;flex-wrap:wrap}.health{display:flex;gap:8px;flex-wrap:wrap}.health span{padding:5px 9px;border-radius:8px;background:#253239}.health .down{background:#4a2220}.console{border-color:#5a707a;background:linear-gradient(135deg,#1b2a31,#142025)}.console-head{display:flex;justify-content:space-between;gap:12px;align-items:start}.console-head h2{margin:0}.control-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:10px;margin-top:15px}.control-action{border:1px solid var(--line);border-radius:9px;padding:12px;background:#10191d}.control-action h3{font-size:14px;margin-bottom:5px}.control-action p{font-size:13px;min-height:39px}.status-line{margin:14px 0 0;padding:10px 12px;border-radius:8px;background:#10191d;border-left:3px solid var(--steel)}.status-line.stopping{border-left-color:var(--bad);background:#2b1b1a}.status-line.idle{border-left-color:var(--ok)}.prompt-meta{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.prompt-tools{display:flex;align-items:center;gap:2px;flex-wrap:wrap}.prompt-tools button{margin-top:10px}button:disabled{cursor:not-allowed;opacity:.5}
.timeline a.stage{color:inherit;text-decoration:none;display:block}.timeline a.stage:hover{border-color:var(--accent)}.stage.active{outline:2px solid var(--accent);outline-offset:1px}.badge.gate{background:#3b2f47;color:#dcc7f0;margin-top:4px}.badge.na{background:#2a3238;color:#8f9ba0}
.attempt{display:flex;align-items:baseline;gap:10px;margin:20px 0 8px}.attempt h3{margin:0}.evidence details{margin-top:8px}.evidence summary{cursor:pointer;color:var(--muted);font-size:13px}.metrics{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 0}
.decisions{list-style:none;padding:0;margin:6px 0 0}.decisions li{border-left:2px solid var(--line);padding:6px 0 6px 12px;margin:6px 0}.decisions .approve{border-left-color:var(--ok)}.decisions .reject,.decisions .rollback{border-left-color:var(--bad)}.decisions .edit,.decisions .retry{border-left-color:var(--steel)}.decisions .skip{border-left-color:var(--muted)}
.run-card .bar{margin:12px 0 6px}.needs{background:var(--wait)!important;color:#1b1405!important}.crumb{display:flex;gap:10px;align-items:center;margin-bottom:12px}.crumb a{color:#c8dce5}
.bar.overall{height:12px;margin:8px 0 4px}.bar span{background:linear-gradient(90deg,var(--accent),#f0a154);transition:width .35s ease}.progress-label{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:13px}select{width:100%;padding:11px;border:1px solid var(--line);border-radius:7px;background:#0d1417;color:var(--text)}.options{margin-top:18px;border:1px solid var(--line);border-radius:9px;padding:12px;background:#10191d}.options summary{cursor:pointer;font-weight:650}.option-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:4px 14px}.service-status{margin-top:10px;padding:9px 11px;border-radius:7px;background:#0d1417}.service-status.good{border-left:3px solid var(--ok)}.service-status.warning{border-left:3px solid var(--wait)}
.bar.gpu span{background:linear-gradient(90deg,var(--steel),#75bdd9)}
.glb-preview{display:block;width:100%;height:420px;touch-action:none;cursor:grab;background:radial-gradient(circle,#263238,#080c0e 70%);border:1px solid var(--line);border-radius:8px}.glb-preview:active{cursor:grabbing}.viewer-note{font-size:12px;color:var(--muted)}
@media(max-width:800px){.wrap{padding:15px}header{padding:14px}.hero{padding:18px}}
"""


def _page(title: str, body: str) -> bytes:
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        '<body><header><h1>Text2Model Forge Studio</h1><a href="/">Runs</a><a href="/new">New asset</a>'
        '<a href="/golden">Golden corpus</a><a href="/doctor">System</a></header><main class="wrap">'
        f'{body}</main><script src="/static/glb-viewer.js" defer></script></body></html>'
    ).encode("utf-8")


# Polls /api/run/<id> while a stage is running and updates the active stage's
# progress bar and status text in place. The CSP has no 'unsafe-inline' on
# script-src (default-src 'self' covers same-origin <script src>, not inline
# blocks), so this must be served as a real same-origin file, not embedded --
# see build_server()'s CSP header. A full <meta http-equiv=refresh> reload
# used to do this job, but it reset scroll position every five seconds on any
# long-running stage; this only reloads the page when run.state or the
# current stage actually changes, and otherwise just updates numbers in place.
STUDIO_JS = """
(function () {
  function percentage(value) { return Math.round(Math.max(0, Math.min(1, value)) * 100); }

  function overallProgress(run) {
    var applicable = run.stages.filter(function (stage) { return stage.applicable; });
    if (!applicable.length) return 0;
    return applicable.reduce(function (sum, stage) { return sum + stage.progress; }, 0) / applicable.length;
  }

  function initRunProgress() {
    var hero = document.querySelector('[data-run-id]');
    if (!hero) return;
    var runId = hero.getAttribute('data-run-id');
    var knownState = hero.getAttribute('data-state');
    var knownStage = hero.getAttribute('data-current-stage');

    function apply(run) {
      if (run.state !== knownState || run.current_stage !== knownStage) {
        window.location.reload();
        return;
      }
      var stage = null;
      for (var i = 0; i < run.stages.length; i++) {
        if (run.stages[i].stage_id === run.current_stage) { stage = run.stages[i]; break; }
      }
      if (!stage) return;
      var stagePercent = percentage(stage.progress);
      var bar = document.getElementById('active-stage-bar');
      if (bar) {
        bar.style.width = stagePercent + '%';
        bar.parentNode.setAttribute('aria-valuenow', String(stagePercent));
      }
      var overallPercent = percentage(overallProgress(run));
      var overall = document.getElementById('overall-run-bar');
      if (overall) {
        overall.style.width = overallPercent + '%';
        overall.parentNode.setAttribute('aria-valuenow', String(overallPercent));
      }
      var overallLabel = document.getElementById('overall-run-label');
      if (overallLabel) overallLabel.textContent = overallPercent + '% overall';
      var message = document.getElementById('stage-message');
      if (message) message.textContent = stage.message;
      var work = document.getElementById('stage-work');
      if (work) {
        var count = stage.progress_total ? ' · ' + stage.progress_current + '/' + stage.progress_total + ' ' + stage.progress_unit : '';
        work.textContent = stage.progress_phase + count;
      }
      var gpu = document.getElementById('gpu-run-bar');
      if (gpu && stage.gpu_total_gb) {
        var gpuUsed = stage.gpu_used_gb || 0;
        var gpuPercent = Math.round(gpuUsed / stage.gpu_total_gb * 100);
        gpu.style.width = gpuPercent + '%';
        gpu.parentNode.setAttribute('aria-valuenow', String(gpuPercent));
      }
      var gpuLabel = document.getElementById('gpu-run-label');
      if (gpuLabel && stage.gpu_total_gb) {
        gpuLabel.textContent = (stage.gpu_used_gb || 0).toFixed(2) + '/' + stage.gpu_total_gb.toFixed(2) + ' GiB';
      }
      var status = document.getElementById('status-line');
      if (status && stage.state === 'running') {
        status.textContent = 'Running ' + stage.stage_id + ' at ' + stagePercent + '%.';
      }
    }

    function poll() {
      fetch('/api/run/' + encodeURIComponent(runId), { cache: 'no-store' })
        .then(function (response) { return response.json(); })
        .then(function (run) { apply(run); window.setTimeout(poll, 4000); })
        .catch(function () { window.setTimeout(poll, 8000); });
    }

    window.setTimeout(poll, 4000);
  }

  function fillDatalist(id, values) {
    var list = document.getElementById(id);
    if (!list) return;
    while (list.firstChild) list.removeChild(list.firstChild);
    values.forEach(function (value) {
      var option = document.createElement('option');
      option.value = value;
      list.appendChild(option);
    });
  }

  function initSetupOptions() {
    var form = document.querySelector('[data-setup-options]');
    if (!form) return;
    var profile = form.querySelector('[name="profile"]');
    var status = document.getElementById('setup-service-status');

    function load() {
      if (status) {
        status.className = 'service-status';
        status.textContent = 'Checking local AI services and installed models...';
      }
      fetch('/api/setup/options?profile=' + encodeURIComponent(profile.value), { cache: 'no-store' })
        .then(function (response) { return response.json(); })
        .then(function (data) {
          Object.keys(data.defaults).forEach(function (name) {
            var field = form.querySelector('[name="' + name + '"]');
            if (!field) return;
            if (field.tagName === 'SELECT' && field.options.length && field.options[0].value === '') {
              field.options[0].textContent = 'Profile default: ' + data.defaults[name];
            } else {
              field.setAttribute('placeholder', 'Profile default: ' + data.defaults[name]);
            }
          });
          fillDatalist('installed-checkpoints', data.checkpoints);
          fillDatalist('installed-review-models', data.review_models);
          if (status) {
            var ready = data.services.comfyui.ready && data.services.reviewer.ready;
            status.className = 'service-status ' + (ready ? 'good' : 'warning');
            status.textContent =
              'ComfyUI: ' + data.services.comfyui.detail + ' (' + data.checkpoints.length +
              ' checkpoints, ' + data.diffusion_models.length + ' diffusion models). Reviewer: ' +
              data.services.reviewer.detail + ' (' + data.review_models.length + ' models). Qwen Image 2512: ' +
              (data.qwen_image_2512_ready ? 'ready.' : 'required files not all detected.') + ' Z-Image Turbo: ' +
              (data.z_image_turbo_ready ? 'ready.' : 'required files not all detected.');
          }
        })
        .catch(function () {
          if (status) {
            status.className = 'service-status warning';
            status.textContent = 'Could not inspect local AI services. You can still use profile defaults or type model names.';
          }
        });
    }

    profile.addEventListener('change', load);
    load();
  }

  initRunProgress();
  initSetupOptions();
})();
"""


# Dependency-free local GLB preview. It reads mesh POSITION/index accessors and
# draws a sampled, depth-sorted shaded view on Canvas 2D. This is intentionally
# a review aid rather than a replacement for Blender or the D10 runtime gate.
GLB_VIEWER_JS = r"""
(function () {
  function components(type) { return {SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16}[type] || 1; }
  function componentBytes(type) { return {5120:1,5121:1,5122:2,5123:2,5125:4,5126:4}[type]; }
  function readComponent(view, offset, type) {
    if (type===5120) return view.getInt8(offset); if (type===5121) return view.getUint8(offset);
    if (type===5122) return view.getInt16(offset,true); if (type===5123) return view.getUint16(offset,true);
    if (type===5125) return view.getUint32(offset,true); return view.getFloat32(offset,true);
  }
  function parseGlb(buffer) {
    var file=new DataView(buffer); if(file.getUint32(0,true)!==0x46546c67||file.getUint32(4,true)!==2) throw Error('Only GLB 2.0 is supported');
    var offset=12,json=null,binary=null;
    while(offset+8<=buffer.byteLength){var length=file.getUint32(offset,true),type=file.getUint32(offset+4,true);offset+=8;
      if(type===0x4e4f534a) json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,offset,length)));
      if(type===0x004e4942) binary=buffer.slice(offset,offset+length); offset+=length;}
    if(!json||!binary) throw Error('GLB has no JSON or binary chunk');
    function accessor(index){var a=json.accessors[index],b=json.bufferViews[a.bufferView],count=components(a.type),size=componentBytes(a.componentType),stride=b.byteStride||count*size;
      var data=new DataView(binary),start=(b.byteOffset||0)+(a.byteOffset||0),out=new Array(a.count*count);
      for(var i=0;i<a.count;i++) for(var c=0;c<count;c++) out[i*count+c]=readComponent(data,start+i*stride+c*size,a.componentType); return out;}
    var triangles=[];
    (json.meshes||[]).forEach(function(mesh){(mesh.primitives||[]).forEach(function(p){if(!p.attributes||p.attributes.POSITION===undefined||(p.mode!==undefined&&p.mode!==4))return;
      var xyz=accessor(p.attributes.POSITION),indices=p.indices===undefined?null:accessor(p.indices),count=indices?indices.length:xyz.length/3;
      var step=Math.max(3,Math.ceil(count/150000/3)*3);
      for(var i=0;i+2<count;i+=step){var ia=indices?indices[i]:i,ib=indices?indices[i+1]:i+1,ic=indices?indices[i+2]:i+2;
        triangles.push([[xyz[ia*3],xyz[ia*3+1],xyz[ia*3+2]],[xyz[ib*3],xyz[ib*3+1],xyz[ib*3+2]],[xyz[ic*3],xyz[ic*3+1],xyz[ic*3+2]]]);}});});
    if(!triangles.length) throw Error('No triangle POSITION data was found'); return triangles;
  }
  function start(canvas){var context=canvas.getContext('2d'),triangles,yaw=-0.6,pitch=-0.25,zoom=0.82,drag=null,status='Loading GLB preview…';
    function resize(){var ratio=Math.min(devicePixelRatio||1,2),rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,rect.width*ratio);canvas.height=Math.max(1,rect.height*ratio);draw();}
    function draw(){var w=canvas.width,h=canvas.height;context.clearRect(0,0,w,h);context.fillStyle='#9fb0b7';context.font=(14*(devicePixelRatio||1))+'px system-ui';
      if(!triangles){context.fillText(status,18,28);return;} var points=[],min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
      triangles.forEach(function(t){t.forEach(function(p){for(var k=0;k<3;k++){min[k]=Math.min(min[k],p[k]);max[k]=Math.max(max[k],p[k]);}});});
      var center=[(min[0]+max[0])/2,(min[1]+max[1])/2,(min[2]+max[2])/2],extent=Math.max(max[0]-min[0],max[1]-min[1],max[2]-min[2],1e-6),cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),scale=Math.min(w,h)*zoom/extent;
      function project(p){var x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],rx=cy*x+sy*z,rz=-sy*x+cy*z,ry=cx*y-sx*rz;rz=sx*y+cx*rz;return [w/2+rx*scale,h/2-ry*scale,rz];}
      triangles.forEach(function(t){var a=project(t[0]),b=project(t[1]),c=project(t[2]),cross=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);if(Math.abs(cross)<0.01)return;points.push({p:[a,b,c],z:(a[2]+b[2]+c[2])/3,light:Math.max(.16,Math.min(.9,.45+cross/(Math.abs(cross)+9000)*.4))});});
      points.sort(function(a,b){return a.z-b.z;});points.forEach(function(t){var shade=Math.round(95+t.light*105);context.beginPath();context.moveTo(t.p[0][0],t.p[0][1]);context.lineTo(t.p[1][0],t.p[1][1]);context.lineTo(t.p[2][0],t.p[2][1]);context.closePath();context.fillStyle='rgb('+Math.round(shade*.72)+','+Math.round(shade*.86)+','+shade+')';context.fill();context.strokeStyle='rgba(8,14,17,.18)';context.stroke();});}
    canvas.addEventListener('pointerdown',function(e){drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',function(e){if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw();});
    canvas.addEventListener('pointerup',function(){drag=null;});canvas.addEventListener('wheel',function(e){e.preventDefault();zoom=Math.max(.2,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
    fetch(canvas.dataset.glbSrc,{cache:'no-store'}).then(function(r){if(!r.ok)throw Error('HTTP '+r.status);return r.arrayBuffer();}).then(function(b){triangles=parseGlb(b);resize();}).catch(function(e){status='Preview unavailable: '+e.message;draw();});
    addEventListener('resize',resize);resize();
  }
  document.querySelectorAll('canvas[data-glb-src]').forEach(start);
})();
"""


def _probe(url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200, f"HTTP {response.status}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _json_get(url: str) -> tuple[Any | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return json.load(response), f"HTTP {response.status}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _form(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        handler.close_connection = True
        raise ValueError("the request had no readable length") from exc
    if length < 0 or length > 1_000_000:
        # The body is deliberately not read, so this connection can no longer
        # be trusted to start the next request at a message boundary. Closing
        # it keeps an oversized form from desynchronising a keep-alive
        # connection and corrupting whatever request follows.
        handler.close_connection = True
        raise ValueError("form is too large")
    parsed = parse_qs(handler.rfile.read(length).decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def _slug() -> str:
    from datetime import datetime, timezone

    return "asset-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5]


def _artifact_url(run_id: str, relative_path: str) -> str:
    return f"/artifact/{quote(run_id)}/{quote(relative_path, safe='/')}"


def _stage_url(run_id: str, stage_id: str) -> str:
    return f"/run/{quote(run_id)}/stage/{quote(stage_id)}"


def _run_progress(run) -> float:
    """Normalized progress across stages that apply to this asset.

    A stage's own fractional progress counts while it is running, instead of
    making the dashboard look frozen until a whole stage settles. D0 is the
    only stage that can change applicability; excluded stages are therefore
    omitted once the compiled contract identifies them.
    """
    applicable = [stage for stage in run.stages if stage.applicable]
    if not applicable:
        return 0.0
    return sum(stage.progress for stage in applicable) / len(applicable)


def _progress_bar(
    value: float,
    *,
    label: str,
    bar_id: str | None = None,
    extra_class: str = "",
) -> str:
    percent = round(max(0.0, min(1.0, value)) * 100)
    identifier = f' id="{html.escape(bar_id)}"' if bar_id else ""
    return (
        f'<div class="bar overall {html.escape(extra_class)}" role="progressbar" aria-label="{html.escape(label)}" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
        f'<span{identifier} style="width:{percent}%"></span></div>'
    )


def _timeline(run, *, active: str | None = None, mark_progress_id: bool = False) -> str:
    """Every stage as a link to its own detail page.

    The timeline used to be inert, which meant an approved stage's evidence
    became unreachable in the browser the moment the run moved past it --
    the run page only ever renders the *current* stage.

    `mark_progress_id` gives the active tile's bar a stable id so STUDIO_JS
    can update its width without a full page reload. Only the run page (where
    `active` really is the run's live current_stage) passes it; the stage
    detail page's `active` is just whichever stage the human is looking at.
    """
    tiles = []
    for item in run.stages:
        classes = "stage " + item.state + (" active" if item.stage_id == active else "")
        if item.applicable:
            state_badge = f'<span class="badge">{html.escape(item.state.replace("_", " "))}</span>'
        else:
            state_badge = '<span class="badge na">not applicable</span>'
        gate = '<span class="badge gate">human gate</span>' if item.gate_required else ""
        bar_id = ' id="active-stage-bar"' if mark_progress_id and item.stage_id == active else ""
        percent = round(item.progress * 100)
        tiles.append(
            f'<a class="{classes}" href="{_stage_url(run.run_id, item.stage_id)}">'
            f"<strong>{item.stage_id}</strong><small>{html.escape(item.label)}</small>"
            f'{state_badge}{gate}<div class="bar" role="progressbar" aria-label="{html.escape(item.stage_id)} progress" '
            f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
            f'<span{bar_id} style="width:{percent}%"></span></div></a>'
        )
    columns = max(len(run.stages), 1)
    return (
        f'<div class="timeline" style="grid-template-columns:repeat({columns},minmax(88px,1fr))">'
        + "".join(tiles)
        + "</div>"
    )


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
    return _one_qwen_review(stage.qwen_reviews[-1])


def _one_qwen_review(review) -> str:
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


# Metrics every evidence card carries for bookkeeping rather than for the
# reader; they are still in the full JSON, just not worth a badge each.
_ROUTINE_METRICS = frozenset({"iteration", "selectable", "role"})


def _metric_badges(metrics: dict) -> str:
    """The handful of values worth reading at a glance, above the full JSON."""
    interesting = [
        (key, value)
        for key, value in metrics.items()
        if key not in _ROUTINE_METRICS and value is not None and value != ""
    ]
    if not interesting:
        return ""
    shown = "".join(
        f"<span class=badge>{html.escape(str(key))} {html.escape(str(value))}</span>"
        for key, value in interesting[:6]
    )
    return f'<p class="metrics">{shown}</p>'


def _evidence_iteration(item, stage) -> int:
    """Which attempt produced this artefact.

    Evidence written without an explicit iteration belongs to the attempt in
    progress -- the same rule the flat renderer used when it decided whether
    an item was still selectable.
    """
    raw = item.metrics.get("iteration")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return stage.iteration
    return int(raw)


def _evidence(run, stage, *, allow_selection: bool = True) -> tuple[str, list[str]]:
    """One stage's evidence, grouped by attempt, newest attempt first.

    A gate that has been rejected and retried a few times accumulates every
    attempt's artefacts on the same stage. Rendered as one flat grid, the two
    candidates actually up for decision were indistinguishable from eight
    superseded ones, and only the radio buttons hinted at which was which.
    """
    recommended = stage.qwen_reviews[-1].recommended_evidence_id if stage.qwen_reviews else None
    groups: dict[int, list] = {}
    for item in stage.evidence:
        groups.setdefault(_evidence_iteration(item, stage), []).append(item)
    if not groups:
        return '<p class="muted">This stage has not produced any evidence yet.</p>', []

    choices: list[str] = []
    sections: list[str] = []
    for iteration in sorted(groups, reverse=True):
        current = iteration == stage.iteration
        cards = []
        for item in groups[iteration]:
            url = _artifact_url(run.run_id, item.relative_path)
            pick = (
                '<span class="badge recommended">Qwen pick</span>'
                if item.evidence_id == recommended
                else ""
            )
            details = (
                "<details><summary>All recorded metrics and content hash</summary>"
                f"<pre>{html.escape(json.dumps({**item.metrics, 'sha256': item.sha256}, indent=2))}</pre></details>"
            )
            if item.media_type.startswith("image/"):
                choice = ""
                if (
                    allow_selection
                    and current
                    and item.metrics.get("selectable") is True
                    and stage.state == "awaiting_review"
                ):
                    checked = " checked" if item.evidence_id == recommended else ""
                    choice = (
                        f'<label class=choice><input type=radio name=selected_evidence_id '
                        f'value="{html.escape(item.evidence_id)}"{checked}>'
                        f"Select {html.escape(item.label)}</label>"
                    )
                    choices.append(choice)
                body = (
                    f'<a href="{url}" target=_blank><img loading=lazy src="{url}" '
                    f'alt="{html.escape(item.label)}"></a>'
                )
            elif item.media_type in {"model/gltf-binary", "model/gltf+json"} or item.relative_path.lower().endswith(".glb"):
                choice = ""
                body = (
                    f'<canvas class=glb-preview data-glb-src="{url}" aria-label="Interactive 3D preview of {html.escape(item.label)}"></canvas>'
                    '<p class=viewer-note>Drag to orbit; use the wheel to zoom. This local sampled preview is for inspection only.</p>'
                    f'<p><a class=button href="{url}" target=_blank>Download original GLB</a></p>'
                )
            else:
                choice = ""
                body = f'<p><a class=button href="{url}" target=_blank>Open evidence</a></p>'
            cards.append(
                '<section class="card evidence">'
                f"<h3>{html.escape(item.label)} {pick}</h3>{body}"
                f"{_metric_badges(item.metrics)}{details}{choice}</section>"
            )
        heading = (
            f'<div class="attempt"><h3>Attempt {iteration}</h3>'
            + (
                '<span class="badge recommended">current attempt</span>'
                if current
                else '<span class=badge>superseded</span>'
            )
            + "</div>"
        )
        sections.append(heading + '<div class="grid">' + "".join(cards) + "</div>")
    return "".join(sections), choices


def _decision_history(stage) -> str:
    """The append-only human record for one stage.

    Gate decisions are the product's core claim, and until now they were
    only visible as raw JSON inside the last thirty run events -- which a
    long run pushes out entirely.
    """
    if not stage.human_decisions:
        return '<p class="muted">No human decision has been recorded at this gate yet.</p>'
    rows = []
    for item in stage.human_decisions:
        facts = [f"<span class=badge>{len(item.evidence_hashes)} artefacts hash-bound</span>"]
        if item.selected_evidence_id:
            facts.append(f"<span class=badge>selected {html.escape(item.selected_evidence_id)}</span>")
        if item.target_stage_id:
            facts.append(f"<span class=badge>target {html.escape(item.target_stage_id)}</span>")
        if item.overrides:
            facts.append(f"<span class=badge>overrides {html.escape(json.dumps(item.overrides))}</span>")
        if item.assisted_by_review_id:
            facts.append(
                f"<span class='badge recommended'>AI-assisted {html.escape(item.assisted_by_review_id)}</span>"
            )
        rows.append(
            f'<li class="{html.escape(item.decision)}"><strong>{html.escape(item.decision)}</strong> '
            f'<span class=muted>{html.escape(item.created_at.isoformat())}</span>'
            f"<p>{html.escape(item.comment) or '<span class=muted>No comment recorded.</span>'}</p>"
            f'<p class="metrics">{"".join(facts)}</p></li>'
        )
    return f'<ul class="decisions">{"".join(rows)}</ul>'


def _ai_recommendation_form(run, stage, csrf: str) -> str:
    """Render a review recommendation as an explicit human confirmation.

    The model never submits the gate. The review ID is posted only by this
    dedicated confirmation form, so an ordinary manual decision is not
    mislabeled as AI-assisted.
    """
    if stage.state != "awaiting_review" or not stage.qwen_reviews:
        return ""
    review = stage.qwen_reviews[-1]
    if review.stage_id != stage.stage_id or review.iteration != stage.iteration:
        return ""
    recommended_item = next(
        (
            item
            for item in stage.evidence
            if item.evidence_id == review.recommended_evidence_id
            and item.metrics.get("selectable") is True
        ),
        None,
    )
    decision = (
        "approve"
        if review.hard_requirements_satisfied and recommended_item is not None
        else "reject"
    )
    selected_input = (
        f'<input type=hidden name=selected_evidence_id value="{html.escape(recommended_item.evidence_id, quote=True)}">'
        if recommended_item is not None
        else ""
    )
    details = review.recommended_changes or review.issues
    comment = review.summary
    if details:
        comment += " Suggested changes: " + "; ".join(details)
    verdict = "Approve recommended candidate" if decision == "approve" else "Reject and iterate"
    return (
        '<section class="card review"><h2>AI review recommendation</h2>'
        f'<p><strong>{html.escape(verdict)}</strong> '
        f'<span class=badge>confidence {review.confidence:.0%}</span></p>'
        f'<p>{html.escape(review.summary)}</p>'
        '<p class=muted>This is advisory. Inspect the evidence, then confirm to create a human-owned, '
        'hash-bound gate decision.</p>'
        f'<form method=post action="/run/{html.escape(run.run_id, quote=True)}/decision">'
        f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
        f'<input type=hidden name=stage_id value="{html.escape(stage.stage_id, quote=True)}">'
        f'<input type=hidden name=decision value="{decision}">'
        f'<input type=hidden name=comment value="{html.escape(comment, quote=True)}">'
        f'<input type=hidden name=assisted_by_review_id value="{html.escape(review.review_id, quote=True)}">'
        f'{selected_input}<button class=primary type=submit>Confirm AI recommendation: {html.escape(verdict)}</button>'
        f'<small class=muted>Review {html.escape(review.review_id)}</small></form></section>'
    )


def _decision_form(run, stage, csrf: str, choices: list[str]) -> str:
    if stage.state != "awaiting_review":
        return ""
    stage_index = next(
        (i for i, item in enumerate(run.stages) if item.stage_id == stage.stage_id), 0
    )
    rollback_targets = [
        item
        for item in run.stages[:stage_index]
        # A stage the asset contract ruled out cannot be rolled back to: it
        # was never decided, and StudioStore.decide() would refuse it.
        if item.applicable and item.state in {"approved", "skipped", "rejected", "failed"}
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
    return _ai_recommendation_form(run, stage, csrf) + (
        '<section class="card"><h2>Your decision</h2><p>Approve a selected candidate, reject with a comment for '
        "Qwen's next attempt, retry the same stage fresh, edit with a concrete correction, skip this stage, or "
        'roll back to an earlier stage. Full history is preserved either way.</p>'
        f'<form method=post action="/run/{html.escape(run.run_id)}/decision">'
        f'<input type=hidden name=csrf value="{csrf}"><input type=hidden name=stage_id value="{stage.stage_id}">'
        + "".join(choices)
        + '<label>Comment</label><textarea name=comment placeholder="Required to reject or skip. An edit needs a '
        'comment, overrides, or both. Optional for approve, retry, and roll back."></textarea>'
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
        'placeholder="Example: hand-painted clockwork courier with a canvas satchel on its left side and a brass lantern in its right hand; neutral studio background, clear proportions."></textarea>'
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
        status = f"Running {stage.stage_id} at {stage.progress:.0%}. This page updates automatically."
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
    archive_action = (
        f'<form method=post action="/run/{html.escape(run.run_id)}/unarchive">'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<button class=secondary type=submit>Unarchive this run</button></form>'
        if run.archived
        else f'<form method=post action="/run/{html.escape(run.run_id)}/archive">'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<button class=secondary type=submit>Archive this run</button></form>'
    )
    return (
        '<section class="card console"><div class=console-head><div><h2>Studio control console</h2>'
        '<p class=muted>Controls apply only to the local Text2Model Forge Studio job and local ComfyUI service.</p></div>'
        f'<span class=badge>{"stopping" if stopping else "running" if busy else "idle"}</span></div>'
        f'<div id=status-line class="status-line {status_class}">{html.escape(status)}</div>'
        '<div class=control-grid><section class=control-action><h3>Current work</h3>'
        '<p>Interrupts the tracked ComfyUI workflow and preserves all prior evidence. Resume is always explicit.</p>'
        f'{stop_action}</section><section class=control-action><h3>GPU / model memory</h3>'
        '<p>Unloads ComfyUI models and releases its execution cache. The next render will take longer while models reload.</p>'
        f'{memory_action}</section><section class=control-action><h3>Prompt workspace</h3>'
        '<p>The direct-render form includes a local Clear prompt and seed button. It never deletes stored evidence.</p>'
        '<p class=muted>ComfyUI is a shared local service: avoid unrelated Comfy jobs while stopping a Studio render.</p>'
        '</section><section class=control-action><h3>Dashboard visibility</h3>'
        '<p>Hides this run from the default dashboard list. Never deletes anything and is always reversible; '
        'the run stays reachable at its own URL either way.</p>'
        f'{archive_action}</section></div></section>'
    )


# Substrings that show up in an unmet-dependency error: a missing Blender
# executable, an absent config.local.toml worker binding, a ComfyUI model
# that was never installed. Heuristic, not exhaustive -- worst case a real
# bug's error just doesn't get the extra link, which is what happened for
# every failure before this existed.
_DEPENDENCY_ERROR_HINTS = (
    "not installed",
    "is required",
    "does not exist",
    "config.local.toml",
    "worker binding",
    "executable",
    "not reachable",
    "connection refused",
)


def _error_block(stage) -> str:
    """A failed/blocked stage's raw error, plus what a human can actually do
    about it. Previously this was a bare <pre> of the exception text with no
    guidance beyond the Resume button already shown elsewhere on the page --
    fine for "the render itself failed," unhelpful for "ComfyUI was never
    started," which reads identically to a human until they already know to
    check /doctor."""
    if not stage.error:
        return ""
    hint = ""
    lowered = stage.error.lower()
    if any(needle in lowered for needle in _DEPENDENCY_ERROR_HINTS):
        hint = (
            '<p class=muted>This looks like a missing local dependency or configuration, not a bad '
            'result. Check <a href="/doctor">the system page</a> for what is and is not reachable, fix '
            "it, then use Resume below to retry this stage from its saved inputs -- no evidence is lost."
            "</p>"
        )
    return f'<pre class=error>{html.escape(stage.error)}</pre>{hint}'


def _duration_badge(stage) -> str:
    """started_at/finished_at are written on every stage transition -- eleven
    call sites in studio_pipeline.py -- but nothing ever read them back; no
    duration was reachable anywhere in the browser. Render what's available:
    a finished duration, an in-progress elapsed time, or neither."""
    if stage.started_at is None:
        return ""
    if stage.finished_at is not None:
        seconds = (stage.finished_at - stage.started_at).total_seconds()
        label = f"ran {seconds:.0f}s"
    elif stage.state == "running":
        seconds = (utc_now() - stage.started_at).total_seconds()
        label = f"running {seconds:.0f}s"
    else:
        return f'<span class=badge>started {html.escape(stage.started_at.isoformat(timespec="seconds"))}</span>'
    return f"<span class=badge>{html.escape(label)}</span>"


def _stage_page(store: StudioStore, run_id: str, stage_id: str) -> str:
    """One stage's full record, reachable at any time from the timeline.

    The run page shows only the current stage, so before this every earlier
    stage's evidence, Qwen reviews, and human decisions were unreachable from
    the browser as soon as the run advanced past them.
    """
    run = store.load(run_id)
    try:
        stage = run.stage(stage_id)
    except KeyError as exc:
        raise FileNotFoundError(f"unknown stage: {stage_id}") from exc
    evidence, _ = _evidence(run, stage, allow_selection=False)
    reviews = (
        "".join(
            f"<h3>Attempt {review.iteration} — {html.escape(review.review_id)}</h3>"
            + _one_qwen_review(review)
            for review in reversed(stage.qwen_reviews)
        )
        or '<p class="muted">Qwen did not review this stage.</p>'
    )
    gate = "Stops for your decision" if stage.gate_required else "Runs automatically"
    error = _error_block(stage)
    live = (
        f'<p><a class="button primary" href="/run/{quote(run_id)}">This stage is waiting for your '
        "decision — open the review gate</a></p>"
        if stage.state == "awaiting_review" and run.current_stage == stage.stage_id
        else ""
    )
    return (
        f'<div class=crumb><a href="/">Runs</a><span class=muted>/</span>'
        f'<a href="/run/{quote(run_id)}">{html.escape(run.title)}</a>'
        f'<span class=muted>/</span><span>{html.escape(stage.stage_id)}</span></div>'
        '<section class="card hero">'
        f"<h1>{html.escape(stage.stage_id)} — {html.escape(stage.label)}</h1>"
        f'<p><span class=badge>{html.escape(stage.state.replace("_", " "))}</span> '
        f"<span class=badge>{html.escape(gate)}</span> "
        f"<span class=badge>attempt {stage.iteration}</span> "
        f'<span class=badge>{"applies to this asset" if stage.applicable else "not applicable"}</span> '
        f"{_duration_badge(stage)}</p>"
        f'<div class=progress-label><span>Stage progress</span><span>{stage.progress:.0%}</span></div>'
        f'{_progress_bar(stage.progress, label=f"{stage.stage_id} progress")} '
        f"<p>{html.escape(stage.message)}</p>{error}{live}"
        f"{_timeline(run, active=stage.stage_id)}</section>"
        '<div class=grid style="margin-top:16px">'
        f'<section class=card><h2>Human decisions</h2>{_decision_history(stage)}</section>'
        '<section class=card><h2>Stage metrics</h2>'
        f'<pre>{html.escape(json.dumps(stage.metrics, indent=2, default=str))}</pre></section></div>'
        f'<section class="card" style="margin-top:16px"><h2>Qwen reviews</h2>{reviews}</section>'
        f"<h2 style='margin-top:22px'>Evidence</h2>{evidence}"
    )


def _run_page(store: StudioStore, coordinator: StudioCoordinator, run_id: str, csrf: str) -> str:
    run = store.load(run_id)
    stage = run.stage(run.current_stage)
    evidence, choices = _evidence(run, stage)
    events = store.read_events(run_id)[-30:]
    work_detail = html.escape(stage.progress_phase)
    if stage.progress_total:
        work_detail += (
            f" · {stage.progress_current}/{stage.progress_total} "
            f"{html.escape(stage.progress_unit)}"
        )
    gpu_fraction = (
        min(1.0, (stage.gpu_used_gb or 0.0) / stage.gpu_total_gb)
        if stage.gpu_total_gb
        else 0.0
    )
    gpu_label = (
        f"{stage.gpu_used_gb or 0.0:.2f}/{stage.gpu_total_gb:.2f} GiB"
        if stage.gpu_total_gb
        else "Waiting for telemetry"
    )
    gpu_progress = (
        '<div class=progress-label><span>Last observed GPU memory</span>'
        f'<span id=gpu-run-label>{gpu_label}</span></div>'
        f'{_progress_bar(gpu_fraction, label="Live GPU memory", bar_id="gpu-run-bar", extra_class="gpu")}'
    )
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
    error = _error_block(stage)
    overall_progress = _run_progress(run)
    body = (
        f'<section class="card hero" data-run-id="{html.escape(run.run_id)}" '
        f'data-state="{html.escape(run.state)}" data-current-stage="{html.escape(run.current_stage)}">'
        f"<h1>{html.escape(run.title)}</h1><p>{html.escape(run.description)}</p>"
        f'<p><span class=badge>{html.escape(run.run_id)}</span> <span class=badge>{html.escape(run.state)}</span> '
        f'<span class=badge>concept backend: {html.escape(run.concept_backend)}</span> '
        f'<span class=badge>device policy: {html.escape(run.device_policy)}</span> '
        f'<span class=badge>created {html.escape(run.created_at.isoformat(timespec="seconds"))}</span>'
        + (' <span class=badge>Archived</span>' if run.archived else "")
        + "</p>"
        f'<div class=progress-label><span>Whole pipeline</span><span id=overall-run-label>{overall_progress:.0%} overall</span></div>'
        f'{_progress_bar(overall_progress, label="Whole pipeline progress", bar_id="overall-run-bar")}'
        f'<div class=progress-label><span>Active work</span><span id=stage-work>{work_detail}</span></div>'
        f'{gpu_progress}'
        f"{_timeline(run, active=stage.stage_id, mark_progress_id=True)}"
        f'<h2><a href="{_stage_url(run.run_id, stage.stage_id)}">{stage.stage_id} — '
        f"{html.escape(stage.label)}</a></h2>"
        + (f'<p>{_duration_badge(stage)}</p>' if stage.started_at else "")
        + f'<p id=stage-message>{html.escape(stage.message)}</p>{error}{actions}</section>'
        '<div class=grid style="margin-top:16px"><section class=card><h2>Qwen production contract</h2>'
        f"{_spec(run)}</section><section class=card><h2>Qwen gate review</h2>{_qwen_review(stage)}"
        f'<p class=muted><a href="{_stage_url(run.run_id, stage.stage_id)}">'
        f"See every review and decision recorded at this stage</a></p></section></div>"
        f"{_studio_controls(run, stage, coordinator, csrf, busy=busy)}"
        f"<h2 style='margin-top:22px'>Evidence</h2>{evidence}"
        f"{_manual_qwen_image_form(run, stage, csrf, busy=busy)}"
        f"{_decision_form(run, stage, csrf, choices)}"
        f'<section class=card style="margin-top:16px"><h2>Run history</h2><div class=events>{event_html}</div></section>'
        + ('<script src="/static/studio.js" defer></script>' if busy else "")
    )
    return body


def _run_card(run) -> str:
    settled = sum(1 for item in run.stages if item.state in {"approved", "skipped"})
    total = max(len(run.stages), 1)
    overall_progress = _run_progress(run)
    waiting = run.state == "awaiting_review"
    attention = (
        '<span class="badge needs">Needs your decision</span>'
        if waiting
        else (
            '<span class="badge" style="background:#4a2220">Stopped</span>'
            if run.state in {"failed", "blocked"}
            else f"<span class=badge>{html.escape(run.state)}</span>"
        )
    )
    description = run.description if len(run.description) <= 240 else run.description[:237] + "..."
    stage = run.stage(run.current_stage) if any(
        item.stage_id == run.current_stage for item in run.stages
    ) else run.stages[0]
    archived_badge = '<span class=badge>Archived</span> ' if run.archived else ""
    return (
        '<section class="card run-card">'
        f'<h2><a href="/run/{quote(run.run_id)}">{html.escape(run.title)}</a></h2>'
        f'<p>{archived_badge}{attention} <span class=badge>{html.escape(stage.stage_id)} '
        f"{html.escape(stage.label)}</span> <span class=badge>{settled}/{total} stages settled</span></p>"
        f'<div class=progress-label><span>Pipeline progress</span><span>{overall_progress:.0%}</span></div>'
        f'{_progress_bar(overall_progress, label=f"{run.title} pipeline progress")}'
        f"<p>{html.escape(description)}</p>"
        f'<p class=muted>{html.escape(run.run_id)} · profile {html.escape(run.profile)} · '
        f"updated {html.escape(run.updated_at.isoformat(timespec='seconds'))}</p>"
        f'<a class="button{" primary" if waiting else ""}" href="/run/{quote(run.run_id)}">'
        f'{"Review now" if waiting else "Open run"}</a></section>'
    )


def _dashboard(store: StudioStore, *, show_archived: bool = False) -> str:
    # Runs arrive newest-first; float the ones blocked on a human above them,
    # because in a human-gated compiler an idle gate is the only thing that
    # actually stops the machine. Archived runs are hidden by default -- see
    # StudioRun.archived -- but never dropped from StudioStore.list() itself,
    # only from what this page chooses to show.
    all_runs = store.list()
    archived_count = sum(1 for run in all_runs if run.archived)
    runs = all_runs if show_archived else [run for run in all_runs if not run.archived]
    runs = sorted(runs, key=lambda item: item.state != "awaiting_review")
    waiting = sum(1 for run in runs if run.state == "awaiting_review")
    cards = "".join(_run_card(run) for run in runs)
    if not cards:
        cards = '<section class=card><h2>No runs yet</h2><p>Describe any asset; Text2Model Forge compiles the production contract.</p><a class="button primary" href=/new>Create asset</a></section>'
    summary = (
        f"{len(runs)} run{'' if len(runs) == 1 else 's'}, "
        f"{waiting} waiting for your decision."
        if runs
        else "One description in. Qwen acts, critiques, and mediates bounded corrections; "
        "deterministic gates preserve evidence and explicit human decisions."
    )
    toggle = ""
    if archived_count:
        toggle = (
            '<p class=muted><a href="/?archived=1">Show ' + str(archived_count) + " archived run"
            + ("" if archived_count == 1 else "s") + "</a></p>"
            if not show_archived
            else '<p class=muted><a href="/">Hide archived runs</a></p>'
        )
    return (
        '<section class="card hero"><h1>Asset production runs</h1>'
        f'<p class=muted>{html.escape(summary)}</p>{toggle}'
        '<a class="button primary" href=/new>Create asset</a></section>'
        '<div class=grid style="margin-top:16px">' + cards + "</div>"
    )


def _available_profiles() -> list[str]:
    directory = profiles_dir()
    if not directory.is_dir():
        return ["simple"]
    names = sorted(p.stem for p in directory.glob("*.toml") if p.stem != "base")
    return names or ["simple"]


def _setup_options(profile: str) -> dict[str, Any]:
    """Profile defaults plus live model choices for the new-run form.

    Service discovery is advisory: an offline ComfyUI or reviewer returns an
    empty list and a visible status, while the form remains usable with typed
    model ids. This keeps startup and profile editing independent of external
    service availability.
    """
    defaults = studio_overrides(resolve_settings(profile=profile))
    comfy_url = str(defaults.get("comfy_url", "http://127.0.0.1:8188")).rstrip("/")
    reviewer_url = str(defaults.get("localdeploy_url", "http://127.0.0.1:8000/v1")).rstrip("/")
    endpoints = {
        "checkpoints": f"{comfy_url}/models/checkpoints",
        "diffusion_models": f"{comfy_url}/models/diffusion_models",
        "text_encoders": f"{comfy_url}/models/text_encoders",
        "vae": f"{comfy_url}/models/vae",
        "reviewer": f"{reviewer_url}/models",
    }
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futures = {name: pool.submit(_json_get, url) for name, url in endpoints.items()}
        results = {name: future.result() for name, future in futures.items()}

    def string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return sorted(str(item) for item in value if isinstance(item, str))

    checkpoints = string_list(results["checkpoints"][0])
    diffusion_models = string_list(results["diffusion_models"][0])
    text_encoders = string_list(results["text_encoders"][0])
    vaes = string_list(results["vae"][0])
    reviewer_payload = results["reviewer"][0]
    reviewer_models = []
    if isinstance(reviewer_payload, dict) and isinstance(reviewer_payload.get("data"), list):
        reviewer_models = sorted(
            str(item["id"])
            for item in reviewer_payload["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
    qwen_ready = all(
        required in installed
        for required, installed in (
            ("qwen_image_2512_fp8_e4m3fn.safetensors", diffusion_models),
            ("qwen_2.5_vl_7b_fp8_scaled.safetensors", text_encoders),
            ("qwen_image_vae.safetensors", vaes),
        )
    )
    z_image_ready = all(
        required in installed
        for required, installed in (
            ("z_image_turbo_int8_convrot.safetensors", diffusion_models),
            ("qwen_3_4b_fp8_mixed.safetensors", text_encoders),
            ("ae.safetensors", vaes),
        )
    )
    comfy_ready = any(results[name][0] is not None for name in endpoints if name != "reviewer")
    return {
        "defaults": {
            key: value
            for key, value in defaults.items()
            if key in {
                "concept_backend",
                "checkpoint",
                "model",
                "spec_strategy",
                "concept_steps",
                "concept_cfg",
                "concept_candidates",
                "device_policy",
            }
            and value is not None
        },
        "checkpoints": checkpoints,
        "diffusion_models": diffusion_models,
        "review_models": reviewer_models,
        "qwen_image_2512_ready": qwen_ready,
        "z_image_turbo_ready": z_image_ready,
        "services": {
            "comfyui": {
                "ready": comfy_ready,
                "detail": results["checkpoints"][1] if comfy_ready else results["diffusion_models"][1],
            },
            "reviewer": {
                "ready": reviewer_payload is not None,
                "detail": results["reviewer"][1],
            },
        },
    }


def _new_form(csrf: str, preset_description: str = "") -> str:
    profiles = _available_profiles()
    recommended_profile = recommend_stack(detect_hardware()).profile
    default = (
        recommended_profile
        if recommended_profile in profiles
        else "simple"
        if "simple" in profiles
        else profiles[0]
    )
    defaults = studio_overrides(resolve_settings(profile=default))
    options = "".join(
        f'<option value="{html.escape(name)}"{" selected" if name == default else ""}>{html.escape(name)}</option>'
        for name in profiles
    )
    backend_default = html.escape(str(defaults.get("concept_backend", "auto")))
    strategy_default = html.escape(str(defaults.get("spec_strategy", "monolithic")))
    checkpoint_default = html.escape(str(defaults.get("checkpoint", "")))
    model_default = html.escape(str(defaults.get("model", "")))
    steps_default = html.escape(str(defaults.get("concept_steps", 30)))
    cfg_default = html.escape(str(defaults.get("concept_cfg", 6.0)))
    candidates_default = html.escape(str(defaults.get("concept_candidates", 3)))
    device_default = html.escape(str(defaults.get("device_policy", "prefer_gpu")))
    return (
        '<section class="card hero"><h1>Describe one original asset</h1>'
        '<p>This is the only production input. It may be a character, creature, door, wall, prop, environment, material, or VFX. Include handedness, moving pieces, and required states when they matter; Qwen compiles the rest.</p>'
        '<form method=post action=/runs data-setup-options>'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<label>Description</label><textarea name=description minlength=20 required '
        'placeholder="Examples: a weathered stone well with an iron crank; or a clockwork courier with a right-hand lantern and two walking states...">'
        f'{html.escape(preset_description)}</textarea>'
        f'<label>Configuration profile</label><select name=profile>{options}</select>'
        '<details class=options open><summary>Text-to-2D and reviewer options</summary>'
        '<p class=muted>Leave any field blank to inherit the selected profile. Installed model names appear as suggestions when the local services are running.</p>'
        '<div class=option-grid>'
        # Each option states the measured cost of choosing it. These numbers
        # are wall-clock per candidate at 768x1024 on an 8 GB RTX 3080, taken
        # from a real side-by-side run of one prompt through every backend --
        # not vendor claims. D1 renders `concept_candidates` of these per
        # iteration and re-runs the whole set on rejection, so the per-image
        # figure is the one that decides whether a run finishes tonight.
        '<div><label>Text-to-2D backend</label><select name=concept_backend>'
        f'<option value="">Profile default: {backend_default}</option>'
        '<option value=auto>Auto — Z-Image if installed, else Qwen, else SDXL</option>'
        '<option value=z_image_turbo>Z-Image Turbo — stylized, ~50 s/image (recommended)</option>'
        '<option value=qwen_image_2512>Qwen Image 2512 — best quality, ~10 min/image</option>'
        '<option value=sdxl>SDXL checkpoint — fastest, ~20 s/image</option>'
        '</select>'
        '<p class=muted data-backend-note>Times measured on this machine at 768&times;1024. '
        'Qwen and Z-Image ignore the checkpoint field below; SDXL uses it.</p></div>'
        '<div><label>SDXL / custom checkpoint</label>'
        f'<input name=checkpoint list=installed-checkpoints maxlength=300 placeholder="Profile default: {checkpoint_default}">'
        '<datalist id=installed-checkpoints></datalist></div>'
        '<div><label>Qwen reviewer model</label>'
        f'<input name=model list=installed-review-models maxlength=300 placeholder="Profile default: {model_default}">'
        '<datalist id=installed-review-models></datalist></div>'
        '<div><label>D0 spec strategy</label><select name=spec_strategy>'
        f'<option value="">Profile default: {strategy_default}</option>'
        '<option value=chunked>Chunked — best for 7–8B local models</option>'
        '<option value=monolithic>Monolithic — qualified 27B model</option>'
        '</select></div>'
        '<div><label>Concept steps</label>'
        f'<input name=concept_steps type=number min=1 max=150 placeholder="Profile default: {steps_default}"></div>'
        '<div><label>Concept CFG</label>'
        f'<input name=concept_cfg type=number min=0.1 max=30 step=0.1 placeholder="Profile default: {cfg_default}"></div>'
        '<div><label>Sequential candidate budget</label>'
        f'<input name=concept_candidates type=number min=2 max=12 placeholder="Profile default: {candidates_default}"></div>'
        '<div><label>Device policy</label><select name=device_policy>'
        f'<option value="">Profile default: {device_default}</option>'
        '<option value=gpu_compute_only>GPU compute only — requires live telemetry</option>'
        '<option value=prefer_gpu>Prefer GPU — CPU inference allowed</option>'
        '<option value=strict_device_only>Strict device only — experimental</option>'
        '</select></div>'
        '</div><div id=setup-service-status class=service-status>Checking local AI services and installed models...</div>'
        '</details><button class=primary type=submit>Compile asset and start</button></form>'
        '<p class=muted>D1 is human-gated: reject, edit, or retry as many times as needed before the approved image is allowed into 3D.</p>'
        '</section><script src="/static/studio.js" defer></script>'
    )


_CONCEPT_BACKENDS = {"auto", "qwen_image_2512", "qwen_image_edit_2511", "sdxl", "z_image_turbo"}
_SPEC_STRATEGIES = {"monolithic", "chunked"}
_DEVICE_POLICIES = {"prefer_gpu", "gpu_compute_only", "strict_device_only"}


def _new_run_overrides(values: dict[str, str], profile: str) -> dict[str, Any]:
    """Merge optional browser fields over one resolved profile.

    Empty fields intentionally do not become empty run settings; they mean
    "inherit" so switching profiles keeps working exactly as it did before
    the advanced controls were added.
    """
    result = studio_overrides(resolve_settings(profile=profile))
    backend = values.get("concept_backend", "").strip()
    if backend:
        if backend not in _CONCEPT_BACKENDS:
            raise ValueError(f"unknown text-to-2D backend: {backend}")
        result["concept_backend"] = backend
    strategy = values.get("spec_strategy", "").strip()
    if strategy:
        if strategy not in _SPEC_STRATEGIES:
            raise ValueError(f"unknown D0 spec strategy: {strategy}")
        result["spec_strategy"] = strategy
    device_policy = values.get("device_policy", "").strip()
    if device_policy:
        if device_policy not in _DEVICE_POLICIES:
            raise ValueError(f"unknown device policy: {device_policy}")
        result["device_policy"] = device_policy
    for key in ("checkpoint", "model"):
        value = values.get(key, "").strip()
        if value:
            result[key] = value
    raw_steps = values.get("concept_steps", "").strip()
    if raw_steps:
        try:
            steps = int(raw_steps)
        except ValueError as exc:
            raise ValueError("concept steps must be a whole number from 1 to 150") from exc
        if not 1 <= steps <= 150:
            raise ValueError("concept steps must be a whole number from 1 to 150")
        result["concept_steps"] = steps
    raw_cfg = values.get("concept_cfg", "").strip()
    if raw_cfg:
        try:
            cfg = float(raw_cfg)
        except ValueError as exc:
            raise ValueError("concept CFG must be a number greater than 0 and no more than 30") from exc
        if not 0 < cfg <= 30:
            raise ValueError("concept CFG must be a number greater than 0 and no more than 30")
        result["concept_cfg"] = cfg
    raw_candidates = values.get("concept_candidates", "").strip()
    if raw_candidates:
        try:
            candidates = int(raw_candidates)
        except ValueError as exc:
            raise ValueError("concept candidates must be a whole number from 2 to 12") from exc
        if not 2 <= candidates <= 12:
            raise ValueError("concept candidates must be a whole number from 2 to 12")
        result["concept_candidates"] = candidates
    return result


def _worker_report() -> list[dict[str, Any]]:
    """Live worker readiness, the same check `python -m text2model_forge workers` runs.

    /doctor used to list only the *names* bound in config.local.toml, which
    says nothing about whether any of them can actually run -- the one
    question the page exists to answer. Failures are reported per worker
    rather than raised: a single unreadable manifest must not blank the page.
    """
    try:
        manifests = load_manifests()
    except Exception as exc:
        return [{"worker_id": "(manifests)", "ready": False, "health_error": f"{type(exc).__name__}: {exc}"}]
    config = load_local_config()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            worker_id: pool.submit(
                preflight,
                manifest,
                command_prefix=(binding.command_prefix if (binding := worker_binding(config, worker_id)) else None),
                timeout_seconds=1.5,
            )
            for worker_id, manifest in manifests.items()
        }
        report = []
        for worker_id, future in futures.items():
            try:
                report.append(future.result())
            except Exception as exc:
                report.append(
                    {"worker_id": worker_id, "ready": False, "health_error": f"{type(exc).__name__}: {exc}"}
                )
    return sorted(report, key=lambda item: (item.get("ready") is not True, item["worker_id"]))



def _preflight_report():
    """Hardware + every cross-stage check, for the System page."""
    from .hardware import detect_hardware, recommend_stack
    from .preflight import run_preflight

    try:
        hardware, checks = run_preflight(profile="simple", deep=False)
        return hardware, checks, recommend_stack(hardware)
    except Exception as exc:  # a broken check must not blank the page
        from .hardware import HardwareProfile
        from .preflight import Check

        blank = HardwareProfile()
        return (
            blank,
            [Check(name="preflight", status="fail", detail=f"{type(exc).__name__}: {exc}")],
            recommend_stack(blank),
        )


_STATUS_BADGE = {"ok": "", "warn": " needs", "fail": " needs", "skip": ""}


def _preflight_html(hardware, checks, recommendation) -> str:
    failures = [check for check in checks if check.status == "fail"]
    rows = "".join(
        f'<li><span class="badge{_STATUS_BADGE.get(check.status, "")}">{html.escape(check.status)}</span> '
        f"<strong>{html.escape(check.name)}</strong><br>"
        f"<span class=muted>{html.escape(check.detail)}</span>"
        + (
            f'<br><span class=warning>fix: {html.escape(check.remedy)}</span>'
            if check.remedy and check.status in {"fail", "warn"}
            else ""
        )
        + "</li>"
        for check in checks
    )
    headline = (
        f'<p class=error>{len(failures)} assumption{"" if len(failures) == 1 else "s"} '
        "will fail this run. Each one below would otherwise surface several stages in.</p>"
        if failures
        else '<p class="good">Every cross-stage assumption holds on this machine.</p>'
    )
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in recommendation.reasons)
    return (
        '<h2 style="margin-top:18px">Preflight</h2>'
        f"{headline}"
        f'<ul class="decisions">{rows}</ul>'
        f'<details style="margin-top:12px"><summary>Recommended stack for this hardware '
        f"(profile {html.escape(recommendation.profile)})</summary>"
        f"<p><span class=badge>reviewer {html.escape(recommendation.reviewer_size)}</span> "
        f"<span class=badge>spec_strategy {html.escape(recommendation.spec_strategy)}</span> "
        f"<span class=badge>voxel_fraction {recommendation.voxel_fraction}</span></p>"
        f"<ul>{reasons}</ul></details>"
    )


def _doctor() -> str:
    defaults = studio_overrides(resolve_settings(profile="simple"))
    reviewer_url = str(defaults.get("localdeploy_url", "http://127.0.0.1:8000/v1")).rstrip("/")
    comfy_url = str(defaults.get("comfy_url", "http://127.0.0.1:8188")).rstrip("/")
    with ThreadPoolExecutor(max_workers=3) as pool:
        # Each probe blocks for up to two seconds and the worker preflight
        # blocks for longer; run them together so the page is not the sum of
        # every timeout on a machine with nothing running.
        localdeploy_future = pool.submit(_probe, f"{reviewer_url}/models")
        comfy_future = pool.submit(_probe, f"{comfy_url}/system_stats")
        workers_future = pool.submit(_worker_report)
        # The cross-stage assumption checks. Same report `text2model_forge doctor`
        # prints, because a mismatch that only shows up in the CLI is a
        # mismatch the browser user still runs into three stages later.
        preflight_future = pool.submit(_preflight_report)
        localdeploy, localdeploy_detail = localdeploy_future.result()
        comfy, comfy_detail = comfy_future.result()
        workers = workers_future.result()
        hardware, checks, recommendation = preflight_future.result()
    config = load_local_config()
    ready = sum(1 for item in workers if item.get("ready") is True)
    rows = "".join(
        f'<li><strong>{html.escape(str(item["worker_id"]))}</strong> '
        f'<span class="badge{"" if item.get("ready") else " needs"}">'
        f'{"ready" if item.get("ready") else "not ready"}</span> '
        f'<span class=muted>{html.escape(str(item.get("executable") or item.get("health_error") or item.get("declared_lifecycle") or ""))}</span></li>'
        for item in workers
    )
    return (
        '<section class="card hero"><h1>Local production system</h1><div class=health>'
        f'<span class="{"" if localdeploy else "down"}">Qwen reviewer ({html.escape(reviewer_url)}): {html.escape(localdeploy_detail)}</span>'
        f'<span class="{"" if comfy else "down"}">ComfyUI ({html.escape(comfy_url)}): {html.escape(comfy_detail)}</span>'
        f'<span class="{"" if config else "down"}">Text2Model Forge config: {"loaded" if config else "missing (copy machine.example.toml to config.local.toml)"}</span>'
        f'<span class="{"" if ready else "down"}">Workers ready: {ready}/{len(workers)}</span>'
        f'<span class="{"" if hardware.detected else "down"}">GPU: '
        f'{html.escape((hardware.primary.name if hardware.primary else "not detected"))}'
        f'{f" &mdash; {hardware.vram_total_gb} GB" if hardware.vram_total_gb else ""}</span>'
        '</div>'
        + _preflight_html(hardware, checks, recommendation)
        + '<h2 style="margin-top:18px">Deterministic worker preflight</h2>'
        f'<ul class="decisions">{rows or "<li>No worker manifests were found.</li>"}</ul>'
        '<p class=muted>Studio binds to loopback by default. The Docker profile uses an explicit container-only bind and publishes it on host loopback. Qwen proposes structured decisions; it never executes code or edits artifacts.</p></section>'
    )


def _golden_dashboard(store: StudioStore) -> str:
    from .golden import load_corpus
    from text2model_forge.paths import resource_root

    corpus = load_corpus(resource_root() / "golden" / "static-props.json")
    runs_by_description: dict[str, list] = {}
    for run in store.list():
        runs_by_description.setdefault(run.description.strip(), []).append(run)
    cards: list[str] = []
    attempted = completed = 0
    for case in corpus.cases:
        matches = runs_by_description.get(case.prompt.strip(), [])
        run = matches[0] if matches else None
        if run is not None:
            attempted += 1
            completed += int(run.state == "completed")
            progress = _run_progress(run)
            action = f'<a class=button href="/run/{quote(run.run_id)}">Open latest run</a>'
            state = html.escape(run.state)
        else:
            progress = 0
            action = f'<a class="button primary" href="/new?prompt={quote(case.prompt)}">Start this case</a>'
            state = "not attempted"
        features = "".join(f"<li>{html.escape(item)}</li>" for item in case.required_features)
        cards.append(
            '<section class="card run-card">'
            f'<h2>{html.escape(case.case_id)}</h2><p><span class=badge>{html.escape(case.category)}</span> '
            f'<span class=badge>{state}</span></p><p>{html.escape(case.prompt)}</p>'
            f'<ul>{features}</ul><div class=progress-label><span>Run progress</span><span>{progress:.0%}</span></div>'
            f'{_progress_bar(progress, label=case.case_id + " corpus progress")}{action}</section>'
        )
    overall = completed / corpus.required_attempts
    return (
        '<section class="card hero"><h1>Live 8 GB static-prop qualification</h1>'
        f'<p>{attempted}/{corpus.required_attempts} attempted; {completed}/{corpus.required_attempts} completed. '
        f'Publication threshold: at least {corpus.minimum_passing_cases} human-reviewed passes after every case is attempted.</p>'
        f'<div class=progress-label><span>Completed live runs</span><span>{overall:.0%}</span></div>'
        f'{_progress_bar(overall, label="Golden corpus completion")} '
        '<p class=muted>Completion alone is not a pass. Export the human assessment report and run '
        '<code>text2model-forge golden evaluate</code>; the evaluator verifies the stored run evidence and fails closed.</p>'
        '</section><div class=grid style="margin-top:16px">' + "".join(cards) + "</div>"
    )
class StudioServer(NamedTuple):
    """A constructed but not-yet-running Studio server, plus the pieces a
    caller needs to drive or inspect it. Split out of serve() so the HTTP
    layer -- routing, CSRF, form parsing, error rendering -- can be tested
    against a real loopback server instead of only by reading it."""

    server: ThreadingHTTPServer
    store: StudioStore
    coordinator: StudioCoordinator
    csrf: str
    recovered: list[str]

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"


def build_server(
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    allow_non_loopback: bool = False,
    coordinator_factory=None,
) -> StudioServer:
    """Construct the Studio HTTP server without running it.

    Pass port=0 for an ephemeral port. `coordinator_factory` takes the
    server's own StudioStore and returns a StudioCoordinator, so a test can
    supply fake Qwen/ComfyUI/worker providers. It deliberately receives the
    store rather than accepting a pre-built coordinator: two StudioStore
    instances over one directory hold independent locks, so a coordinator
    built on a different store than the request handlers use would race on
    run.json writes.
    """
    if host not in {"127.0.0.1", "::1", "localhost"} and not allow_non_loopback:
        raise ValueError(
            "Text2Model Forge Studio may bind only to a loopback address unless "
            "allow_non_loopback=True is explicitly set"
        )
    store = StudioStore(workspace)
    recovered = store.recover_interrupted_runs()
    coordinator = (coordinator_factory or StudioCoordinator)(store)
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

        def page(self, title: str, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = _page(title, body)
            self._headers(status, "text/html; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def redirect(self, path: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", path)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/":
                    show_archived = parse_qs(parsed.query).get("archived", [""])[0] == "1"
                    self.page("Text2Model Forge Studio", _dashboard(store, show_archived=show_archived))
                elif path == "/new":
                    preset = parse_qs(parsed.query).get("prompt", [""])[0]
                    self.page("New asset", _new_form(csrf, preset))
                elif path == "/golden":
                    self.page("Golden corpus", _golden_dashboard(store))
                elif path == "/doctor":
                    self.page("System", _doctor())
                elif path == "/favicon.ico":
                    self._headers(HTTPStatus.NO_CONTENT, "image/x-icon", 0)
                elif path.startswith("/run/"):
                    parts = path.split("/")
                    if len(parts) >= 5 and parts[3] == "stage":
                        run_id = unquote(parts[2])
                        stage_id = unquote(parts[4])
                        self.page("Stage detail", _stage_page(store, run_id, stage_id))
                    else:
                        run_id = unquote(parts[2])
                        self.page("Production run", _run_page(store, coordinator, run_id, csrf))
                elif path == "/static/studio.js":
                    payload = STUDIO_JS.encode("utf-8")
                    self._headers(HTTPStatus.OK, "text/javascript; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path == "/static/glb-viewer.js":
                    payload = GLB_VIEWER_JS.encode("utf-8")
                    self._headers(HTTPStatus.OK, "text/javascript; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path == "/api/setup/options":
                    profile = parse_qs(parsed.query).get("profile", ["simple"])[0]
                    if profile not in _available_profiles():
                        raise ValueError(f"unknown configuration profile: {profile}")
                    payload = json.dumps(_setup_options(profile)).encode("utf-8")
                    self._headers(HTTPStatus.OK, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
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
                    # resolve_settings() turns this straight into a
                    # profiles/<name>.toml path, so accept only a profile the
                    # selector actually offers rather than any string a form
                    # post happens to carry.
                    if profile not in _available_profiles():
                        raise ValueError(f"unknown configuration profile: {profile}")
                    run_id = _slug()
                    store.create(
                        run_id,
                        description,
                        {**_new_run_overrides(values, profile), "profile": profile},
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
                        assisted_by_review_id=values.get("assisted_by_review_id") or None,
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
                elif path.startswith("/run/") and path.endswith("/archive"):
                    run_id = unquote(path.split("/")[2])
                    store.set_archived(run_id, True)
                    self.redirect("/")
                elif path.startswith("/run/") and path.endswith("/unarchive"):
                    run_id = unquote(path.split("/")[2])
                    store.set_archived(run_id, False)
                    self.redirect("/run/" + quote(run_id))
                else:
                    self.page("Not found", "<h1>Not found</h1>", HTTPStatus.NOT_FOUND)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                self.page(
                    "Text2Model Forge Studio error",
                    f'<section class=card><h1 class=error>Could not apply that action</h1><pre>{html.escape(str(exc))}</pre><a class=button href="/">Back to runs</a></section>',
                    HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"Text2Model Forge Studio {self.address_string()}: {format % args}")

    return StudioServer(
        server=ThreadingHTTPServer((host, port), Handler),
        store=store,
        coordinator=coordinator,
        csrf=csrf,
        recovered=recovered,
    )


def serve(
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = False,
    allow_non_loopback: bool = False,
) -> None:
    studio = build_server(
        workspace,
        host=host,
        port=port,
        allow_non_loopback=allow_non_loopback,
    )
    if studio.recovered:
        print(f"Text2Model Forge Studio recovered interrupted runs: {', '.join(studio.recovered)}")
    print(f"Text2Model Forge Studio: {studio.url}")
    if open_browser:
        webbrowser.open(studio.url + "/new")
    try:
        studio.server.serve_forever()
    finally:
        studio.coordinator.close()
        studio.server.server_close()
