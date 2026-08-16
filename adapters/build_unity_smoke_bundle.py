"""Build a portable, non-promoting Unity smoke project for a Darkness sprite candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


UNITY_RUNTIME_MUTABLE_FILES = (
    "UnitySmokeProject/ProjectSettings/ProjectVersion.txt",
)
UNITY_VERSION = "6000.5.1f1"
UNITY_VERSION_FAMILY = "6000.5"
UNITY_REVISION = "0d9463e84828"
EXPECTED_ACTIONS = {f"{clip}/{direction}" for clip in ("idle", "walk", "attack", "death") for direction in ("north", "south", "east", "west")}


def _arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=root / "unity_smoke_template")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_package_file(package: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError(f"candidate file must be a package-local filename: {name!r}")
    path = package / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_candidate(package: Path) -> tuple[dict[str, object], Path]:
    manifest_path = package / "candidate_unit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "human_review_candidate"
        or manifest.get("automatic_gate_passed") is not True
        or manifest.get("human_approval_required") is not True
        or manifest.get("human_approved") is not False
        or not isinstance(manifest.get("asset_id"), str)
    ):
        raise ValueError("candidate manifest is not an unapproved, automatic-gate-passing review candidate")
    actions = manifest.get("actions")
    if not isinstance(actions, list) or len(actions) != 16:
        raise ValueError("candidate must contain exactly sixteen directional actions")
    observed = {f"{action.get('name')}/{action.get('direction')}" for action in actions}
    if observed != EXPECTED_ACTIONS:
        raise ValueError("candidate action/direction set is incomplete or duplicated")
    for action in actions:
        sheet = _safe_package_file(package, str(action.get("sheet", "")))
        if _sha256(sheet).lower() != str(action.get("sha256", "")).lower():
            raise ValueError(f"candidate sheet hash mismatch: {sheet.name}")
    master = Path(str(manifest.get("source_master", ""))).resolve()
    if not master.is_file() or _sha256(master).lower() != str(manifest.get("source_master_sha256", "")).lower():
        raise ValueError("candidate source-master hash mismatch")
    return manifest, master


def _runner_text() -> str:
    return r'''param(
    [string]$Unity
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Version = "6000.5.1f1"
if (-not $Unity) {
    $Candidates = @(
        "C:\Program Files\Unity\Hub\Editor\$Version\Editor\Unity.exe",
        "C:\UnityLocal\$Version\Editor\Unity.exe"
    )
    $Unity = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Unity -or -not (Test-Path -LiteralPath $Unity)) {
    throw "Unity $Version was not found. Pass -Unity <path-to-Unity.exe>."
}

$Project = Join-Path $Root "UnitySmokeProject"
$Package = Join-Path $Root "candidate"
$Result = Join-Path $Root "result"
New-Item -ItemType Directory -Path $Result -Force | Out-Null
Remove-Item -LiteralPath (Join-Path $Result "unity_candidate_validation.json") -Force -ErrorAction SilentlyContinue
$env:DARKNESS_CANDIDATE_PACKAGE = $Package
$env:DARKNESS_CANDIDATE_OUTPUT = $Result
$env:DARKNESS_BUNDLE_MANIFEST = Join-Path $Root "bundle_manifest.json"

$Arguments = @(
    "-quit",
    "-batchmode",
    "-projectPath", ('"' + $Project + '"'),
    "-executeMethod", "Darkness.EditorTools.DarknessCandidateValidator.ValidateFromBatch",
    "-logFile", ('"' + (Join-Path $Result "unity.log") + '"')
)
$Process = Start-Process -FilePath $Unity -ArgumentList $Arguments -PassThru -Wait -WindowStyle Hidden
if ($Process.ExitCode -ne 0) {
    throw "Unity smoke validation failed with exit code $($Process.ExitCode). See result\unity.log."
}
$Report = Join-Path $Result "unity_candidate_validation.json"
if (-not (Test-Path -LiteralPath $Report)) {
    throw "Unity exited without producing result\unity_candidate_validation.json."
}
Write-Output "Darkness Unity smoke validation passed. Return the result folder to the Darkness pipeline."
'''


def _interactive_runner_text() -> str:
    return rf'''param(
    [string]$Unity
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Version = "{UNITY_VERSION}"
if (-not $Unity) {{
    $Candidates = @(
        "C:\Program Files\Unity\Hub\Editor\$Version\Editor\Unity.exe",
        "C:\UnityLocal\$Version\Editor\Unity.exe"
    )
    $Unity = $Candidates | Where-Object {{ Test-Path -LiteralPath $_ }} | Select-Object -First 1
}}
if (-not $Unity -or -not (Test-Path -LiteralPath $Unity)) {{
    throw "Unity $Version was not found. Pass -Unity <path-to-Unity.exe>."
}}

$Project = Join-Path $Root "UnitySmokeProject"
$env:DARKNESS_CANDIDATE_PACKAGE = Join-Path $Root "candidate"
$Arguments = @("-projectPath", ('"' + $Project + '"'))
Start-Process -FilePath $Unity -ArgumentList $Arguments
Write-Output "Unity review project launched. The Candidate Review window opens automatically; use Darkness > Candidate Review if it was closed."
'''


def _review_html(manifest: dict[str, object]) -> str:
    actions = [
        {
            "name": action["name"],
            "direction": action["direction"],
            "frames": action["frames"],
            "fps": action["fps"],
            "loop": action["loop"],
            "sheet": action["sheet"],
        }
        for action in manifest["actions"]
    ]
    data = json.dumps(actions, separators=(",", ":"))
    display_name = str(manifest.get("display_name", manifest["asset_id"]))
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VettedMesh Candidate Motion Review</title>
<style>
  :root {{ color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; }}
  body {{ margin:0; background:#0a0e14; color:#e7edf4; display:grid; place-items:center; min-height:100vh; }}
  main {{ width:min(94vw,760px); padding:22px; box-sizing:border-box; }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub {{ color:#91a1b3; margin-bottom:16px; }}
  .row {{ display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }}
  button {{ border:1px solid #334356; border-radius:6px; padding:9px 14px; background:#182230; color:#dce7f3; cursor:pointer; }}
  button.active {{ background:#2d6683; border-color:#65b7dc; color:white; }}
  canvas {{ display:block; width:min(100%,560px); aspect-ratio:1; margin:14px auto; background:#09101a; border:1px solid #28394b; border-radius:8px; }}
  .status {{ text-align:center; color:#b8c6d5; font-variant-numeric:tabular-nums; }}
  .hint {{ background:#111a25; border-left:3px solid #5da7ca; padding:12px; color:#aebdcb; line-height:1.45; margin-top:14px; }}
  label {{ margin-left:auto; align-self:center; color:#aebdcb; }}
</style>
</head>
<body>
<main>
  <h1>VettedMesh Candidate — Human Motion Review</h1>
  <div class="sub">{display_name} · real generated sprite sheets · non-promoting review</div>
  <div id="actions" class="row"></div>
  <div id="directions" class="row"></div>
  <div class="row">
    <button id="play">Pause [Space]</button><button id="restart">Restart [R]</button>
    <label><input id="tour" type="checkbox" checked> Auto-tour all motions</label>
  </div>
  <canvas id="view" width="560" height="560"></canvas>
  <div id="status" class="status"></div>
  <div class="hint">Review foot sliding, attack contact and strength, elbow/knee/hip collapse, silhouette pops, and whether the death reads clearly. Keys 1–4 select motion; arrow keys select direction.</div>
</main>
<script>
const clips={data};
const names=['idle','walk','attack','death'], directions=['north','south','east','west'];
let name='idle', direction='south', started=performance.now(), paused=false, pausedAt=0, tour=true, image=null;
const canvas=document.querySelector('#view'), ctx=canvas.getContext('2d'), status=document.querySelector('#status');
ctx.imageSmoothingEnabled=true;
function clip(){{return clips.find(x=>x.name===name&&x.direction===direction)}}
function duration(c){{return c.loop?2500:(c.frames/c.fps*1000)+800}}
function setClip(nextName=name,nextDirection=direction){{name=nextName;direction=nextDirection;started=performance.now();paused=false;load();buttons()}}
function load(){{const c=clip();image=new Image();image.src='candidate/'+c.sheet}}
function buttonGroup(root, values, labels, current, setter){{root.innerHTML='';values.forEach((v,i)=>{{const b=document.createElement('button');b.textContent=labels[i];b.className=v===current?'active':'';b.onclick=()=>setter(v);root.appendChild(b)}})}}
function buttons(){{buttonGroup(document.querySelector('#actions'),names,['Idle [1]','Walk [2]','Attack [3]','Death [4]'],name,v=>setClip(v));buttonGroup(document.querySelector('#directions'),directions,['North [↑]','South [↓]','East [→]','West [←]'],direction,v=>setClip(name,v));document.querySelector('#play').textContent=paused?'Play [Space]':'Pause [Space]'}}
function toggle(){{if(paused){{started+=performance.now()-pausedAt;paused=false}}else{{pausedAt=performance.now();paused=true}}buttons()}}
function draw(now){{const c=clip(),elapsed=(paused?pausedAt:now)-started;if(tour&&!paused&&elapsed>=duration(c)){{setClip(names[(names.indexOf(name)+1)%names.length]);requestAnimationFrame(draw);return}}let frame=Math.floor(elapsed/1000*c.fps);frame=c.loop?frame%c.frames:Math.min(frame,c.frames-1);ctx.fillStyle='#09101a';ctx.fillRect(0,0,560,560);ctx.strokeStyle='#365068';ctx.globalAlpha=.65;ctx.beginPath();ctx.moveTo(38,459);ctx.lineTo(522,459);ctx.stroke();ctx.globalAlpha=1;if(image&&image.complete)ctx.drawImage(image,frame*256,0,256,256,0,0,560,560);status.textContent=`${{name}} / ${{direction}} · frame ${{frame+1}}/${{c.frames}} · ${{c.fps}} fps · ${{c.loop?'loop':'one-shot'}}`;requestAnimationFrame(draw)}}
document.querySelector('#play').onclick=toggle;document.querySelector('#restart').onclick=()=>setClip();document.querySelector('#tour').onchange=e=>tour=e.target.checked;
addEventListener('keydown',e=>{{if(e.key>='1'&&e.key<='4')setClip(names[+e.key-1]);else if(e.key==='ArrowUp')setClip(name,'north');else if(e.key==='ArrowDown')setClip(name,'south');else if(e.key==='ArrowRight')setClip(name,'east');else if(e.key==='ArrowLeft')setClip(name,'west');else if(e.key===' ')toggle();else if(e.key.toLowerCase()==='r')setClip();}});
load();buttons();requestAnimationFrame(draw);
</script>
</body>
</html>
'''


def _readme_text() -> str:
    return f"""# Darkness standalone Unity smoke check

This bundle checks the candidate in an isolated Unity project. It does not import anything into your game project or
modify one.

## Human motion review

- Open `review.html` for an immediate animation preview using the exact candidate sheets.
- For the Unity-native viewer, ensure Unity `{UNITY_VERSION}` is licensed, then run
  `powershell -ExecutionPolicy Bypass -File .\\open_unity_review.ps1`.
- The Unity Candidate Review window opens automatically. Reopen it from `Darkness > Candidate Review` if needed.

Both viewers provide Idle, Walk, Attack, and Death buttons, four directions, auto-tour, pause, and restart.

## Automated Unity smoke proof

1. Copy this complete folder to the Unity computer.
2. Ensure Unity `{UNITY_VERSION}` is licensed on that computer.
3. Run `powershell -ExecutionPolicy Bypass -File .\\run_unity_smoke.ps1`.
4. Copy the generated `result` folder back with this bundle.
5. Resume the Darkness pipeline; it will verify every returned hash before accepting the Unity proof.

The candidate remains non-promoting and requires human approval even after the smoke check passes.
"""


def build(package: Path, output: Path, template: Path) -> dict[str, object]:
    package = package.resolve()
    output = output.resolve()
    template = template.resolve()
    if not package.is_dir() or not template.is_dir():
        raise FileNotFoundError(package if not package.is_dir() else template)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest, master = _validate_candidate(package)

    candidate = output / "candidate"
    project = output / "UnitySmokeProject"
    candidate.mkdir()
    shutil.copytree(template, project)
    for source in package.iterdir():
        if source.is_file() and source.name != "candidate_unit_manifest.json":
            shutil.copy2(source, candidate / source.name)
    source_dir = candidate / "source"
    source_dir.mkdir()
    portable_master = source_dir / master.name
    shutil.copy2(master, portable_master)
    portable_manifest = dict(manifest)
    portable_manifest["source_master"] = f"source/{master.name}"
    portable_manifest_path = candidate / "candidate_unit_manifest.json"
    _write_json(portable_manifest_path, portable_manifest)

    (output / "run_unity_smoke.ps1").write_text(_runner_text(), encoding="utf-8")
    (output / "open_unity_review.ps1").write_text(_interactive_runner_text(), encoding="utf-8")
    (output / "review.html").write_text(_review_html(manifest), encoding="utf-8")
    (output / "README.md").write_text(_readme_text(), encoding="utf-8")
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    bundle = {
        "schema_version": 1,
        "bundle_kind": "darkness_standalone_unity_smoke",
        "asset_id": manifest["asset_id"],
        "unity_version": UNITY_VERSION,
        "unity_version_family": UNITY_VERSION_FAMILY,
        "unity_revision": UNITY_REVISION,
        "entrypoint": "run_unity_smoke.ps1",
        "project": "UnitySmokeProject",
        "candidate_manifest": "candidate/candidate_unit_manifest.json",
        "candidate_manifest_sha256": _sha256(portable_manifest_path),
        "source_master_sha256": manifest["source_master_sha256"],
        "directional_actions": 16,
        "expected_decoded_sprites": sum(int(action["frames"]) for action in manifest["actions"]),
        "unity_runtime_mutable_files": list(UNITY_RUNTIME_MUTABLE_FILES),
        "human_approval_required": True,
        "human_approved": False,
        "files": files,
    }
    _write_json(output / "bundle_manifest.json", bundle)
    return bundle


def main() -> int:
    args = _arguments()
    bundle = build(args.package, args.output_directory, args.template)
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
