# VettedMesh

VettedMesh is a developer preview of an auditable, human-gated compiler for
local AI-assisted 3D assets. It records every candidate, review decision,
correction, model lineage, and downstream invalidation across eleven named
stages. The deterministic orchestration and browser control plane are tested;
the complete live generation chain is still experimental.

The first supported product target is a static-prop path:

`description → approved concept → generated geometry → cleanup → surface → validated GLB`

Rigging, skinning, motion, directional sprites, and Unity validation remain
available as research stages, not as production-ready promises. No real asset
has yet completed the entire D0–D10 chain on one machine; see
[Known gaps](#known-gaps) and [the support matrix](SUPPORT.md) before installing
large models.

The control plane is free and local. An 8 GB NVIDIA profile exists for a
specific sequential ComfyUI workflow, but it is experimental until the
published golden corpus passes on independently reproduced 8 GB hardware.
Optional models have their own licenses and territorial restrictions; see
[Third-party and model terms](THIRD_PARTY_LICENSES.md). The deliberately cold
[public-readiness assessment](docs/public-readiness.md) compares the closest
open repositories and defines the release gates used here.

## The pipeline

```
D0  Brief                    -- your description, compiled to a typed spec
D1  Concept            [gate] -- 2D concept art (text -> 2D). Retry/edit here.
D2  3D generation             -- the image YOU APPROVED becomes the mesh (2D -> 3D)
D3  Cleanup                   -- topology repair, retopology
D4  Canonical structure [gate] -- landmarks, skeleton
D5  Rig / articulation         -- bones, joints
D6  Skinning / deformation      -- weights
D7  Motion              [gate] -- animation, including donor-motion retargeting
D8  Surface painting    [gate] -- materials, textures
D9  Sprite rendering            -- directional sheets, LODs, atlases
D10 Runtime validation  [gate] -- an isolated Unity smoke check
```

Static assets (props, architecture, materials) skip the stages that only apply to
animated, rigged things -- see `asset_kind` and `behavior` in a run's spec.

| Scope | Current evidence |
|---|---|
| Contracts, gates, retries, lineage, configuration, browser UI | Automated matrix targets Windows, Ubuntu, and macOS; current hosted result pending |
| D0–D10 deterministic demo | Uses real parseable fixtures and fake generation providers |
| Individual Blender and geometry workers | Partial qualification records under `qualifications/` |
| Complete static-prop live chain | Golden-corpus qualification pending |
| Complete animated live chain | Not qualified |

The live threshold is executable, not a prose promise: `python -m darkness
golden show` prints ten versioned static-prop prompts, and `python -m darkness
golden evaluate --workspace <runs> --results <report.json> --html
golden-report.html` verifies the underlying completed runs, final hash-bound
human approvals, exact model revisions, 8 GB environment, and an 8/10 minimum.
The checked-in example report intentionally fails until real evidence exists.

## Quick start

### One-command setup and start

On Windows, run the PowerShell launcher from a fresh checkout:

```powershell
.\vettedmesh.ps1
```

On Linux or macOS, use the native Bash launcher:

```bash
bash ./vettedmesh.sh
```

The launchers check for Python 3.12+, create the reusable `.venv`, install
the editable project when needed, create a machine-local config only when one
does not already exist, run the offline D0-D10 smoke test, start selected
local services, and opens Studio. Linux/macOS use a pinned per-user `uv`
bootstrap when Python is missing, then fall back to Homebrew or the native
system package manager only where a tool genuinely needs it. Windows tries
WinGet user scope, normal scope, and finally UAC elevation. The first run
offers four setups:

- **Full Qwen stack** -- Ollama + a small Qwen reviewer, ComfyUI, native Qwen
  Image 2512, SDXL for surface work, Hunyuan3D-2, and Blender.
- **Full SDXL stack** -- the same local base stack with SDXL selected for D1.
- **Existing services** -- keep using ComfyUI/reviewer/models already installed.
- **Core only** -- install and run the control plane without live generators.

Large model downloads show progress and require accepting their own model
terms. For unattended setup, make every choice explicit:

```powershell
.\vettedmesh.ps1 -Action install -AiStack qwen -NonInteractive `
  -AcceptSdxlLicense -AcceptHunyuanLicense
.\vettedmesh.ps1 -AiStack existing       # later runs: check core, then start
.\vettedmesh.ps1 -Action repair           # retry/repair the selected setup
.\vettedmesh.ps1 -Action doctor           # concise service + worker readiness
```

```bash
bash ./vettedmesh.sh install --ai-stack qwen --non-interactive \
  --accept-sdxl-license --accept-hunyuan-license
bash ./vettedmesh.sh start --ai-stack existing
bash ./vettedmesh.sh repair
bash ./vettedmesh.sh doctor
```

Package setup is bounded by three attempts by default (`-MaxAttempts` on
Windows, `--max-attempts` on Linux/macOS). Pass `-NoElevation` or
`--no-elevation` to prohibit the administrator fallback; non-interactive sudo
uses `sudo -n` so it cannot hang waiting for a password. `-NoBrowser` or
`--no-browser` suppresses both browser tabs. Existing
`config.local.toml` files are never overwritten. The launcher installs the
usable local D0-D4/D8-D9 base, but the System page still reports specialized
workers, motion sources, and Unity validation that require separate
qualification or tooling.

### Docker Compose

Docker runs Studio plus the local Qwen reviewer with persistent named volumes:

```bash
docker compose up --build
```

Open `http://127.0.0.1:8766`. The first run also pulls
`qwen3-vl:8b-instruct`; model data and Studio runs survive container rebuilds.
Studio is published on host loopback only because it has no account/login
boundary. On an NVIDIA Docker host with the NVIDIA Container Toolkit, enable
GPU access for Ollama with:

```bash
docker compose -f compose.yaml -f compose.nvidia.yaml up --build
```

ComfyUI stays host-native in this portable Compose setup so Windows, Linux,
and Apple Silicon can each use their supported GPU runtime. The supplied
container config looks for it at `host.docker.internal:8188`; a host ComfyUI
instance must use a container-reachable bind, with the host firewall keeping
8188 private. Override the mounted typed config with
`VETTEDMESH_DOCKER_CONFIG=/absolute/path/config.local.toml` when services live
elsewhere. `docker compose config` is a fast configuration check, and
`docker compose down` stops the stack without deleting either named volume.

### Manual/core-only setup

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements.lock
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e .
.venv\Scripts\python -m darkness studio --workspace ./VettedMeshRuns --open-browser
```

Linux or macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m darkness studio --workspace ./VettedMeshRuns --open-browser
```

This opens the local browser control plane at `http://127.0.0.1:8766`. Describe
one asset, pick a configuration profile, optionally override the text-to-2D
backend (Auto, native Qwen Image 2512, or any installed SDXL checkpoint),
reviewer model, spec strategy, steps, and CFG, then step through the gates. The
form discovers installed ComfyUI and reviewer models when those services are
online. Every
decision -- approve, reject, retry, edit, skip, or roll back -- is explained on
the page; see [Human control](#human-control) below for what each one does.

No GPU, Blender, ComfyUI, or Unity is required to explore the control plane
itself; those are only needed when a real worker actually runs. To see the
orchestration substrate work end to end without any of them:

```powershell
python -m darkness demo --workspace C:/VettedMeshRuns/demo
```

## Check the machine first

```powershell
python -m darkness doctor          # ~1s: hardware, recommended stack, every assumption
python -m darkness doctor --deep   # ~17s: also launches Blender to verify its API and your donor rig
```

`doctor` answers two questions `darkness workers` cannot. First, *what stack
does this hardware imply* -- the reviewer model's real budget is what is left
after ComfyUI, not the size of the card, and getting that wrong makes every
vision review look like a timeout. Second, *do the assumptions between stages
hold here* -- whether ComfyUI has the nodes your D2 backend needs, whether the
D3 voxel is fine enough for the fingers D7 will look for, whether the donor
motion's bones match the retarget contract.

Those are the failures that otherwise surface three stages and twenty minutes
into a run, one at a time. `doctor` finds all of them at once, each with the
fix, and exits non-zero if any would fail.

Pick the matching profile rather than tuning settings individually:

```powershell
python -m darkness config show --profile 8gb
```

`profiles/8gb.toml` is one decision -- "8 GB card, ComfyUI and the reviewer
co-resident" -- and every value in it follows from that. See
`docs/free-local-8gb-setup.md` for the measurements behind each one.

## Configuration

Every setting lives in a layered TOML configuration, resolved in this order
(later wins):

1. `darkness/profiles/base.toml` -- every default, documented, bundled with the
   package.
2. `profiles/<name>.toml` -- a named profile. `simple` (the ~10 knobs worth
   naming: quality, optimizer on/off) and `advanced` (every stage parameter) ship
   with the package; add your own under a project's own `profiles/` directory to
   override them.
3. `config.local.toml`'s `[studio_defaults]` table -- this machine's own
   overrides. Gitignored; copy `machine.example.toml` to get started.
4. A run's own settings, set once when it is created.
5. A stage-level override, supplied at a retry or edit decision.

See the resolved result at any time:

```powershell
python -m darkness config show --profile advanced
```

`config.local.toml` also carries worker bindings -- which binary or interpreter
launches each generation worker (Blender, ComfyUI, TripoSG, TRELLIS.2, ...) and
what environment variables it needs. Copy `machine.example.toml` to
`config.local.toml` and fill in real paths for your machine; see
`docs/darkness-worker-guide.md`.

## Human control

Every gated stage stops and waits. At a gate you can:

| Action | Effect |
|---|---|
| **Approve** | Pick a candidate; the next stage may run. |
| **Reject** | Comment required. The next attempt sees your comment and the full history. |
| **Retry** | Reroll the same stage fresh -- no comment, no quality judgement implied, and no targeted correction. Pass override values (JSON) to change parameters for the next attempt; at D1, `{"seed": 4242}` pins the first seed so you can re-roll a candidate deterministically. |
| **Edit** | Like reject, but the correction is concrete: a comment, override values, or both. Your comment reaches the model exactly as a rejection's would, and drives the same targeted-correction path. |
| **Skip** | Mark a stage not applicable, with a required reason. Unlike the others, this does not invalidate anything downstream. |
| **Roll back** | Reopen an earlier, already-decided stage and invalidate everything from there forward. |

Every decision hash-binds the evidence it saw and is recorded in an append-only
history -- nothing is silently overwritten, and evidence from an invalidated
attempt stays in the run's event log even after a stage resets.

Invalidation never reopens a stage the compiled D0 contract ruled out. A static
prop has no skeleton, rig, skin weights, or motion, so rejecting its concept --
or rolling back past those stages -- leaves them skipped rather than scheduling
work the asset does not need. Only re-running D0, which recompiles the spec,
can change whether a stage applies.

Every stage in the timeline is a link to its own page, so an approved stage's
evidence, every Qwen review it received, and the full human decision record
stay reachable for the life of the run -- not only while it is the current
stage.

**Scope note on override values:** all six actions work at every gated stage,
but the JSON *override values* carried by retry/edit are consumed only where
the underlying stage actually has a per-attempt parameter to change:

- **D1 (Concept)** -- `{"seed": N}` pins the first concept seed.
- **D4 (Canonical structure), non-rigid path only** -- `{"landmark_adjustments":
  {...}, "weight_adjustments": [...], "render_size": N, "maximum_material_change_fraction": F,
  "maximum_bone_influences": N}` reach the Blender rig-proposal worker
  directly; `adapters/blender_worker.py` applies landmark offsets and
  joint-pair weight transfers and validates their deep shape itself.

D7 (Motion), D8 (Surface), and D10 (Runtime validation) are not seed-driven
or parameter-driven the way D1/D4 are -- D7 retargets from a fixed donor
motion catalog, D8's paint bake is deterministic from the spec text, and D10
only packages already-approved D9 evidence. There is no per-attempt lever in
any of the three to wire an override onto; adding one there would be a
control with nothing behind it. A rejection or edit at those stages still
works exactly as everywhere else -- the comment carries a correction to
Qwen, which is how D7/D8's own mediator loop actually improves an attempt.

### Driving a gate headlessly

The browser is not the only way to record a decision. `darkness studio`
without a subcommand launches it exactly as before; three subcommands drive
the same `StudioStore`/`StudioCoordinator` without it:

```powershell
python -m darkness studio list --workspace C:/VettedMeshRuns
python -m darkness studio show --workspace C:/VettedMeshRuns --run-id asset-...
python -m darkness studio decide --workspace C:/VettedMeshRuns --run-id asset-... `
  --stage-id D1 --decision approve --selected-evidence-id d1-i01-candidate-1 `
  --comment "Looks good."
```

`decide` calls the identical `StudioStore.decide()` the web form's
`/decision` route calls, so validation and state-machine effect are exactly
the same either way. By default it then drives the pipeline on the calling
thread until the next stopping point (a gate, completion, failure, or block)
and prints the resulting run state; pass `--no-resume` to only record the
decision, matching a browser tab that records a decision and is then closed
without the coordinator being resubmitted. `--overrides` takes the same JSON
object the web form's Overrides field does.

**Do not run `darkness studio decide` against the same `--workspace` while
`darkness studio serve` is running against it.** `StudioStore`'s lock is
in-process only, not a cross-process file lock; a concurrent load-modify-save
from each process can silently clobber the other's write.

## The LLM optimizer is optional

`darkness/optimizer.py` is a constrained diagnostic loop: given evidence and a
list of registered operations, it may only propose operation ids that already
exist, is schema- and parameter-validated, and gets bounded correction rounds
against a deterministic validator before failing closed. It is not wired into
the main D0-D10 flow automatically -- it is invoked explicitly (see
`adapters/optimize_short_biped_rig.py`), and `[optimizer].enabled` in
`profiles/base.toml` documents that off-by-default intent. Its first real
episode retained the starting baseline rather than improving it, so it is not
recommended as a default until it has a measured win.

## Motion: donor retargeting

`adapters/retarget_humanoid_motion.py` retargets a donor animation (e.g. a walk
cycle from a separate CC0 pack) onto a newly generated rig, producing a
rest-relative retarget with numeric proof, an LLM critic review, and an
independent mediator. `[stages.D7].donor_motion_id` in the config is the hook
for selecting a specific donor clip at that gate.

## Known gaps

Stated plainly so nobody discovers these the hard way:

- **The generators themselves are never exercised by the test suite.** Both
  subprocess boundaries are injectable -- `worker_executor` for the typed
  worker protocol (D2-D5, D9) and `script_runner` for the `adapters/` helper
  scripts (D7's motion chain, D8's surface bake) -- and both full chains, a
  static prop and a deformable character, run end to end against fakes. What
  that proves is orchestration: stage sequencing, gates, evidence, hashes,
  adoption contracts, and config reaching real worker requests. It proves
  nothing about whether TRELLIS.2 makes a good mesh or whether the surface
  bake looks right. Only a real GPU run does that.
- **No asset has been produced end to end on real hardware.** The chair test
  proves the prop path's stages run and hand off correctly; it does not prove
  a recognizable chair comes out. Expect the D1 concept stage in particular to
  need work for non-character subjects -- it is built around turnarounds,
  OpenPose ControlNet, and equipment layout.
- **Retry/edit override values are intentionally limited to D1 and the
  non-rigid D4 path.** Those are the stages with real per-attempt controls.
  D2 has no human gate, while D7, D8, and D10 currently improve through the
  correction comment rather than unimplemented numeric knobs.
- **A human retry is not subject to the automatic iteration budget.** D1 stops
  Qwen's own correction loop after six iterations, but an explicit human retry
  resets the stage to `pending` and bypasses that ceiling. This is deliberate
  -- a person clicking retry should not be rate-limited by a guard meant for a
  runaway model -- but it does mean retry is unbounded by design.
- **`[adapter_defaults]` in `base.toml` is reference, not configuration.**
  Those values mirror the argparse defaults of standalone scripts under
  `adapters/` that you invoke by hand. Editing them changes nothing; only
  `[studio]`, `[quality.*]`, and `[stages.*]` are read by the pipeline.
- **Coverage is measured in CI, but is not a live-worker qualification.** The
  least-covered paths are the ones that talk to real model services and
  subprocess workers. A line-coverage percentage cannot substitute for the
  versioned live corpus above.

## Repository layout

```
darkness/            The D0-D10 compiler: schemas, orchestration, worker
                      protocol, human gates, config resolver, browser UI.
assetforge/           The older directional-sprite renderer; darkness's D8/D9
                      stages reuse its ComfyUI client and Blender view baker.
adapters/             Worker entry-point scripts (Blender, TripoSG, TRELLIS.2,
                      InstantMesh, AutoRemesher, motion retargeting, ...).
blender/               Blender-side scripts assetforge's renderer shells out to.
workers/               Portable worker manifests (lifecycle, licensing state).
qualifications/         Hashed evidence records from qualifying each worker.
registry/               Candidate registry consulted by `forge package`.
unity_smoke_template/    A minimal, pinned Unity project for D10 validation --
                          no game-project dependency.
characters/, creatures/, genesis/, presets/, assets/
                          assetforge example fixtures and species/family schemas.
docs/                    Detailed historical references (darkness-worker-guide,
                          legacy-sprite-pipeline, the art-direction rubric).
tests/                   Both packages' test suites (pytest).
```

## Development

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-dev.lock
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e .
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m darkness demo --workspace C:/VettedMeshRuns/demo
```

`pyproject.toml` declares supported dependency ranges. `uv.lock` is the full
cross-platform resolution, and the four `requirements*.lock` exports pin every
artifact by SHA-256 for runtime, development, local-AI, and combined installs.
The launchers, Dockerfile, and CI use those lock exports. The three unpinned
`requirements*.txt` files remain convenience entry points for dependency
upgrades; do not use them for release or CI installs. See
[`docs/packaging.md`](docs/packaging.md) for the refresh procedure.

The demo command exercises the full D0-D10 orchestration substrate
deterministically, without a GPU or any of the real generation workers -- it is
the fast regression check for anything that touches the compiler.

Anything that touches Blender, ComfyUI, Ollama/LocalDeploy, WSL2, or Unity needs
those tools installed and running locally. Everything else runs in CI on
Windows, Ubuntu, and macOS (`.github/workflows/ci.yml`), which also enforces two
coupling guards: no reference to a specific consuming project, and no
pre-extraction repo-root path assumptions.

Test order is randomized on every run by `pytest-randomly` -- it earned its
place by catching an order-dependent race in the web test fixture that a fixed
order never surfaced. Reproduce a specific ordering with the seed printed on
failure:

```powershell
python -m pytest tests -q -p randomly --randomly-seed=12345
python -m pytest tests -q -p no:randomly            # disable shuffling
python -m pytest tests -q --cov=darkness --cov=assetforge --cov-report=term-missing
```

### Writing tests for stages

Two injection seams let a test drive real stage logic without real tools, and
`tests/test_studio.py`'s fakes show the pattern:

- `worker_executor=` replaces the typed worker protocol (D2-D5, D9).
- `script_runner=` replaces the `adapters/` helper scripts (D7, D8).

Two rules worth keeping, because both caught real bugs:

1. **Fakes must produce real artifacts.** `FakeWorkerExecutor` writes an
   actual `trimesh` mesh because D2 genuinely parses it and computes
   vertex/face/watertightness from it. A stub file would pass the test and
   hide the defect.
2. **Prove a regression test fails without its fix.** Revert the fix, confirm
   the exact original error, restore. A test that passes both ways is
   decoration.

## Using this from another project

```
# requirements.txt or pyproject.toml dependency
vettedmesh @ git+https://github.com/iodriller/vettedmesh@v0.2.0-rc.1
```

Pin a tag, not a branch. The wheel carries `adapters/`, `workers/`, `blender/`,
`qualifications/`, and the other runtime resources under the environment's
`share/vettedmesh` directory, so an ordinary VCS install is supported:

```powershell
pip install "vettedmesh @ git+https://github.com/iodriller/vettedmesh@v0.2.0-rc.1"
```

A consuming project that wants its own profile overrides can add a
`profiles/base.toml` in its own working directory; `darkness.settings` prefers a
project's own profiles over the ones bundled with the package.
