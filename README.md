# Asset Forge

Text in, human-gated 3D asset out. Describe a character, creature, prop, or piece
of architecture; Asset Forge takes it through concept art, 3D generation, cleanup,
rigging, skinning, motion, surface painting, sprite rendering, and runtime
validation, stopping at named human gates so you stay in control of every stage.

This repository was extracted from [EmberDefense](https://github.com/iodriller/EmberDefense),
where the `darkness` compiler and the older `assetforge` sprite renderer it reuses
were originally built. It is still used there, now as a pinned dependency rather
than embedded game code.

## The pipeline

```
D0  Brief                    -- your description, compiled to a typed spec
D1  Concept            [gate] -- 2D turnaround art (text -> 2D)
D2  3D generation             -- candidate geometry (2D -> 3D)
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

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\pip install -e .
$env:PYTHONPATH = "."  # not required once installed, shown for clarity
python -m darkness studio --workspace C:/AssetForgeRuns --open-browser
```

This opens the local browser control plane at `http://127.0.0.1:8766`. Describe
one asset, pick a configuration profile, and step through the gates. Every
decision -- approve, reject, retry, edit, skip, or roll back -- is explained on
the page; see [Human control](#human-control) below for what each one does.

No GPU, Blender, ComfyUI, or Unity is required to explore the control plane
itself; those are only needed when a real worker actually runs. To see the
orchestration substrate work end to end without any of them:

```powershell
python -m darkness demo --workspace C:/AssetForgeRuns/demo
```

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

**Scope note on override values:** all six actions work at every gated stage,
but the JSON *override values* carried by retry/edit are currently consumed
only by D1 (Concept), where `{"seed": N}` pins the first concept seed. The
other stage runners accept and record overrides but do not yet read them --
see "Known gaps" below.

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
- **Retry/edit override values reach D1 only.** The mechanism is generic
  (`_begin()` returns a stage's pending overrides), but D1 is the only stage
  that reads them today. D2 additionally has no human gate, so a stage-level
  override could not reach it through the review flow regardless.
- **A human retry is not subject to the automatic iteration budget.** D1 stops
  Qwen's own correction loop after six iterations, but an explicit human retry
  resets the stage to `pending` and bypasses that ceiling. This is deliberate
  -- a person clicking retry should not be rate-limited by a guard meant for a
  runaway model -- but it does mean retry is unbounded by design.
- **`[adapter_defaults]` in `base.toml` is reference, not configuration.**
  Those values mirror the argparse defaults of standalone scripts under
  `adapters/` that you invoke by hand. Editing them changes nothing; only
  `[studio]`, `[quality.*]`, and `[stages.*]` are read by the pipeline.
- **Coverage is ~72%.** The untested remainder is concentrated in code that
  talks to real services: `darkness/studio_qwen.py`'s live LLM calls (52%),
  `external_worker.py`'s real subprocess path (22%), and `cli.py` (26%).

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
.venv\Scripts\pip install -e ".[dev]"
python -m pytest tests -q
python -m darkness demo --workspace C:/AssetForgeRuns/demo
```

The demo command exercises the full D0-D10 orchestration substrate
deterministically, without a GPU or any of the real generation workers -- it is
the fast regression check for anything that touches the compiler.

Anything that touches Blender, ComfyUI, Ollama/LocalDeploy, WSL2, or Unity needs
those tools installed and running locally. Everything else runs in CI on both
Windows and Ubuntu (`.github/workflows/ci.yml`), which also enforces two
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
asset-forge @ git+https://github.com/iodriller/asset-forge@v0.1.0
```

Pin a tag, not a branch. Because `adapters/`, `workers/`, `blender/`,
`qualifications/`, and the other resource directories are real repo-relative
files (worker subprocess scripts and manifests, not pure importable Python),
install in **editable** mode so they resolve correctly:

```powershell
pip install -e "git+https://github.com/iodriller/asset-forge@v0.1.0#egg=asset-forge"
```

A consuming project that wants its own profile overrides can add a
`profiles/base.toml` in its own working directory; `darkness.settings` prefers a
project's own profiles over the ones bundled with the package.
