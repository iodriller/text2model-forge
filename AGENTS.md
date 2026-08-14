# AGENTS.md

## Project

Asset Forge is a text-to-3D asset compiler: a description goes in, and a
human-gated animated (or static) asset comes out through eleven named stages,
D0 through D10. `darkness/` is the compiler -- orchestration, contracts,
worker protocol, human gates, config resolution, and the browser control
plane. `assetforge/` is the older directional-sprite renderer darkness's D8/D9
stages reuse for its ComfyUI multiview client and Blender view baker.

This repository was extracted from EmberDefense, where both packages were
originally built and are still used as a pinned dependency. Nothing here may
depend on EmberDefense, any other specific game project, or any specific
character/creature identity -- this is a general-purpose tool. If you find a
reference back to a specific consuming project, that is a regression; fix it
at the root rather than adding a special case.

Use these durable sources of truth:

- `README.md` for the pipeline overview, configuration, and human-control
  actions.
- `docs/darkness-worker-guide.md` for detailed worker-by-worker commands and
  qualification history.
- `docs/legacy-sprite-pipeline.md` for the assetforge renderer's own commands.
- `darkness/profiles/base.toml` for every configuration default and its
  meaning -- this file's comments are the parameter reference.
- `darkness/schemas.py` and `darkness/studio_models.py` for the data
  contracts every stage and gate decision must satisfy.

## Commands

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
python -m pytest tests -q
python -m darkness demo --workspace C:/AssetForgeRuns/demo
python -m darkness config show --profile advanced
python -m darkness studio --workspace C:/AssetForgeRuns --open-browser
python -m darkness workers
```

`demo` and the test suite need no GPU, Blender, ComfyUI, or Unity -- they are
the fast, always-runnable checks. `studio` and `workers` reflect live worker
availability from `config.local.toml`, which is machine-specific and
gitignored; copy `machine.example.toml` to get one.

## Architecture Boundaries

- `darkness/schemas.py`, `darkness/studio_models.py`: the typed contracts.
  Adding a field means updating the contract, not reading an untyped dict at
  the call site.
- `darkness/compiler.py`, `darkness/workers.py`, `darkness/external_worker.py`:
  the generic job lifecycle (validate -> plan -> execute -> collect artifacts
  -> hard gates -> human approval -> promote). Stage-specific logic belongs in
  `darkness/studio_pipeline.py`'s `_run_dN` methods, not here.
- `darkness/lineage.py`, `darkness/export_policy.py`: license/lineage
  enforcement. This is the most mature, most safety-critical part of the
  system -- it is what stops a gated or territory-restricted model from
  silently reaching a release export. Never weaken a check here to unblock a
  single asset; fix the asset's lineage instead.
- `darkness/settings.py`: the layered configuration resolver. It only merges
  plain data and knows nothing about StudioRun; `studio_overrides()` is the
  one place that translates resolved config into StudioRun constructor
  fields. Adding a new tunable parameter means adding it to
  `darkness/profiles/base.toml` with a comment explaining it, not inventing a
  new ad hoc default somewhere else.
- `adapters/*.py`: one worker's subprocess entry point each, speaking the
  `ExternalWorkerRequest`/response file contract. An adapter must not import
  from `darkness/studio_pipeline.py`; the dependency runs the other way.
- Repo-relative resource resolution (`Path(__file__).resolve().parents[N]`)
  is real and load-bearing throughout `adapters/`, `darkness/`, and
  `assetforge/` -- these packages assume they are running from within a full
  checkout (editable install), not as pure importable libraries. If you
  change a file's directory depth, grep for `parents[` and fix every
  offset; the test suite mostly (not entirely) catches this by exercising
  the affected code paths, so also check by hand.

## Human Gate Invariants

- A gate decision (`StudioStore.decide()`) always hash-binds the evidence it
  saw and appends to an append-only `human_decisions` list. Never make a
  decision mutate or delete history -- that traceability is the whole point
  of the human-review contract.
- `reject`, `retry`, and `edit` invalidate everything downstream of the
  stage they're recorded against; `skip` does not; `rollback` invalidates
  from its target stage forward, inclusive. Get this wrong and a run can
  promote work built on since-invalidated evidence.
- Rejection (and edit, when it carries no explicit comment) requires a
  comment -- so the record explains what should change, not just that
  something was wrong.
- Do not add a new decision type without updating `_DECISIONS`,
  `_invalidate_from`'s callers, the CLI/web layer, and a test that exercises
  its state-machine effect end to end.

## Change Style

- Inspect the relevant schema, config default, and existing test before
  editing.
- Make the smallest coherent change; reuse `darkness/settings.py`,
  `darkness/schemas.py`, and the existing worker protocol rather than adding
  a parallel mechanism.
- A new stage parameter belongs in `darkness/profiles/base.toml`, documented,
  with a WIRED or DOCUMENTED marker in its section comment depending on
  whether `studio_pipeline.py` actually reads it yet.
- Keep `darkness/` and `assetforge/` decoupled from any specific consuming
  project. `tests/test_unity_candidate_contract.py` asserts the string
  "EmberDefense" never appears in the Unity smoke-check template; treat any
  similarly specific, non-generic identifier the same way.
- Prefer extending an existing adapter's typed request/response contract
  over adding a new ad hoc script.

## Verification

- Any change to `darkness/` or `assetforge/`: run `pytest tests -q` and
  `python -m darkness demo --workspace <tmp>`. Both are fast, deterministic,
  and require no external tools -- there is no excuse to skip them.
- A change to the gate/decision state machine (`studio_store.py`,
  `studio_models.py`): add or update a test in `tests/test_studio.py` that
  exercises the state transition directly, following the existing
  `_awaiting_stage_with_one_candidate`-style pattern rather than driving the
  full coordinator when you only need to test `decide()`'s mechanics.
- A change to `darkness/settings.py` or `profiles/*.toml`: add a case to
  `tests/test_settings.py` and confirm `darkness config show` still resolves
  correctly for `simple` and `advanced`.
- A change to `darkness/config.py` (worker bindings): re-verify
  `machine.example.toml` still round-trips through `load_local_config`.
- A change to an adapter, a Blender/ComfyUI script, or anything requiring a
  real worker: state plainly that it was not exercised end to end unless you
  actually ran it against real Blender/ComfyUI/GPU/Unity, and say so in the
  handoff. Do not claim a live-worker check ran without observed evidence.

## Git and Safety

- Preserve unrelated changes and keep commits focused.
- Use the configured repository-owner identity.
- Do not add assistant names, co-author trailers, session links, or tool
  attribution to Git artifacts.
- Never commit `config.local.toml` (real machine paths) -- it is gitignored;
  keep it that way.
- `qualifications/*.json` records real hardware and licensing findings about
  third-party models (some gated, some territory-restricted). Read one before
  assuming it is safe to change or remove; the lineage/export-policy system
  depends on these records staying accurate.
- Finish with what changed, what was verified, what was skipped, and
  remaining risks -- especially anything that needed a live worker this
  environment did not have.
