# Contributing to VettedMesh

VettedMesh accepts focused bug fixes, tests, documentation corrections, worker
adapters, and qualification evidence. It is an alpha-stage local application;
backward compatibility matters, but verified safety and honest capability labels
matter more.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-dev.lock
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e .
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m darkness demo --workspace C:/VettedMeshRuns/demo
```

On Linux or macOS use `.venv/bin/python`; the native launchers select the same
lock files automatically. Pull requests that touch
`darkness/` or `assetforge/` must run the full tests and deterministic demo.
Changes to a real worker must also state whether that worker was exercised on
real hardware; fakes do not qualify model quality.

## Change rules

- Open an issue before adding a dependency, worker, stage, or public contract.
- Keep model weights, credentials, `config.local.toml`, downloaded tools, and
  run outputs out of Git.
- Put portable parameters in `darkness/profiles/base.toml`; machine paths belong
  only in the ignored machine configuration.
- Add typed fields to the relevant Pydantic contract rather than passing
  untyped dictionaries through the pipeline.
- Preserve append-only human decisions and downstream invalidation semantics.
- Add a regression test and prove that it fails without the fix.
- Never describe fake-provider or fixture output as live model evidence.
- Record model code, weight, dependency, territory, and output terms before
  changing a worker to `research_ready` or `available`.

## Pull requests

Use a focused branch and fill out the pull-request template. Include exact
commands and outcomes. Installation reports must redact usernames, tokens,
private paths, prompts, and proprietary input assets.

This project uses the [Developer Certificate of Origin 1.1](DCO.txt), not a
separate contributor license agreement. Sign each commit with `git commit -s`
to certify that you have the right to submit it under the project license.
