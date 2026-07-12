# Asset Forge Darkness

This is the independent implementation described by
`DESIGN/asset_forge_darkness_master_plan.md`. It does not import or depend on the legacy
`tools/asset_forge` package.

The first foundation slice provides:

- a schema-validated model/worker registry with explicit priority and licensing state;
- recursive artifact-lineage checks that permit research while failing closed for release;
- bounded, schema-validating LocalDeploy structured-output retries;
- digest-pinned runtime qualification records that separate transport, schema, and semantic evidence;
- a qualification gate for any legacy component proposed for reuse.

The first live optimizer record is
`qualifications/qwen3.6-27b_ollama_rtx5090.json`. It qualifies two-image semantic review and four-image transport for
the exact recorded Qwen/Ollama digest. Eight-image use is not qualified: the tested backend returned an empty completion
inside a successful HTTP response, which the Darkness client correctly rejected.

Model weights and generated artifacts do not belong in this directory. Store them in external or gitignored
workspaces and record exact digests in run lineage.

Run tests from the EmberDefense root with a Python environment containing the project dependencies:

```powershell
python -m pytest tools/asset_forge_darkness/tests -q
```
