from pathlib import Path

from darkness.golden import evaluate, load_corpus, load_report
from darkness.studio_store import StudioStore
from vettedmesh_paths import source_revision


ROOT = Path(__file__).resolve().parents[1]


def test_static_prop_corpus_has_ten_diverse_versioned_cases() -> None:
    corpus = load_corpus(ROOT / "golden" / "static-props.json")
    assert corpus.required_attempts == len(corpus.cases) == 10
    assert corpus.minimum_passing_cases == 8
    assert len({item.category for item in corpus.cases}) >= 7
    assert all(len(item.required_features) >= 3 for item in corpus.cases)


def test_placeholder_report_fails_closed_with_actionable_reasons(tmp_path: Path) -> None:
    corpus = load_corpus(ROOT / "golden" / "static-props.json")
    report = load_report(ROOT / "golden" / "results.example.json")
    result = evaluate(corpus, report, tmp_path)
    assert result.eligible is False
    assert result.attempted == 0
    assert result.passed == 0
    assert len(result.cases) == 10
    assert all(item.reasons == ["not attempted"] for item in result.cases)
    assert any("live workers" in reason for reason in result.global_reasons)
    assert any("commit_sha" in reason for reason in result.global_reasons)


def test_new_runs_bind_the_observed_source_revision(tmp_path: Path) -> None:
    observed = source_revision()
    assert observed is not None
    run = StudioStore(tmp_path).create(
        "revision-proof",
        "A static qualification fixture with enough detail for the contract.",
    )
    assert run.source_revision == observed
