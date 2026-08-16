"""Fail-closed evaluation and gallery generation for live qualification corpora."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import shutil
from typing import Literal

from pydantic import Field, model_validator

from .schemas import StrictModel
from .studio_store import StudioStore
from vettedmesh_paths import source_revision


class GoldenCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: str = Field(min_length=1)
    prompt: str = Field(min_length=20)
    required_features: list[str] = Field(min_length=1)


class GoldenCorpus(StrictModel):
    schema_version: Literal[1] = 1
    corpus_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1)
    target_profile: str = Field(min_length=1)
    required_attempts: int = Field(gt=0)
    minimum_passing_cases: int = Field(gt=0)
    cases: list[GoldenCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_thresholds(self):
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("golden corpus case ids must be unique")
        if self.required_attempts != len(self.cases):
            raise ValueError("required_attempts must equal the number of cases")
        if self.minimum_passing_cases > self.required_attempts:
            raise ValueError("minimum_passing_cases cannot exceed required_attempts")
        return self


class GoldenEnvironment(StrictModel):
    profile: str = Field(min_length=1)
    gpu_name: str = Field(min_length=1)
    vram_total_gb: float = Field(gt=0)
    operating_system: str = Field(min_length=1)
    real_workers: bool
    deterministic_fixture: bool
    model_revisions: dict[str, str]


class GoldenAssessment(StrictModel):
    case_id: str
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    reviewer: str = Field(min_length=1)
    concept_recognizable: bool
    geometry_usable: bool
    surface_usable: bool
    required_features_present: bool
    notes: str = Field(min_length=1)


class GoldenReport(StrictModel):
    schema_version: Literal[1] = 1
    corpus_id: str
    commit_sha: str
    environment: GoldenEnvironment
    assessments: list[GoldenAssessment]


class GoldenCaseResult(StrictModel):
    case_id: str
    run_id: str | None = None
    passed: bool
    reasons: list[str]
    preview_path: str | None = None


class GoldenEvaluation(StrictModel):
    corpus_id: str
    eligible: bool
    attempted: int
    passed: int
    required_attempts: int
    minimum_passing_cases: int
    global_reasons: list[str]
    cases: list[GoldenCaseResult]


def load_corpus(path: Path) -> GoldenCorpus:
    return GoldenCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def load_report(path: Path) -> GoldenReport:
    return GoldenReport.model_validate_json(path.read_text(encoding="utf-8"))


def _final_preview(run) -> str | None:
    images = [item.relative_path for item in run.stage("D10").evidence if item.media_type.startswith("image/")]
    return images[-1] if images else None


def evaluate(corpus: GoldenCorpus, report: GoldenReport, workspace: Path) -> GoldenEvaluation:
    global_reasons: list[str] = []
    if report.corpus_id != corpus.corpus_id:
        global_reasons.append("report corpus_id does not match the selected corpus")
    if not re.fullmatch(r"[0-9a-f]{40}", report.commit_sha) or set(report.commit_sha) == {"0"}:
        global_reasons.append("commit_sha must identify the non-placeholder tested commit")
    observed_revision = source_revision()
    if observed_revision is None:
        global_reasons.append(
            "the evaluator cannot prove its source commit; use a Git checkout or set VETTEDMESH_SOURCE_REVISION"
        )
    elif report.commit_sha != observed_revision:
        global_reasons.append(
            f"report commit_sha does not match the evaluator source commit {observed_revision}"
        )
    if report.environment.profile != corpus.target_profile:
        global_reasons.append(f"profile must be {corpus.target_profile!r}")
    if not (7.0 <= report.environment.vram_total_gb <= 8.5):
        global_reasons.append("the 8 GB qualification requires an observed GPU between 7.0 and 8.5 GB")
    if not report.environment.real_workers or report.environment.deterministic_fixture:
        global_reasons.append("qualification must use live workers, not the deterministic fixture")
    if not report.environment.model_revisions or any(not value.strip() for value in report.environment.model_revisions.values()):
        global_reasons.append("exact model revisions must be recorded")

    by_id: dict[str, GoldenAssessment] = {}
    duplicates: set[str] = set()
    for item in report.assessments:
        if item.case_id in by_id:
            duplicates.add(item.case_id)
        by_id[item.case_id] = item
    if duplicates:
        global_reasons.append("duplicate assessments: " + ", ".join(sorted(duplicates)))
    unknown = sorted(set(by_id) - {item.case_id for item in corpus.cases})
    if unknown:
        global_reasons.append("unknown corpus cases: " + ", ".join(unknown))

    store = StudioStore(workspace)
    results: list[GoldenCaseResult] = []
    for case in corpus.cases:
        assessment = by_id.get(case.case_id)
        reasons: list[str] = []
        preview: str | None = None
        if assessment is None:
            reasons.append("not attempted")
            results.append(GoldenCaseResult(case_id=case.case_id, passed=False, reasons=reasons))
            continue
        try:
            run = store.load(assessment.run_id)
        except (FileNotFoundError, ValueError) as exc:
            reasons.append(f"run evidence is unavailable: {exc}")
        else:
            if run.description.strip() != case.prompt.strip():
                reasons.append("run description does not exactly match the corpus prompt")
            if run.source_revision != report.commit_sha:
                reasons.append("run source revision does not match the report commit")
            if run.spec is None or run.spec.behavior != "static":
                reasons.append("D0 did not compile the case as a static asset")
            if run.state != "completed":
                reasons.append(f"run state is {run.state!r}, not 'completed'")
            incomplete = [
                stage.stage_id
                for stage in run.stages
                if stage.applicable and stage.state != "approved"
            ]
            if incomplete:
                reasons.append("applicable stages not approved: " + ", ".join(incomplete))
            final_decisions = run.stage("D10").human_decisions
            if not final_decisions or final_decisions[-1].decision != "approve":
                reasons.append("D10 has no final hash-bound human approval")
            preview = _final_preview(run)
            if preview is None:
                reasons.append("D10 has no final image evidence")
        if not assessment.concept_recognizable:
            reasons.append("human review marked the concept unrecognizable")
        if not assessment.geometry_usable:
            reasons.append("human review marked the geometry unusable")
        if not assessment.surface_usable:
            reasons.append("human review marked the surface unusable")
        if not assessment.required_features_present:
            reasons.append("human review found required features missing")
        results.append(
            GoldenCaseResult(
                case_id=case.case_id,
                run_id=assessment.run_id,
                passed=not reasons,
                reasons=reasons,
                preview_path=preview,
            )
        )

    attempted = sum(item.case_id in by_id for item in corpus.cases)
    passed = sum(item.passed for item in results)
    eligible = (
        not global_reasons
        and attempted == corpus.required_attempts
        and passed >= corpus.minimum_passing_cases
    )
    return GoldenEvaluation(
        corpus_id=corpus.corpus_id,
        eligible=eligible,
        attempted=attempted,
        passed=passed,
        required_attempts=corpus.required_attempts,
        minimum_passing_cases=corpus.minimum_passing_cases,
        global_reasons=global_reasons,
        cases=results,
    )


def write_gallery(evaluation: GoldenEvaluation, workspace: Path, output: Path) -> Path:
    media = output.parent / f"{output.stem}-media"
    media.mkdir(parents=True, exist_ok=True)
    store = StudioStore(workspace)
    cards: list[str] = []
    for item in evaluation.cases:
        preview_html = "<p>No final preview.</p>"
        if item.run_id and item.preview_path:
            source = store.artifact_path(item.run_id, item.preview_path)
            if source.is_file():
                target = media / f"{item.case_id}{source.suffix.lower()}"
                shutil.copyfile(source, target)
                preview_html = f'<img src="{html.escape(media.name + "/" + target.name)}" alt="{html.escape(item.case_id)}">'
        reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in item.reasons) or "<li>All qualification gates passed.</li>"
        cards.append(
            f'<article class="{("pass" if item.passed else "fail")}"><h2>{html.escape(item.case_id)}</h2>'
            f"{preview_html}<p><strong>{'PASS' if item.passed else 'FAIL'}</strong></p><ul>{reasons}</ul></article>"
        )
    document = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(evaluation.corpus_id)}</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:auto;padding:2rem;background:#111;color:#eee}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}}article{{border:2px solid #a44;padding:1rem;border-radius:.6rem;background:#1b1b1b}}article.pass{{border-color:#4a6}}img{{width:100%;height:240px;object-fit:contain;background:#080808}}</style>
<h1>{html.escape(evaluation.corpus_id)}</h1><p>{evaluation.passed}/{evaluation.required_attempts} passed; threshold {evaluation.minimum_passing_cases}. Eligibility: <strong>{evaluation.eligible}</strong>.</p>
<main>{''.join(cards)}</main>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
