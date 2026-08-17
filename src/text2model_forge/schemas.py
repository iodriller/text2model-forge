"""Strict versioned contracts for the initial Text2Model safety controls."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidatePriority(str, Enum):
    open_source_first = "open_source_first"
    research_primary = "research_primary"
    later_candidate = "later_candidate"


class ReleaseEligibility(str, Enum):
    permitted = "permitted"
    review_required = "review_required"
    blocked = "blocked"


class AssetStage(str, Enum):
    brief = "D0_brief"
    concept = "D1_concept"
    geometry = "D2_geometry"
    anatomy = "D3_anatomy"
    topology = "D4_topology"
    rig = "D5_rig"
    skinning = "D6_skinning"
    motion = "D7_motion"
    surface = "D8_surface"
    optimization = "D9_optimization"
    export = "D10_export"


class AssetComponent(StrictModel):
    component_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    semantic_type: Literal[
        "deformable_body",
        "deformable_clothing",
        "rigid_attachment",
        "rigid_equipment",
        "effect_origin",
        "collision_proxy",
    ]
    rig_policy: str = Field(min_length=1)
    required: bool = True


class DestinationContract(StrictModel):
    engine: Literal["unity", "generic"] = "unity"
    delivery: list[Literal["blend", "fbx", "glb", "prefab", "sprite_sheet"]] = Field(min_length=1)
    render_pipeline: str | None = None
    rig_type: Literal["humanoid", "generic", "none"] = "generic"
    unit_scale_meters: float = Field(default=1.0, gt=0)
    triangle_budget_lod0: int = Field(default=50000, ge=3)
    texture_resolution: int = Field(default=2048, ge=64)
    release_regions: list[str] = Field(default_factory=lambda: ["global"])


class AssetBrief(StrictModel):
    schema_version: Literal[1] = 1
    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    asset_type: Literal["character", "prop", "environment", "material", "vfx"]
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    style: str = Field(min_length=1)
    anatomy_family: str | None = None
    height_m: float | None = Field(default=None, gt=0)
    components: list[AssetComponent] = Field(default_factory=list)
    required_animations: list[str] = Field(default_factory=list)
    destination: DestinationContract
    locked_features: list[str] = Field(default_factory=list)
    approval_required: bool = True


class GpuMemoryEnvelope(StrictModel):
    """Peak-memory evidence attached to one exact worker configuration.

    ``estimated`` is useful for planning but never sufficient for strict
    admission.  ``measured`` means the peak came from a real run identified by
    ``qualification_id``; changing a model, driver, precision, or resolution
    requires a new envelope rather than silently reusing the old number.
    """

    peak_vram_gb: float = Field(gt=0)
    status: Literal["estimated", "measured"] = "estimated"
    backend: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    qualification_id: str | None = None
    cpu_compute_allowed: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def measured_has_evidence(self) -> "GpuMemoryEnvelope":
        if self.status == "measured" and not self.qualification_id:
            raise ValueError("a measured GPU memory envelope requires qualification_id")
        return self


class WorkerCapability(StrictModel):
    schema_version: Literal[1] = 1
    worker_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    worker_version: str = Field(min_length=1)
    candidate_id: str | None = None
    stages: list[AssetStage] = Field(min_length=1)
    operations: list[str] = Field(min_length=1)
    requires_gpu: bool = False
    exclusive_gpu: bool = False
    # A declaration, not a promise: only ``measured`` envelopes may be used
    # for fail-closed admission.  Unknown workers remain discoverable without
    # pretending an estimate is qualification evidence.
    gpu_memory: GpuMemoryEnvelope | None = None
    supports_cancel: bool = True
    deterministic: bool = False
    environment_digest: str | None = None
    input_media_types: list[str] = Field(default_factory=list)
    output_media_types: list[str] = Field(default_factory=list)


class WorkerManifest(StrictModel):
    schema_version: Literal[1] = 1
    capability: WorkerCapability
    adapter: Literal["builtin", "localdeploy", "http", "subprocess"]
    lifecycle: Literal["available", "research_ready", "discovered", "blocked"]
    source_url: str | None = None
    pinned_revision: str | None = None
    environment_kind: Literal["builtin", "windows", "wsl2", "container", "remote"]
    executable_candidates: list[str] = Field(default_factory=list)
    health_url: str | None = None
    license_gate: ReleaseEligibility
    blockers: list[str] = Field(default_factory=list)


class WorkerBinding(StrictModel):
    command_prefix: list[str] = Field(min_length=1)
    environment: dict[str, str] = Field(default_factory=dict)


class Text2ModelLocalConfig(StrictModel):
    schema_version: Literal[1] = 1
    workspace_root: str = Field(min_length=1)
    workers: dict[str, WorkerBinding] = Field(default_factory=dict)
    # This machine's [studio_defaults] overlay. src/text2model_forge/settings.py reads it
    # out of the SAME config.local.toml (see resolve_settings' layer 3) and
    # README documents it, but this model is strict -- so without the field
    # declared here, writing the documented table made load_local_config()
    # raise extra_forbidden and took the worker bindings down with it.
    # Deliberately untyped values: settings.py owns their meaning, this model
    # only owns the worker bindings, and duplicating the studio schema here
    # would give two places to update for every new tunable.
    studio_defaults: dict[str, Any] = Field(default_factory=dict)
    # Per-stage counterpart of studio_defaults; see settings.resolve_settings.
    stage_defaults: dict[str, Any] = Field(default_factory=dict)


class ExternalWorkerRequest(StrictModel):
    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    operation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    stage: AssetStage
    inputs: list[ArtifactRecord] = Field(default_factory=list)
    input_paths: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_directory: str = Field(min_length=1)
    device_policy: Literal["prefer_gpu", "gpu_compute_only", "strict_device_only"] = "prefer_gpu"
    gpu_safety_margin_gb: float = Field(default=0.75, ge=0)


class ExternalWorkerOutput(StrictModel):
    path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    role: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalWorkerResponse(StrictModel):
    schema_version: Literal[1] = 1
    job_id: str
    status: Literal["succeeded", "failed", "cancelled"]
    outputs: list[ExternalWorkerOutput] = Field(default_factory=list)
    diagnostics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class OperationDefinition(StrictModel):
    schema_version: Literal[1] = 1
    operation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    worker_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    stages: list[AssetStage] = Field(min_length=1)
    description: str = Field(min_length=1)
    input_count_min: int = Field(default=1, ge=0)
    input_count_max: int = Field(default=1, ge=0)
    output_media_type: str = Field(min_length=1)
    deterministic: bool = False
    parameter_schema: dict[str, Any] = Field(default_factory=dict)


class CandidateRecord(StrictModel):
    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    license_url: str | None = None
    priority: CandidatePriority
    research_allowed: bool
    release_eligibility: ReleaseEligibility
    min_vram_gb: float | None = Field(default=None, ge=0)
    max_vram_gb: float | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)


class CandidateRegistry(StrictModel):
    schema_version: Literal[1] = 1
    candidates: list[CandidateRecord]


class ArtifactLineage(StrictModel):
    schema_version: Literal[1] = 1
    artifact_id: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage: str = Field(min_length=1)
    producer_candidate_ids: list[str] = Field(default_factory=list)
    parent_artifact_ids: list[str] = Field(default_factory=list)
    source_license_ids: list[str] = Field(default_factory=list)
    source_license_status: Literal["cleared", "review_required", "blocked"] = "cleared"


class ArtifactRecord(StrictModel):
    schema_version: Literal[1] = 1
    artifact_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    stage: AssetStage
    blob_path: str = Field(min_length=1)
    created_at: datetime
    lineage: ArtifactLineage
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(StrictModel):
    artifact_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    region: str | None = None


class EvidenceBundle(StrictModel):
    schema_version: Literal[1] = 1
    evidence_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    stage: AssetStage
    goal: str = Field(min_length=1)
    items: list[EvidenceItem] = Field(min_length=1)
    numeric_diagnostics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    locked_features: list[str] = Field(default_factory=list)


class OperationProposal(StrictModel):
    schema_version: Literal[1] = 1
    proposal_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    operation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    stage: AssetStage
    input_artifact_ids: list[str] = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_changes: dict[str, float | int | bool | str] = Field(default_factory=dict)
    preserve: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class OptimizerObservation(StrictModel):
    region: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    severity: float = Field(ge=0, le=1)


class RootCauseHypothesis(StrictModel):
    cause: str = Field(min_length=1)
    probability: float = Field(ge=0, le=1)


class OptimizerComparison(StrictModel):
    preferred: Literal["previous", "current", "tie", "uncertain"]
    visual_delta: float = Field(ge=-1, le=1)
    reason: str = Field(min_length=1)


class OptimizerDecision(StrictModel):
    schema_version: Literal[1] = 1
    goal_satisfied: bool
    visual_score: float | None = Field(default=None, ge=0, le=1)
    technical_score: float = Field(ge=0, le=1)
    observations: list[OptimizerObservation] = Field(default_factory=list)
    root_causes: list[RootCauseHypothesis] = Field(default_factory=list)
    proposals: list[OperationProposal] = Field(default_factory=list, max_length=3)
    preserve: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    request_human_review: bool = False
    comparison: OptimizerComparison | None = None


class MetricResult(StrictModel):
    name: str = Field(min_length=1)
    value: float | int | bool | str
    passed: bool
    hard_failure: bool = False
    threshold: float | int | bool | str | None = None


class CandidateEvaluation(StrictModel):
    schema_version: Literal[1] = 1
    evaluation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    candidate_artifact_id: str = Field(min_length=1)
    baseline_artifact_id: str | None = None
    metrics: list[MetricResult] = Field(min_length=1)
    visual_score: float | None = Field(default=None, ge=0, le=1)
    technical_score: float = Field(ge=0, le=1)
    locked_feature_regressions: list[str] = Field(default_factory=list)
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


class ApprovalRecord(StrictModel):
    schema_version: Literal[1] = 1
    approval_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    artifact_id: str = Field(min_length=1)
    stage: AssetStage
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_by: str = Field(min_length=1)
    approved_at: datetime
    locked_features: list[str] = Field(default_factory=list)
    notes: str = ""


class ExportDecision(StrictModel):
    schema_version: Literal[1] = 1
    artifact_id: str = Field(min_length=1)
    destination: DestinationContract
    release_policy: PolicyDecision
    approvals_current: bool
    contract_tests_passed: bool
    allowed: bool
    reasons: list[str] = Field(default_factory=list)


class RunEvent(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    sequence: int = Field(ge=1)
    event_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    event_type: Literal[
        "run_created",
        "artifact_recorded",
        "candidate_staged",
        "candidate_evaluated",
        "candidate_promoted",
        "candidate_rejected",
        "approval_recorded",
        "approval_invalidated",
        "stage_advanced",
        "run_failed",
        "run_resumed",
        "run_completed",
    ]
    occurred_at: datetime
    stage: AssetStage
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_event_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class RunState(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    stage: AssetStage = AssetStage.brief
    status: Literal["created", "running", "failed", "completed"] = "created"
    sequence: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    staged_candidates: list[str] = Field(default_factory=list)
    rejected_candidates: list[str] = Field(default_factory=list)
    promoted_by_stage: dict[str, str] = Field(default_factory=dict)
    approvals: dict[str, ApprovalRecord] = Field(default_factory=dict)
    failure: str | None = None


class WorkerJob(StrictModel):
    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    worker_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    command: list[str] = Field(min_length=1)
    cwd: str
    timeout_seconds: float = Field(default=300, gt=0)
    exclusive_gpu: bool = False
    gpu_memory_gb: float | None = Field(default=None, gt=0)
    gpu_safety_margin_gb: float = Field(default=0.75, ge=0)
    require_gpu_measurement: bool = False
    device_policy: Literal["prefer_gpu", "gpu_compute_only", "strict_device_only"] = "prefer_gpu"
    environment: dict[str, str] = Field(default_factory=dict)


class WorkerResult(StrictModel):
    schema_version: Literal[1] = 1
    job_id: str
    worker_id: str
    return_code: int | None
    timed_out: bool = False
    cancelled: bool = False
    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float = Field(ge=0)
    stdout_path: str
    stderr_path: str
    gpu_before: dict[str, Any] | None = None
    gpu_after: dict[str, Any] | None = None
    gpu_peak_used_gb: float | None = Field(default=None, ge=0)
    gpu_admission_required_gb: float | None = Field(default=None, ge=0)
    gpu_admission_free_gb: float | None = Field(default=None, ge=0)


class PolicyDecision(StrictModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    traversed_artifact_ids: list[str] = Field(default_factory=list)


class LegacyQualification(StrictModel):
    component_id: str = Field(min_length=1)
    contract_tests_total: int = Field(ge=1)
    contract_tests_passed: int = Field(ge=0)
    hard_failures: list[str] = Field(default_factory=list)
    blinded_trials: int = Field(ge=0)
    blinded_wins: int = Field(ge=0)
    evidence_ids: list[str] = Field(default_factory=list)


class LegacyQualificationDecision(StrictModel):
    reusable: bool
    reasons: list[str] = Field(default_factory=list)


class RuntimeProbe(StrictModel):
    probe_id: str = Field(min_length=1)
    image_count: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    http_status: int = Field(ge=100, le=599)
    elapsed_seconds: float = Field(ge=0)
    schema_valid: bool
    semantic_review: Literal["passed", "pending", "failed", "not_applicable"]
    notes: list[str] = Field(default_factory=list)


class ModelRuntimeQualification(StrictModel):
    """Observed facts for one exact locally executed model artifact."""

    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    backend: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    tested_at: datetime
    host_gpu: str = Field(min_length=1)
    installed_bytes: int = Field(ge=1)
    observed_vram_bytes: int = Field(ge=1)
    context_limit: int = Field(ge=1)
    profile_max_images: int = Field(ge=1)
    transport_qualified_max_images: int = Field(ge=1)
    semantic_qualified_max_images: int = Field(ge=1)
    structured_output_policy: Literal["native_schema_plus_client_validation_retry"]
    status: Literal["passed", "partial", "failed"]
    probes: list[RuntimeProbe] = Field(min_length=1)
    known_risks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerRuntimeQualification(StrictModel):
    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    weights_revision: str | None = None
    code_license: str = Field(min_length=1)
    weights_license: str = Field(min_length=1)
    tested_at: datetime
    host_gpu: str
    environment: dict[str, str] = Field(default_factory=dict)
    weight_sha256: dict[str, str] = Field(default_factory=dict)
    status: Literal["passed", "partial", "failed"]
    observed: dict[str, Any] = Field(default_factory=dict)
    excluded_dependencies: list[str] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)


class PackageFile(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)


class DeliveryPackageManifest(StrictModel):
    """Portable, digest-pinned engine-neutral delivery bundle."""

    schema_version: Literal[1] = 1
    package_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    candidate_id: str = Field(min_length=1)
    mode: Literal["research", "release"]
    eligibility: Literal["research_only", "release_ready", "blocked"]
    code_license: str = Field(min_length=1)
    weights_license: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    weights_revision: str | None = None
    qualification_status: Literal["passed", "partial", "failed"]
    files: list[PackageFile] = Field(min_length=1)
    blockers: list[str] = Field(default_factory=list)
