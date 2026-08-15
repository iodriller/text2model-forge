from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

from PIL import Image
import pytest

from darkness.studio_comfy import (
    concept_workflow,
    inpaint_workflow,
    make_chroma_alpha,
    make_footman_equipment_layout_guide,
    make_humanoid_openpose_guide,
    qwen_image_2512_workflow,
    qwen_image_edit_2511_workflow,
)
from darkness.studio_models import (
    STAGE_DEFINITIONS,
    StudioAssetSpec,
    StudioCharacterSpec,
    StudioComponent,
    StudioEquipment,
    StudioQwenReview,
    StudioRun,
)
from darkness.studio_pipeline import (
    StudioCoordinator,
    _composite_inpaint_crop,
    _prepare_inpaint_crop,
)
from darkness.schemas import (
    DarknessLocalConfig,
    ExternalWorkerOutput,
    ExternalWorkerRequest,
    ExternalWorkerResponse,
)
from darkness.studio_qwen import ConceptCorrectionPlan, ConceptPlan, GeometrySeedPlan, StudioQwen
from darkness.studio_qwen import RigidPartPlan, RigidStructurePlan, _history
from darkness.studio_store import StudioStore


DESCRIPTION = (
    "An original chunky heroic dark-fantasy footman with one arming sword in his right hand "
    "and one broad shield on his left arm."
)


def footman_spec() -> StudioCharacterSpec:
    return StudioCharacterSpec(
        asset_id="original_footman",
        title="Original Iron Footman",
        description=DESCRIPTION,
        creative_direction="Original stylized broad mobile-readable defender.",
        anatomy_family="humanoid",
        height_m=1.82,
        silhouette=["broad shoulders", "large shield"],
        materials=["worn steel", "blue cloth", "leather"],
        equipment=[
            StudioEquipment(
                equipment_id="sword",
                category="weapon",
                side="right",
                socket="hand_right.grip",
                grip="palm_and_fingers",
                description="Original short arming sword.",
            ),
            StudioEquipment(
                equipment_id="shield",
                category="shield",
                side="left",
                socket="forearm_left.shield",
                grip="forearm_strap",
                description="Original broad strapped shield.",
            ),
        ],
        animations=["idle", "walk", "attack", "hit", "death", "block"],
        locked_features=["sword right", "shield left"],
        negative_constraints=["no copied IP", "no wrong handedness"],
        gameplay_readability=["equipment readable at sprite scale"],
    )


class FakeQwen:
    correction_calls = 0

    def compile_spec(self, description):
        assert description == DESCRIPTION
        return footman_spec()

    def concept_plan(self, spec, stage):
        return ConceptPlan(
            positive_prompt="A " * 45 + "complete original stylized footman, right sword, left shield.",
            negative_prompt="cropped, wrong hands, copied design",
            seeds=[101 + stage.iteration, 202 + stage.iteration],
            rationale="Compare two deterministic seeds.",
        )

    def review_concepts(self, spec, stage, images, *, comparison_board=None):
        ids = [item[0] for item in images]
        return StudioQwenReview(
            review_id=f"review-{stage.iteration}",
            stage_id="D1",
            iteration=stage.iteration,
            summary="Candidate one has the clearer right sword and left shield.",
            strengths=["readable equipment"],
            issues=["check grip close-up"],
            candidate_ranking=ids,
            recommended_evidence_id=ids[0],
            confidence=0.8,
        )

    def concept_correction_plan(self, spec, stage, candidate_ids, comparison_board=None):
        self.correction_calls += 1
        assert stage.human_decisions[-1].comment == "Make the shield larger."
        return ConceptCorrectionPlan(
            operation_id="regenerate_complete_asset",
            base_evidence_id=candidate_ids[0],
            edit_box_normalized=[0.0, 0.0, 1.0, 1.0],
            positive_prompt="A " * 45 + "complete original footman with a larger left shield and right sword.",
            negative_prompt="small shield, missing shield, missing sword, wrong hands",
            seeds=[303, 404],
            denoise=0.8,
            diagnosis="Shield was too small.",
            preserve=["Right-hand sword."],
        )

    def geometry_seed_plan(self, spec, stage, selected_concept):
        return GeometrySeedPlan(
            positive_prompt="A " * 45 + "unarmed A-pose geometry seed, neutral studio lighting.",
            negative_prompt="equipment, weapons, cropped limbs, wrong pose",
            seed=606 + stage.iteration,
            rationale="Unarmed A-pose isolates body geometry from equipment.",
        )

    def review_geometry(self, spec, stage, selected_concept, diagnostic, metrics):
        review = StudioQwenReview(
            review_id=f"d2-review-{stage.iteration}",
            stage_id="D2",
            iteration=stage.iteration,
            summary="Geometry candidate preserves the approved silhouette and passes numeric gates.",
            strengths=["watertight", "proportions match the approved concept"],
            issues=[],
            candidate_ranking=["geometry-candidate"],
            recommended_evidence_id="geometry-candidate",
            confidence=0.85,
            hard_requirements_satisfied=True,
        )
        return review, True

    def review_cleanup(self, spec, stage, selected_concept, diagnostic, metrics):
        review = StudioQwenReview(
            review_id=f"d3-review-{stage.iteration}",
            stage_id="D3",
            iteration=stage.iteration,
            summary="Cleanup preserved identity; no floating or missing structural pieces.",
            strengths=["single connected component", "silhouette unchanged"],
            issues=[],
            candidate_ranking=["cleanup-candidate"],
            recommended_evidence_id="cleanup-candidate",
            confidence=0.85,
            hard_requirements_satisfied=True,
        )
        return review, True


class FakeComfy:
    def __init__(self) -> None:
        self.workflows: list[dict] = []

    def checkpoints(self):
        return ["dreamshaper_xl_v2_turbo.safetensors"]

    def controlnets(self):
        return ["controlnet_openpose_sdxl_xinsir.safetensors"]

    def models(self, kind: str):
        return ["Warcraft style.safetensors"] if kind == "loras" else []

    def upload_image(self, name: str, data: bytes, subfolder: str = "darkness_studio") -> str:
        return f"{subfolder}/{name}"

    def generate(self, *, workflow, destination: Path, timeout_seconds=900):
        self.workflows.append(workflow)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "concept.png"
        # An edge-connected green border around a non-green center, so this
        # image is also valid input to make_chroma_alpha() (D2's geometry
        # seed path key it out) -- not just any RGB PNG, exactly like
        # test_chroma_alpha_removes_only_edge_connected_green's fixture.
        image = Image.new("RGB", (64, 96), (0, 220, 0))
        for y in range(16, 80):
            for x in range(12, 52):
                image.putpixel((x, y), (70, 90, 120))
        image.save(target)
        return [target]


class FakeWorkerExecutor:
    """Stands in for _execute_worker's real path (config.local.toml lookup +
    a real subprocess via WorkerManager/SubprocessWorkerAdapter). Writes a
    genuinely valid synthetic mesh via trimesh -- D2's real code path loads
    the GLB it receives and computes real geometric properties from it
    (vertices, faces, watertightness), so a placeholder file that merely
    exists is not enough; it has to actually parse."""

    def __init__(self) -> None:
        self.requests: list[ExternalWorkerRequest] = []

    def __call__(self, worker_id: str, request: ExternalWorkerRequest, *, timeout_seconds: float):
        self.requests.append(request)
        output_root = Path(request.output_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        if worker_id == "trellis2.4b":
            import trimesh

            # subdivisions=4 clears D2's hard gate (>=1000 vertices, >=1000 faces)
            # by a wide margin and is cheap to build/export in a test.
            mesh = trimesh.creation.icosphere(subdivisions=4)
            glb_path = output_root / "geometry_candidate.glb"
            mesh.export(glb_path)
            return ExternalWorkerResponse(
                job_id=request.job_id,
                status="succeeded",
                outputs=[
                    ExternalWorkerOutput(
                        path=str(glb_path), media_type="model/gltf-binary", role="geometry_candidate"
                    )
                ],
                diagnostics={"synthetic": True},
            )
        if worker_id == "blender" and request.operation_id == "blender.repair":
            front = output_root / "candidate_front.png"
            Image.new("RGB", (64, 64), (90, 100, 80)).save(front)
            geometry = output_root / "candidate_geometry.glb"
            # Re-export the same synthetic sphere as the "cleaned" geometry --
            # D3 does not care whether it changed, only that a valid
            # model/gltf-binary output with role candidate_geometry exists.
            import trimesh

            trimesh.creation.icosphere(subdivisions=4).export(geometry)
            return ExternalWorkerResponse(
                job_id=request.job_id,
                status="succeeded",
                outputs=[
                    ExternalWorkerOutput(path=str(front), media_type="image/png", role="candidate_front"),
                    ExternalWorkerOutput(
                        path=str(geometry), media_type="model/gltf-binary", role="candidate_geometry"
                    ),
                ],
                diagnostics={"synthetic": True, "hard_gate_passed": True},
            )
        raise AssertionError(f"FakeWorkerExecutor has no fixture for worker_id={worker_id!r} operation_id={request.operation_id!r}")


class QwenImageFakeComfy(FakeComfy):
    def __init__(self) -> None:
        self.workflows: list[dict] = []

    def models(self, kind: str):
        values = {
            "diffusion_models": ["qwen_image_2512_fp8_e4m3fn.safetensors"],
            "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
            "vae": ["qwen_image_vae.safetensors"],
        }
        return values.get(kind, [])

    # generate() is inherited from FakeComfy, which already records into
    # self.workflows -- do not also append here, or every call double-counts.


class ControlFakeComfy(QwenImageFakeComfy):
    def __init__(self) -> None:
        super().__init__()
        self.interrupt_calls = 0
        self.memory_releases: list[tuple[bool, bool]] = []

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        self.memory_releases.append((unload_models, free_memory))


class BlockingControlFakeComfy(ControlFakeComfy):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.interrupted = threading.Event()

    def generate(self, *, workflow, destination: Path, timeout_seconds=900):
        self.workflows.append(workflow)
        self.started.set()
        if not self.interrupted.wait(timeout=3):
            raise AssertionError("test did not request a ComfyUI interrupt")
        raise RuntimeError("ComfyUI workflow interrupted")

    def interrupt(self) -> None:
        super().interrupt()
        self.interrupted.set()


def wait_for(store: StudioStore, run_id: str, state: str, timeout: float = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = store.load(run_id)
        if run.state == state:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run never reached {state}: {store.load(run_id).model_dump()}")


def _wait_for_stage_settled(store: StudioStore, run_id: str, stage_id: str, timeout: float = 5):
    """Wait for one stage to reach a terminal state. Unlike wait_for(), which
    watches run.state, this watches a specific stage -- needed for stages like
    D2 that have no human gate and so never park the run in awaiting_review."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = store.load(run_id)
        if run.stage(stage_id).state in {"approved", "rejected", "failed", "skipped", "blocked"}:
            return run
        time.sleep(0.02)
    raise AssertionError(
        f"{stage_id} never settled: {store.load(run_id).stage(stage_id).model_dump()}"
    )


def test_stage_contract_covers_d0_through_d10() -> None:
    assert [item[0] for item in STAGE_DEFINITIONS] == [f"D{i}" for i in range(11)]
    assert [item[0] for item in STAGE_DEFINITIONS if item[2]] == ["D1", "D4", "D7", "D8", "D10"]


def test_workflow_uses_only_qualified_core_nodes() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="original footman",
        negative="copied design",
        seed=42,
        prefix="DarknessStudio/test",
    )
    assert {item["class_type"] for item in workflow.values()} == {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
    assert workflow["4"]["inputs"] == {"width": 768, "height": 1024, "batch_size": 1}


def test_inpaint_workflow_uses_bounded_core_mask() -> None:
    workflow = inpaint_workflow(
        checkpoint="model.safetensors",
        positive="A " * 45,
        negative="missing equipment and wrong grip",
        seed=42,
        prefix="DarknessStudio/test",
        source_image="studio/source.png",
        mask_image="studio/mask.png",
        denoise=0.7,
    )
    assert workflow["5"] == {
        "class_type": "LoadImageMask",
        "inputs": {"image": "studio/mask.png", "channel": "red"},
    }
    assert workflow["6"]["class_type"] == "VAEEncodeForInpaint"
    assert workflow["6"]["inputs"]["grow_mask_by"] == 12
    assert workflow["7"]["class_type"] == "DifferentialDiffusion"
    assert workflow["8"]["inputs"]["denoise"] == 0.7


def test_concept_workflow_passes_steps_and_cfg_through_to_the_sampler() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="original footman",
        negative="copied design",
        seed=42,
        prefix="DarknessStudio/test",
        steps=45,
        cfg=7.0,
    )
    samplers = [item for item in workflow.values() if item["class_type"] == "KSampler"]
    assert samplers[0]["inputs"]["steps"] == 45
    assert samplers[0]["inputs"]["cfg"] == 7.0


def test_qwen_image_edit_workflow_uses_native_dual_image_conditioning() -> None:
    workflow = qwen_image_edit_2511_workflow(
        prompt="Preserve the footman and fix only the sword grip.",
        negative_prompt="extra character, equipment rack",
        seed=42,
        prefix="DarknessStudio/test",
        source_image="studio/source.png",
    )
    assert workflow["1"]["class_type"] == "UNETLoader"
    assert workflow["4"]["inputs"]["type"] == "qwen_image"
    assert workflow["4"]["inputs"]["device"] == "cpu"
    assert workflow["8"]["class_type"] == "TextEncodeQwenImageEdit"
    assert workflow["8"]["inputs"]["image"] == ["7", 0]
    assert workflow["12"]["class_type"] == "VAEEncode"
    assert workflow["13"]["inputs"]["steps"] == 40


def test_qwen_image_2512_workflow_is_native_portrait_text_to_image() -> None:
    workflow = qwen_image_2512_workflow(
        prompt="One original human footman with a sword and shield.",
        negative_prompt="pixel art, duplicate character",
        seed=42,
        prefix="DarknessStudio/test",
    )
    assert workflow["1"]["inputs"]["unet_name"] == "qwen_image_2512_fp8_e4m3fn.safetensors"
    assert workflow["4"]["inputs"] == {
        "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "type": "qwen_image",
        "device": "cpu",
    }
    assert workflow["8"]["inputs"] == {"width": 1104, "height": 1472, "batch_size": 1}
    assert workflow["9"]["inputs"]["steps"] == 50


def test_concept_workflow_chains_loras_and_pose_control_around_one_body() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="complete original footman, right hand sword, left arm shield",
        negative="missing equipment, duplicate body",
        seed=42,
        prefix="DarknessStudio/test",
        loras=[("style.safetensors", 0.85), ("armor.safetensors", 0.6)],
        control_guides=[("openpose.safetensors", "pose.png", 0.85, 0.0, 0.85)],
    )
    lora_nodes = [item for item in workflow.values() if item["class_type"] == "LoraLoader"]
    controls = [item for item in workflow.values() if item["class_type"] == "ControlNetApplyAdvanced"]
    samplers = [item for item in workflow.values() if item["class_type"] == "KSampler"]
    assert [item["inputs"]["lora_name"] for item in lora_nodes] == ["style.safetensors", "armor.safetensors"]
    assert [item["inputs"]["strength_model"] for item in lora_nodes] == [0.85, 0.6]
    assert [item["inputs"]["strength"] for item in controls] == [0.85]
    last_lora_id = next(
        node_id for node_id, item in workflow.items() if item is lora_nodes[-1]
    )
    # The LoRA chain must feed the sampler's model input -- there is only ever one figure, never
    # a second region-conditioned body.
    assert samplers[0]["inputs"]["model"] == [last_lora_id, 0]
    assert "ConditioningSetArea" not in {item["class_type"] for item in workflow.values()}
    assert "ConditioningCombine" not in {item["class_type"] for item in workflow.values()}


def test_concept_workflow_with_no_lora_or_control_matches_prior_core_node_shape() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="original footman",
        negative="copied design",
        seed=42,
        prefix="DarknessStudio/test",
    )
    assert {item["class_type"] for item in workflow.values()} == {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }


def test_humanoid_guide_and_local_crop_preserve_unmasked_pixels(tmp_path: Path) -> None:
    pose = make_humanoid_openpose_guide(tmp_path / "guides")
    assert Image.open(pose).size == (768, 1024)

    source = tmp_path / "source.png"
    Image.new("RGB", (300, 400), "blue").save(source)
    crop, mask, crop_box = _prepare_inpaint_crop(
        source,
        [0.05, 0.2, 0.45, 0.8],
        tmp_path / "repair",
    )
    assert Image.open(crop).size == (768, 1024)
    generated = tmp_path / "generated.png"
    Image.new("RGB", (768, 1024), "red").save(generated)
    output = _composite_inpaint_crop(source, generated, mask, crop_box, tmp_path / "output.png")
    with Image.open(output).convert("RGB") as result:
        assert result.getpixel((299, 399)) == (0, 0, 255)
        assert result.getpixel((75, 200))[0] > result.getpixel((75, 200))[2]


def test_footman_equipment_layout_guide_has_a_fixed_portrait_canvas(tmp_path: Path) -> None:
    guide = make_footman_equipment_layout_guide(tmp_path / "footman_layout.png")
    assert Image.open(guide).size == (768, 1024)


def test_generic_static_asset_contract_accepts_wall_without_character_fields() -> None:
    wall = StudioAssetSpec(
        asset_id="fortress_wall",
        title="Original Fortress Wall",
        description="A worn modular defensive wall.",
        creative_direction="Readable original stylized stone construction.",
        asset_kind="architecture",
        behavior="static",
        anatomy_family=None,
        height_m=3.0,
        dimensions_m=[4.0, 3.0, 0.8],
        silhouette=["crenellated top"],
        materials=["worn stone"],
        animations=[],
        locked_features=["modular ends"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["doorway scale readable"],
    )
    assert wall.asset_kind == "architecture"
    assert wall.behavior == "static"


class _SpecQwen:
    """Minimal fake Qwen that returns a caller-supplied spec, for testing D0's
    asset_kind/behavior-driven stage-skip logic without needing D1-D10 to
    actually work for a non-character asset."""

    def __init__(self, spec: StudioAssetSpec) -> None:
        self._spec = spec

    def compile_spec(self, description):
        return self._spec


def _run_d0_only(spec: StudioAssetSpec, run_id: str, tmp_path: Path) -> StudioRun:
    store = StudioStore(tmp_path)
    run = store.create(run_id, spec.description)
    coordinator = StudioCoordinator(store, qwen_factory=lambda run: _SpecQwen(spec))
    coordinator._run_d0(run)
    store.save(run)
    return store.load(run_id)


def test_static_prop_skips_only_rig_and_motion_stages(tmp_path: Path) -> None:
    chair = StudioAssetSpec(
        asset_id="oak_chair",
        title="Original Oak Chair",
        description="A simple original sturdy wooden dining chair with a straight back.",
        creative_direction="Readable original cottage-style furniture, no franchise motifs.",
        asset_kind="prop",
        behavior="static",
        anatomy_family=None,
        height_m=0.9,
        dimensions_m=[0.45, 0.9, 0.45],
        silhouette=["straight back", "four legs"],
        materials=["oak wood"],
        animations=[],
        locked_features=["four legs", "straight back"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["silhouette reads at prop scale"],
    )
    run = _run_d0_only(chair, "chair-v1", tmp_path)
    assert run.stage("D0").state == "approved"
    # a static prop still needs geometry, cleanup, surface, sprites, and validation
    for stage_id in ("D2", "D3", "D8", "D9", "D10"):
        assert run.stage(stage_id).state != "skipped", stage_id
        assert run.stage(stage_id).applicable is True, stage_id
    # but no skeleton, rig, deforming weights, or motion
    for stage_id in ("D4", "D5", "D6", "D7"):
        stage = run.stage(stage_id)
        assert stage.state == "skipped", stage_id
        assert stage.applicable is False, stage_id
        assert stage.progress == 1


def test_material_asset_skips_every_geometry_and_articulation_stage(tmp_path: Path) -> None:
    rust_iron = StudioAssetSpec(
        asset_id="rust_iron_material",
        title="Weathered Rust Iron",
        description="An original weathered rust-streaked iron surface material.",
        creative_direction="Readable original grunge metal, tileable at gameplay scale.",
        asset_kind="material",
        behavior="static",
        anatomy_family=None,
        height_m=None,
        silhouette=["flat tileable swatch"],
        materials=["rust", "iron"],
        animations=[],
        locked_features=["rust streak pattern"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["reads at tile scale"],
    )
    run = _run_d0_only(rust_iron, "material-v1", tmp_path)
    assert run.stage("D0").state == "approved"
    for stage_id in ("D2", "D3", "D4", "D5", "D6", "D7"):
        stage = run.stage(stage_id)
        assert stage.state == "skipped", stage_id
        assert stage.applicable is False, stage_id
    # a material still goes through concept, surface, sprites, and validation
    for stage_id in ("D1", "D8", "D9", "D10"):
        assert run.stage(stage_id).state != "skipped", stage_id


def test_rigid_articulated_asset_skips_skinning_and_motion_without_clips(tmp_path: Path) -> None:
    gate = StudioAssetSpec(
        asset_id="iron_gate",
        title="Original Hinged Iron Gate",
        description="An original hinged iron gate with a single swinging door.",
        creative_direction="Readable original ironwork, open/close states.",
        asset_kind="architecture",
        behavior="rigid_articulated",
        anatomy_family=None,
        height_m=2.4,
        dimensions_m=[1.6, 2.4, 0.2],
        silhouette=["vertical iron bars"],
        materials=["iron"],
        components=[
            StudioComponent(
                component_id="door",
                role="movable_part",
                connection="hinge to frame",
                motion="rigid",
                description="Single swinging door leaf.",
            )
        ],
        animations=[],
        locked_features=["single door leaf"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["open/close states readable"],
    )
    run = _run_d0_only(gate, "gate-v1", tmp_path)
    assert run.stage("D6").state == "skipped"
    assert run.stage("D6").message == "Rigid articulated assets do not require deforming skin weights."
    assert run.stage("D7").state == "skipped"
    assert run.stage("D7").message == "No rigid motion clips were requested."
    # rigid articulation still needs a skeleton/rig for its hinge, unlike a static prop
    assert run.stage("D4").state != "skipped"
    assert run.stage("D5").state != "skipped"


def test_rigid_structure_contract_normalizes_bounds_and_limits() -> None:
    plan = RigidStructurePlan(
        parts=[
            RigidPartPlan(
                component_id="door_left",
                front_box_normalized=[0.1, 0.2, 0.45, 0.9],
                pivot_normalized=[0.1, 0.55],
                rotation_axis="z",
                minimum_degrees=95,
                maximum_degrees=0,
                neutral_degrees=0,
                rationale="Side hinge.",
            )
        ],
        static_component_ids=["frame"],
        confidence=0.8,
    )
    assert plan.parts[0].minimum_degrees == 0
    assert plan.parts[0].maximum_degrees == 95


def test_stage_selector_treats_typed_skips_as_completed() -> None:
    from darkness.studio_models import new_studio_run

    state = new_studio_run("static-wall", "An original static stone wall for a mobile game.")
    state.stage("D0").state = "approved"
    state.stage("D1").state = "approved"
    state.stage("D2").state = "approved"
    state.stage("D3").state = "approved"
    for stage_id in ("D4", "D5", "D6", "D7"):
        state.stage(stage_id).state = "skipped"
        state.stage(stage_id).applicable = False
    assert StudioCoordinator._next_stage(state).stage_id == "D8"


def test_numeric_history_stays_bounded_after_many_iterations() -> None:
    from darkness.studio_models import new_studio_run

    stage = new_studio_run("history", DESCRIPTION).stage("D1")
    stage.iteration = 12
    for iteration in range(1, 13):
        stage.qwen_reviews.append(
            StudioQwenReview(
                review_id=f"review-{iteration}",
                stage_id="D1",
                iteration=iteration,
                summary="long diagnosis " * 200,
                issues=["issue " * 100],
                candidate_ranking=[f"candidate-{iteration}"],
                recommended_evidence_id=f"candidate-{iteration}",
                confidence=0.5,
            )
        )
    assert len(_history(stage)) < 12_000


def test_chroma_alpha_removes_only_edge_connected_green(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "rgba.png"
    image = Image.new("RGB", (100, 100), (0, 220, 0))
    for y in range(20, 80):
        for x in range(30, 70):
            image.putpixel((x, y), (80, 90, 120))
    # A green emblem inside the foreground must remain because it is not edge-connected.
    for y in range(40, 50):
        for x in range(45, 55):
            image.putpixel((x, y), (0, 200, 0))
    image.save(source)
    metrics = make_chroma_alpha(source, output)
    assert metrics["meaningful_alpha"] is True
    with Image.open(output).convert("RGBA") as result:
        assert result.getpixel((0, 0))[3] == 0
        assert result.getpixel((50, 45))[3] == 255


def test_explicit_handedness_validator_fails_closed() -> None:
    wrong = footman_spec().model_copy(deep=True)
    wrong.equipment[0].side = "left"
    with pytest.raises(ValueError, match="right-hand sword"):
        StudioQwen._validate_explicit_handedness(DESCRIPTION, wrong)


def test_pipeline_stops_for_review_and_rejection_reuses_comment(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.create("footman-v1", DESCRIPTION)
    qwen = FakeQwen()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: FakeComfy(),
            executor=executor,
        )
        assert coordinator.submit("footman-v1") is True
        first = wait_for(store, "footman-v1", "awaiting_review")
        assert first.current_stage == "D1"
        assert first.stage("D0").state == "approved"
        assert first.stage("D1").iteration == 1
        candidates = [
            item
            for item in first.stage("D1").evidence
            if item.media_type == "image/png" and item.metrics.get("selectable") is True
        ]
        assert len(candidates) == 2

        store.decide("footman-v1", "D1", "reject", "Make the shield larger.", candidates[0].evidence_id)
        coordinator.submit("footman-v1")
        second = wait_for(store, "footman-v1", "awaiting_review")
        assert second.stage("D1").iteration == 2
        assert qwen.correction_calls == 1
        assert len(second.stage("D1").human_decisions) == 1
        assert len(
            [
                item
                for item in second.stage("D1").evidence
                if item.media_type == "image/png" and item.metrics.get("selectable") is True
            ]
        ) == 4
        event_types = [item["event_type"] for item in store.read_events("footman-v1")]
        assert "gate_rejected" in event_types


def test_pipeline_prefers_installed_qwen_image_generation_for_a_typed_humanoid_brief(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("footman-qwen-v1", DESCRIPTION)
    run.concept_backend = "qwen_image_2512"
    store.save(run)
    qwen = FakeQwen()
    comfy = QwenImageFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit("footman-qwen-v1") is True
        result = wait_for(store, "footman-qwen-v1", "awaiting_review")
    stage = result.stage("D1")
    assert stage.metrics["workflow_strategy"] == "qwen_image_2512_t2i_v1"
    assert not any(item.evidence_id.endswith("qwen-edit-layout-guide") for item in stage.evidence)
    assert len(comfy.workflows) == 2
    assert all(workflow["1"]["class_type"] == "UNETLoader" for workflow in comfy.workflows)


def test_manual_qwen_image_prompt_is_preserved_rendered_and_reviewed(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("footman-manual-qwen", DESCRIPTION)
    run.spec = footman_spec()
    run.title = run.spec.title
    run.current_stage = "D1"
    run.state = "awaiting_review"
    run.stage("D0").state = "approved"
    run.stage("D1").state = "awaiting_review"
    store.save(run)
    qwen = FakeQwen()
    comfy = QwenImageFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit_manual_qwen_image(
            "footman-manual-qwen",
            "Use natural human proportions, a broad opaque shield, and a straight readable sword.",
            seed=12345,
        ) is True
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = store.load("footman-manual-qwen")
            stage = result.stage("D1")
            if stage.state == "awaiting_review" and stage.iteration == 1 and stage.qwen_reviews:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("manual Qwen Image candidate never reached review")
    stage = result.stage("D1")
    candidate = next(item for item in stage.evidence if item.metrics.get("human_prompt") is True)
    request = next(item for item in stage.evidence if item.metrics.get("role") == "human_direct_prompt")
    assert candidate.metrics["seed"] == 12345
    assert candidate.metrics["workflow_strategy"] == "qwen_image_2512_manual_prompt_v1"
    assert "natural human proportions" in store.artifact_path(result.run_id, request.relative_path).read_text(encoding="utf-8")
    assert comfy.workflows[-1]["1"]["class_type"] == "UNETLoader"


def _awaiting_d1_run(store: StudioStore, run_id: str) -> None:
    run = store.create(run_id, DESCRIPTION)
    run.spec = footman_spec()
    run.title = run.spec.title
    run.current_stage = "D1"
    run.state = "awaiting_review"
    run.stage("D0").state = "approved"
    run.stage("D1").state = "awaiting_review"
    store.save(run)


def test_control_can_release_comfy_memory_only_while_idle(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    _awaiting_d1_run(store, "footman-memory")
    comfy = ControlFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        released, message = coordinator.release_comfy_memory("footman-memory")
    assert released is True
    assert "released" in message
    assert comfy.memory_releases == [(True, True)]
    assert "comfy_memory_released" in [item["event_type"] for item in store.read_events("footman-memory")]


def test_control_stops_active_direct_qwen_render_and_preserves_run(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    _awaiting_d1_run(store, "footman-stop")
    comfy = BlockingControlFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit_manual_qwen_image(
            "footman-stop",
            "Render a natural original footman with an opaque shield and clearly gripped sword.",
        )
        assert comfy.started.wait(timeout=2)
        accepted, message = coordinator.stop("footman-stop")
        assert accepted is True
        assert "Stop requested" in message
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            stopped = store.load("footman-stop")
            if stopped.state == "blocked":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("stopped render did not become resumable")
    assert stopped.stage("D1").state == "blocked"
    assert comfy.interrupt_calls == 1
    assert "human_stopped_job" in [item["event_type"] for item in store.read_events("footman-stop")]


def test_rejection_requires_comment_and_hash_binds_evidence(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("footman-v2", DESCRIPTION)
    stage = run.stage("D1")
    stage.state = "awaiting_review"
    image = store.run_root(run.run_id) / "D1_concept" / "candidate.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(image)
    item = store.evidence(
        run,
        "D1",
        image,
        evidence_id="candidate-1",
        label="Candidate",
        media_type="image/png",
        metrics={"selectable": True},
    )
    store.save(run)
    with pytest.raises(ValueError, match="rejection comment"):
        store.decide(run.run_id, "D1", "reject", "", item.evidence_id)
    decided = store.decide(run.run_id, "D1", "approve", "Looks good.", item.evidence_id)
    assert decided.stage("D1").human_decisions[-1].evidence_hashes == {"candidate-1": item.sha256}


def _awaiting_stage_with_one_candidate(store: StudioStore, run_id: str, stage_id: str):
    run = store.create(run_id, DESCRIPTION) if run_id not in {item.run_id for item in store.list()} else store.load(run_id)
    stage = run.stage(stage_id)
    stage.state = "awaiting_review"
    image = store.run_root(run.run_id) / f"{stage_id}_stage" / "candidate.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "blue").save(image)
    item = store.evidence(
        run,
        stage_id,
        image,
        evidence_id=f"{stage_id.lower()}-candidate-1",
        label="Candidate",
        media_type="image/png",
        metrics={"selectable": True},
    )
    store.save(run)
    return run, item


def test_retry_requires_no_comment_and_carries_overrides_into_pending_overrides(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "retry-v1", "D1")
    decided = store.decide(
        run.run_id, "D1", "retry", "", item.evidence_id, overrides={"seed": 42}
    )
    stage = decided.stage("D1")
    assert stage.state == "pending"
    assert stage.pending_overrides == {"seed": 42}
    assert stage.human_decisions[-1].decision == "retry"
    event_types = [entry["event_type"] for entry in store.read_events(run.run_id)]
    assert "gate_retried" in event_types


@pytest.mark.parametrize(
    "bad_overrides, message",
    [
        ({"seed": -5}, "non-negative whole number"),
        ({"seed": 1.5}, "non-negative whole number"),
        ({"seed": True}, "non-negative whole number"),  # bool is an int subclass
        ({"seed": "42"}, "non-negative whole number"),
        ({"concept_steps": 0}, "between 1 and 150"),
        ({"concept_steps": 500}, "between 1 and 150"),
        ({"concept_cfg": 0}, "greater than 0"),
        ({"concept_cfg": 99}, "greater than 0 and at most 30"),
    ],
)
def test_decide_rejects_malformed_overrides_immediately(
    tmp_path: Path, bad_overrides: dict, message: str
) -> None:
    """Regression: decide() accepted any JSON object, so {"seed": -5} was
    reported as success and only surfaced later as an asynchronously failed
    stage, far from the input that caused it. The human (or API caller) must
    get the error at the moment they submit it."""
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "bad-overrides-v1", "D1")
    with pytest.raises(ValueError, match=message):
        store.decide(run.run_id, "D1", "retry", "", item.evidence_id, overrides=bad_overrides)
    # rejected cleanly -- the stage is untouched, not left half-decided
    assert store.load(run.run_id).stage("D1").state == "awaiting_review"
    assert store.load(run.run_id).stage("D1").pending_overrides == {}


def test_decide_allows_valid_and_unknown_override_keys(tmp_path: Path) -> None:
    """Valid known keys pass, and an unknown key is deliberately allowed so
    this validator does not become a chokepoint every new stage-specific
    override has to be registered in before it can be used."""
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "ok-overrides-v1", "D1")
    decided = store.decide(
        run.run_id,
        "D1",
        "retry",
        "",
        item.evidence_id,
        overrides={"seed": 0, "concept_steps": 150, "concept_cfg": 30, "some_future_key": "x"},
    )
    assert decided.stage("D1").pending_overrides["seed"] == 0
    assert decided.stage("D1").pending_overrides["some_future_key"] == "x"


def test_edit_requires_comment_or_overrides(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "edit-v1", "D1")
    with pytest.raises(ValueError, match="comment, override values, or both"):
        store.decide(run.run_id, "D1", "edit", "", item.evidence_id)
    decided = store.decide(
        run.run_id, "D1", "edit", "", item.evidence_id, overrides={"prompt_suffix": "bigger shield"}
    )
    stage = decided.stage("D1")
    assert stage.state == "rejected"
    assert stage.pending_overrides == {"prompt_suffix": "bigger shield"}


def test_skip_requires_a_reason_and_does_not_invalidate_downstream(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "skip-v1", "D1")
    with pytest.raises(ValueError, match="skip reason"):
        store.decide(run.run_id, "D1", "skip", "", item.evidence_id)
    decided = store.decide(run.run_id, "D1", "skip", "material asset, no concept needed", item.evidence_id)
    stage = decided.stage("D1")
    assert stage.state == "skipped"
    # skip does not wipe evidence or state on later stages the way reject/retry/edit do
    assert decided.stage("D4").state == "pending"
    assert decided.stage("D4").message == "Waiting for the preceding stage."


def test_rollback_reopens_an_earlier_stage_and_invalidates_everything_after_it(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("rollback-v1", DESCRIPTION)
    run.stage("D1").state = "approved"
    store.save(run)
    run, later_item = _awaiting_stage_with_one_candidate(store, "rollback-v1", "D4")
    with pytest.raises(ValueError, match="earlier stage"):
        store.decide(run.run_id, "D4", "rollback", "actually the concept was wrong", later_item.evidence_id, target_stage_id="D7")
    decided = store.decide(
        run.run_id, "D4", "rollback", "actually the concept was wrong", None, target_stage_id="D1"
    )
    assert decided.current_stage == "D1"
    assert decided.stage("D1").state == "pending"
    assert decided.stage("D4").state == "pending"
    assert decided.stage("D4").evidence == []
    # the rollback decision itself is recorded against the stage the human was standing at
    assert decided.stage("D4").human_decisions[-1].decision == "rollback"
    assert decided.stage("D4").human_decisions[-1].target_stage_id == "D1"
    event_types = [entry["event_type"] for entry in store.read_events(run.run_id)]
    assert "gate_rolled_back" in event_types


def test_rollback_target_must_have_a_prior_decision(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "rollback-v2", "D4")
    with pytest.raises(ValueError, match="no prior decision"):
        store.decide(run.run_id, "D4", "rollback", "go back", item.evidence_id, target_stage_id="D1")


class _CorrectionRecordingQwen(FakeQwen):
    """FakeQwen without the fixed-comment assertion, so a test can drive any
    correction-carrying decision and inspect what actually reached Qwen."""

    def __init__(self) -> None:
        self.correction_calls = 0
        self.seen_comments: list[str] = []

    def concept_correction_plan(self, spec, stage, candidate_ids, comparison_board=None):
        self.correction_calls += 1
        self.seen_comments.append(stage.human_decisions[-1].comment)
        return ConceptCorrectionPlan(
            operation_id="regenerate_complete_asset",
            base_evidence_id=candidate_ids[0],
            edit_box_normalized=[0.0, 0.0, 1.0, 1.0],
            positive_prompt="A " * 45 + "corrected original footman.",
            negative_prompt="missing shield, missing sword, wrong hands",
            seeds=[303, 404],
            denoise=0.8,
            diagnosis="Applying the human correction.",
            preserve=["Right-hand sword."],
        )


def _drive_to_first_d1_review(store: StudioStore, run_id: str, qwen, executor, *, comfy=None, overrides=None):
    comfy = comfy or FakeComfy()
    store.create(run_id, DESCRIPTION, overrides)
    coordinator = StudioCoordinator(
        store,
        qwen_factory=lambda run: qwen,
        comfy_factory=lambda run: comfy,
        executor=executor,
    )
    assert coordinator.submit(run_id) is True
    wait_for(store, run_id, "awaiting_review")
    return coordinator


def test_edit_decision_reaches_the_correction_path_like_a_rejection(tmp_path: Path) -> None:
    """Regression: every 'latest human correction' lookup used to filter on
    decision == "reject" alone, so an edit's comment was silently discarded
    and the stage re-ran as an unrelated fresh generation."""
    store = StudioStore(tmp_path)
    qwen = _CorrectionRecordingQwen()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = _drive_to_first_d1_review(store, "edit-path-v1", qwen, executor)
        first = store.load("edit-path-v1")
        candidate = next(
            item
            for item in first.stage("D1").evidence
            if item.metrics.get("selectable") is True
        )
        store.decide(
            "edit-path-v1",
            "D1",
            "edit",
            "Widen the shield boss.",
            candidate.evidence_id,
        )
        coordinator.submit("edit-path-v1")
        second = wait_for(store, "edit-path-v1", "awaiting_review")
    assert qwen.correction_calls == 1, "an edit must drive the targeted correction path"
    assert qwen.seen_comments == ["Widen the shield boss."]
    assert second.stage("D1").iteration == 2
    event_types = [item["event_type"] for item in store.read_events("edit-path-v1")]
    assert "gate_edited" in event_types


def test_plain_retry_does_not_use_the_correction_path(tmp_path: Path) -> None:
    """A retry is an explicit 'no judgement, just roll again', so unlike an
    edit it must NOT be turned into a targeted correction."""
    store = StudioStore(tmp_path)
    qwen = _CorrectionRecordingQwen()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = _drive_to_first_d1_review(store, "retry-path-v1", qwen, executor)
        first = store.load("retry-path-v1")
        candidate = next(
            item
            for item in first.stage("D1").evidence
            if item.metrics.get("selectable") is True
        )
        store.decide("retry-path-v1", "D1", "retry", "", candidate.evidence_id)
        coordinator.submit("retry-path-v1")
        wait_for(store, "retry-path-v1", "awaiting_review")
    assert qwen.correction_calls == 0


def test_retry_overrides_pin_the_seed_for_exactly_one_attempt(tmp_path: Path) -> None:
    """Regression: pending_overrides was written by decide() and cleared by
    _invalidate_from(), but nothing ever read it -- a retry with {"seed": N}
    silently did nothing."""
    store = StudioStore(tmp_path)
    qwen = _CorrectionRecordingQwen()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = _drive_to_first_d1_review(store, "seed-override-v1", qwen, executor)
        first = store.load("seed-override-v1")
        candidate = next(
            item
            for item in first.stage("D1").evidence
            if item.metrics.get("selectable") is True
        )
        store.decide(
            "seed-override-v1",
            "D1",
            "retry",
            "",
            candidate.evidence_id,
            overrides={"seed": 4242},
        )
        coordinator.submit("seed-override-v1")
        second = wait_for(store, "seed-override-v1", "awaiting_review")

    stage = second.stage("D1")
    second_attempt_seeds = {
        item.metrics.get("seed")
        for item in stage.evidence
        if int(item.metrics.get("iteration", 0)) == 2 and item.metrics.get("seed") is not None
    }
    assert 4242 in second_attempt_seeds, "the pinned seed must actually reach the render"
    # consumed exactly once, so a later automatic iteration is not silently re-pinned
    assert stage.pending_overrides == {}
    applied = [
        item
        for item in store.read_events("seed-override-v1")
        if item["event_type"] == "stage_overrides_applied"
    ]
    assert len(applied) == 1
    assert applied[0]["payload"]["overrides"] == {"seed": 4242}


def test_a_run_configured_with_high_quality_actually_submits_high_quality_workflows(
    tmp_path: Path,
) -> None:
    """End-to-end proof that [quality.high]'s concept_steps/concept_cfg reach
    the real KSampler node ComfyUI would receive, not just that the numbers
    parse. FakeComfy.generate() never touches a GPU; it only records what
    was submitted, exactly like the seed-override regression test above."""
    store = StudioStore(tmp_path)
    comfy = FakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        _drive_to_first_d1_review(
            store,
            "quality-high-v1",
            FakeQwen(),
            executor,
            comfy=comfy,
            overrides={"concept_steps": 45, "concept_cfg": 7.0},
        )
    assert comfy.workflows, "D1 must have submitted at least one workflow"
    # Only concept_workflow() outputs carry quality-tier steps/cfg; the
    # footman spec's deferred shield repair also submits an inpaint_workflow()
    # KSampler with its own fixed steps/cfg, distinguishable by EmptyLatentImage
    # (concept_workflow always creates a fresh latent; inpaint never does).
    concept_workflows = [
        workflow
        for workflow in comfy.workflows
        if any(node["class_type"] == "EmptyLatentImage" for node in workflow.values())
    ]
    assert concept_workflows, "no concept_workflow() output was submitted"
    samplers = [
        node for workflow in concept_workflows for node in workflow.values() if node["class_type"] == "KSampler"
    ]
    assert samplers, "no KSampler node was submitted"
    assert all(node["inputs"]["steps"] == 45 for node in samplers)
    assert all(node["inputs"]["cfg"] == 7.0 for node in samplers)


def test_a_run_with_no_quality_override_submits_todays_real_default_workflow(tmp_path: Path) -> None:
    """Companion to the test above: a run with no quality override at all
    must submit exactly the steps=30/cfg=6.0 that concept_workflow() has
    always defaulted to -- proving the new quality-tier wiring introduced
    no behavior change for the common case."""
    store = StudioStore(tmp_path)
    comfy = FakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        _drive_to_first_d1_review(store, "quality-default-v1", FakeQwen(), executor, comfy=comfy)
    concept_workflows = [
        workflow
        for workflow in comfy.workflows
        if any(node["class_type"] == "EmptyLatentImage" for node in workflow.values())
    ]
    assert concept_workflows, "no concept_workflow() output was submitted"
    samplers = [
        node for workflow in concept_workflows for node in workflow.values() if node["class_type"] == "KSampler"
    ]
    assert samplers
    assert all(node["inputs"]["steps"] == 30 for node in samplers)
    assert all(node["inputs"]["cfg"] == 6.0 for node in samplers)


def test_d2_geometry_generation_runs_end_to_end_through_a_fake_worker(tmp_path: Path) -> None:
    """First real pipeline-level test of D2, which has never been driven
    through the coordinator before -- it always stopped at D1's gate. Proves
    the worker_executor injection seam works and that _run_d2's real
    evidence/gate logic (vertices/faces/watertight from an actually-parsed
    GLB, hard_gate_passed, Qwen review) reaches 'approved', not just that a
    canned response satisfies a mock.

    Takes no config.local.toml monkeypatch: D2's blob-path computation now
    goes through _workspace_relative() and is based on the store's own
    workspace, so a run driven by an injected worker_executor needs no
    machine config at all.
    """
    store = StudioStore(tmp_path)
    qwen = FakeQwen()
    comfy = FakeComfy()
    worker = FakeWorkerExecutor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            worker_executor=worker,
            executor=executor,
        )
        store.create("d2-e2e-v1", DESCRIPTION)
        assert coordinator.submit("d2-e2e-v1") is True
        first = wait_for(store, "d2-e2e-v1", "awaiting_review")
        assert first.current_stage == "D1"
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("d2-e2e-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("d2-e2e-v1")
        # D2 has no human gate, so the run never parks in awaiting_review for it.
        run = _wait_for_stage_settled(store, "d2-e2e-v1", "D2")

    d2 = run.stage("D2")
    assert d2.state == "approved", d2.error
    assert d2.metrics["vertices"] == 2562  # trimesh.creation.icosphere(subdivisions=4), computed for real
    assert d2.metrics["faces"] == 5120
    assert d2.metrics["watertight"] is True
    assert d2.metrics["hard_gate_passed"] is True
    assert worker.requests[0].operation_id == "geometry.generate_from_rgba"
    glb_evidence = [item for item in d2.evidence if item.media_type == "model/gltf-binary"]
    assert len(glb_evidence) == 1


def test_d2_geometry_seed_render_uses_the_runs_configured_quality(tmp_path: Path) -> None:
    """D2's geometry-seed render honors the run's resolved quality tier, the
    same as D1's concept render -- this exercises D2's own concept_workflow()
    call site, which is a separate one from D1's.

    (Named for what it asserts: this is run-level quality config, NOT the
    per-attempt retry/edit seed override -- D2 has no human gate, so a
    stage-level override can never reach it through the review flow.)
    """
    store = StudioStore(tmp_path)
    qwen = FakeQwen()
    comfy = FakeComfy()
    worker = FakeWorkerExecutor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            worker_executor=worker,
            executor=executor,
        )
        store.create("d2-seed-v1", DESCRIPTION, {"concept_steps": 45, "concept_cfg": 7.0})
        coordinator.submit("d2-seed-v1")
        first = wait_for(store, "d2-seed-v1", "awaiting_review")
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("d2-seed-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("d2-seed-v1")
        run = _wait_for_stage_settled(store, "d2-seed-v1", "D2")

    assert run.stage("D2").state == "approved", run.stage("D2").error
    # Isolate D2's own render by its filename_prefix -- asserting across every
    # submitted workflow would pass on D1's two concept renders alone, even if
    # D2's call site were never reached.
    geometry_seed_workflows = [
        workflow
        for workflow in comfy.workflows
        if any(
            node["class_type"] == "SaveImage"
            and "/geometry/" in str(node["inputs"]["filename_prefix"])
            for node in workflow.values()
        )
    ]
    assert geometry_seed_workflows, "D2's geometry-seed workflow was never submitted"
    samplers = [
        node
        for workflow in geometry_seed_workflows
        for node in workflow.values()
        if node["class_type"] == "KSampler"
    ]
    assert samplers
    assert all(node["inputs"]["steps"] == 45 for node in samplers)
    assert all(node["inputs"]["cfg"] == 7.0 for node in samplers)


def test_d3_cleanup_runs_end_to_end_through_the_same_fake_worker(tmp_path: Path) -> None:
    """Validates the README's claim that D3 is 'one FakeQwen method away'
    from the treatment D2 got: FakeQwen.review_cleanup and
    FakeWorkerExecutor's blender.repair fixture already existed, so this
    needed no new production code and no new fake -- only a test. D3 runs
    the real Blender-worker request path (via the injected executor), builds
    a real image board from the returned PNG, and applies its real Qwen
    identity gate."""
    store = StudioStore(tmp_path)
    worker = FakeWorkerExecutor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=worker,
            executor=executor,
        )
        store.create("d3-e2e-v1", DESCRIPTION)
        coordinator.submit("d3-e2e-v1")
        first = wait_for(store, "d3-e2e-v1", "awaiting_review")
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("d3-e2e-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("d3-e2e-v1")
        run = _wait_for_stage_settled(store, "d3-e2e-v1", "D3")

    d3 = run.stage("D3")
    assert d3.state == "approved", d3.error
    assert run.stage("D2").state == "approved"
    repair_requests = [item for item in worker.requests if item.operation_id == "blender.repair"]
    assert len(repair_requests) == 1
    # A deformable character must ask Blender for a single connected component.
    assert repair_requests[0].parameters["component_policy"] == "keep_largest"
    assert repair_requests[0].parameters["maximum_connected_components"] == 1
    # D3 must carry forward a usable cleaned mesh for D4.
    assert any(
        item.media_type == "model/gltf-binary" and item.metrics.get("role") == "candidate_geometry"
        for item in d3.evidence
    )


def test_d2_survives_a_studio_workspace_that_differs_from_config_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `darkness studio --workspace X` explicitly allows X to
    differ from config.local.toml's workspace_root (see cli.py's
    `args.workspace or ...`), but D2 computed its artifact blob_path as
    rgba_seed.relative_to(config.workspace_root). When they diverged that
    raised an opaque ValueError -- "'...' is not in the subpath of '...'" --
    and failed the stage. The base must be the store's own workspace, which
    every Studio-run file lives under by construction."""
    studio_workspace = tmp_path / "studio_workspace"
    config_workspace = tmp_path / "config_workspace"
    config_workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "darkness.studio_pipeline.load_local_config",
        lambda *a, **k: DarknessLocalConfig(workspace_root=str(config_workspace), workers={}),
    )
    store = StudioStore(studio_workspace)
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=FakeWorkerExecutor(),
            executor=executor,
        )
        store.create("diverging-ws-v1", DESCRIPTION)
        coordinator.submit("diverging-ws-v1")
        first = wait_for(store, "diverging-ws-v1", "awaiting_review")
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("diverging-ws-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("diverging-ws-v1")
        run = _wait_for_stage_settled(store, "diverging-ws-v1", "D2")
    assert run.stage("D2").state == "approved", run.stage("D2").error


def test_d2_needs_no_machine_config_when_a_worker_executor_is_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: D2 was the one load_local_config() call site of five with
    no None guard, so running without config.local.toml crashed with
    AttributeError: 'NoneType' object has no attribute 'workspace_root'
    instead of the clear RuntimeError every other stage raises. The blob-path
    computation no longer consults machine config at all, so this path is
    simply gone rather than merely better-worded."""
    monkeypatch.setattr("darkness.studio_pipeline.load_local_config", lambda *a, **k: None)
    store = StudioStore(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=FakeWorkerExecutor(),
            executor=executor,
        )
        store.create("no-machine-config-v1", DESCRIPTION)
        coordinator.submit("no-machine-config-v1")
        first = wait_for(store, "no-machine-config-v1", "awaiting_review")
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("no-machine-config-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("no-machine-config-v1")
        run = _wait_for_stage_settled(store, "no-machine-config-v1", "D2")
    assert run.stage("D2").state == "approved", run.stage("D2").error


def test_save_retries_a_transient_windows_permission_error_on_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for an intermittently-failing test suite: a real run seen
    while building the D2 worker-executor seam (test_pipeline_prefers_
    installed_qwen_image_generation..., no relation to D2 itself) failed with
    PermissionError: [WinError 5] Access is denied on run.json.tmp -> run.json,
    then passed cleanly on rerun. Path.replace() is atomic on POSIX but can
    transiently fail on Windows when another reader briefly has the file
    open; save() must absorb a few such failures rather than fail the whole
    stage over a race that clears in milliseconds."""
    from pathlib import Path as PathType

    store = StudioStore(tmp_path)
    run = store.create("flaky-save-v1", DESCRIPTION)

    real_replace = PathType.replace
    calls = {"count": 0}

    def flaky_replace(self, target):
        calls["count"] += 1
        if calls["count"] <= 2 and self.name == "run.json.tmp":
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(PathType, "replace", flaky_replace)
    store.save(run)  # must not raise, despite the first two attempts failing
    assert calls["count"] == 3
    assert store.load("flaky-save-v1").run_id == "flaky-save-v1"


def test_save_gives_up_after_persistent_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry is bounded -- a real, non-transient lock must still surface
    as an error rather than hang or silently drop the write."""
    from pathlib import Path as PathType

    store = StudioStore(tmp_path)
    run = store.create("stuck-save-v1", DESCRIPTION)

    def always_fails(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(PathType, "replace", always_fails)
    with pytest.raises(PermissionError):
        store.save(run)


def test_artifact_paths_cannot_escape_run(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.create("footman-v3", DESCRIPTION)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        store.artifact_path("footman-v3", "../../../outside.txt")
