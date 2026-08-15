"""Asynchronous, resumable execution of Darkness Studio stages."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from typing import Protocol

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat

from .config import load_local_config, worker_binding
from .external_worker import SubprocessWorkerAdapter
from .hashing import sha256_file
from .manifests import load_manifests
from .schemas import ArtifactLineage, ArtifactRecord, AssetStage, ExternalWorkerRequest
from .studio_comfy import (
    StudioComfyClient,
    concept_workflow,
    inpaint_workflow,
    make_chroma_alpha,
    make_humanoid_openpose_guide,
    qwen_image_2512_workflow,
)
from .studio_models import (
    StudioQwenReview,
    StudioRun,
    awaiting_correction,
    latest_correction,
    utc_now,
)
from .studio_qwen import ConceptCorrectionPlan, ConceptPlan, StudioQwen
from .studio_store import StudioStore
from .workers import WorkerManager


class QwenProvider(Protocol):
    def compile_spec(self, description: str): ...
    def concept_plan(self, spec, stage): ...
    def concept_correction_plan(self, spec, stage, candidate_ids, comparison_board=None): ...
    def review_concepts(self, spec, stage, images, comparison_board=None): ...
    def revision_plan(self, spec, stage): ...


class ComfyProvider(Protocol):
    def checkpoints(self) -> list[str]: ...
    def controlnets(self) -> list[str]: ...
    def models(self, kind: str) -> list[str]: ...
    def upload_image(self, name: str, data: bytes, subfolder: str = "darkness_studio") -> str: ...
    def generate(self, *, workflow, destination: Path, timeout_seconds: float = 900) -> list[Path]: ...


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _image_metrics(path: Path) -> dict[str, float | int | bool | str | None]:
    with Image.open(path).convert("RGB") as image:
        stats = ImageStat.Stat(image)
        return {
            "width": image.width,
            "height": image.height,
            "aspect_ratio": round(image.width / image.height, 4),
            "mean_luminance_255": round(
                0.2126 * stats.mean[0] + 0.7152 * stats.mean[1] + 0.0722 * stats.mean[2], 3
            ),
            "nonblank": max(stats.var) > 1.0,
        }


def _concept_comparison_board(
    store: StudioStore,
    run: StudioRun,
    stage,
    candidates: list[tuple[str, Path, dict[str, object]]],
    output: Path,
) -> Path:
    records: list[tuple[str, Path]] = []
    latest_rejection = latest_correction(stage)
    if latest_rejection and latest_rejection.selected_evidence_id:
        prior = next(
            (
                item
                for item in stage.evidence
                if item.evidence_id == latest_rejection.selected_evidence_id
            ),
            None,
        )
        if prior is not None and prior.media_type.startswith("image/"):
            records.append((f"PREVIOUS REJECTED: {prior.evidence_id}", store.artifact_path(run.run_id, prior.relative_path)))
    records.extend((f"CURRENT: {evidence_id}", path) for evidence_id, path, _ in candidates)
    cell_width, cell_height, detail_height, header = 480, 640, 230, 62
    board = Image.new(
        "RGB",
        (cell_width * len(records), cell_height + detail_height + header + 34),
        "#10161a",
    )
    draw = ImageDraw.Draw(board)
    for index, (label, path) in enumerate(records):
        with Image.open(path).convert("RGB") as image:
            fitted = ImageOps.contain(image, (cell_width - 16, cell_height - 16))
            crop = image.crop(
                (0, round(image.height * 0.27), image.width, round(image.height * 0.72))
            )
            detail = ImageOps.contain(crop, (cell_width - 16, detail_height - 25))
        x = index * cell_width + (cell_width - fitted.width) // 2
        y = header + (cell_height - fitted.height) // 2
        board.paste(fitted, (x, y))
        detail_x = index * cell_width + (cell_width - detail.width) // 2
        detail_y = header + cell_height + 27
        board.paste(detail, (detail_x, detail_y))
        draw.text(
            (index * cell_width + 10, header + cell_height + 7),
            "HAND / EQUIPMENT DETAIL",
            fill="#9ed8ed",
        )
        draw.text((index * cell_width + 10, 15), label, fill="white")
        if index == 0 and latest_rejection:
            draw.text(
                (index * cell_width + 10, 36),
                ("HUMAN: " + latest_rejection.comment)[:72],
                fill="#ffb59f",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output)
    return output


def _prepare_inpaint_crop(
    source: Path,
    box: list[float],
    output_directory: Path,
) -> tuple[Path, Path, tuple[int, int, int, int]]:
    """Expand a defect box, scale it for detail work, and mask only the requested defect."""
    with Image.open(source).convert("RGB") as image:
        width, height = image.size
    x0, y0, x1, y1 = box
    defect = (
        max(0, min(width - 1, round(x0 * width))),
        max(0, min(height - 1, round(y0 * height))),
        max(1, min(width, round(x1 * width))),
        max(1, min(height, round(y1 * height))),
    )
    defect = (defect[0], defect[1], max(defect[0] + 1, defect[2]), max(defect[1] + 1, defect[3]))
    margin_x = max(24, round((defect[2] - defect[0]) * 0.16))
    margin_y = max(24, round((defect[3] - defect[1]) * 0.12))
    crop_box = (
        max(0, defect[0] - margin_x),
        max(0, defect[1] - margin_y),
        min(width, defect[2] + margin_x),
        min(height, defect[3] + margin_y),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    target_size = (768, 1024)
    with Image.open(source).convert("RGB") as image:
        crop = image.crop(crop_box).resize(target_size, Image.Resampling.LANCZOS)
    crop_path = output_directory / "correction_crop.png"
    crop.save(crop_path)
    scale_x = target_size[0] / (crop_box[2] - crop_box[0])
    scale_y = target_size[1] / (crop_box[3] - crop_box[1])
    mask_box = (
        round((defect[0] - crop_box[0]) * scale_x),
        round((defect[1] - crop_box[1]) * scale_y),
        round((defect[2] - crop_box[0]) * scale_x),
        round((defect[3] - crop_box[1]) * scale_y),
    )
    mask = Image.new("L", target_size, 0)
    ImageDraw.Draw(mask).rectangle(mask_box, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
    mask_path = output_directory / "correction_mask.png"
    mask.save(mask_path)
    return crop_path, mask_path, crop_box


def _composite_inpaint_crop(
    source: Path,
    generated_crop: Path,
    mask: Path,
    crop_box: tuple[int, int, int, int],
    output: Path,
) -> Path:
    crop_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    with Image.open(source).convert("RGB") as base:
        with Image.open(generated_crop).convert("RGB") as generated:
            repaired = generated.resize(crop_size, Image.Resampling.LANCZOS)
        with Image.open(mask).convert("L") as mask_image:
            feather = mask_image.resize(crop_size, Image.Resampling.LANCZOS)
        original_crop = base.crop(crop_box)
        merged = Image.composite(repaired, original_crop, feather)
        base.paste(merged, crop_box[:2])
        output.parent.mkdir(parents=True, exist_ok=True)
        base.save(output)
    return output


def _deferred_shield_equipment(spec):
    """A shield is deferred to a dedicated local-repair pass rather than described in the global
    prompt. A rectangular held prop (shield) fits a box-shaped inpaint mask naturally; a thin
    diagonal prop (sword) does not and degrades into a flat rectangular panel if forced through
    the same mechanism, so the global pass renders it directly instead. Mentioning both props in
    one prompt also measurably increases prop-bleed (mixed/duplicated weapons) in testing, so the
    global pass is told the shield arm is empty rather than asked to render two held props."""
    return next((item for item in spec.equipment if item.category == "shield"), None)


def _limb_repair_box(side: str) -> list[float]:
    """A tight box around the forearm/hand only (not a full body-half); tuned empirically so the
    inpaint has enough room for a shield without also including so much open background that the
    model reinterprets the whole masked region as a flat surface."""
    return [0.50, 0.30, 0.94, 0.72] if side == "left" else [0.06, 0.30, 0.50, 0.72]


def _deferred_shield_repair_prompts(item) -> tuple[str, str]:
    positive = (
        "World of Warcraft cinematic style, close-up of a chunky heroic armored footman's arm and hand, "
        f"(one complete flat round shield, disc-shaped, strapped to the forearm and gripped by the hand, "
        f"{item.description}:1.5), matching the existing armor color and lighting, sharp focus"
    )
    negative = (
        "grainy, blurry, low quality, empty hand, no shield, sword, weapon, extra fingers, deformed hand, "
        "different armor color, different style, two shields, barrel, keg, cask, cylinder, crate, drum, "
        "wine barrel, wooden box"
    )
    return positive, negative


def _whole_asset_prompt_guard(
    spec, positive: str, negative: str, *, defer_shield: bool = True
) -> tuple[str, str]:
    """Keep a correction prompt from accidentally turning into an isolated prop sheet.

    Handedness is stated once in plain language and reinforced by the OpenPose skeleton at
    generation time; there is no region-compositing here, so there is only ever one body. Any
    For the ordinary SDXL route the shield may be deliberately left out (see
    _deferred_shield_equipment) and added by a local-repair pass. Qwen Image Edit receives
    the complete composition guide and must keep both pieces together, so it disables that
    SDXL-only deferral.
    """
    if spec.asset_kind != "character":
        return positive, negative
    materials = ", ".join(spec.materials[:4])
    deferred_shield = _deferred_shield_equipment(spec) if defer_shield else None
    deferred_ids = {deferred_shield.equipment_id} if deferred_shield else set()
    right_equipment = [
        item for item in spec.equipment if item.side == "right" and item.equipment_id not in deferred_ids
    ]
    left_equipment = [
        item for item in spec.equipment if item.side == "left" and item.equipment_id not in deferred_ids
    ]
    right_text = "; ".join(item.description for item in right_equipment)
    left_text = "; ".join(item.description for item in left_equipment)
    left_has_forearm_shield = any(item.category == "shield" for item in left_equipment)
    guarded_positive = (
        "ONE SINGLE COMPLETE LIVING FULL-BODY HUMAN CHARACTER, not an equipment display: a stylized chunky heroic "
        "figure standing centered in a neutral A-pose, with one connected head, face, torso, two shoulders, two "
        "attached arms, two attached hands, two legs, and boots all visible inside frame. Three-quarter-front "
        f"production concept on a plain studio background. The same person wears {materials}. "
        + (
            f"(the character's right hand firmly gripping the hilt of {right_text}, clearly extended away from "
            "the body and away from the shield-side arm:1.4). "
            if right_text
            else ""
        )
        + (
            f"(the character's left forearm visibly and securely strapped into {left_text}, with the left hand "
            "shown at the shield straps and not holding a second weapon:1.4). "
            if left_has_forearm_shield
            else f"(the character's left hand firmly gripping the hilt of {left_text}, clearly extended away from "
            "the body:1.4). "
            if left_text
            else ""
        )
        + (
            "The character's other arm and hand are relaxed and empty at the side, without any shield or extra "
            "prop. "
            if deferred_shield
            else ""
        )
        + positive
    )
    guarded_negative = (
        "isolated equipment, equipment inventory, weapon display, armor rack, wall-mounted swords, disembodied "
        "armor, empty suit of armor, missing human body, invisible person, prop sheet, multiple characters, "
        "extra limbs, extra arms, duplicated body, floating weapon, floating shield, "
        + ("empty right hand, unarmed, weapon on ground, weapon on back, sheathed weapon, " if right_text else "")
        + ("empty left hand, unarmed, " if left_text else "")
        + ("shield, " if deferred_shield else "")
        + negative
    )
    return guarded_positive, guarded_negative


def _image_board(records: list[tuple[str, Path]], output: Path, *, columns: int = 4) -> Path:
    if not records:
        raise ValueError("an evidence board requires at least one image")
    cell_width, cell_height, header = 360, 420, 44
    columns = max(1, min(columns, len(records)))
    rows = (len(records) + columns - 1) // columns
    board = Image.new("RGB", (columns * cell_width, rows * (cell_height + header)), "#10161a")
    draw = ImageDraw.Draw(board)
    for index, (label, path) in enumerate(records):
        column, row = index % columns, index // columns
        with Image.open(path).convert("RGB") as source:
            image = ImageOps.contain(source, (cell_width - 16, cell_height - 16))
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * (cell_height + header) + header + (cell_height - image.height) // 2
        board.paste(image, (x, y))
        draw.text((column * cell_width + 9, row * (cell_height + header) + 13), label, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output)
    return output


def _rigid_structure_overlay(source: Path, plan, output: Path) -> Path:
    with Image.open(source).convert("RGB") as image:
        canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = ("#ff9e72", "#64d4ff", "#d4e65d", "#d487ff")
    for index, part in enumerate(plan.parts):
        x0, y0, x1, y1 = part.front_box_normalized
        box = (x0 * canvas.width, y0 * canvas.height, x1 * canvas.width, y1 * canvas.height)
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=max(2, canvas.width // 300))
        pivot = (part.pivot_normalized[0] * canvas.width, part.pivot_normalized[1] * canvas.height)
        radius = max(5, canvas.width // 80)
        draw.ellipse(
            (pivot[0] - radius, pivot[1] - radius, pivot[0] + radius, pivot[1] + radius),
            outline=color,
            width=max(2, canvas.width // 300),
        )
        draw.text(
            (box[0] + 4, box[1] + 4),
            f"{part.component_id} | {part.rotation_axis} | {part.minimum_degrees:g}..{part.maximum_degrees:g}",
            fill=color,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


class StudioCoordinator:
    """Runs one GPU-heavy Studio stage at a time and stops at every human gate."""

    def __init__(
        self,
        store: StudioStore,
        *,
        qwen_factory=None,
        comfy_factory=None,
        worker_executor=None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.store = store
        self._qwen_factory = qwen_factory or (
            lambda run: StudioQwen(base_url=run.localdeploy_url, model=run.model)
        )
        self._comfy_factory = comfy_factory or (lambda run: StudioComfyClient(run.comfy_url))
        # D2-D5/D9's typed subprocess worker protocol (blender, trellis2.4b, ...),
        # injectable the same way qwen_factory/comfy_factory are. None (the
        # default) means _execute_worker does what it always has: read
        # config.local.toml and run a real subprocess via WorkerManager. A
        # test supplies a callable(worker_id, request, *, timeout_seconds) ->
        # ExternalWorkerResponse instead, exactly like FakeQwen/FakeComfy.
        self._worker_executor = worker_executor
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="darkness-studio")
        self._owns_executor = executor is None
        self._lock = threading.Lock()
        self._jobs: dict[str, Future[None]] = {}
        self._active_comfy: dict[str, ComfyProvider] = {}
        self._stop_requested: set[str] = set()

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, run_id: str) -> bool:
        with self._lock:
            current = self._jobs.get(run_id)
            if current is not None and not current.done():
                return False
            self._stop_requested.discard(run_id)
            self._jobs[run_id] = self._executor.submit(self._drive, run_id)
            return True

    def submit_manual_qwen_image(
        self,
        run_id: str,
        prompt: str,
        *,
        seed: int | None = None,
    ) -> bool:
        """Render one human-authored D1 candidate through Qwen Image 2512.

        The direct prompt is intentionally an additional *candidate*, not a
        bypass of the Studio contract.  It is preserved as evidence, expanded
        by the LocalDeploy Qwen prompt worker when available, then reviewed by
        the same Qwen gate critic as ordinary candidates.
        """
        prompt = prompt.strip()
        if len(prompt) < 12:
            raise ValueError("the direct Qwen Image prompt must contain at least 12 characters")
        if seed is not None and seed < 0:
            raise ValueError("the optional seed must be zero or a positive integer")
        with self._lock:
            current = self._jobs.get(run_id)
            if current is not None and not current.done():
                return False
            self._stop_requested.discard(run_id)
            self._jobs[run_id] = self._executor.submit(
                self._run_manual_qwen_image,
                run_id,
                prompt,
                seed,
            )
            return True

    def busy(self, run_id: str) -> bool:
        with self._lock:
            current = self._jobs.get(run_id)
            return current is not None and not current.done()

    def stopping(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._stop_requested

    def _register_comfy(self, run_id: str, comfy: ComfyProvider) -> None:
        with self._lock:
            self._active_comfy[run_id] = comfy

    def _clear_comfy(self, run_id: str) -> None:
        with self._lock:
            self._active_comfy.pop(run_id, None)

    def _was_stopped(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._stop_requested

    def _mark_stopped(self, run_id: str) -> None:
        run = self.store.load(run_id)
        stage = run.stage(run.current_stage)
        stage.state = "blocked"
        stage.error = None
        stage.finished_at = utc_now()
        stage.message = "Stopped by you. Evidence is preserved; use Resume when you are ready to continue."
        run.state = "blocked"
        self.store.event(
            run,
            "human_stopped_job",
            {"stage_id": stage.stage_id, "iteration": stage.iteration},
        )

    def _record_stop_requested(self, run_id: str) -> None:
        run = self.store.load(run_id)
        self.store.event(
            run,
            "human_stop_requested",
            {"stage_id": run.current_stage, "iteration": run.stage(run.current_stage).iteration},
        )

    def stop(self, run_id: str) -> tuple[bool, str]:
        """Interrupt only the currently tracked Studio workflow, never an arbitrary process."""
        with self._lock:
            current = self._jobs.get(run_id)
            if current is None or current.done():
                return False, "No Studio job is running for this asset."
            self._stop_requested.add(run_id)
            queued_cancelled = current.cancel()
            comfy = self._active_comfy.get(run_id)
        self._record_stop_requested(run_id)
        if queued_cancelled:
            self._mark_stopped(run_id)
            return True, "The queued Studio job was cancelled before it reached ComfyUI."
        if comfy is None:
            return True, "Stop requested. The current Studio step will stop before it launches ComfyUI."
        interrupt = getattr(comfy, "interrupt", None)
        if not callable(interrupt):
            return True, "Stop requested. The active adapter cannot interrupt ComfyUI, so Studio will stop at its next safe boundary."
        try:
            interrupt()
        except Exception as exc:
            return True, f"Stop requested; ComfyUI did not acknowledge the interrupt ({type(exc).__name__}: {exc})."
        return True, "Stop requested; ComfyUI was told to interrupt the active Studio workflow."

    def release_comfy_memory(self, run_id: str) -> tuple[bool, str]:
        """Unload ComfyUI models only when this Studio run is idle."""
        if self.busy(run_id):
            return False, "Stop or wait for the active Studio job before releasing ComfyUI model memory."
        run = self.store.load(run_id)
        comfy = self._comfy_factory(run)
        release = getattr(comfy, "free_memory", None)
        if not callable(release):
            return False, "This ComfyUI adapter does not expose memory release controls."
        try:
            release(unload_models=True, free_memory=True)
        except Exception as exc:
            return False, f"ComfyUI did not release memory: {type(exc).__name__}: {exc}"
        stage = run.stage(run.current_stage)
        stage.message = (
            "ComfyUI models and its execution cache were released. The next render will reload the required models."
        )
        self.store.event(
            run,
            "comfy_memory_released",
            {"stage_id": stage.stage_id, "unload_models": True, "free_memory": True},
        )
        return True, "ComfyUI models and execution cache were released."

    def _drive(self, run_id: str) -> None:
        try:
            while True:
                if self._was_stopped(run_id):
                    self._mark_stopped(run_id)
                    return
                run = self.store.load(run_id)
                stage = self._next_stage(run)
                if stage is None:
                    return
                if stage.stage_id == "D0":
                    self._run_d0(run)
                elif stage.stage_id == "D1":
                    self._run_d1(run)
                elif stage.stage_id == "D2":
                    self._run_d2(run)
                elif stage.stage_id == "D3":
                    self._run_d3(run)
                elif stage.stage_id == "D4":
                    self._run_d4(run)
                elif stage.stage_id == "D5":
                    self._run_d5(run)
                elif stage.stage_id == "D6":
                    self._run_d6(run)
                elif stage.stage_id == "D7":
                    self._run_d7(run)
                elif stage.stage_id == "D8":
                    self._run_d8(run)
                elif stage.stage_id == "D9":
                    self._run_d9(run)
                elif stage.stage_id == "D10":
                    self._run_d10(run)
                else:
                    stage.state = "blocked"
                    stage.message = (
                        "The Studio control contract is ready, but this production adapter has not yet been "
                        "connected to the generic character/equipment pipeline."
                    )
                    run.state = "blocked"
                    run.current_stage = stage.stage_id
                    self.store.event(
                        run,
                        "stage_blocked",
                        {"stage_id": stage.stage_id, "reason": stage.message},
                    )
                    return
                if self._was_stopped(run_id):
                    self._mark_stopped(run_id)
                    return
                if self.store.load(run_id).state == "blocked":
                    return
        except Exception as exc:
            if self._was_stopped(run_id):
                self._mark_stopped(run_id)
                return
            run = self.store.load(run_id)
            stage = run.stage(run.current_stage)
            stage.state = "failed"
            stage.error = f"{type(exc).__name__}: {exc}"
            stage.message = "Stage failed. Fix the dependency or use Resume to retry without losing history."
            stage.finished_at = utc_now()
            run.state = "failed"
            self.store.event(
                run,
                "stage_failed",
                {"stage_id": stage.stage_id, "error": stage.error},
            )
        finally:
            self._clear_comfy(run_id)

    @staticmethod
    def _next_stage(run: StudioRun):
        for stage in run.stages:
            if stage.state == "skipped":
                continue
            if stage.state in {"pending", "queued", "rejected", "failed", "blocked"}:
                preceding = run.stages[: run.stages.index(stage)]
                if any(item.state not in {"approved", "skipped"} for item in preceding):
                    return None
                return stage
            if stage.state == "awaiting_review":
                return None
        run.state = "completed"
        run.current_stage = "D10"
        return None

    def _begin(self, run: StudioRun, stage_id: str, message: str) -> dict[str, Any]:
        """Start one attempt of a stage and return the human-supplied parameter
        overrides that apply to *this* attempt.

        A retry/edit decision parks its overrides on the stage; they are
        consumed here (exactly once, then cleared) so a later automatic
        iteration does not silently keep re-applying a one-off correction.
        """
        stage = run.stage(stage_id)
        retrying_pre_output_failure = stage.state == "failed" and not any(
            item.metrics.get("iteration") == stage.iteration for item in stage.evidence
        )
        stage.state = "running"
        stage.progress = 0.01
        if not retrying_pre_output_failure:
            stage.iteration += 1
        stage.started_at = utc_now()
        stage.finished_at = None
        stage.error = None
        stage.message = message
        run.state = "running"
        run.current_stage = stage_id
        overrides = dict(stage.pending_overrides)
        stage.pending_overrides = {}
        self.store.event(
            run,
            "stage_started",
            {"stage_id": stage_id, "iteration": stage.iteration, "message": message},
        )
        if overrides:
            self.store.event(
                run,
                "stage_overrides_applied",
                {"stage_id": stage_id, "iteration": stage.iteration, "overrides": overrides},
            )
        return overrides

    def _progress(self, run: StudioRun, stage_id: str, value: float, message: str) -> None:
        stage = run.stage(stage_id)
        stage.progress = value
        stage.message = message
        self.store.save(run)

    def _run_manual_qwen_image(self, run_id: str, human_prompt: str, seed: int | None) -> None:
        """Run the direct-prompt Qwen Image 2512 path from the D1 human gate."""
        try:
            run = self.store.load(run_id)
            stage = run.stage("D1")
            if run.current_stage != "D1" or stage.state != "awaiting_review":
                raise ValueError("a direct Qwen Image render is available only while the D1 concept gate awaits review")
            if run.spec is None:
                raise RuntimeError("D1 requires the compiled D0 specification")

            self._begin(run, "D1", "Preparing your direct prompt for Qwen Image 2512.")
            stage = run.stage("D1")
            iteration = stage.iteration
            attempt_root = (
                self.store.run_root(run.run_id)
                / "D1_concept"
                / f"iteration-{iteration:02d}"
                / "manual_qwen_image"
            )
            actual_seed = seed if seed is not None else secrets.randbelow(2_147_483_647)
            direct_positive, direct_negative = _whole_asset_prompt_guard(
                run.spec,
                human_prompt,
                (
                    "pixel art, retro sprites, 8-bit graphics, voxel blocks, hard black outlines, chibi, "
                    "duplicate people, extra limbs, floating equipment, text, logos, watermark"
                ),
                defer_shield=False,
            )
            # ConceptPlan gives the existing Qwen prompt rewriter the same
            # typed contract as a pipeline-authored candidate.  Pad terse but
            # valid human prompts with neutral rendering requirements rather
            # than silently rejecting a useful instruction.
            if len(direct_positive) < 80:
                direct_positive += " Render one complete original asset in a neutral studio presentation."
            plan = ConceptPlan(
                positive_prompt=direct_positive,
                negative_prompt=direct_negative,
                seeds=[actual_seed, actual_seed + 1],
                rationale="A human-authored direct Qwen Image candidate at the D1 review gate.",
            )
            qwen = self._qwen_factory(run)
            instruction = None
            instruction_error = ""
            effective_prompt = plan.positive_prompt
            effective_negative = plan.negative_prompt
            rewriter = getattr(qwen, "image_edit_instruction", None)
            if callable(rewriter):
                try:
                    instruction = rewriter(
                        run.spec,
                        stage,
                        plan,
                        source_handling="new_text_to_image",
                    )
                    effective_prompt = instruction.prompt
                    effective_negative = instruction.negative_prompt
                except Exception as exc:
                    instruction_error = f"{type(exc).__name__}: {exc}"

            request_path = attempt_root / "manual_qwen_image_request.json"
            _write_json(
                request_path,
                {
                    "schema_version": 1,
                    "human_prompt": human_prompt,
                    "seed": actual_seed,
                    "backend": "qwen_image_2512",
                    "source_handling": "new_text_to_image",
                    "qwen_instruction": instruction.model_dump(mode="json") if instruction else None,
                    "qwen_instruction_error": instruction_error or None,
                },
            )
            self.store.evidence(
                run,
                "D1",
                request_path,
                evidence_id=f"d1-i{iteration:02d}-manual-qwen-image-request",
                label=f"Your direct Qwen Image prompt, iteration {iteration}",
                media_type="application/json",
                metrics={
                    "iteration": iteration,
                    "selectable": False,
                    "role": "human_direct_prompt",
                    "workflow_strategy": "qwen_image_2512_manual_prompt_v1",
                    "seed": actual_seed,
                },
            )
            self._progress(run, "D1", 0.16, "Checking that the native Qwen Image 2512 model is available.")
            comfy = self._comfy_factory(run)
            self._register_comfy(run_id, comfy)
            models_provider = getattr(comfy, "models", None)
            installed_diffusion_models = models_provider("diffusion_models") if callable(models_provider) else []
            installed_text_encoders = models_provider("text_encoders") if callable(models_provider) else []
            installed_vaes = models_provider("vae") if callable(models_provider) else []
            required_models = (
                ("qwen_image_2512_fp8_e4m3fn.safetensors", installed_diffusion_models),
                ("qwen_2.5_vl_7b_fp8_scaled.safetensors", installed_text_encoders),
                ("qwen_image_vae.safetensors", installed_vaes),
            )
            if not all(required in installed for required, installed in required_models):
                raise RuntimeError(
                    "Direct Qwen Image requires the Qwen Image 2512 model, text encoder, and VAE in ComfyUI. "
                    "Run adapters/install_qwen_image_edit_models.py --profile image-2512."
                )
            profile_path = attempt_root / "workflow_profile.json"
            _write_json(
                profile_path,
                {
                    "schema_version": 1,
                    "strategy": "qwen_image_2512_manual_prompt_v1",
                    "backend": "qwen_image_2512",
                    "seed": actual_seed,
                    "human_prompt_preserved": True,
                    "qwen_prompt_rewritten": instruction is not None,
                    "qwen_prompt_rewrite_error": instruction_error or None,
                    "effective_positive_prompt": effective_prompt,
                    "effective_negative_prompt": effective_negative,
                },
            )
            self.store.evidence(
                run,
                "D1",
                profile_path,
                evidence_id=f"d1-i{iteration:02d}-manual-qwen-image-profile",
                label=f"Direct Qwen Image workflow profile, iteration {iteration}",
                media_type="application/json",
                metrics={
                    "iteration": iteration,
                    "selectable": False,
                    "role": "workflow_profile",
                    "workflow_strategy": "qwen_image_2512_manual_prompt_v1",
                },
            )
            self._progress(run, "D1", 0.28, f"Qwen Image 2512 is rendering your direct candidate (seed {actual_seed}).")
            outputs = comfy.generate(
                workflow=qwen_image_2512_workflow(
                    prompt=effective_prompt,
                    negative_prompt=effective_negative,
                    seed=actual_seed,
                    prefix=f"DarknessStudio/{run.run_id}/manual/i{iteration:02d}",
                ),
                destination=attempt_root / "candidate-1",
                timeout_seconds=1200,
            )
            if self._was_stopped(run_id):
                self._mark_stopped(run_id)
                return
            if not outputs:
                raise RuntimeError("Qwen Image 2512 returned no output image")
            image = outputs[0]
            evidence_id = f"d1-i{iteration:02d}-candidate-qwen-image-2512-manual-01"
            metrics = _image_metrics(image)
            metrics.update(
                {
                    "seed": actual_seed,
                    "iteration": iteration,
                    "selectable": True,
                    "operation_id": "human_direct_qwen_image_prompt",
                    "workflow_strategy": "qwen_image_2512_manual_prompt_v1",
                    "concept_backend": "qwen_image_2512",
                    "human_prompt": True,
                    "qwen_prompt_rewritten": instruction is not None,
                }
            )
            self.store.evidence(
                run,
                "D1",
                image,
                evidence_id=evidence_id,
                label=f"Your direct Qwen Image candidate, iteration {iteration}",
                media_type="image/png",
                metrics=metrics,
            )
            candidates = [(evidence_id, image, metrics)]
            comparison_board = _concept_comparison_board(
                self.store,
                run,
                stage,
                candidates,
                attempt_root / "previous_vs_direct_candidate.png",
            )
            self.store.evidence(
                run,
                "D1",
                comparison_board,
                evidence_id=f"d1-i{iteration:02d}-manual-qwen-image-comparison-board",
                label=f"Direct Qwen Image comparison, iteration {iteration}",
                media_type="image/png",
                metrics={"iteration": iteration, "selectable": False},
            )
            self._progress(run, "D1", 0.82, "Qwen critic is reviewing your direct candidate against the typed contract.")
            try:
                review = qwen.review_concepts(
                    run.spec,
                    stage,
                    candidates,
                    comparison_board=comparison_board,
                )
            except Exception as exc:
                review = StudioQwenReview(
                    review_id=f"d1.manual-qwen.unavailable-{iteration:02d}",
                    stage_id="D1",
                    iteration=iteration,
                    summary="The direct Qwen Image candidate rendered, but the Qwen critic did not complete.",
                    issues=[f"Qwen critic error: {type(exc).__name__}: {exc}"],
                    candidate_ranking=[evidence_id],
                    recommended_evidence_id=None,
                    recommended_changes=["Use your review comment to direct the next candidate."],
                    confidence=0.0,
                    hard_requirements_satisfied=False,
                    request_human_review=True,
                )
            stage.qwen_reviews.append(review)
            review_path = attempt_root / "qwen_review.json"
            _write_json(review_path, review.model_dump(mode="json"))
            self.store.evidence(
                run,
                "D1",
                review_path,
                evidence_id=f"d1-i{iteration:02d}-manual-qwen-image-review",
                label=f"Qwen review of your direct candidate, iteration {iteration}",
                media_type="application/json",
                metrics={"iteration": iteration, "selectable": False, "confidence": review.confidence},
            )
            stage.metrics = {
                "iteration": iteration,
                "candidate_count": 1,
                "qwen_confidence": review.confidence,
                "recommended_evidence_id": review.recommended_evidence_id or "",
                "hard_requirements_satisfied": review.hard_requirements_satisfied,
                "workflow_strategy": "qwen_image_2512_manual_prompt_v1",
                "human_prompt": True,
            }
            stage.progress = 1
            stage.finished_at = utc_now()
            stage.state = "awaiting_review"
            stage.message = "Your direct Qwen Image candidate and Qwen review are ready for approval or rejection."
            run.state = "awaiting_review"
            self.store.event(
                run,
                "manual_qwen_image_ready",
                {
                    "stage_id": "D1",
                    "iteration": iteration,
                    "evidence_id": evidence_id,
                    "recommended_evidence_id": review.recommended_evidence_id,
                },
            )
        except Exception as exc:
            if self._was_stopped(run_id):
                self._mark_stopped(run_id)
                return
            run = self.store.load(run_id)
            stage = run.stage(run.current_stage)
            stage.state = "failed"
            stage.error = f"{type(exc).__name__}: {exc}"
            stage.message = "Direct Qwen Image render failed. Fix the dependency or use Resume without losing history."
            stage.finished_at = utc_now()
            run.state = "failed"
            self.store.event(
                run,
                "manual_qwen_image_failed",
                {"stage_id": stage.stage_id, "error": stage.error},
            )
        finally:
            self._clear_comfy(run_id)

    def _run_d0(self, run: StudioRun) -> None:
        self._begin(run, "D0", "Qwen is compiling the description into a production contract.")
        qwen = self._qwen_factory(run)
        spec = qwen.compile_spec(run.description)
        run.spec = spec
        run.title = spec.title
        root = self.store.run_root(run.run_id)
        spec_path = root / "D0_brief" / "asset_spec.json"
        _write_json(spec_path, spec.model_dump(mode="json"))
        self.store.evidence(
            run,
            "D0",
            spec_path,
            evidence_id="d0-asset-spec",
            label="Qwen-compiled asset specification",
            media_type="application/json",
            metrics={
                "equipment_count": len(spec.equipment),
                "animation_count": len(spec.animations),
                "locked_feature_count": len(spec.locked_features),
            },
        )
        stage = run.stage("D0")
        skip_reasons: dict[str, str] = {}
        if spec.asset_kind == "material":
            skip_reasons.update(
                {stage_id: "Material assets do not require geometry or articulation." for stage_id in ("D2", "D3", "D4", "D5", "D6", "D7")}
            )
        elif spec.behavior == "static":
            skip_reasons.update(
                {stage_id: "Static assets do not require skeleton, rig, deformation, or motion." for stage_id in ("D4", "D5", "D6", "D7")}
            )
        elif spec.behavior == "rigid_articulated":
            skip_reasons["D6"] = "Rigid articulated assets do not require deforming skin weights."
            if not spec.animations:
                skip_reasons["D7"] = "No rigid motion clips were requested."
        elif spec.behavior == "simulated":
            skip_reasons.update(
                {stage_id: "Simulated assets use a simulation contract rather than a skeletal rig." for stage_id in ("D4", "D5", "D6")}
            )
        for skipped_id, reason in skip_reasons.items():
            skipped = run.stage(skipped_id)
            skipped.applicable = False
            skipped.state = "skipped"
            skipped.progress = 1
            skipped.message = reason
        stage.metrics = {
            "asset_kind": spec.asset_kind,
            "behavior": spec.behavior,
            "equipment_count": len(spec.equipment),
            "animation_count": len(spec.animations),
            "explicit_right_weapon": any(
                item.category == "weapon" and item.side == "right" for item in spec.equipment
            ),
            "explicit_left_shield": any(
                item.category == "shield" and item.side == "left" for item in spec.equipment
            ),
        }
        stage.progress = 1
        stage.state = "approved"
        stage.finished_at = utc_now()
        stage.message = "Description compiled into a typed asset contract and deterministic constraints passed."
        self.store.event(run, "automatic_gate_passed", {"stage_id": "D0", "metrics": stage.metrics})

    def _run_d1(self, run: StudioRun) -> None:
        stage = run.stage("D1")
        structural_iterations = {
            int(item.metrics.get("iteration", 0))
            for item in stage.evidence
            if str(item.metrics.get("workflow_strategy", "")).endswith(("_v2", "_v3"))
        }
        qwen_image_iterations = {
            int(item.metrics.get("iteration", 0))
            for item in stage.evidence
            if str(item.metrics.get("workflow_strategy", "")).startswith("qwen_image_")
        }
        qwen_retry_selected = run.concept_backend.startswith("qwen_image_") or (
            run.concept_backend == "auto" and bool(qwen_image_iterations)
        )
        retry_budget_exhausted = (
            len(qwen_image_iterations) >= 3 if qwen_retry_selected else len(structural_iterations) >= 2
        )
        if stage.iteration >= 6 and stage.state == "rejected" and retry_budget_exhausted:
            stage.state = "blocked"
            stage.message = (
                "The bounded concept-backend retry budget was rejected after the original prompt-only attempts. "
                "The evidence is preserved for a human strategy decision instead of spending more GPU time."
            )
            run.state = "blocked"
            self.store.event(
                run,
                "iteration_budget_exhausted",
                {
                    "stage_id": "D1",
                    "workflow_strategy": "qwen_image_2512" if qwen_retry_selected else "pose_guided_lora_v3",
                },
            )
            return
        stage_overrides = self._begin(
            run, "D1", "Qwen is planning the next two concept candidates from the full history."
        )
        if run.spec is None:
            raise RuntimeError("D1 requires the compiled D0 specification")
        qwen = self._qwen_factory(run)
        correction: ConceptCorrectionPlan | None = None
        is_rejected_retry = awaiting_correction(stage)
        if is_rejected_retry:
            all_selectable = [
                item
                for item in stage.evidence
                if item.media_type.startswith("image/")
                and item.metrics.get("selectable") is not False
                and "candidate" in item.evidence_id
            ]
            latest_evidence_iteration = max(
                (int(item.metrics.get("iteration", 0)) for item in all_selectable),
                default=0,
            )
            visible_candidates = [
                item
                for item in all_selectable
                if int(item.metrics.get("iteration", 0)) == latest_evidence_iteration
            ]
            latest_rejection = latest_correction(stage)
            visible_ids = {item.evidence_id for item in visible_candidates}
            if latest_rejection and latest_rejection.selected_evidence_id:
                visible_ids.add(latest_rejection.selected_evidence_id)
            candidate_ids = sorted(visible_ids)
            comparison_item = next(
                (
                    item
                    for item in reversed(stage.evidence)
                    if item.evidence_id.endswith("comparison-board")
                ),
                None,
            )
            comparison_path = (
                self.store.artifact_path(run.run_id, comparison_item.relative_path)
                if comparison_item is not None
                else None
            )
            if comparison_path is None and visible_candidates:
                comparison_path = _concept_comparison_board(
                    self.store,
                    run,
                    stage,
                    [
                        (
                            item.evidence_id,
                            self.store.artifact_path(run.run_id, item.relative_path),
                            dict(item.metrics),
                        )
                        for item in visible_candidates
                    ],
                    self.store.run_root(run.run_id)
                    / "D1_concept"
                    / f"correction_history_before_i{stage.iteration:02d}.png",
                )
            correction = qwen.concept_correction_plan(
                run.spec,
                stage,
                candidate_ids,
                comparison_board=comparison_path,
            )
            plan: ConceptPlan | ConceptCorrectionPlan = correction
        else:
            plan = qwen.concept_plan(run.spec, stage)
        # A human "retry"/"edit" may pin the first seed for this one attempt, so
        # a candidate they liked can be re-rolled deterministically instead of
        # accepting whatever Qwen proposes next. The second seed stays Qwen's so
        # the attempt still offers a genuine alternative to compare against.
        pinned_seed = stage_overrides.get("seed")
        if pinned_seed is not None:
            if not isinstance(pinned_seed, int) or isinstance(pinned_seed, bool) or pinned_seed < 0:
                raise ValueError("the 'seed' stage override must be a non-negative whole number")
            other = next((item for item in plan.seeds if item != pinned_seed), pinned_seed + 1)
            plan.seeds = [pinned_seed, other]
        attempt_root = self.store.run_root(run.run_id) / "D1_concept" / f"iteration-{stage.iteration:02d}"
        _write_json(
            attempt_root / "qwen_plan.json",
            {
                "plan": plan.model_dump(mode="json"),
                "mode": "targeted_correction" if correction else "new_generation",
                "history_iteration": stage.iteration,
            },
        )
        self._progress(run, "D1", 0.12, "Qwen plan is ready; selecting the installed ComfyUI concept backend.")
        comfy = self._comfy_factory(run)
        self._register_comfy(run.run_id, comfy)
        controlnet_provider = getattr(comfy, "controlnets", None)
        installed_controlnets = controlnet_provider() if callable(controlnet_provider) else []
        models_provider = getattr(comfy, "models", None)
        installed_loras = models_provider("loras") if callable(models_provider) else []
        installed_diffusion_models = models_provider("diffusion_models") if callable(models_provider) else []
        installed_text_encoders = models_provider("text_encoders") if callable(models_provider) else []
        installed_vaes = models_provider("vae") if callable(models_provider) else []
        qwen_image_2512_models_ready = all(
            required in models
            for required, models in (
                ("qwen_image_2512_fp8_e4m3fn.safetensors", installed_diffusion_models),
                ("qwen_2.5_vl_7b_fp8_scaled.safetensors", installed_text_encoders),
                ("qwen_image_vae.safetensors", installed_vaes),
            )
        )
        if run.concept_backend in {"qwen_image_2512", "qwen_image_edit_2511"} and not qwen_image_2512_models_ready:
            raise RuntimeError(
                "Qwen Image 2512 was selected for concept generation but its required ComfyUI model files are missing. "
                "Run adapters/install_qwen_image_edit_models.py --profile image-2512 or select the SDXL backend."
            )
        use_qwen_image_generation = qwen_image_2512_models_ready and run.concept_backend in {
            "auto",
            "qwen_image_2512",
            # Legacy persisted value: D1 now uses the proper generator model.
            "qwen_image_edit_2511",
        }
        if not use_qwen_image_generation and run.checkpoint not in comfy.checkpoints():
            raise RuntimeError(f"required ComfyUI checkpoint is not installed: {run.checkpoint}")
        pose_guided = not use_qwen_image_generation and run.spec.anatomy_family == "humanoid" and (
            correction is None or correction.operation_id == "regenerate_complete_asset"
        )
        deferred_shield = _deferred_shield_equipment(run.spec) if pose_guided else None
        workflow_strategy = (
            "qwen_image_2512_t2i_v1"
            if use_qwen_image_generation
            else "pose_guided_lora_v3"
            if pose_guided
            else "local_crop_inpaint_v2"
            if correction is not None and correction.operation_id != "regenerate_complete_asset"
            else "global_text_v1"
        )
        control_guides: list[tuple[str, str, float, float, float]] = []
        guide_models: list[str] = []
        effective_positive, effective_negative = _whole_asset_prompt_guard(
            run.spec,
            plan.positive_prompt,
            plan.negative_prompt,
            defer_shield=not use_qwen_image_generation,
        )
        applied_loras: list[tuple[str, float]] = [
            (name, strength)
            for name, strength in (
                (run.style_lora, run.style_lora_strength),
                (run.prop_lora, run.prop_lora_strength),
            )
            if not use_qwen_image_generation and name and name in installed_loras
        ]
        lora_trigger_words = ", ".join(
            trigger
            for name, trigger in ((run.style_lora, run.style_lora_trigger),)
            if not use_qwen_image_generation and trigger and name and name in installed_loras
        )
        if lora_trigger_words:
            effective_positive = f"{lora_trigger_words}, {effective_positive}"
        plain_positive = f"{lora_trigger_words}, {plan.positive_prompt}" if lora_trigger_words else plan.positive_prompt
        if pose_guided:
            pose_path = make_humanoid_openpose_guide(attempt_root / "layout_guides")
            pose_model = next(
                (item for item in installed_controlnets if "openpose" in item.lower()),
                None,
            )
            self.store.evidence(
                run,
                "D1",
                pose_path,
                evidence_id=f"d1-i{stage.iteration:02d}-openpose-guide",
                label="Deterministic OpenPose structural guide",
                media_type="image/png",
                metrics={
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "structural_guide",
                    "workflow_strategy": workflow_strategy,
                    "controlnet_model": pose_model or "not-installed",
                },
            )
            if pose_model is None:
                raise RuntimeError(
                    "A humanoid concept generation requires an installed OpenPose ControlNet so the "
                    "figure stays one coherent body (e.g. controlnet_openpose_sdxl_xinsir.safetensors "
                    "in ComfyUI's models/controlnet folder). None of the installed ControlNets matched "
                    f"'openpose'. Installed: {installed_controlnets or 'none'}."
                )
            upload_folder = f"darkness_studio/{run.run_id}/i{stage.iteration:02d}"
            uploaded = comfy.upload_image(pose_path.name, pose_path.read_bytes(), upload_folder)
            control_guides.append((pose_model, uploaded, 0.85, 0.0, 0.85))
            guide_models.append(pose_model)
        qwen_edit_instruction = None
        qwen_instruction_error = ""
        qwen_edit_prompt = effective_positive
        qwen_edit_negative = " " if use_qwen_image_generation else effective_negative
        if use_qwen_image_generation:
            rewriter = getattr(qwen, "image_edit_instruction", None)
            if callable(rewriter):
                try:
                    qwen_edit_instruction = rewriter(
                        run.spec,
                        stage,
                        plan,
                        source_handling="new_text_to_image",
                    )
                    qwen_edit_prompt = qwen_edit_instruction.prompt
                    qwen_edit_negative = qwen_edit_instruction.negative_prompt
                    instruction_path = attempt_root / "qwen_image_edit_instruction.json"
                    _write_json(instruction_path, qwen_edit_instruction.model_dump(mode="json"))
                    self.store.evidence(
                        run,
                        "D1",
                        instruction_path,
                        evidence_id=f"d1-i{stage.iteration:02d}-qwen-image-edit-instruction",
                        label="Qwen-rewritten native image-edit instruction",
                        media_type="application/json",
                        metrics={
                            "iteration": stage.iteration,
                            "selectable": False,
                            "workflow_strategy": workflow_strategy,
                            "source_handling": qwen_edit_instruction.source_handling,
                        },
                    )
                except Exception as exc:
                    qwen_instruction_error = f"{type(exc).__name__}: {exc}"
        profile_path = attempt_root / "workflow_profile.json"
        _write_json(
            profile_path,
            {
                "schema_version": 1,
                "strategy": workflow_strategy,
                "concept_backend": "qwen_image_2512" if use_qwen_image_generation else "sdxl",
                "checkpoint": run.checkpoint,
                "qwen_image_models_ready": qwen_image_2512_models_ready,
                "qwen_diffusion_model": "qwen_image_2512_fp8_e4m3fn.safetensors" if use_qwen_image_generation else None,
                "qwen_text_encoder": "qwen_2.5_vl_7b_fp8_scaled.safetensors" if use_qwen_image_generation else None,
                "qwen_vae": "qwen_image_vae.safetensors" if use_qwen_image_generation else None,
                "qwen_prompt_rewritten": qwen_edit_instruction is not None,
                "qwen_prompt_rewrite_error": qwen_instruction_error or None,
                "installed_controlnets": installed_controlnets,
                "applied_controlnets": guide_models,
                "installed_loras": installed_loras,
                "applied_loras": [name for name, _ in applied_loras],
                "pose_guided": pose_guided,
                "local_high_resolution_crop": workflow_strategy == "local_crop_inpaint_v2",
                "ordinary_checkpoint_inpaint_encoder": "VAEEncodeForInpaint",
                "whole_asset_prompt_guard": run.spec.asset_kind == "character",
                "lora_trigger_words": lora_trigger_words,
                "effective_positive_prompt": qwen_edit_prompt if use_qwen_image_generation else (effective_positive if pose_guided else plain_positive),
                "effective_negative_prompt": qwen_edit_negative if use_qwen_image_generation else (effective_negative if pose_guided else plan.negative_prompt),
            },
        )
        self.store.evidence(
            run,
            "D1",
            profile_path,
            evidence_id=f"d1-i{stage.iteration:02d}-workflow-profile",
            label=f"ComfyUI workflow profile, iteration {stage.iteration}",
            media_type="application/json",
            metrics={
                "iteration": stage.iteration,
                "selectable": False,
                "role": "workflow_profile",
                "workflow_strategy": workflow_strategy,
            },
        )
        candidates: list[tuple[str, Path, dict[str, object]]] = []
        source_upload = None
        mask_upload = None
        correction_source: Path | None = None
        correction_mask: Path | None = None
        correction_crop_box: tuple[int, int, int, int] | None = None
        if correction and not use_qwen_image_generation and correction.operation_id != "regenerate_complete_asset":
            source_item = next(
                (item for item in stage.evidence if item.evidence_id == correction.base_evidence_id),
                None,
            )
            if source_item is None or not source_item.media_type.startswith("image/"):
                raise RuntimeError("Qwen correction selected a base that is not a visual candidate")
            correction_source = self.store.artifact_path(run.run_id, source_item.relative_path)
            crop, correction_mask, correction_crop_box = _prepare_inpaint_crop(
                correction_source,
                correction.edit_box_normalized,
                attempt_root / "local_repair",
            )
            upload_folder = f"darkness_studio/{run.run_id}/i{stage.iteration:02d}"
            source_upload = comfy.upload_image(crop.name, crop.read_bytes(), upload_folder)
            mask_upload = comfy.upload_image(
                correction_mask.name,
                correction_mask.read_bytes(),
                upload_folder,
            )
            for role, path, label in (
                ("repair_crop", crop, "High-resolution local correction crop"),
                ("correction_mask", correction_mask, "Feathered local correction mask"),
            ):
                self.store.evidence(
                    run,
                    "D1",
                    path,
                    evidence_id=f"d1-i{stage.iteration:02d}-{role}",
                    label=f"{label}, iteration {stage.iteration}",
                    media_type="image/png",
                    metrics={
                        "iteration": stage.iteration,
                        "selectable": False,
                        "operation_id": correction.operation_id,
                        "base_evidence_id": correction.base_evidence_id,
                        "workflow_strategy": workflow_strategy,
                    },
                )
        for index, seed in enumerate(plan.seeds, start=1):
            evidence_id = f"d1-i{stage.iteration:02d}-candidate-{index}"
            self._progress(
                run,
                "D1",
                0.15 + (index - 1) * 0.32,
                f"ComfyUI is rendering concept {index} of {len(plan.seeds)} (seed {seed}).",
            )
            if use_qwen_image_generation:
                workflow = qwen_image_2512_workflow(
                    prompt=qwen_edit_prompt,
                    negative_prompt=qwen_edit_negative,
                    seed=seed,
                    prefix=f"DarknessStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}",
                )
            elif (
                correction
                and correction.operation_id != "regenerate_complete_asset"
                and source_upload
                and mask_upload
            ):
                workflow = inpaint_workflow(
                    checkpoint=run.checkpoint,
                    positive=plain_positive,
                    negative=plan.negative_prompt,
                    seed=seed,
                    prefix=f"DarknessStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}",
                    source_image=source_upload,
                    mask_image=mask_upload,
                    denoise=correction.denoise,
                    loras=applied_loras,
                )
            else:
                workflow = concept_workflow(
                    checkpoint=run.checkpoint,
                    positive=effective_positive if pose_guided else plain_positive,
                    negative=effective_negative if pose_guided else plan.negative_prompt,
                    seed=seed,
                    prefix=f"DarknessStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}",
                    loras=applied_loras,
                    control_guides=control_guides,
                    steps=run.concept_steps,
                    cfg=run.concept_cfg,
                )
            outputs = comfy.generate(
                workflow=workflow,
                destination=attempt_root / f"candidate-{index}",
            )
            image = outputs[0]
            if correction_source and correction_mask and correction_crop_box:
                image = _composite_inpaint_crop(
                    correction_source,
                    image,
                    correction_mask,
                    correction_crop_box,
                    attempt_root / f"candidate-{index}" / "composited_full_image.png",
                )
            shield_repaired = False
            if not use_qwen_image_generation and deferred_shield is not None and not (
                correction and correction.operation_id != "regenerate_complete_asset"
            ):
                shield_dir = attempt_root / f"candidate-{index}" / "shield_repair"
                shield_box = _limb_repair_box(deferred_shield.side)
                shield_crop, shield_mask, shield_crop_box = _prepare_inpaint_crop(image, shield_box, shield_dir)
                shield_upload_folder = f"darkness_studio/{run.run_id}/i{stage.iteration:02d}"
                shield_crop_upload = comfy.upload_image(
                    shield_crop.name, shield_crop.read_bytes(), shield_upload_folder
                )
                shield_mask_upload = comfy.upload_image(
                    shield_mask.name, shield_mask.read_bytes(), shield_upload_folder
                )
                shield_positive, shield_negative = _deferred_shield_repair_prompts(deferred_shield)
                shield_workflow = inpaint_workflow(
                    checkpoint=run.checkpoint,
                    positive=shield_positive,
                    negative=shield_negative,
                    seed=seed + 1,
                    prefix=f"DarknessStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}_shield",
                    source_image=shield_crop_upload,
                    mask_image=shield_mask_upload,
                    denoise=0.75,
                    loras=applied_loras,
                )
                shield_generated = comfy.generate(
                    workflow=shield_workflow,
                    destination=shield_dir / "generated",
                )[0]
                image = _composite_inpaint_crop(
                    image, shield_generated, shield_mask, shield_crop_box, shield_dir / "composited.png"
                )
                shield_repaired = True
            metrics = _image_metrics(image)
            metrics.update(
                {
                    "seed": seed,
                    "iteration": stage.iteration,
                    "selectable": True,
                    "operation_id": correction.operation_id if correction else "regenerate_complete_asset",
                    "base_evidence_id": correction.base_evidence_id if correction else "",
                    "workflow_strategy": workflow_strategy,
                    "concept_backend": "qwen_image_2512" if use_qwen_image_generation else "sdxl",
                    "controlnet_count": len(control_guides),
                    "lora_count": len(applied_loras),
                    "local_high_resolution_crop": correction_crop_box is not None,
                    "deferred_shield_repaired": shield_repaired,
                }
            )
            self.store.evidence(
                run,
                "D1",
                image,
                evidence_id=evidence_id,
                label=f"Concept {index}, iteration {stage.iteration}",
                media_type="image/png",
                metrics=metrics,
            )
            candidates.append((evidence_id, image, metrics))
        comparison_board = _concept_comparison_board(
            self.store,
            run,
            stage,
            candidates,
            attempt_root / "previous_vs_current.png",
        )
        self.store.evidence(
            run,
            "D1",
            comparison_board,
            evidence_id=f"d1-i{stage.iteration:02d}-comparison-board",
            label=f"Previous rejection versus current candidates, iteration {stage.iteration}",
            media_type="image/png",
            metrics={"iteration": stage.iteration, "selectable": False},
        )
        self._progress(run, "D1", 0.82, "Qwen critic is reading the labeled rejection-versus-current evidence board.")
        try:
            review = qwen.review_concepts(
                run.spec,
                stage,
                candidates,
                comparison_board=comparison_board,
            )
        except Exception as exc:
            review = StudioQwenReview(
                review_id=f"d1.qwen.unavailable-{stage.iteration:02d}",
                stage_id="D1",
                iteration=stage.iteration,
                summary=(
                    "The Qwen critic did not finish inside the bounded review window. The generated images are "
                    "available now; no candidate is automatically recommended."
                ),
                issues=[f"Qwen critic error: {type(exc).__name__}: {exc}"],
                candidate_ranking=[item[0] for item in candidates],
                recommended_evidence_id=None,
                recommended_changes=["Use the human decision as the next optimizer instruction."],
                confidence=0.0,
                hard_requirements_satisfied=False,
                request_human_review=True,
            )
        stage.qwen_reviews.append(review)
        review_path = attempt_root / "qwen_review.json"
        _write_json(review_path, review.model_dump(mode="json"))
        self.store.evidence(
            run,
            "D1",
            review_path,
            evidence_id=f"d1-i{stage.iteration:02d}-qwen-review",
            label=f"Qwen comparison, iteration {stage.iteration}",
            media_type="application/json",
            metrics={"confidence": review.confidence, "selectable": False},
        )
        stage.metrics = {
            "iteration": stage.iteration,
            "candidate_count": len(candidates),
            "qwen_confidence": review.confidence,
            "recommended_evidence_id": review.recommended_evidence_id or "",
            "hard_requirements_satisfied": review.hard_requirements_satisfied,
            "workflow_strategy": workflow_strategy,
            "controlnet_count": len(control_guides),
            "lora_count": len(applied_loras),
        }
        stage.progress = 1
        stage.finished_at = utc_now()
        automatically_retry = (
            is_rejected_retry
            and not review.hard_requirements_satisfied
            and review.confidence >= 0.6
            and stage.iteration < 6
        )
        if automatically_retry:
            stage.state = "rejected"
            stage.message = (
                "Qwen referee rejected this correction because a locked requirement is still visibly missing. "
                "The next bounded correction will run automatically."
            )
            run.state = "running"
            self.store.event(
                run,
                "qwen_referee_rejected",
                {
                    "stage_id": "D1",
                    "iteration": stage.iteration,
                    "summary": review.summary,
                },
            )
            return
        stage.state = "awaiting_review"
        stage.message = "Concept candidates and Qwen comparison are ready for your approval or rejection."
        run.state = "awaiting_review"
        self.store.event(
            run,
            "human_gate_ready",
            {
                "stage_id": "D1",
                "iteration": stage.iteration,
                "recommended_evidence_id": review.recommended_evidence_id,
            },
        )

    def _selected_evidence_path(self, run: StudioRun, stage_id: str) -> Path:
        stage = run.stage(stage_id)
        approvals = [item for item in stage.human_decisions if item.decision == "approve"]
        if not approvals or not approvals[-1].selected_evidence_id:
            raise RuntimeError(f"{stage_id} has no approved selected evidence")
        evidence_id = approvals[-1].selected_evidence_id
        evidence = next((item for item in stage.evidence if item.evidence_id == evidence_id), None)
        if evidence is None:
            raise RuntimeError(f"selected {stage_id} evidence no longer exists")
        return self.store.artifact_path(run.run_id, evidence.relative_path)

    def _latest_evidence_path(
        self,
        run: StudioRun,
        stage_id: str,
        *,
        media_type: str | None = None,
        role: str | None = None,
    ) -> Path:
        for item in reversed(run.stage(stage_id).evidence):
            if media_type is not None and item.media_type != media_type:
                continue
            if role is not None and item.metrics.get("role") != role:
                continue
            return self.store.artifact_path(run.run_id, item.relative_path)
        raise RuntimeError(f"{stage_id} has no evidence matching media_type={media_type!r}, role={role!r}")

    @staticmethod
    def _artifact_for_path(run: StudioRun, path: Path, *, artifact_id: str) -> ArtifactRecord:
        config = load_local_config()
        if config is None:
            raise RuntimeError("Darkness config.local.toml is required")
        workspace = Path(config.workspace_root).resolve()
        resolved = path.resolve()
        digest = sha256_file(resolved)
        return ArtifactRecord(
            artifact_id=artifact_id,
            sha256=digest,
            size_bytes=resolved.stat().st_size,
            media_type=(
                "application/x-blender"
                if resolved.suffix.lower() == ".blend"
                else "model/gltf-binary"
            ),
            stage=AssetStage.geometry,
            blob_path=resolved.relative_to(workspace).as_posix(),
            created_at=datetime.now(timezone.utc),
            lineage=ArtifactLineage(
                artifact_id=artifact_id,
                artifact_sha256=digest,
                stage=AssetStage.geometry.value,
                source_license_ids=["user-authored", "project-generated"],
            ),
            metadata={"studio_run_id": run.run_id},
        )

    def _execute_worker(
        self,
        worker_id: str,
        request: ExternalWorkerRequest,
        *,
        timeout_seconds: float,
    ):
        if self._worker_executor is not None:
            return self._worker_executor(worker_id, request, timeout_seconds=timeout_seconds)
        config = load_local_config()
        if config is None:
            raise RuntimeError("Darkness config.local.toml is required")
        binding = worker_binding(config, worker_id)
        if binding is None:
            raise RuntimeError(f"Darkness worker has no machine binding: {worker_id}")
        manifest = load_manifests()[worker_id]
        workspace = Path(config.workspace_root).resolve()
        adapter = SubprocessWorkerAdapter(
            WorkerManager(workspace, allowed_roots=[workspace]),
            manifest,
            binding.command_prefix,
            environment=binding.environment,
        )
        return adapter.execute(request, timeout_seconds=timeout_seconds)

    def _run_d2(self, run: StudioRun) -> None:
        stage = run.stage("D2")
        if run.spec is None:
            raise RuntimeError("D2 requires the compiled specification")
        selected_concept = self._selected_evidence_path(run, "D1")
        self._begin(run, "D2", "Qwen is translating the approved identity into a riggable geometry seed.")
        qwen = self._qwen_factory(run)
        plan = qwen.geometry_seed_plan(run.spec, stage, selected_concept)
        attempt_root = self.store.run_root(run.run_id) / "D2_geometry" / f"iteration-{stage.iteration:02d}"
        _write_json(attempt_root / "qwen_geometry_seed_plan.json", plan.model_dump(mode="json"))
        comfy = self._comfy_factory(run)
        self._register_comfy(run.run_id, comfy)
        if run.checkpoint not in comfy.checkpoints():
            raise RuntimeError(f"required ComfyUI checkpoint is not installed: {run.checkpoint}")
        self._progress(run, "D2", 0.08, "ComfyUI is rendering the unarmed A-pose geometry seed.")
        workflow = concept_workflow(
            checkpoint=run.checkpoint,
            positive=plan.positive_prompt,
            negative=plan.negative_prompt,
            seed=plan.seed,
            prefix=f"DarknessStudio/{run.run_id}/geometry/i{stage.iteration:02d}",
            steps=run.concept_steps,
            cfg=run.concept_cfg,
        )
        raw_seed = comfy.generate(workflow=workflow, destination=attempt_root / "seed")[0]
        rgba_seed = attempt_root / "geometry_seed_rgba.png"
        alpha_metrics = make_chroma_alpha(raw_seed, rgba_seed)
        alpha_metrics.update({"seed": plan.seed, "iteration": stage.iteration})
        self.store.evidence(
            run,
            "D2",
            rgba_seed,
            evidence_id=f"d2-i{stage.iteration:02d}-rgba-seed",
            label=f"Geometry seed with deterministic alpha, iteration {stage.iteration}",
            media_type="image/png",
            metrics=alpha_metrics,
        )
        self._progress(run, "D2", 0.2, "TRELLIS.2 is generating the 3D candidate from owned RGBA input.")
        artifact_id = f"{run.spec.asset_id}.d2.seed.i{stage.iteration:02d}"
        digest = sha256_file(rgba_seed)
        artifact = ArtifactRecord(
            artifact_id=artifact_id,
            sha256=digest,
            size_bytes=rgba_seed.stat().st_size,
            media_type="image/png",
            stage=AssetStage.concept,
            blob_path=rgba_seed.relative_to(Path(load_local_config().workspace_root).resolve()).as_posix(),
            created_at=datetime.now(timezone.utc),
            lineage=ArtifactLineage(
                artifact_id=artifact_id,
                artifact_sha256=digest,
                stage=AssetStage.concept.value,
                source_license_ids=["user-authored", "project-generated"],
            ),
            metadata={"meaningful_alpha": True, "studio_run_id": run.run_id},
        )
        output_root = attempt_root / "trellis2"
        request = ExternalWorkerRequest(
            job_id=f"studio.{run.run_id}.d2.i{stage.iteration:02d}",
            run_id=run.run_id,
            operation_id="geometry.generate_from_rgba",
            stage=AssetStage.geometry,
            inputs=[artifact],
            input_paths={artifact_id: str(rgba_seed)},
            parameters={
                "seed": plan.seed,
                "decimation_target": 300000,
                "texture_size": 2048,
                "remesh": True,
            },
            output_directory=str(output_root),
        )
        response = self._execute_worker("trellis2.4b", request, timeout_seconds=1800)
        geometry_output = next(
            (Path(item.path) for item in response.outputs if item.media_type == "model/gltf-binary"),
            None,
        )
        if geometry_output is None:
            raise RuntimeError("TRELLIS.2 returned no GLB candidate")
        self._progress(run, "D2", 0.78, "Building deterministic four-view geometry evidence.")
        diagnostic = attempt_root / "geometry_diagnostic.png"
        script = Path(__file__).resolve().parents[1] / "adapters" / "render_glb_diagnostic.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--input", str(geometry_output), "--output", str(diagnostic)],
            text=True,
            capture_output=True,
            timeout=300,
        )
        if completed.returncode != 0 or not diagnostic.is_file():
            raise RuntimeError(f"geometry diagnostic failed: {completed.stderr[-2000:]}")
        import trimesh

        loaded = trimesh.load(geometry_output, force="scene")
        meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
        metrics: dict[str, object] = {
            **response.diagnostics,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "scene_meshes": len(meshes),
            "bounds_x": float(mesh.extents[0]),
            "bounds_y": float(mesh.extents[1]),
            "bounds_z": float(mesh.extents[2]),
            "hard_gate_passed": len(mesh.vertices) >= 1000 and len(mesh.faces) >= 1000,
        }
        self.store.evidence(
            run,
            "D2",
            geometry_output,
            evidence_id=f"d2-i{stage.iteration:02d}-glb",
            label=f"TRELLIS.2 geometry candidate, iteration {stage.iteration}",
            media_type="model/gltf-binary",
            metrics=metrics,
        )
        diagnostic_evidence = self.store.evidence(
            run,
            "D2",
            diagnostic,
            evidence_id=f"d2-i{stage.iteration:02d}-diagnostic",
            label=f"Four-view geometry diagnostic, iteration {stage.iteration}",
            media_type="image/png",
            metrics={**metrics, "iteration": stage.iteration},
        )
        self._progress(run, "D2", 0.88, "Qwen is comparing the mesh against the approved identity and numeric gates.")
        review, qwen_passed = qwen.review_geometry(
            run.spec,
            stage,
            selected_concept,
            diagnostic,
            metrics,
        )
        stage.qwen_reviews.append(review)
        _write_json(attempt_root / "qwen_geometry_review.json", review.model_dump(mode="json"))
        hard_passed = bool(metrics["hard_gate_passed"])
        stage.metrics = {**metrics, "qwen_goal_satisfied": qwen_passed}
        stage.progress = 1
        stage.finished_at = utc_now()
        if hard_passed and qwen_passed:
            stage.state = "approved"
            stage.message = "Geometry passed hard gates and Qwen judged it credible for deterministic cleanup."
            self.store.event(run, "automatic_gate_passed", {"stage_id": "D2", "metrics": stage.metrics})
        elif stage.iteration < 3:
            stage.state = "rejected"
            stage.message = "Geometry did not clear the combined gate; Qwen will revise one bounded seed attempt."
            self.store.event(
                run,
                "automatic_gate_rejected",
                {"stage_id": "D2", "iteration": stage.iteration, "metrics": stage.metrics},
            )
        else:
            stage.gate_required = True
            stage.state = "awaiting_review"
            stage.message = "Three geometry attempts are preserved. Human direction is required before more GPU work."
            run.state = "awaiting_review"
            review.recommended_evidence_id = diagnostic_evidence.evidence_id
            self.store.event(run, "human_gate_ready", {"stage_id": "D2", "iteration": stage.iteration})

    def _run_d3(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D3 requires the compiled asset specification")
        source = self._latest_evidence_path(run, "D2", media_type="model/gltf-binary")
        selected_concept = self._selected_evidence_path(run, "D1")
        self._begin(run, "D3", "Blender is validating and checkpointing deterministic geometry cleanup.")
        stage = run.stage("D3")
        attempt_root = self.store.run_root(run.run_id) / "D3_cleanup" / f"iteration-{stage.iteration:02d}"
        artifact_id = f"{run.spec.asset_id}.d3.input.i{stage.iteration:02d}"
        artifact = self._artifact_for_path(run, source, artifact_id=artifact_id)
        deformable = run.spec.behavior == "deformable_animated"
        maximum_components = 1 if deformable else max(1, len(run.spec.components) * 4, 32)
        request = ExternalWorkerRequest(
            job_id=f"studio.{run.run_id}.d3.i{stage.iteration:02d}",
            run_id=run.run_id,
            operation_id="blender.repair",
            stage=AssetStage.geometry,
            inputs=[artifact],
            input_paths={artifact_id: str(source)},
            parameters={
                "component_policy": "keep_largest" if deformable else "none",
                "weld_distance": 0.00005 if deformable else 0.0,
                "render_size": 512,
                "maximum_material_change_fraction": 0.03,
                "minimum_connected_components": 1,
                "maximum_connected_components": maximum_components,
            },
            output_directory=str(attempt_root / "blender"),
        )
        response = self._execute_worker("blender", request, timeout_seconds=900)
        image_records: list[tuple[str, Path]] = []
        for output in response.outputs:
            path = Path(output.path)
            safe_role = output.role.replace("_", "-")
            evidence_id = f"d3-i{stage.iteration:02d}-{safe_role}"
            metrics = {
                "iteration": stage.iteration,
                "selectable": False,
                "role": output.role,
            }
            self.store.evidence(
                run,
                "D3",
                path,
                evidence_id=evidence_id,
                label=f"Cleanup {output.role.replace('_', ' ')}",
                media_type=output.media_type,
                metrics=metrics,
            )
            if output.media_type.startswith("image/") and output.role.startswith("candidate_"):
                image_records.append((output.role, path))
        if not image_records:
            raise RuntimeError("Blender cleanup returned no candidate diagnostic views")
        diagnostic = _image_board(
            image_records,
            attempt_root / "cleanup_diagnostic.png",
            columns=min(4, len(image_records)),
        )
        diagnostic_evidence = self.store.evidence(
            run,
            "D3",
            diagnostic,
            evidence_id=f"d3-i{stage.iteration:02d}-diagnostic",
            label=f"Cleanup fixed-view diagnostic, iteration {stage.iteration}",
            media_type="image/png",
            metrics={
                **response.diagnostics,
                "iteration": stage.iteration,
                "selectable": False,
                "role": "cleanup_diagnostic",
            },
        )
        self._progress(run, "D3", 0.86, "Qwen is checking identity preservation against numeric cleanup gates.")
        qwen = self._qwen_factory(run)
        review, qwen_passed = qwen.review_cleanup(
            run.spec,
            stage,
            selected_concept,
            diagnostic,
            dict(response.diagnostics),
        )
        stage.qwen_reviews.append(review)
        _write_json(attempt_root / "qwen_cleanup_review.json", review.model_dump(mode="json"))
        stage.metrics = {
            **response.diagnostics,
            "qwen_goal_satisfied": qwen_passed,
            "maximum_connected_components": maximum_components,
        }
        stage.progress = 1
        stage.finished_at = utc_now()
        if qwen_passed:
            stage.state = "approved"
            stage.message = "Cleanup passed Blender export gates and Qwen identity review."
            self.store.event(run, "automatic_gate_passed", {"stage_id": "D3", "metrics": stage.metrics})
        elif stage.iteration < 3:
            stage.state = "rejected"
            stage.message = "Cleanup evidence failed Qwen identity review; one bounded retry will run."
            self.store.event(
                run,
                "automatic_gate_rejected",
                {"stage_id": "D3", "iteration": stage.iteration, "evidence": diagnostic_evidence.evidence_id},
            )
        else:
            stage.gate_required = True
            stage.state = "awaiting_review"
            stage.message = "Three cleanup attempts are preserved; human direction is required."
            diagnostic_evidence.metrics["selectable"] = True
            review.recommended_evidence_id = diagnostic_evidence.evidence_id
            run.state = "awaiting_review"
            self.store.event(run, "human_gate_ready", {"stage_id": "D3", "iteration": stage.iteration})

    def _run_d4(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D4 requires the compiled asset specification")
        source = self._latest_evidence_path(
            run,
            "D3",
            media_type="model/gltf-binary",
            role="candidate_geometry",
        )
        selected_concept = self._selected_evidence_path(run, "D1")
        self._begin(run, "D4", "Building the typed canonical structure and its review evidence.")
        stage = run.stage("D4")
        attempt_root = self.store.run_root(run.run_id) / "D4_structure" / f"iteration-{stage.iteration:02d}"
        qwen = self._qwen_factory(run)
        if run.spec.behavior == "rigid_articulated":
            diagnostic = self._latest_evidence_path(run, "D3", role="candidate_front")
            plan = qwen.rigid_structure_plan(run.spec, stage, selected_concept, diagnostic)
            plan_path = attempt_root / "rigid_structure_plan.json"
            _write_json(plan_path, plan.model_dump(mode="json"))
            overlay = _rigid_structure_overlay(
                diagnostic,
                plan,
                attempt_root / "rigid_structure_overlay.png",
            )
            self.store.evidence(
                run,
                "D4",
                plan_path,
                evidence_id=f"d4-i{stage.iteration:02d}-rigid-plan",
                label="Qwen rigid structure contract",
                media_type="application/json",
                metrics={"iteration": stage.iteration, "selectable": False, "role": "rigid_structure_plan"},
            )
            overlay_item = self.store.evidence(
                run,
                "D4",
                overlay,
                evidence_id=f"d4-i{stage.iteration:02d}-rigid-overlay",
                label="Rigid component boxes, pivots, axes, and limits",
                media_type="image/png",
                metrics={"iteration": stage.iteration, "selectable": True, "role": "rigid_structure_overlay"},
            )
            review = StudioQwenReview(
                review_id=f"d4.rigid-plan.iteration-{stage.iteration:02d}",
                stage_id="D4",
                iteration=stage.iteration,
                summary=(
                    f"Qwen proposed {len(plan.parts)} typed rigid movable parts. Human approval is required for "
                    "component segmentation, pivots, rotation axes, and joint limits before Blender separates geometry."
                ),
                candidate_ranking=[overlay_item.evidence_id],
                recommended_evidence_id=overlay_item.evidence_id,
                confidence=plan.confidence,
                hard_requirements_satisfied=True,
                request_human_review=True,
            )
            stage.qwen_reviews.append(review)
            stage.metrics = {"rigid_parts": len(plan.parts), "confidence": plan.confidence}
        else:
            artifact_id = f"{run.spec.asset_id}.d4.input.i{stage.iteration:02d}"
            artifact = self._artifact_for_path(run, source, artifact_id=artifact_id)
            request = ExternalWorkerRequest(
                job_id=f"studio.{run.run_id}.d4.i{stage.iteration:02d}",
                run_id=run.run_id,
                operation_id="blender.propose_short_biped_rig",
                stage=AssetStage.rig,
                inputs=[artifact],
                input_paths={artifact_id: str(source)},
                parameters={
                    "render_size": 512,
                    "maximum_material_change_fraction": 0.03,
                    "landmark_adjustments": {},
                    "weight_adjustments": [],
                },
                output_directory=str(attempt_root / "blender"),
            )
            response = self._execute_worker("blender", request, timeout_seconds=1200)
            stress_records: list[tuple[str, Path]] = []
            for output in response.outputs:
                path = Path(output.path)
                evidence_id = f"d4-i{stage.iteration:02d}-{output.role.replace('_', '-')}"
                self.store.evidence(
                    run,
                    "D4",
                    path,
                    evidence_id=evidence_id,
                    label=f"Rig {output.role.replace('_', ' ')}",
                    media_type=output.media_type,
                    metrics={
                        "iteration": stage.iteration,
                        "selectable": False,
                        "role": output.role,
                    },
                )
                if output.media_type.startswith("image/") and output.role.startswith("rig_"):
                    stress_records.append((output.role, path))
            stress_board = _image_board(
                stress_records,
                attempt_root / "rig_stress_board.png",
                columns=4,
            )
            board_item = self.store.evidence(
                run,
                "D4",
                stress_board,
                evidence_id=f"d4-i{stage.iteration:02d}-rig-stress-board",
                label="Neutral and critical-joint deformation board",
                media_type="image/png",
                metrics={
                    **response.diagnostics,
                    "iteration": stage.iteration,
                    "selectable": True,
                    "role": "rig_stress_board",
                },
            )
            review = qwen.review_deformable_rig(
                run.spec,
                stage,
                selected_concept,
                stress_board,
                dict(response.diagnostics),
            )
            review.recommended_evidence_id = board_item.evidence_id
            stage.qwen_reviews.append(review)
            stage.metrics = dict(response.diagnostics)
        stage.progress = 1
        stage.state = "awaiting_review"
        stage.finished_at = utc_now()
        stage.message = "Canonical structure evidence is ready for your approval or rejection."
        run.state = "awaiting_review"
        self.store.event(run, "human_gate_ready", {"stage_id": "D4", "iteration": stage.iteration})

    def _adopt_d4_output(self, run: StudioRun, stage_id: str, roles: tuple[str, ...]) -> None:
        stage = run.stage(stage_id)
        self._begin(run, stage_id, f"Adopting hash-bound D4 outputs for {stage.label} validation.")
        adopted = 0
        gate_passed = True
        for item in run.stage("D4").evidence:
            role = str(item.metrics.get("role", ""))
            if role not in roles:
                continue
            path = self.store.artifact_path(run.run_id, item.relative_path)
            self.store.evidence(
                run,
                stage_id,
                path,
                evidence_id=f"{stage_id.lower()}-{role.replace('_', '-')}",
                label=f"Adopted {role.replace('_', ' ')}",
                media_type=item.media_type,
                metrics={"selectable": False, "role": role, "source_sha256": item.sha256},
            )
            adopted += 1
            if item.media_type == "application/json":
                value = json.loads(path.read_text(encoding="utf-8"))
                gate_passed = gate_passed and bool(
                    value.get("gate_passed", value.get("automatic_gate_passed", True))
                )
        if not adopted:
            raise RuntimeError(f"D4 did not produce the required {stage_id} evidence roles: {roles}")
        stage.metrics = {"adopted_artifacts": adopted, "source_stage": "D4", "hard_gate_passed": gate_passed}
        stage.progress = 1
        stage.finished_at = utc_now()
        if not gate_passed:
            raise RuntimeError(f"adopted {stage_id} report did not pass its hard gate")
        stage.state = "approved"
        stage.message = "Hash-bound canonical output passed deterministic adoption checks."
        self.store.event(run, "automatic_gate_passed", {"stage_id": stage_id, "metrics": stage.metrics})

    def _run_d5(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D5 requires the compiled asset specification")
        if run.spec.behavior == "rigid_articulated":
            self._begin(run, "D5", "Blender is separating approved rigid parts and authoring bounded pivots/actions.")
            stage = run.stage("D5")
            source = self._latest_evidence_path(
                run,
                "D3",
                media_type="application/x-blender",
                role="candidate_checkpoint",
            )
            plan_path = self._latest_evidence_path(run, "D4", role="rigid_structure_plan")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            attempt_root = self.store.run_root(run.run_id) / "D5_articulation" / f"iteration-{stage.iteration:02d}"
            artifact_id = f"{run.spec.asset_id}.d5.input.i{stage.iteration:02d}"
            artifact = self._artifact_for_path(run, source, artifact_id=artifact_id)
            request = ExternalWorkerRequest(
                job_id=f"studio.{run.run_id}.d5.i{stage.iteration:02d}",
                run_id=run.run_id,
                operation_id="blender.author_rigid_articulation",
                stage=AssetStage.rig,
                inputs=[artifact],
                input_paths={artifact_id: str(source)},
                parameters={
                    "render_size": 512,
                    "structure_plan": plan,
                    "animations": run.spec.animations,
                },
                output_directory=str(attempt_root / "blender"),
            )
            response = self._execute_worker("blender", request, timeout_seconds=1200)
            image_records: list[tuple[str, Path]] = []
            for output in response.outputs:
                path = Path(output.path)
                self.store.evidence(
                    run,
                    "D5",
                    path,
                    evidence_id=f"d5-i{stage.iteration:02d}-{output.role.replace('_', '-')}",
                    label=f"Rigid articulation {output.role.replace('_', ' ')}",
                    media_type=output.media_type,
                    metrics={"iteration": stage.iteration, "selectable": False, "role": output.role},
                )
                if output.media_type.startswith("image/"):
                    image_records.append((output.role, path))
            board = _image_board(image_records, attempt_root / "rigid_articulation_board.png", columns=4)
            self.store.evidence(
                run,
                "D5",
                board,
                evidence_id=f"d5-i{stage.iteration:02d}-articulation-board",
                label="Rigid neutral versus open articulation board",
                media_type="image/png",
                metrics={
                    **response.diagnostics,
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "rigid_articulation_board",
                },
            )
            stage.metrics = dict(response.diagnostics)
            stage.progress = 1
            stage.state = "approved"
            stage.finished_at = utc_now()
            stage.message = "Rigid parts, pivots, degree limits, and requested actions passed deterministic gates."
            self.store.event(run, "automatic_gate_passed", {"stage_id": "D5", "metrics": stage.metrics})
            return
        self._adopt_d4_output(
            run,
            "D5",
            ("rig_contract", "landmarks_contract", "rigged_candidate_checkpoint", "rigged_candidate"),
        )

    def _run_d6(self, run: StudioRun) -> None:
        self._adopt_d4_output(
            run,
            "D6",
            ("skinning_report", "deformation_report", "neutral_comparison_report", "rigged_export_validation"),
        )

    def _run_motion_chain(self, run: StudioRun, *, stop_after: str, timeout_seconds: int = 3600) -> Path:
        if run.spec is None:
            raise RuntimeError("the motion chain requires an asset specification")
        config = load_local_config()
        if config is None:
            raise RuntimeError("Darkness config.local.toml is required")
        binding = worker_binding(config, "blender")
        if binding is None or not binding.command_prefix:
            raise RuntimeError("the Blender worker binding is required")
        blender = Path(binding.command_prefix[0]).resolve()
        if not blender.is_file():
            raise RuntimeError(f"configured Blender executable does not exist: {blender}")
        target = self._latest_evidence_path(
            run,
            "D5",
            media_type="application/x-blender",
            role="rigged_candidate_checkpoint",
        )
        motion_source = (
            Path(config.workspace_root)
            / "sources"
            / "quaternius_universal_animation_library_standard"
            / "Universal Animation Library[Standard]"
            / "Unreal-Godot"
            / "UAL1_Standard.glb"
        ).resolve()
        if not motion_source.is_file():
            raise RuntimeError(f"qualified CC0 motion source is missing: {motion_source}")
        spec_path = self._latest_evidence_path(run, "D0", media_type="application/json")
        chain_root = self.store.run_root(run.run_id) / "D7_D10_chain"
        script = Path(__file__).resolve().parents[1] / "adapters" / "run_motion_candidate_pipeline.py"
        command = [
            sys.executable,
            str(script),
            "--target",
            str(target),
            "--motion-source",
            str(motion_source),
            "--character-spec",
            str(spec_path),
            "--output-root",
            str(chain_root),
            "--blender",
            str(blender),
            "--repo-root",
            str(Path(__file__).resolve().parents[1]),
            "--model",
            run.model,
            "--comfy-url",
            run.comfy_url,
            "--surface-checkpoint",
            run.checkpoint,
            "--timeout-seconds",
            "1200",
            "--stop-after",
            stop_after,
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"resumable D7-D10 chain failed at {stop_after}: "
                f"{(completed.stdout + completed.stderr)[-5000:]}"
            )
        return chain_root

    def _run_d7(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D7 requires the compiled asset specification")
        self._begin(run, "D7", "Building typed motion evidence and independent Qwen review.")
        stage = run.stage("D7")
        selected_concept = self._selected_evidence_path(run, "D1")
        if run.spec.behavior == "rigid_articulated":
            source = self._latest_evidence_path(run, "D5", role="rigid_articulation_board")
            item = self.store.evidence(
                run,
                "D7",
                source,
                evidence_id=f"d7-i{stage.iteration:02d}-rigid-motion-board",
                label="Rigid neutral versus actuated motion review",
                media_type="image/png",
                metrics={**run.stage("D5").metrics, "iteration": stage.iteration, "selectable": True},
            )
            qwen = self._qwen_factory(run)
            review = qwen.review_rigid_motion(
                run.spec,
                stage,
                selected_concept,
                source,
                dict(run.stage("D5").metrics),
            )
            review.recommended_evidence_id = item.evidence_id
            stage.qwen_reviews.append(review)
            stage.metrics = dict(run.stage("D5").metrics)
        else:
            self._progress(run, "D7", 0.08, "Retargeting qualified CC0 humanoid motions onto the Darkness rig.")
            chain = self._run_motion_chain(run, stop_after="retarget_qwen_review")
            evidence_root = chain / "retarget" / "human_review"
            board = evidence_root / "all_motion_front_keyposes.png"
            mediator_path = evidence_root / "qwen_retarget_mediator.json"
            report_path = chain / "retarget" / "retarget_validation.json"
            if not board.is_file() or not mediator_path.is_file() or not report_path.is_file():
                raise RuntimeError("motion chain did not publish the required D7 evidence packet")
            board_item = self.store.evidence(
                run,
                "D7",
                board,
                evidence_id=f"d7-i{stage.iteration:02d}-motion-board",
                label="Idle, walk, attack, and death key-pose board",
                media_type="image/png",
                metrics={"iteration": stage.iteration, "selectable": True, "role": "motion_board"},
            )
            for name in ("attack_front_keyposes.png", "walk_front_keyposes.png", "death_front_keyposes.png"):
                path = evidence_root / name
                if path.is_file():
                    self.store.evidence(
                        run,
                        "D7",
                        path,
                        evidence_id=f"d7-i{stage.iteration:02d}-{path.stem.replace('_', '-')}",
                        label=path.stem.replace("_", " ").title(),
                        media_type="image/png",
                        metrics={"iteration": stage.iteration, "selectable": False},
                    )
            mediator = json.loads(mediator_path.read_text(encoding="utf-8"))
            passed = mediator.get("corrected_overall") == "ready_for_human_gate"
            review = StudioQwenReview(
                review_id=f"d7.motion-mediator.iteration-{stage.iteration:02d}",
                stage_id="D7",
                iteration=stage.iteration,
                summary=str(mediator.get("reason", "Motion mediator completed.")),
                strengths=[str(item) for item in mediator.get("supported_claims", [])],
                issues=[str(item) for item in mediator.get("unsupported_or_overstated_claims", [])],
                candidate_ranking=[board_item.evidence_id],
                recommended_evidence_id=board_item.evidence_id,
                recommended_changes=[],
                confidence=0.8 if passed else 0.45,
                hard_requirements_satisfied=passed,
                request_human_review=True,
            )
            stage.qwen_reviews.append(review)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            stage.metrics = {
                "actions": len(report.get("actions", [])),
                "automatic_gate_passed": report.get("automatic_gate_passed", True),
                "qwen_mediator_passed": passed,
            }
        stage.progress = 1
        stage.state = "awaiting_review"
        stage.finished_at = utc_now()
        stage.message = "Motion evidence is ready for your approval or rejection."
        run.state = "awaiting_review"
        self.store.event(run, "human_gate_ready", {"stage_id": "D7", "iteration": stage.iteration})

    def _configured_blender(self) -> Path:
        config = load_local_config()
        if config is None:
            raise RuntimeError("Darkness config.local.toml is required")
        binding = worker_binding(config, "blender")
        if binding is None or not binding.command_prefix:
            raise RuntimeError("the Blender worker binding is required")
        executable = Path(binding.command_prefix[0]).resolve()
        if not executable.is_file():
            raise RuntimeError(f"configured Blender executable does not exist: {executable}")
        return executable

    def _run_d8(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D8 requires the compiled asset specification")
        self._begin(run, "D8", "Preparing semantic materials, painting canonical views, and baking one stable master.")
        stage = run.stage("D8")
        attempt_root = self.store.run_root(run.run_id) / "D8_surface" / f"iteration-{stage.iteration:02d}"
        if run.spec.behavior == "deformable_animated":
            master = self.store.run_root(run.run_id) / "D7_D10_chain" / "retarget" / "quaternius_retargeted_candidate.blend"
            surface = self.store.run_root(run.run_id) / "D7_D10_chain" / "surface"
        elif run.spec.behavior == "rigid_articulated":
            master = self._latest_evidence_path(
                run,
                "D5",
                media_type="application/x-blender",
                role="rigid_articulated_checkpoint",
            )
            surface = attempt_root / "production"
        else:
            master = self._latest_evidence_path(
                run,
                "D3",
                media_type="application/x-blender",
                role="candidate_checkpoint",
            )
            surface = attempt_root / "production"
        if not master.is_file():
            raise RuntimeError(f"D8 source master is missing: {master}")
        original_spec = self._latest_evidence_path(run, "D0", media_type="application/json")
        spec_value = json.loads(original_spec.read_text(encoding="utf-8"))
        latest_rejection = latest_correction(stage)
        if latest_rejection is not None:
            qwen = self._qwen_factory(run)
            revision = qwen.revision_plan(run.spec, stage)
            spec_value["creative_direction"] = (
                str(spec_value["creative_direction"])
                + " Human surface correction: "
                + latest_rejection.comment
                + " Qwen bounded changes: "
                + "; ".join(revision.changes)
            )
            spec_value["negative_constraints"] = list(spec_value.get("negative_constraints", [])) + [
                latest_rejection.comment
            ]
        effective_spec = attempt_root / "effective_asset_spec.json"
        _write_json(effective_spec, spec_value)
        blender = self._configured_blender()
        script = Path(__file__).resolve().parents[1] / "adapters" / "bake_darkness_surface.py"
        command = [
            sys.executable,
            str(script),
            "--master",
            str(master),
            "--asset-spec",
            str(effective_spec),
            "--output-directory",
            str(surface),
            "--blender",
            str(blender),
            "--repo-root",
            str(Path(__file__).resolve().parents[1]),
            "--comfy-url",
            run.comfy_url,
            "--checkpoint",
            run.checkpoint,
            "--timeout-seconds",
            "1200",
        ]
        if latest_rejection is not None:
            command.append("--force")
        completed = subprocess.run(command, text=True, capture_output=True, timeout=3600)
        if completed.returncode != 0:
            raise RuntimeError(f"D8 surface bake failed: {(completed.stdout + completed.stderr)[-5000:]}")
        review_script = Path(__file__).resolve().parents[1] / "adapters" / "review_surface_master.py"
        reviewed = subprocess.run(
            [sys.executable, str(review_script), "--surface-directory", str(surface), "--model", run.model],
            text=True,
            capture_output=True,
            timeout=300,
        )
        if reviewed.returncode != 0:
            raise RuntimeError(f"D8 surface review failed: {(reviewed.stdout + reviewed.stderr)[-3000:]}")
        board = surface / "surface_review.png"
        master_output = surface / "darkness_surface_master.blend"
        report_path = surface / "surface_validation.json"
        mediator_path = surface / "qwen_surface_mediator.json"
        for required in (board, master_output, report_path, mediator_path):
            if not required.is_file():
                raise RuntimeError(f"D8 did not publish required evidence: {required}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mediator = json.loads(mediator_path.read_text(encoding="utf-8"))
        board_item = self.store.evidence(
            run,
            "D8",
            board,
            evidence_id=f"d8-i{stage.iteration:02d}-surface-board",
            label="Before versus persistent painted surface master",
            media_type="image/png",
            metrics={
                **report.get("image_metrics", {}),
                "iteration": stage.iteration,
                "selectable": True,
                "automatic_gate_passed": report.get("automatic_gate_passed", False),
            },
        )
        for evidence_id, label, path, media_type, role in (
            (f"d8-i{stage.iteration:02d}-master", "Persistent painted Blender master", master_output, "application/x-blender", "surface_master"),
            (f"d8-i{stage.iteration:02d}-validation", "Surface numeric and provenance validation", report_path, "application/json", "surface_validation"),
            (f"d8-i{stage.iteration:02d}-mediator", "Qwen surface mediator", mediator_path, "application/json", "surface_mediator"),
        ):
            self.store.evidence(
                run,
                "D8",
                path,
                evidence_id=evidence_id,
                label=label,
                media_type=media_type,
                metrics={"iteration": stage.iteration, "selectable": False, "role": role},
            )
        passed = bool(report.get("automatic_gate_passed")) and mediator.get("corrected_overall") in {
            "ready_for_final_render",
            "uncertain",
        }
        review = StudioQwenReview(
            review_id=f"d8.surface-mediator.iteration-{stage.iteration:02d}",
            stage_id="D8",
            iteration=stage.iteration,
            summary=str(mediator.get("reason", "Surface evidence is ready for human review.")),
            strengths=[],
            issues=[str(item) for item in mediator.get("unsupported_or_overstated_claims", [])],
            candidate_ranking=[board_item.evidence_id],
            recommended_evidence_id=board_item.evidence_id,
            recommended_changes=[],
            confidence=0.8 if passed else 0.4,
            hard_requirements_satisfied=bool(report.get("automatic_gate_passed")),
            request_human_review=True,
        )
        stage.qwen_reviews.append(review)
        stage.metrics = {
            "automatic_gate_passed": report.get("automatic_gate_passed", False),
            "surface_master_sha256": report.get("surface_master_sha256", ""),
            "qwen_mediator": mediator.get("corrected_overall", "uncertain"),
        }
        stage.progress = 1
        stage.state = "awaiting_review"
        stage.finished_at = utc_now()
        stage.message = "Persistent painted surface evidence is ready for your approval or rejection."
        run.state = "awaiting_review"
        self.store.event(run, "human_gate_ready", {"stage_id": "D8", "iteration": stage.iteration})

    def _run_d9(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D9 requires the compiled asset specification")
        self._begin(run, "D9", "Rendering deterministic delivery views and validating the package.")
        stage = run.stage("D9")
        if run.spec.behavior == "deformable_animated":
            self._progress(run, "D9", 0.08, "Rendering and packaging directional motion sprites from one surface master.")
            chain = self._run_motion_chain(run, stop_after="sprite_qwen_review")
            package = chain / "sprites" / "package"
            board = package / "sprite_review.png"
            manifest = package / "candidate_unit_manifest.json"
            mediator_path = package / "qwen_sprite_mediator.json"
            for required in (board, manifest, mediator_path):
                if not required.is_file():
                    raise RuntimeError(f"D9 sprite chain did not publish {required}")
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            mediator = json.loads(mediator_path.read_text(encoding="utf-8"))
            passed = bool(manifest_value.get("automatic_gate_passed")) and mediator.get(
                "corrected_overall"
            ) == "ready_for_unity_candidate"
            self.store.evidence(
                run,
                "D9",
                board,
                evidence_id=f"d9-i{stage.iteration:02d}-sprite-board",
                label="Directional motion sprite package",
                media_type="image/png",
                metrics={
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "delivery_board",
                    "automatic_gate_passed": passed,
                },
            )
            for evidence_id, path, label, role in (
                (f"d9-i{stage.iteration:02d}-manifest", manifest, "Hash-bound sprite manifest", "delivery_manifest"),
                (f"d9-i{stage.iteration:02d}-mediator", mediator_path, "Qwen sprite mediator", "delivery_mediator"),
            ):
                self.store.evidence(
                    run,
                    "D9",
                    path,
                    evidence_id=evidence_id,
                    label=label,
                    media_type="application/json",
                    metrics={"iteration": stage.iteration, "selectable": False, "role": role},
                )
            stage.metrics = {
                "automatic_gate_passed": passed,
                "qwen_mediator": mediator.get("corrected_overall", "uncertain"),
                "source_master_sha256": manifest_value.get("source_master_sha256", ""),
            }
        else:
            source = self._latest_evidence_path(
                run,
                "D8",
                media_type="application/x-blender",
                role="surface_master",
            )
            attempt_root = self.store.run_root(run.run_id) / "D9_delivery" / f"iteration-{stage.iteration:02d}"
            artifact_id = f"{run.spec.asset_id}.d9.input.i{stage.iteration:02d}"
            artifact = self._artifact_for_path(run, source, artifact_id=artifact_id)
            request = ExternalWorkerRequest(
                job_id=f"studio.{run.run_id}.d9.i{stage.iteration:02d}",
                run_id=run.run_id,
                operation_id="blender.render_diagnostics",
                stage=AssetStage.optimization,
                inputs=[artifact],
                input_paths={artifact_id: str(source)},
                parameters={"render_size": 768},
                output_directory=str(attempt_root / "blender"),
            )
            response = self._execute_worker("blender", request, timeout_seconds=900)
            records = [
                (output.role, Path(output.path))
                for output in response.outputs
                if output.media_type.startswith("image/")
            ]
            board = _image_board(records, attempt_root / "delivery_board.png", columns=4)
            board_item = self.store.evidence(
                run,
                "D9",
                board,
                evidence_id=f"d9-i{stage.iteration:02d}-delivery-board",
                label="Painted multi-view delivery board",
                media_type="image/png",
                metrics={
                    **response.diagnostics,
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "delivery_board",
                },
            )
            manifest_path = attempt_root / "delivery_manifest.json"
            _write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "asset_id": run.spec.asset_id,
                    "asset_kind": run.spec.asset_kind,
                    "behavior": run.spec.behavior,
                    "source_master": str(source),
                    "source_master_sha256": sha256_file(source),
                    "delivery_board": str(board),
                    "delivery_board_sha256": board_item.sha256,
                    "automatic_gate_passed": bool(records),
                    "human_approval_required": True,
                    "human_approved": False,
                },
            )
            self.store.evidence(
                run,
                "D9",
                manifest_path,
                evidence_id=f"d9-i{stage.iteration:02d}-manifest",
                label="Hash-bound generic delivery manifest",
                media_type="application/json",
                metrics={"iteration": stage.iteration, "selectable": False, "role": "delivery_manifest"},
            )
            stage.metrics = {**response.diagnostics, "automatic_gate_passed": bool(records)}
        stage.progress = 1
        stage.finished_at = utc_now()
        if stage.metrics.get("automatic_gate_passed") is not True:
            raise RuntimeError("D9 delivery package failed its automatic gate")
        stage.state = "approved"
        stage.message = "Delivery renders, hashes, framing, and package contract passed automatic gates."
        self.store.event(run, "automatic_gate_passed", {"stage_id": "D9", "metrics": stage.metrics})

    def _run_d10(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D10 requires the compiled asset specification")
        self._begin(run, "D10", "Building a standalone runtime-review package and final human evidence.")
        stage = run.stage("D10")
        delivery_board = self._latest_evidence_path(run, "D9", role="delivery_board")
        runtime_link: Path | None = None
        if run.spec.behavior == "deformable_animated":
            chain = self._run_motion_chain(run, stop_after="unity_smoke_bundle")
            runtime_link = chain / "unity_smoke_bundle" / "review.html"
            runtime_manifest = chain / "unity_smoke_bundle" / "bundle_manifest.json"
            if not runtime_link.is_file() or not runtime_manifest.is_file():
                raise RuntimeError("D10 standalone Unity/browser bundle is incomplete")
        else:
            output = self.store.run_root(run.run_id) / "D10_runtime"
            output.mkdir(parents=True, exist_ok=True)
            runtime_manifest = output / "runtime_manifest.json"
            delivery_manifest = self._latest_evidence_path(run, "D9", role="delivery_manifest")
            _write_json(
                runtime_manifest,
                {
                    "schema_version": 1,
                    "asset_id": run.spec.asset_id,
                    "asset_kind": run.spec.asset_kind,
                    "behavior": run.spec.behavior,
                    "delivery_manifest": str(delivery_manifest),
                    "delivery_manifest_sha256": sha256_file(delivery_manifest),
                    "runtime_profile": "standalone_generic_asset_review_v1",
                    "automatic_gate_passed": True,
                    "human_approval_required": True,
                    "human_approved": False,
                },
            )
        final_item = self.store.evidence(
            run,
            "D10",
            delivery_board,
            evidence_id=f"d10-i{stage.iteration:02d}-final-board",
            label="Final runtime-scale asset evidence",
            media_type="image/png",
            metrics={
                "iteration": stage.iteration,
                "selectable": True,
                "role": "final_runtime_board",
            },
        )
        self.store.evidence(
            run,
            "D10",
            runtime_manifest,
            evidence_id=f"d10-i{stage.iteration:02d}-runtime-manifest",
            label="Standalone runtime package manifest",
            media_type="application/json",
            metrics={"iteration": stage.iteration, "selectable": False, "role": "runtime_manifest"},
        )
        if runtime_link is not None:
            self.store.evidence(
                run,
                "D10",
                runtime_link,
                evidence_id=f"d10-i{stage.iteration:02d}-browser-review",
                label="Interactive standalone motion review",
                media_type="text/html",
                metrics={"iteration": stage.iteration, "selectable": False, "role": "browser_review"},
            )
        review = StudioQwenReview(
            review_id=f"d10.package.iteration-{stage.iteration:02d}",
            stage_id="D10",
            iteration=stage.iteration,
            summary=(
                "The hash-bound standalone package passed automatic assembly. Human ship approval remains required "
                "for identity, motion/readability when applicable, surface quality, and runtime-scale appearance."
            ),
            candidate_ranking=[final_item.evidence_id],
            recommended_evidence_id=final_item.evidence_id,
            confidence=0.75,
            hard_requirements_satisfied=True,
            request_human_review=True,
        )
        stage.qwen_reviews.append(review)
        stage.metrics = {"automatic_gate_passed": True, "runtime_manifest_sha256": sha256_file(runtime_manifest)}
        stage.progress = 1
        stage.state = "awaiting_review"
        stage.finished_at = utc_now()
        stage.message = "Final standalone package is ready for explicit human ship approval."
        run.state = "awaiting_review"
        self.store.event(run, "human_gate_ready", {"stage_id": "D10", "iteration": stage.iteration})
