import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from text2model_forge.sprites.animate import build_animation_prompt, build_frame_workflow, generate_animation, sampler_settings
from text2model_forge.sprites.core import create_asset, init_workspace, read_json
from text2model_forge.sprites.pose import default_pack, frame_pose, load_pack, materialize_pack, render_action_frames, render_pose
from text2model_forge.sprites.sheets import pack_sheets


class PosePackTests(unittest.TestCase):
    def test_default_pack_covers_required_actions(self):
        pack = default_pack()
        for action in ("idle", "walk", "attack", "shoot", "defend", "cast", "hit", "death"):
            self.assertIn(action, pack["actions"])
            frames = render_action_frames(pack, action)
            self.assertEqual(int(pack["actions"][action]["frames"]), len(frames))

    def test_rendering_is_deterministic(self):
        pack = default_pack()
        first = render_action_frames(pack, "walk")
        second = render_action_frames(pack, "walk")
        for a, b in zip(first, second):
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_tween_moves_between_keyframes(self):
        pack = default_pack()
        action = pack["actions"]["attack"]
        start = frame_pose(action, 0)
        windup = frame_pose(action, 2)
        self.assertLess(windup["r_wrist"][0], start["r_wrist"][0])
        self.assertLess(windup["r_wrist"][1], start["r_wrist"][1])

    def test_walk_cycle_actually_moves(self):
        pack = default_pack()
        frames = render_action_frames(pack, "walk")
        self.assertNotEqual(frames[0].tobytes(), frames[3].tobytes())

    def test_materialized_pack_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "forge"
            init_workspace(workspace, "test_game", "Test Game")
            path = materialize_pack(workspace)
            self.assertTrue(path.is_file())
            loaded = load_pack(workspace)
            self.assertEqual(default_pack()["actions"].keys(), loaded["actions"].keys())
            direct = render_pose(frame_pose(default_pack()["actions"]["idle"], 1), (768, 768))
            reloaded = render_pose(frame_pose(loaded["actions"]["idle"], 1), (768, 768))
            self.assertEqual(direct.tobytes(), reloaded.tobytes())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "forge"
        init_workspace(self.workspace, "test_game", "Test Game")
        create_asset(self.workspace, "iron_guard", "Iron Guard", "character", "shield tank", "defender")

    def tearDown(self):
        self.temporary.cleanup()

    def test_registry_sampler_settings_override_profile(self):
        settings = sampler_settings(self.workspace, "dreamshaper_xl_v2_turbo.safetensors")
        self.assertEqual(7, settings["steps"])
        self.assertEqual(2.5, settings["cfg"])
        base = sampler_settings(self.workspace, "sd_xl_base_1.0.safetensors")
        self.assertEqual(28, base["steps"])

    def test_animation_prompt_reads_brief_and_action(self):
        positive, negative = build_animation_prompt(self.workspace, "iron_guard", "attack")
        self.assertIn("forward strike", positive)
        self.assertIn("facing right", positive)
        self.assertIn("multiple characters", negative)

    def test_frame_workflow_wires_controlnet_between_prompts_and_sampler(self):
        settings = {"steps": 7, "cfg": 2.5, "sampler": "dpmpp_sde", "scheduler": "karras"}
        workflow = build_frame_workflow(
            "pos", "neg", "ckpt.safetensors", "cn.safetensors",
            "src/text2model_forge/sprites/ref.png", "src/text2model_forge/sprites/pose.png", 42, settings, 0.6, 0.85, "prefix",
        )
        self.assertEqual("ControlNetApplyAdvanced", workflow["8"]["class_type"])
        self.assertEqual(["8", 0], workflow["9"]["inputs"]["positive"])
        self.assertEqual(["8", 1], workflow["9"]["inputs"]["negative"])
        self.assertEqual(["5", 0], workflow["9"]["inputs"]["latent_image"])
        self.assertEqual(0.6, workflow["9"]["inputs"]["denoise"])


class FakeComfyHandler(BaseHTTPRequestHandler):
    def _png(self):
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), (90, 40, 20)).save(stream, "PNG")
        return stream.getvalue()

    def do_GET(self):
        if self.path == "/models/checkpoints":
            self.respond(["dreamshaper_xl_v2_turbo.safetensors"])
        elif self.path.startswith("/history/"):
            self.respond({
                self.path.rsplit("/", 1)[1]: {
                    "status": {"completed": True},
                    "outputs": {"11": {"images": [{"filename": "frame.png", "subfolder": "", "type": "output"}]}},
                }
            })
        elif self.path.startswith("/view?"):
            payload = self._png()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/upload/image":
            self.respond({"name": "uploaded.png", "subfolder": "text2model-forge/sprites", "type": "input"})
        elif self.path == "/prompt":
            value = json.loads(body)
            assert "prompt" in value
            self.respond({"prompt_id": "job-1", "number": 1, "node_errors": {}})
        else:
            self.send_error(404)

    def respond(self, value):
        payload = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def fake_cutout(pairs):
    for source, destination in pairs:
        with Image.open(source) as image:
            image.convert("RGBA").save(destination, "PNG")


class AnimateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "forge"
        init_workspace(self.workspace, "test_game", "Test Game")
        create_asset(self.workspace, "iron_guard", "Iron Guard", "character", "shield tank", "defender")
        self.reference = self.workspace / "reference.png"
        Image.new("RGBA", (64, 96), (60, 70, 90, 255)).save(self.reference, "PNG")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_animate_writes_frames_and_provenance(self):
        result = generate_animation(
            self.workspace,
            "iron_guard",
            ["hit"],
            reference=self.reference,
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            timeout_seconds=5,
            cutout=fake_cutout,
        )
        folder = Path(result["actions"]["hit"]["folder"])
        frames = sorted(folder.glob("*.png"))
        self.assertEqual(2, len(frames))
        jobs = list((self.workspace / "assets" / "iron_guard" / "jobs").glob("*.json"))
        self.assertTrue(jobs)
        job = read_json(jobs[-1])
        self.assertEqual("pose-controlled-img2img", job["workflow_type"])
        self.assertEqual(job["pose_pack"]["id"], "humanoid_side_east")
        self.assertTrue(all("final_sha256" in frame for frame in job["frames"]))


class PackSheetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "forge"
        init_workspace(self.workspace, "test_game", "Test Game")
        create_asset(self.workspace, "iron_guard", "Iron Guard", "character", "shield tank", "defender")

    def tearDown(self):
        self.temporary.cleanup()

    def _write_frames(self, action, count, body_height=420):
        folder = self.workspace / "assets" / "iron_guard" / "renders" / "frames" / "iron_guard" / action / "east"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            frame = Image.new("RGBA", (768, 768), (0, 0, 0, 0))
            body = Image.new("RGBA", (180 + index * 6, body_height), (120, 60, 40, 255))
            # Feet on the authored ankle baseline (0.86 * 768), like real generated frames.
            frame.alpha_composite(body, (280, 660 - body_height))
            frame.save(folder / f"{index:02d}.png", "PNG")

    def test_pack_produces_baseline_locked_sheets_and_preview(self):
        self._write_frames("idle", 4)
        self._write_frames("walk", 6)
        report = pack_sheets(self.workspace, "iron_guard")
        self.assertTrue(report["passed"], report["failures"])
        self.assertTrue(Path(report["gameplay_preview"]).is_file())
        manifest = read_json(Path(report["unit_manifest"]))
        self.assertEqual({"idle", "walk"}, {action["name"] for action in manifest["actions"]})
        idle = next(action for action in manifest["actions"] if action["name"] == "idle")
        self.assertTrue(idle["loop"])
        self.assertEqual(4, idle["frames"])
        self.assertEqual(256, manifest["pixels_per_unit"])
        sheet_path = self.workspace / "assets" / "iron_guard" / "sheets" / "iron_guard_idle_east_sheet.png"
        with Image.open(sheet_path) as sheet:
            self.assertEqual((256 * 4, 256), sheet.size)
            bottoms = []
            for index in range(4):
                cell = sheet.crop((index * 256, 0, (index + 1) * 256, 256))
                bottoms.append(cell.getchannel("A").getbbox()[3])
        self.assertLessEqual(max(bottoms) - min(bottoms), 1)

    def test_uniform_scale_keeps_relative_sizes(self):
        self._write_frames("idle", 2, body_height=420)
        self._write_frames("hit", 2, body_height=300)
        pack_sheets(self.workspace, "iron_guard")
        idle = Image.open(self.workspace / "assets" / "iron_guard" / "sheets" / "iron_guard_idle_east_sheet.png")
        hit = Image.open(self.workspace / "assets" / "iron_guard" / "sheets" / "iron_guard_hit_east_sheet.png")
        idle_height = idle.crop((0, 0, 256, 256)).getchannel("A").getbbox()
        hit_height = hit.crop((0, 0, 256, 256)).getchannel("A").getbbox()
        idle.close()
        hit.close()
        ratio = (hit_height[3] - hit_height[1]) / (idle_height[3] - idle_height[1])
        self.assertAlmostEqual(300 / 420, ratio, delta=0.03)


if __name__ == "__main__":
    unittest.main()
