import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from assetforge.comfy import ComfyClient, build_sdxl_workflow
from assetforge.core import (
    ForgeError,
    approval_status,
    approve_artifact,
    build_prompt_pack,
    create_asset,
    init_workspace,
    project_status,
    read_json,
    register_model,
    sha256_file,
)
from assetforge.compliance import build_compliance_report


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "forge"
        init_workspace(self.workspace, "test_game", "Test Game", engine="unity", engine_root="unity/Test")

    def tearDown(self):
        self.temporary.cleanup()

    def test_project_creates_portable_profiles(self):
        project = read_json(self.workspace / "forge-project.json")
        self.assertEqual("test_game", project["project_id"])
        self.assertTrue((self.workspace / "style" / "style-profile.json").is_file())
        self.assertTrue((self.workspace / "models" / "model-registry.json").is_file())
        self.assertTrue((self.workspace / "hardware" / "rtx3080-laptop-8gb.profile.json").is_file())

    def test_asset_prompt_and_workflow_are_deterministic(self):
        create_asset(self.workspace, "iron_guard", "Iron Guard", "character", "readable shield tank", "defender")
        asset_path = self.workspace / "assets" / "iron_guard" / "asset.json"
        asset = read_json(asset_path)
        asset["brief"]["equipment"] = ["broad iron shield", "short sword"]
        asset_path.write_text(json.dumps(asset, indent=2), encoding="utf-8")
        prompt = build_prompt_pack(self.workspace, "iron_guard")
        self.assertIn("broad iron shield", prompt["prompts"]["turnaround"])
        self.assertIn("copyrighted character", prompt["negative"])
        first, provenance_a = build_sdxl_workflow(
            self.workspace, "iron_guard", "turnaround", "sd_xl_base_1.0.safetensors", 1234
        )
        second, provenance_b = build_sdxl_workflow(
            self.workspace, "iron_guard", "turnaround", "sd_xl_base_1.0.safetensors", 1234
        )
        self.assertEqual(first, second)
        self.assertEqual(provenance_a["workflow_sha256"], provenance_b["workflow_sha256"])
        self.assertEqual(1216, first["4"]["inputs"]["width"])
        self.assertEqual(832, first["4"]["inputs"]["height"])

    def test_approval_becomes_stale_when_artifact_changes(self):
        create_asset(self.workspace, "stone_gate", "Stone Gate", "building", "protect the defenders", "defender", "none")
        artifact = self.workspace / "assets" / "stone_gate" / "master" / "gate.blend"
        artifact.write_bytes(b"original-master")
        approve_artifact(self.workspace, "stone_gate", "master", artifact, "tester", "first proof")
        self.assertTrue(approval_status(self.workspace, "stone_gate")["master"]["approved"])
        artifact.write_bytes(b"changed-master")
        status = approval_status(self.workspace, "stone_gate")["master"]
        self.assertFalse(status["approved"])
        self.assertEqual("artifact-changed", status["reason"])

    def test_status_derives_next_gate(self):
        create_asset(self.workspace, "goblin_scout", "Goblin Scout", "creature", "fast enemy", "enemy")
        status = project_status(self.workspace)
        self.assertEqual("brief", status["assets"][0]["next_stage"])

    def test_rejects_unsafe_identifier(self):
        with self.assertRaises(ForgeError):
            create_asset(self.workspace, "../escape", "Bad", "prop", "bad")

    def test_model_registration_and_manual_original_compliance(self):
        model = self.workspace / "models" / "fixture.safetensors"
        model.write_bytes(b"fixture-model")
        registered = register_model(self.workspace, "sdxl-base-1.0", model, "tester")
        self.assertEqual(sha256_file(model), registered["installed_sha256"])
        create_asset(self.workspace, "manual_icon", "Manual Icon", "icon", "ember currency", primary_action="none")
        manifest = self.workspace / "assets" / "manual_icon" / "asset.json"
        asset = read_json(manifest)
        asset["license"]["status"] = "original-confirmed"
        manifest.write_text(json.dumps(asset, indent=2), encoding="utf-8")
        report = build_compliance_report(self.workspace, "manual_icon", self.workspace / "manual-icon-compliance.json")
        self.assertTrue(report["passed"], report["failures"])


class FakeComfyHandler(BaseHTTPRequestHandler):
    png = b"\x89PNG\r\n\x1a\nfixture"

    def do_GET(self):
        if self.path == "/system_stats":
            self.respond({"devices": [{"name": "Fake RTX", "vram_total": 8 * 1024**3}]})
        elif self.path == "/models/checkpoints":
            self.respond(["sd_xl_base_1.0.safetensors"])
        elif self.path == "/history/prompt-1":
            self.respond({"prompt-1": {"status": {"completed": True}, "outputs": {"7": {"images": [{"filename": "candidate.png", "subfolder": "", "type": "output"}]}}}})
        elif self.path.startswith("/view?"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.png)))
            self.end_headers()
            self.wfile.write(self.png)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/prompt":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length))
        if "prompt" not in value:
            self.send_error(400)
            return
        self.respond({"prompt_id": "prompt-1", "number": 1, "node_errors": {}})

    def respond(self, value):
        payload = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


class ComfyClientTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = ComfyClient(f"http://127.0.0.1:{self.server.server_port}")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_local_api_submit_poll_and_download(self):
        self.assertEqual(["sd_xl_base_1.0.safetensors"], self.client.checkpoints())
        prompt_id = self.client.submit({"1": {"class_type": "Fixture", "inputs": {}}})
        history = self.client.wait(prompt_id, timeout_seconds=2, poll_seconds=0.01)
        with tempfile.TemporaryDirectory() as temporary:
            outputs = self.client.download_outputs(history, Path(temporary))
            self.assertEqual(1, len(outputs))
            self.assertEqual(FakeComfyHandler.png, outputs[0].read_bytes())
            self.assertEqual(64, len(sha256_file(outputs[0])))


if __name__ == "__main__":
    unittest.main()
