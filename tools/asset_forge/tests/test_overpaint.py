import tempfile
import unittest
from pathlib import Path

from PIL import Image

from assetforge.overpaint import build_overpaint_workflow, equipment_integrity


class WorkflowTests(unittest.TestCase):
    def test_plain_workflow_has_no_controlnet(self):
        workflow = build_overpaint_workflow("ckpt", "pos", "neg", "img.png", 7, 8, 3.0, 0.42, "p")
        self.assertNotIn("11", workflow)
        self.assertEqual(["2", 0], workflow["6"]["inputs"]["positive"])

    def test_depth_control_wires_between_prompts_and_sampler(self):
        workflow = build_overpaint_workflow(
            "ckpt", "pos", "neg", "img.png", 7, 8, 3.0, 0.42, "p",
            control_image="depth.png", control_strength=0.65,
        )
        self.assertEqual("ControlNetApplyAdvanced", workflow["11"]["class_type"])
        self.assertEqual(["11", 0], workflow["6"]["inputs"]["positive"])
        self.assertEqual(["11", 1], workflow["6"]["inputs"]["negative"])
        self.assertEqual(0.65, workflow["11"]["inputs"]["strength"])

    def test_workflow_is_deterministic(self):
        first = build_overpaint_workflow("ckpt", "pos", "neg", "img.png", 7, 8, 3.0, 0.42, "p")
        second = build_overpaint_workflow("ckpt", "pos", "neg", "img.png", 7, 8, 3.0, 0.42, "p")
        self.assertEqual(first, second)


class EquipmentIntegrityTests(unittest.TestCase):
    def _mask(self, folder, box):
        mask = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        blade = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (255, 255, 255, 255))
        mask.alpha_composite(blade, (box[0], box[1]))
        path = Path(folder) / "mask.png"
        mask.save(path, "PNG")
        return path

    def _painted(self, body_color, blade_color, box):
        painted = Image.new("RGB", (256, 256), (20, 22, 26))
        body = Image.new("RGB", (120, 200), body_color)
        painted.paste(body, (68, 40))
        blade = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), blade_color)
        painted.paste(blade, (box[0], box[1]))
        alpha = Image.new("L", (256, 256), 0)
        alpha.paste(255, (60, 32, 196, 248))
        return painted, alpha

    def test_distinct_equipment_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            box = (120, 60, 140, 180)
            mask_path = self._mask(folder, box)
            painted, alpha = self._painted((60, 70, 90), (200, 210, 220), box)
            ok, distance = equipment_integrity(painted, alpha, mask_path)
            self.assertTrue(ok)
            self.assertGreater(distance, 16.0)

    def test_painted_over_equipment_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            box = (120, 60, 140, 180)
            mask_path = self._mask(folder, box)
            painted, alpha = self._painted((60, 70, 90), (62, 71, 92), box)
            ok, distance = equipment_integrity(painted, alpha, mask_path)
            self.assertFalse(ok)
            self.assertLess(distance, 16.0)

    def test_missing_mask_passes_vacuously(self):
        painted, alpha = self._painted((60, 70, 90), (60, 70, 90), (120, 60, 140, 180))
        ok, distance = equipment_integrity(painted, alpha, Path("does/not/exist.png"))
        self.assertTrue(ok)
        self.assertEqual(-1.0, distance)


if __name__ == "__main__":
    unittest.main()
