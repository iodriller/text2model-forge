import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image

from assetforge.overpaint import (
    build_overpaint_workflow,
    equipment_integrity,
    protected_part_metrics,
    repair_protected_frames,
    restore_protected_parts,
)


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

    def test_missing_mask_fails_closed(self):
        painted, alpha = self._painted((60, 70, 90), (60, 70, 90), (120, 60, 140, 180))
        ok, distance = equipment_integrity(painted, alpha, Path("does/not/exist.png"))
        self.assertFalse(ok)
        self.assertEqual(-1.0, distance)

    def test_protected_equipment_is_restored_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            box = (120, 60, 140, 180)
            mask_path = self._mask(folder, box)
            original, _ = self._painted((60, 70, 90), (210, 215, 225), box)
            painted, _ = self._painted((75, 85, 105), (62, 71, 92), box)
            restored, coverage, error = restore_protected_parts(painted, original, mask_path)
            measured_coverage, measured_error = protected_part_metrics(restored, original, mask_path)
            self.assertGreater(coverage, 30)
            self.assertEqual(coverage, measured_coverage)
            self.assertEqual(0.0, error)
            self.assertEqual(0.0, measured_error)

    def test_visibility_mask_uses_white_id_pixels_not_opaque_black_body(self):
        with tempfile.TemporaryDirectory() as folder:
            box = (120, 60, 140, 180)
            mask = Image.new("RGBA", (256, 256), (0, 0, 0, 255))
            mask.paste((255, 255, 255, 255), box)
            mask_path = Path(folder) / "visible-mask.png"
            mask.save(mask_path)
            original, _ = self._painted((40, 45, 50), (220, 225, 230), box)
            painted, _ = self._painted((80, 90, 110), (62, 71, 92), box)
            restored, coverage, _ = restore_protected_parts(painted, original, mask_path)
            self.assertEqual((box[2] - box[0]) * (box[3] - box[1]), coverage)
            self.assertEqual(painted.getpixel((80, 80)), restored.getpixel((80, 80)))
            self.assertEqual(original.getpixel((125, 80)), restored.getpixel((125, 80)))

    def test_protected_equipment_requires_a_mask(self):
        painted, _ = self._painted((60, 70, 90), (60, 70, 90), (120, 60, 140, 180))
        with self.assertRaisesRegex(Exception, "Required equipment mask is missing"):
            restore_protected_parts(painted, painted, Path("does/not/exist.png"))

    def test_historical_overpaint_repair_is_transactional_and_review_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / "unit.character.json"
            config_path.write_text(json.dumps({"id": "unit", "equipment_objects": ["Sword*"]}), encoding="utf-8")
            relative = Path("attack") / "south" / "00.png"
            raw, _ = self._painted((60, 70, 90), (210, 215, 225), (120, 60, 140, 180))
            painted, _ = self._painted((75, 85, 105), (62, 71, 92), (120, 60, 140, 180))
            raw = raw.convert("RGBA")
            raw.putalpha(255)
            painted = painted.convert("RGBA")
            painted.putalpha(255)
            for unit_folder, image in (("unit", painted), ("unit-raw", raw)):
                path = root / unit_folder / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
            mask_path = root / "unit-equip" / relative
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            self._mask(mask_path.parent, (120, 60, 140, 180)).replace(mask_path)

            report = repair_protected_frames(config_path, root)
            self.assertTrue(report["passed"])
            self.assertFalse(report["production_eligible"])
            with Image.open(root / "unit" / relative) as repaired:
                _, error = protected_part_metrics(repaired, raw, mask_path)
            self.assertEqual(0.0, error)


if __name__ == "__main__":
    unittest.main()
