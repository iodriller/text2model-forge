import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from text2model_forge.sprites.deformation import critique_deformation


def _frame(size=64):
    frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    # one connected "creature": head circle overlapping a body rectangle.
    draw.rectangle((24, 30, 40, 58), fill=(90, 120, 70, 255))
    draw.ellipse((22, 16, 42, 36), fill=(90, 120, 70, 255))
    return frame


def _disconnected_frame(size=64):
    frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    # head floats well away from the body: two separate blobs, no overlap.
    draw.rectangle((24, 40, 40, 58), fill=(90, 120, 70, 255))
    draw.ellipse((22, 2, 42, 16), fill=(90, 120, 70, 255))
    return frame


def _write(frames_root, unit, action, direction, frames):
    folder = Path(frames_root) / unit / action / direction
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(folder / f"{index:02d}.png", "PNG")


def _config(folder, animations, directions=("south", "east"), extra=None):
    payload = {
        "id": "unit",
        "directions": list(directions),
        "cell_size": [256, 256],
        "animations": animations,
        "silhouette_contract": {"proof_views": list(directions)},
    }
    if extra:
        payload.update(extra)
    config_path = Path(folder) / "unit.character.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


class DeformationTests(unittest.TestCase):
    def test_connected_stable_action_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            animations = {"idle": {"frames": 4}}
            config_path = _config(folder, animations)
            frames_root = Path(folder) / "frames"
            for direction in ("south", "east"):
                _write(frames_root, "unit", "idle", direction, [_frame() for _ in range(4)])
            report = critique_deformation(config_path, Path(folder), frames_root, Path(folder) / "report.json")
            self.assertTrue(report["passed"], report["failures"])

    def test_disconnected_limbs_fail_connectivity(self):
        with tempfile.TemporaryDirectory() as folder:
            animations = {"walk": {"frames": 3}}
            config_path = _config(folder, animations, directions=("south",))
            frames_root = Path(folder) / "frames"
            _write(frames_root, "unit", "walk", "south", [_disconnected_frame() for _ in range(3)])
            report = critique_deformation(config_path, Path(folder), frames_root, Path(folder) / "report.json")
            self.assertFalse(report["passed"])
            self.assertTrue(any(f["check"] == "connectivity" for f in report["failures"]))

    def test_ballooning_area_fails_stability(self):
        with tempfile.TemporaryDirectory() as folder:
            animations = {"idle": {"frames": 3}}
            config_path = _config(folder, animations, directions=("south",))
            frames_root = Path(folder) / "frames"
            small = _frame()
            huge = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            ImageDraw.Draw(huge).rectangle((2, 2, 62, 62), fill=(90, 120, 70, 255))
            _write(frames_root, "unit", "idle", "south", [small, huge, small])
            report = critique_deformation(config_path, Path(folder), frames_root, Path(folder) / "report.json")
            self.assertFalse(report["passed"])
            self.assertTrue(any(f["check"] == "stability" for f in report["failures"]))

    def test_feet_drift_fails_grounding_for_walk_but_not_dodge(self):
        with tempfile.TemporaryDirectory() as folder:
            animations = {
                "walk": {"frames": 2, "baseline_tolerance": 1},
                "dodge": {"frames": 2, "baseline_tolerance": 1},
            }
            config_path = _config(folder, animations, directions=("south",))
            frames_root = Path(folder) / "frames"
            grounded = _frame()
            shifted = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            draw = ImageDraw.Draw(shifted)
            draw.rectangle((24, 10, 40, 38), fill=(90, 120, 70, 255))
            draw.ellipse((22, -4, 42, 16), fill=(90, 120, 70, 255))
            _write(frames_root, "unit", "walk", "south", [grounded, shifted])
            _write(frames_root, "unit", "dodge", "south", [grounded, shifted])
            report = critique_deformation(config_path, Path(folder), frames_root, Path(folder) / "report.json")
            self.assertFalse(report["passed"])
            walk_failures = [f for f in report["failures"] if f["animation"] == "walk" and f["check"] == "grounding"]
            dodge_failures = [f for f in report["failures"] if f["animation"] == "dodge" and f["check"] == "grounding"]
            self.assertTrue(walk_failures)
            self.assertFalse(dodge_failures)

    def test_missing_frame_fails_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            animations = {"idle": {"frames": 2}}
            config_path = _config(folder, animations, directions=("south",))
            frames_root = Path(folder) / "frames"
            report = critique_deformation(config_path, Path(folder), frames_root, Path(folder) / "report.json")
            self.assertFalse(report["passed"])
            self.assertTrue(any(f["check"] == "exists" for f in report["failures"]))

    def test_report_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            animations = {"idle": {"frames": 2}}
            config_path = _config(folder, animations, directions=("south",))
            frames_root = Path(folder) / "frames"
            _write(frames_root, "unit", "idle", "south", [_frame(), _frame()])
            report_path = Path(folder) / "report.json"
            critique_deformation(config_path, Path(folder), frames_root, report_path)
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual("deformation", saved["stage"])
            self.assertIn("thresholds", saved)


if __name__ == "__main__":
    unittest.main()
