import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.critic import critique_unit


def textured_frame(size, base, accent):
    """A frame with structure: body block, head, weapon line — readable at 96px."""
    frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    quarter = size // 4
    draw.rectangle((quarter, quarter, 3 * quarter, size - 8), fill=(*base, 255))
    draw.ellipse((quarter + 12, 8, 3 * quarter - 12, quarter + 12), fill=(*accent, 255))
    for x in range(quarter, 3 * quarter, 6):
        draw.line((x, quarter, x, size - 8), fill=(*accent, 255), width=1)
    draw.line((8, quarter, quarter, size // 2), fill=(200, 200, 210, 255), width=4)
    return frame


def write_sheet(path, frames):
    size = frames[0].width
    sheet = Image.new("RGBA", (size * len(frames), size), (0, 0, 0, 0))
    for index, frame in enumerate(frames):
        sheet.paste(frame, (index * size, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "PNG")


def make_config(folder, animations=("idle",), directions=("south", "east")):
    config = {
        "id": "unit",
        "directions": list(directions),
        "animations": {name: {"frames": 3} for name in animations},
        "output_path_pattern": "sheets/unit_{animation}_{direction}.png",
    }
    config_path = Path(folder) / "unit.character.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, config


class CriticTests(unittest.TestCase):
    def test_consistent_dark_unit_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, config = make_config(folder)
            frames = [textured_frame(128, (52, 56, 70), (110, 90, 60)) for _ in range(3)]
            for animation in config["animations"]:
                for direction in config["directions"]:
                    write_sheet(Path(folder) / f"sheets/unit_{animation}_{direction}.png", frames)
            report = critique_unit(config_path, Path(folder), Path(folder) / "report.json")
            self.assertTrue(report["passed"], report["failures"])

    def test_pastel_palette_fails_tone(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, config = make_config(folder, directions=("south",))
            frames = [textured_frame(128, (235, 230, 240), (240, 235, 225)) for _ in range(3)]
            write_sheet(Path(folder) / "sheets/unit_idle_south.png", frames)
            report = critique_unit(config_path, Path(folder), Path(folder) / "report.json")
            self.assertFalse(report["passed"])
            self.assertTrue(any(f["check"] == "tone-value" for f in report["failures"]))

    def test_identity_flicker_fails_coherence(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, config = make_config(folder, directions=("south",))
            frames = [
                textured_frame(128, (140, 30, 30), (150, 40, 40)),
                textured_frame(128, (30, 30, 140), (40, 40, 150)),
                textured_frame(128, (30, 140, 30), (40, 150, 40)),
            ]
            write_sheet(Path(folder) / "sheets/unit_idle_south.png", frames)
            report = critique_unit(config_path, Path(folder), Path(folder) / "report.json")
            self.assertFalse(report["passed"])
            self.assertTrue(any(f["check"] == "frame-coherence" for f in report["failures"]))

    def test_featureless_blob_fails_readability(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, config = make_config(folder, directions=("south",))
            blob = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            ImageDraw.Draw(blob).ellipse((20, 20, 108, 108), fill=(60, 60, 66, 255))
            write_sheet(Path(folder) / "sheets/unit_idle_south.png", [blob] * 3)
            report = critique_unit(config_path, Path(folder), Path(folder) / "report.json")
            self.assertFalse(report["passed"])
            self.assertTrue(any(f["check"] == "readability" for f in report["failures"]))

    def test_missing_sheet_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, _ = make_config(folder, directions=("south",))
            report = critique_unit(config_path, Path(folder), Path(folder) / "report.json")
            self.assertFalse(report["passed"])
            self.assertTrue(any(f["check"] == "exists" for f in report["failures"]))

    def test_report_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, _ = make_config(folder, directions=("south",))
            report_path = Path(folder) / "report.json"
            critique_unit(config_path, Path(folder), report_path)
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual("critic", saved["stage"])
            self.assertIn("thresholds", saved)


if __name__ == "__main__":
    unittest.main()
