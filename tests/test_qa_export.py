import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from text2model_forge.sprites.core import approve_artifact, create_asset, init_workspace
from text2model_forge.sprites.exporter import export_asset
from text2model_forge.sprites.qa import validate_sheets


class QaAndExportTests(unittest.TestCase):
    def test_sheet_qa_and_approved_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sheets = root / "rendered"
            sheets.mkdir()
            directions = ["north", "south", "east", "west"]
            colors = [(70, 100, 140, 255), (130, 85, 55, 255), (60, 130, 80, 255), (140, 60, 70, 255)]
            for direction_index, direction in enumerate(directions):
                image = Image.new("RGBA", (128, 64), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                for frame in range(2):
                    x = frame * 64 + 19 + frame
                    draw.rectangle((x, 18, x + 23, 59), fill=colors[direction_index])
                    draw.point((x + direction_index, 22 + frame), fill=(255, 230, 170, 255))
                image.save(sheets / f"fixture_idle_{direction}.png")
            config = {
                "id": "fixture",
                "cell_size": [64, 64],
                "directions": directions,
                "animations": {"idle": {"actions": ["idle"], "frames": 2, "loop": True}},
                "output_path_pattern": str(sheets / "fixture_{animation}_{direction}.png"),
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            report_path = root / "qa.json"
            contact_path = root / "contact.png"
            report = validate_sheets(config_path, root, report_path, contact_path)
            self.assertTrue(report["passed"], report["failures"])
            self.assertTrue(contact_path.is_file())

            workspace = root / "workspace"
            init_workspace(workspace, "test_export", "Test Export")
            create_asset(workspace, "fixture", "Fixture", "character", "test role")
            asset_sheets = workspace / "assets" / "fixture" / "sheets"
            for source in sheets.glob("*.png"):
                (asset_sheets / source.name).write_bytes(source.read_bytes())
            license_record = workspace / "assets" / "fixture" / "reports" / "license.json"
            license_record.write_text('{"passed":true}', encoding="utf-8")
            approve_artifact(workspace, "fixture", "sheets", report_path, "tester")
            approve_artifact(workspace, "fixture", "license", license_record, "tester")
            exported = export_asset(workspace, "fixture", root / "engine-assets", adapter="fixture")
            self.assertEqual(4, len(exported["files"]))
            self.assertTrue((root / "engine-assets" / "text2model-sprites-export.json").is_file())


if __name__ == "__main__":
    unittest.main()
