import unittest

from PIL import Image

from assetforge.bake import BAKE_SCRIPT, grow_edges


class GrowEdgesTests(unittest.TestCase):
    def test_colors_spread_into_transparent_border(self):
        image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        core = Image.new("RGBA", (8, 8), (200, 120, 40, 255))
        image.paste(core, (12, 12))
        grown = grow_edges(image, steps=4)
        outside = grown.getpixel((9, 16))
        self.assertGreater(outside[0], 0, "color should have spread outward")
        self.assertEqual(0, outside[3], "alpha must remain untouched outside")

    def test_interior_pixels_unchanged(self):
        image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        core = Image.new("RGBA", (8, 8), (200, 120, 40, 255))
        image.paste(core, (12, 12))
        grown = grow_edges(image, steps=4)
        self.assertEqual((200, 120, 40, 255), grown.getpixel((16, 16)))

    def test_alpha_channel_is_preserved_exactly(self):
        image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        image.paste(Image.new("RGBA", (4, 4), (90, 90, 90, 137)), (6, 6))
        grown = grow_edges(image, steps=2)
        self.assertEqual(
            list(image.getchannel("A").getdata()),
            list(grown.getchannel("A").getdata()),
        )


class BakeScriptTests(unittest.TestCase):
    def test_blender_bake_script_exists(self):
        self.assertTrue(BAKE_SCRIPT.is_file(), BAKE_SCRIPT)

    def test_bake_driver_allows_run_scoped_work_and_icons(self):
        source = BAKE_SCRIPT.parent.parent.joinpath("assetforge", "bake.py").read_text(encoding="utf-8")
        self.assertIn('block.get("work_root")', source)
        self.assertIn('block.get("icons_output")', source)


if __name__ == "__main__":
    unittest.main()
