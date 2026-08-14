import argparse
import json
import os

from PIL import Image


def absolute(repo_root, value):
    return value if os.path.isabs(value) else os.path.join(repo_root, value)


def alpha_box(image):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image.getchannel("A").getbbox()


def fit_frame(source, cell_size, height_fraction):
    source = source.convert("RGBA")
    box = alpha_box(source)
    if box is None:
        raise RuntimeError("Rendered frame is fully transparent")
    cropped = source.crop(box)
    cell_width, cell_height = cell_size
    max_width = int(cell_width * 0.84)
    max_height = int(cell_height * height_fraction)
    scale = min(max_width / cropped.width, max_height / cropped.height)
    size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    cropped = cropped.resize(size, Image.Resampling.LANCZOS)
    destination = Image.new("RGBA", (cell_width, cell_height), (0, 0, 0, 0))
    x = (cell_width - size[0]) // 2
    baseline = cell_height - max(3, round(cell_height * 0.035))
    y = baseline - size[1]
    destination.alpha_composite(cropped, (x, y))
    return destination


def fixed_camera_frame(source, cell_size):
    """Preserve one camera-space scale across every pose and direction.

    Per-frame cropping makes a crouch or death grow to standing height and creates
    visible size pumping.  Production masters are rendered through a locked camera,
    so the complete render canvas is the authoritative framing.
    """
    source = source.convert("RGBA")
    if source.size == tuple(cell_size):
        return source.copy()
    return source.resize(tuple(cell_size), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    cell_size = tuple(int(value) for value in config["cell_size"])
    height_fraction = float(config.get("target_height_fraction", 0.78))
    packing_mode = config.get("packing_mode", "fit_each_frame")
    records = []
    for animation_name, animation_config in config["animations"].items():
        frame_count = int(animation_config["frames"])
        for direction in config["directions"]:
            cells = []
            for index in range(frame_count):
                path = os.path.join(args.frames_root, config["id"], animation_name, direction, f"{index:02d}.png")
                if not os.path.isfile(path):
                    raise RuntimeError(f"Missing rendered frame: {path}")
                with Image.open(path) as image:
                    cells.append(
                        fixed_camera_frame(image, cell_size)
                        if packing_mode == "fixed_camera"
                        else fit_frame(image, cell_size, height_fraction)
                    )

            sheet = Image.new("RGBA", (cell_size[0] * frame_count, cell_size[1]), (0, 0, 0, 0))
            for index, cell in enumerate(cells):
                sheet.alpha_composite(cell, (index * cell_size[0], 0))
            relative_output = config["output_path_pattern"].format(animation=animation_name, direction=direction)
            output = absolute(repo_root, relative_output)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            sheet.save(output, "PNG", optimize=True)
            records.append({"animation": animation_name, "direction": direction, "frames": frame_count, "path": relative_output})

    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"character": config["id"], "packing_mode": packing_mode, "sheets": records}, handle, indent=2)
    print(f"ASSET_FORGE_SHEETS={len(records)}")


main()

