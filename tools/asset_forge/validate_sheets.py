import argparse
import json
import os
import sys

from PIL import Image


def absolute(repo_root, value):
    return value if os.path.isabs(value) else os.path.join(repo_root, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    cell_width, cell_height = (int(value) for value in config["cell_size"])
    failures = []
    checked = []
    default_baseline_tolerance = int(config.get("baseline_tolerance", 1))
    for animation_name, animation_config in config["animations"].items():
        frame_count = int(animation_config["frames"])
        baseline_tolerance = int(animation_config.get("baseline_tolerance", default_baseline_tolerance))
        for direction in config["directions"]:
            relative = config["output_path_pattern"].format(animation=animation_name, direction=direction)
            path = absolute(repo_root, relative)
            item = {"path": relative, "animation": animation_name, "direction": direction}
            if not os.path.isfile(path):
                failures.append(f"Missing sheet: {relative}")
                continue
            with Image.open(path) as image:
                image = image.convert("RGBA")
                expected = (cell_width * frame_count, cell_height)
                if image.size != expected:
                    failures.append(f"Wrong dimensions for {relative}: {image.size}, expected {expected}")
                    continue
                baselines = []
                for index in range(frame_count):
                    cell = image.crop((index * cell_width, 0, (index + 1) * cell_width, cell_height))
                    box = cell.getchannel("A").getbbox()
                    if box is None:
                        failures.append(f"Transparent frame {index} in {relative}")
                        continue
                    if box[0] <= 0 or box[2] >= cell_width or box[1] <= 0 or box[3] >= cell_height:
                        failures.append(f"Clipped frame {index} in {relative}: alpha bounds {box}")
                    baselines.append(box[3])
                if baselines and max(baselines) - min(baselines) > baseline_tolerance:
                    failures.append(f"Baseline drift in {relative}: {baselines}")
                item["dimensions"] = list(image.size)
                item["frames"] = frame_count
                item["baselines"] = baselines
                checked.append(item)

    report = {"character": config["id"], "passed": not failures, "checked": checked, "failures": failures}
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"Asset Forge validation passed: {len(checked)} sheets")
    return 0


sys.exit(main())

