from __future__ import annotations

from pathlib import Path

from PIL import Image


ADAPTER = Path(__file__).resolve().parents[1] / "adapters" / "triposg_worker.py"


def _load_adapter_namespace() -> dict:
    namespace = {"__name__": "triposg_contract_test"}
    exec(compile(ADAPTER.read_text(encoding="utf-8"), str(ADAPTER), "exec"), namespace)
    return namespace


def test_triposg_requires_meaningful_rgba_mask(tmp_path) -> None:
    valid = tmp_path / "valid.png"
    image = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
    for x in range(5, 15):
        for y in range(5, 15):
            image.putpixel((x, y), (50, 150, 50, 255))
    image.save(valid)

    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (20, 20), (255, 255, 255)).save(rgb)
    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(opaque)

    valid_alpha = _load_adapter_namespace()["_valid_alpha"]
    assert valid_alpha(valid) is True
    assert valid_alpha(rgb) is False
    assert valid_alpha(opaque) is False
