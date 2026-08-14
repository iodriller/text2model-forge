from __future__ import annotations

from pathlib import Path

from PIL import Image


ADAPTER = Path(__file__).resolve().parents[1] / "adapters" / "instantmesh_worker.py"


def _load_adapter_namespace() -> dict:
    namespace = {"__name__": "instantmesh_contract_test"}
    exec(compile(ADAPTER.read_text(encoding="utf-8"), str(ADAPTER), "exec"), namespace)
    return namespace


def test_instantmesh_requires_meaningful_rgba_mask(tmp_path) -> None:
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


def test_instantmesh_composites_owned_rgba_over_white(tmp_path) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    for x in range(2, 8):
        for y in range(2, 8):
            image.putpixel((x, y), (10, 20, 30, 255))
    image.save(source)

    composite_on_white = _load_adapter_namespace()["_composite_on_white"]
    result = composite_on_white(source)

    assert result.mode == "RGB"
    assert result.size == (10, 10)
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((5, 5)) == (10, 20, 30)
