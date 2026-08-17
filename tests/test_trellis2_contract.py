from __future__ import annotations

from pathlib import Path

from PIL import Image


ADAPTER = Path(__file__).resolve().parents[1] / "resources" / "adapters" / "trellis2_worker.py"


def _load_adapter_namespace() -> dict:
    namespace = {"__name__": "trellis2_contract_test"}
    exec(compile(ADAPTER.read_text(encoding="utf-8"), str(ADAPTER), "exec"), namespace)
    return namespace


def test_trellis2_requires_meaningful_rgba_mask(tmp_path) -> None:
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


def test_trellis2_no_network_rembg_is_inert_but_fails_loudly_if_invoked() -> None:
    # preprocess_image() in trellis2's own pipeline only calls rembg_model
    # when the input lacks real alpha; this worker always supplies real
    # alpha, so construction/to/cuda/cpu must be safe no-ops (from_pretrained
    # constructs the object unconditionally) but __call__ must raise rather
    # than silently falling back to the gated, non-commercial briaai/RMBG-2.0.
    stub_cls = _load_adapter_namespace()["_NoNetworkRembg"]
    stub = stub_cls("briaai/RMBG-2.0")  # constructor args must be accepted and ignored
    stub.to("cuda")
    stub.cuda()
    stub.cpu()

    import pytest

    with pytest.raises(RuntimeError, match="RMBG"):
        stub(object())
