# TripoSG DiSo compatibility shim

This shim is loaded only by the Text2Model TripoSG worker on machines without the
MSVC/CUDA extension toolchain. It does not implement or imitate DiSo. It allows
TripoSG's own `use_flash_decoder=False` path to import, and fails immediately if
the flash decoder is accidentally requested.
