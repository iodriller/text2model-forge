"""Fail-closed TripoSG compatibility shim for environments without compiled DiSo.

TripoSG imports ``DiffDMC`` even when its official ``use_flash_decoder=False``
scikit-image path is selected. Text2Model prepends this package only for that
worker. Any accidental flash-decoder use fails explicitly instead of silently
changing extraction behavior.
"""


class DiffDMC:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "DiSo is unavailable in this worker. Set use_flash_decoder=False to use TripoSG's CPU marching-cubes path."
        )
