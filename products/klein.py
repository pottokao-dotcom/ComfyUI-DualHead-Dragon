"""FLUX.2 klein (comfy CLIPLoader type "flux2"): outputs of layers 8/17/26 side by side, padded to 512 tokens. The DiT
reads the pad positions, so they are encoded the way comfy masks them.

  klein 4B: Qwen3-4B, 3 x 2560 = 7680 wide
  klein 9B: Qwen3-8B, 3 x 4096 = 12288 wide (comfy also accepts the Qwen3-VL-8B language model here; not the official
            encoder -- its layer 8/17/26 states are ~0.88 cos to Qwen3-8B's, measured 2026-09-26)
The size is taken from the loaded GGUF, as comfy picks klein_te(model_type=...) from the weights.
"""
from comfy.text_encoders import flux

from . import Product, chatml_qwen3, register
from .qwen3_4b import DHQwen3Clip, make_clip, te_class


class _KleinClip(DHQwen3Clip):
    LAYERS = [8, 17, 26]  # comfy's layer=[9, 18, 27] keeps x BEFORE those layers

    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}, **kwargs):
        self._dh_init(model_options, layer=[9, 18, 27], layer_idx=None)


class DHKleinClip(_KleinClip, flux.Qwen3_4BModel):
    pass


class DHKlein9BClip(_KleinClip, flux.Qwen3_8BModel):
    pass


DHKleinTE = te_class(flux.klein_te(model_type="qwen3_4b"), DHKleinClip, name="qwen3_4b")
DHKlein9BTE = te_class(flux.klein_te(model_type="qwen3_8b"), DHKlein9BClip, name="qwen3_8b")


def _make(backend):
    if backend.n_embd == 2560:
        return make_clip(flux.KleinTokenizer, DHKleinTE, backend)
    if backend.n_embd == 4096:
        return make_clip(flux.KleinTokenizer8B, DHKlein9BTE, backend)
    raise ValueError("flux2 (klein) needs a 4B (2560) or 8B (4096) Qwen3 encoder, this GGUF has n_embd=%d" % backend.n_embd)


class KleinProduct(Product):
    def matches(self, arch, n_embd):  # auto: plain Qwen3 only; a Qwen3-VL GGUF stays Qwen-Image's unless type=flux2
        return arch == "qwen3" and n_embd in (2560, 4096)


register(KleinProduct("flux2", "FLUX.2 klein 4B / 9B", arch="qwen3", n_embd=None, taps=True, make_clip=_make, chat=chatml_qwen3))
