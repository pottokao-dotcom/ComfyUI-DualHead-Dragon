"""Z-Image (comfy CLIPLoader type "lumina2"): Qwen3-4B, output of layer index 34 (layer_idx=-2), no padding."""
from comfy.text_encoders import z_image

from . import Product, chatml_qwen3, register
from .qwen3_4b import DHQwen3Clip, make_clip, te_class


class DHZImageClip(DHQwen3Clip, z_image.Qwen3_4BModel):
    LAYERS = [34]  # comfy keeps x after layer 36 + (-2) = 34

    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}, **kwargs):
        self._dh_init(model_options, layer="hidden", layer_idx=-2)


DHZImageTE = te_class(z_image.te(), DHZImageClip)

register(Product("lumina2", "Z-Image", arch="qwen3", n_embd=2560, taps=True,
                 make_clip=lambda backend: make_clip(z_image.ZImageTokenizer, DHZImageTE, backend), chat=chatml_qwen3))
