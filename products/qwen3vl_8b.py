"""Qwen3-VL-8B DiTs other than Qwen-Image 2.1 (comfy CLIPLoader types "ideogram4", "boogu", "joyimage").

Same idea as the Qwen3-4B products: comfy's own TE class, tokenizer and template, with only the inner clip's transformer
swapped for the dual-head engine. Qwen-Image 2.1 (in __init__.py) reads the last layer without the final norm; these two
read it differently:
  * Ideogram 4: comfy layer=[1,4,...,34,36] keeps x BEFORE those layers = raw outputs of layers 0,3,...,33,35 (13 taps,
    no final norm), stacked [B, 13, N, 4096]; comfy's Ideogram4 TE reshapes them to 53248 wide.
  * Boogu: last layer AFTER the final RMSNorm (layer_norm_hidden_state=True) = llama.cpp's embedding output as is.
  * JoyAI-Image-Edit: last layer WITHOUT the final norm, raw (tap of layer 35, not the h/rms(h) Qwen-Image gets); comfy's
    JoyImage TE then drops the first 34 (system-prompt) positions. Its own TE is a lightly fine-tuned Qwen3-VL-8B
    (JoyAI-Image-Und); comfy preprocesses its reference images differently from llama.cpp's mtmd (bicubic, mean 0.5),
    so image inputs are not checked yet -- text-to-image only.
Neither is picked by type=auto: a Qwen3-VL-8B GGUF stays Qwen-Image's unless the loader type says otherwise.
"""
import torch
from comfy.text_encoders import boogu, ideogram4, joyimage

from . import Product, register
from .qwen3_4b import make_clip, te_class

PAD = 151643


class DHQwen3VLClip:
    """Mixin for comfy's Qwen3VLClipModel subclasses: keep their attributes, run the engine instead of the transformer."""
    LAYERS = None       # engine layer taps (0-based raw layer outputs), None = last layer
    SINGLE = False      # one tapped layer returned as [B, N, E] (comfy's layer_idx), not stacked [B, 1, N, E]
    FINAL_NORM = False  # last layer after the final RMSNorm

    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}, **kwargs):
        torch.nn.Module.__init__(self)  # skip SDClipModel.__init__: it would build the 8B torch transformer
        self.backend = model_options["dual_head_backend"]
        self.special_tokens = {"pad": PAD}
        self.num_layers = 36
        self.layer, self.layer_idx, self.return_projected_pooled = "hidden", -1, True
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)
        self.layer_norm_hidden_state = self.FINAL_NORM
        self.enable_attention_masks = attention_mask
        self.return_attention_masks = attention_mask
        self.zero_out_masked = False
        self.execution_device = None
        self.image_spans = []

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        self.execution_device = None

    def forward(self, tokens):
        hs, spans0 = [], None
        for seq in tokens:
            h, spans = self.backend.encode(seq, layers=self.LAYERS, final_norm=self.FINAL_NORM)  # [n, E] or [n, L, E]
            hs.append(h)
            if spans0 is None:
                spans0 = spans
        n = max(h.shape[0] for h in hs)
        z = torch.zeros((len(hs), n) + tuple(hs[0].shape[1:]), dtype=torch.float32)
        mask = torch.zeros(len(hs), n, dtype=torch.long)
        for i, h in enumerate(hs):
            z[i, :h.shape[0]] = h
            mask[i, :h.shape[0]] = 1
        if self.LAYERS:
            z = z[:, :, 0] if self.SINGLE else z.permute(0, 2, 1, 3)  # [B, L, N, E]: comfy stacks tapped layers on dim 1
        self.image_spans = spans0 or []
        if self.return_attention_masks:
            return z, None, {"attention_mask": mask}
        return z, None

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        return [], []


class DHIdeogram4Clip(DHQwen3VLClip, ideogram4.Ideogram4Qwen3VLClipModel):
    LAYERS = [i - 1 for i in ideogram4.IDEOGRAM4_TAP_LAYERS]  # comfy keeps x before layer i = output of layer i-1


class DHBooguClip(DHQwen3VLClip, boogu.BooguQwen3VLClipModel):
    FINAL_NORM = True


class DHJoyImageClip(DHQwen3VLClip, joyimage._JoyImageClipModel):
    LAYERS, SINGLE = [35], True  # raw last layer (layer_norm_hidden_state=False)


DHIdeogram4TE = te_class(ideogram4.Ideogram4Qwen3VLTEModel, DHIdeogram4Clip, name="qwen3vl_8b")
DHBooguTE = te_class(boogu.BooguTEModel, DHBooguClip, name="qwen3vl_8b")
DHJoyImageTE = te_class(joyimage.JoyImageTEModel, DHJoyImageClip, name="qwen3vl_8b")


class _ExplicitProduct(Product):
    def matches(self, arch, n_embd):  # never chosen by type=auto (Qwen-Image 2.1 keeps the Qwen3-VL-8B default)
        return False


register(_ExplicitProduct("ideogram4", "Ideogram 4", arch="qwen3vl", n_embd=4096, taps=True,
                          make_clip=lambda backend: make_clip(ideogram4.Ideogram4Qwen3VLTokenizer, DHIdeogram4TE, backend)))
register(_ExplicitProduct("boogu", "Boogu-Image", arch="qwen3vl", n_embd=4096, taps=False,
                          make_clip=lambda backend: make_clip(boogu.BooguTokenizer, DHBooguTE, backend)))
register(_ExplicitProduct("joyimage", "JoyAI-Image-Edit", arch="qwen3vl", n_embd=4096, taps=True,
                          make_clip=lambda backend: make_clip(joyimage.JoyImageTokenizer, DHJoyImageTE, backend)))
