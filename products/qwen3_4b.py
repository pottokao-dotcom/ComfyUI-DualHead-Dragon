"""Qwen3-4B family (Z-Image, FLUX.2 klein 4B): comfy's own clip classes with the torch transformer replaced by the
dual-head engine's layer tap. Tokenizer, template, padding and the conditioning shape stay comfy's.

What comfy computes (comfy/text_encoders/llama.py, sd1_clip.py), and how the engine reproduces it:
  * layer_idx=-2 (Z-Image): x kept AFTER layer index 34                -> tap l_out-34
  * layer=[9,18,27] (klein): x kept BEFORE layers 9/18/27             -> tap l_out-8/17/26
  * layer_norm_hidden_state=False: raw hidden states, no final norm   -> l_out is pre-norm
  * attention mask: 1 up to the first pad, 0 from there; pad positions still get outputs, attending to the real tokens
    only (not to themselves or each other). klein pads to 512 and its DiT reads the pads -> encode(key_limit=n_real).
Checked against comfy's TE on CPU: tools/val_4b_encode.py.
"""
import torch

import comfy.sd
import comfy.supported_models_base

PAD = 151643


class DHQwen3Clip:
    """Mixin for a comfy SDClipModel subclass: keep its attributes, run the engine instead of the torch transformer."""
    LAYERS = None  # engine layer indices, set by the product

    def _dh_init(self, model_options, layer, layer_idx):
        torch.nn.Module.__init__(self)  # skip SDClipModel.__init__: it would build the torch transformer
        self.backend = model_options["dual_head_backend"]
        self.special_tokens = {"pad": PAD}
        self.num_layers = 36
        self.layer, self.layer_idx, self.return_projected_pooled = layer, layer_idx, True
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)
        self.layer_norm_hidden_state = False
        self.enable_attention_masks = True
        self.return_attention_masks = True
        self.zero_out_masked = False
        self.execution_device = None

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        self.execution_device = None

    def forward(self, tokens):
        hs, masks = [], []
        for seq in tokens:
            ids = [int(t) for t in seq]
            n = next((i for i, t in enumerate(ids) if t == PAD), len(ids))
            h, _ = self.backend.encode(ids, layers=self.LAYERS, key_limit=n if n < len(ids) else None)  # [N, L, E]
            hs.append(h)
            masks.append([1] * n + [0] * (len(ids) - n))
        if len({h.shape[0] for h in hs}) != 1:
            raise RuntimeError("dual-head: sequences in one batch have different lengths %s" % [h.shape[0] for h in hs])
        z = torch.stack([h.permute(1, 0, 2) for h in hs]).float()  # [B, L, N, E] (comfy stacks tapped layers on dim 1)
        if self.layer_idx is not None:  # a single layer_idx gives [B, N, E]
            z = z[:, 0]
        return z, None, {"attention_mask": torch.tensor(masks, dtype=torch.long)}

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        return [], []


def make_clip(tokenizer_cls, te_cls, backend):
    target = comfy.supported_models_base.ClipTarget(tokenizer_cls, te_cls)
    # the CLIP holds no torch weights; llama.cpp owns them
    return comfy.sd.CLIP(target=target, model_options={"dual_head_backend": backend})



def te_class(official_te, clip_cls, name="qwen3_4b"):
    """comfy's own TE class for the product, with only its inner clip (qwen3_4b / qwen3_8b) swapped (comfy's documented
    model_options["<name>_class"] override, as the Qwen-Image 2.1 loader does)."""
    class DHTE(official_te):
        disable_offload = True  # llama.cpp owns the weights; nothing for comfy to move

        def __init__(self, device="cpu", dtype=None, model_options={}):
            mo = dict(model_options)
            mo[name + "_class"] = clip_cls
            super().__init__(device=device, dtype=dtype, model_options=mo)
    DHTE.__name__ = "DH" + official_te.__name__
    return DHTE
