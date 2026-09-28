"""Z-Image / FLUX.2 klein 4B encoder check: comfy's official TE (torch, CPU) vs the dual-head backend (llama.cpp layer tap).

Both sides get the SAME token ids from comfy's own tokenizer, so any gap is the encoder itself.

    python tools/val_4b_encode.py --comfy ~/comfy_ref --te qwen_3_4b.safetensors --gguf q8.gguf [--gguf bf16.gguf] --model zimage
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPTS = [
    "a red apple on a wooden table",
    "Overcast late-afternoon light falls across a narrow night-market alley in Taipei; steam rises from a bamboo "
    "steamer at the left edge while a vendor in a faded blue apron hands a paper bowl of oyster vermicelli to a "
    "schoolgirl whose reflection wobbles in a puddle. Shot on 35mm, shallow depth of field, neon signs soft in the "
    "background, grain visible in the shadows.",
]

# layers each model reads (see comfy/text_encoders/z_image.py, flux.py and the loop in llama.py)
TAPS = {"zimage": [34], "klein": [8, 17, 26]}


def comfy_encode(te_path, model, prompts):
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.sd
    ctype = {"zimage": comfy.sd.CLIPType.LUMINA2, "klein": comfy.sd.CLIPType.FLUX2}[model]
    clip = comfy.sd.load_clip(ckpt_paths=[te_path], clip_type=ctype)
    print("comfy TE:", type(clip.cond_stage_model).__mro__[1].__name__, flush=True)
    out = []
    for p in prompts:
        tokens = clip.tokenize(p)
        key = next(iter(tokens))
        ids = [t[0] for t in tokens[key][0]]
        cond, extra = clip.encode_from_tokens(tokens, return_dict=False), None
        if isinstance(cond, tuple):
            cond = cond[0]
        out.append((ids, cond[0].float()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--te", required=True)
    ap.add_argument("--gguf", action="append", required=True)
    ap.add_argument("--model", choices=sorted(TAPS), default="zimage")
    ap.add_argument("--enc-lora", default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.expanduser(a.comfy))
    ref = comfy_encode(os.path.expanduser(a.te), a.model, PROMPTS)

    sys.path.insert(0, HERE)
    from dh_backend import DualHeadBackend
    for g in a.gguf:
        be = DualHeadBackend(os.path.expanduser(g), None, n_ctx=4096, taps=True)
        if a.enc_lora:
            be.enc_lora = [(os.path.expanduser(a.enc_lora), 1.0)]
        print("\n==", os.path.basename(g), "taps", TAPS[a.model])
        for (ids, zc), p in zip(ref, PROMPTS):
            n_real = next((i for i, t in enumerate(ids) if t == 151643), len(ids))
            h, _ = be.encode(ids, layers=TAPS[a.model], key_limit=n_real if n_real < len(ids) else None)
            zd = h.reshape(h.shape[0], -1).float()      # klein: layers side by side, same as comfy's stack/reshape
            if zd.shape != zc.shape:
                print("  shape mismatch: dh %s comfy %s" % (tuple(zd.shape), tuple(zc.shape)))
                continue
            cos = torch.nn.functional.cosine_similarity(zd, zc, dim=-1)
            rel = (zd - zc).norm(dim=-1) / zc.norm(dim=-1)
            pad = torch.tensor(ids) == 151643
            for name, m in (("real", ~pad), ("pad", pad)):
                if m.any():
                    print("  %-4s %3d tok  cos min %.5f mean %.5f  |  rel err mean %.4f max %.4f  |  norm ratio %.4f" % (
                        name, int(m.sum()), cos[m].min(), cos[m].mean(), rel[m].mean(), rel[m].max(),
                        (zd[m].norm() / zc[m].norm()).item()))
        be.close()


if __name__ == "__main__":
    main()
