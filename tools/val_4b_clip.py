"""End-to-end check of a product: comfy CLIP built by products/<name> (dual-head engine inside) vs comfy's official
CLIPLoader for the same type, both through clip.tokenize + clip.encode_from_tokens (conditioning + attention mask).

    python tools/val_4b_clip.py --comfy ~/comfy_ref --te qwen_3_4b.safetensors --gguf q.gguf --type lumina2|flux2
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = ["a red apple on a wooden table",
           "Overcast light in a narrow Taipei night-market alley; steam rises from a bamboo steamer while a vendor "
           "hands a paper bowl of oyster vermicelli to a schoolgirl. 35mm, shallow depth of field, grain in the shadows."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--te", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--type", required=True)
    a = ap.parse_args()
    sys.argv = [sys.argv[0], "--cpu"]
    sys.path.insert(0, os.path.expanduser(a.comfy))
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.sd
    sys.path.insert(0, HERE)
    import products
    import products.klein  # noqa: F401  (registers)
    import products.zimage  # noqa: F401
    from dh_backend import DualHeadBackend

    official = comfy.sd.load_clip(ckpt_paths=[os.path.expanduser(a.te)],
                                  clip_type={"lumina2": comfy.sd.CLIPType.LUMINA2, "flux2": comfy.sd.CLIPType.FLUX2}[a.type])
    prod = products.resolve(a.type, os.path.expanduser(a.gguf))
    be = DualHeadBackend(os.path.expanduser(a.gguf), None, n_ctx=4096, taps=prod.taps)
    dh = prod.make_clip(be)
    print("official TE %s | dual-head TE %s" % (type(official.cond_stage_model).__name__, type(dh.cond_stage_model).__name__))
    for p in PROMPTS:
        outs = []
        for clip in (official, dh):
            tok = clip.tokenize(p)
            c, extra = clip.encode_from_tokens(tok, return_dict=False), {}
            d = clip.encode_from_tokens(tok, return_dict=True)
            outs.append((d["cond"][0].float(), d.get("attention_mask")))
        (zo, mo), (zd, md) = outs
        same_mask = (mo is None and md is None) or (mo is not None and md is not None and torch.equal(mo.cpu(), md.cpu()))
        if zo.shape != zd.shape:
            print("  shape official %s vs dual-head %s" % (tuple(zo.shape), tuple(zd.shape)))
            continue
        cos = torch.nn.functional.cosine_similarity(zd, zo, dim=-1)
        print("  cond %s  mask same: %s  cos min %.5f mean %.5f" % (tuple(zo.shape), same_mask, cos.min(), cos.mean()))


if __name__ == "__main__":
    main()
