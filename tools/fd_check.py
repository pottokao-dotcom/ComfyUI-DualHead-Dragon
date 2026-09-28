"""Finite-difference check of sens_fisher.py's gradient path (8B QI2.1, fp32 weights).

For one layer group's Q4_K error dW and one random probe v (same construction as sens_fisher):
    grad:  <J^T v, dW> from the hooks (sum over the group's linears of <dy, dW x>)
    fd:    v . (y(W + e dW) - y(W - e dW)) / (2e), plain no-grad comfy forward
They must agree if the backward is right. Reports both for a few layers.

    COMFYUI_DIR=... DH_LIB=... python tools/fd_check.py --te-hf ... --qi21-dit ... --prompts p.json --imatrix x.gguf
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sens_fisher as sf  # noqa: E402  (applies the torch.add / RoPE patches, sets up comfy)
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from fakequant import GROUPS, fake_quant  # noqa: E402
from sens_dit import dit_qi21, linears  # noqa: E402
from weight_err import load_imatrix  # noqa: E402


def exact(clip, lin, dit, imat, prompts, layers):
    """Per layer group: true damage (mean 1-cos, unrounded) at Q4_K vs the exact linearized damage
    0.5 ||P dy||^2 / ||y||^2 with dy from central differences (e = 0.25), same token/prompt averaging as sens_dit."""
    import torch.nn.functional as F
    print("layer  true_1-cos    lin_exact     lin/true   worst token share", flush=True)
    toks = [clip.tokenize(p["text"]) for p in prompts]
    with torch.no_grad():
        y0s = [dit(clip.encode_from_tokens(t)[0].float().cuda()) for t in toks]
    for L in layers:
        keys = [(L, hf) for hf, _ in GROUPS["ffn"]]
        orig = {k: lin[k].weight.data.clone() for k in keys}
        dW = {k: (orig[k] - fake_quant(orig[k], "q4_k", imat.get("%d.%s" % k)).to(orig[k].device)) for k in keys}

        def run(e):
            for k in keys:
                lin[k].weight.data.copy_(orig[k] + e * dW[k])
            with torch.no_grad():
                return [dit(clip.encode_from_tokens(t)[0].float().cuda()) for t in toks]
        yq, yp, ym = run(-1.0), run(0.25), run(-0.25)
        for k in keys:
            lin[k].weight.data.copy_(orig[k])
        true, lin_, worst = [], [], []
        for y0, a1, p, m in zip(y0s, yq, yp, ym):
            true.append(float((1 - F.cosine_similarity(a1, y0, dim=-1)).mean()))
            d = (p - m) / 0.5 * -1.0                     # J dW with the sign of the Q4 move (W - dW)
            yh = y0 / y0.norm(dim=-1, keepdim=True)
            dp = d - (d * yh).sum(-1, keepdim=True) * yh
            per = 0.5 * dp.pow(2).sum(-1) / y0.pow(2).sum(-1)
            lin_.append(float(per.mean()))
            worst.append(float(per.max() / per.sum()))
        t, l = sum(true) / len(true), sum(lin_) / len(lin_)
        print("ffn%-3d %.4e    %.4e    %.2f       %.2f" % (L, t, l, l / t, max(worst)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te-hf", required=True)
    ap.add_argument("--qi21-dit", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--layers", default="3,8,20,34")
    ap.add_argument("--lowmem", action="store_true")
    ap.add_argument("--exact", action="store_true", help="exact linearized damage from finite differences vs true 1-cos")
    a = ap.parse_args(sf.OUR_ARGV)

    sd = {}
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.te_hf), "model-*.safetensors"))):
        sd.update(load_file(f))
    clip = comfy.sd.load_text_encoder_state_dicts([sd], clip_type=comfy.sd.CLIPType.QWEN_IMAGE, disable_dynamic=True,
                                                  model_options={"dtype": torch.float32})
    del sd
    import ctypes
    import gc
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
    dit = dit_qi21(a.qi21_dit)
    lin = {k: m for k, m in linears(clip).items() if k[0] != "embed"}
    if a.lowmem:
        sf.lowmem_linears(lin)
    imat = load_imatrix(a.imatrix)
    emb = next(m for n, m in clip.cond_stage_model.named_modules() if n.endswith("embed_tokens") and "visual" not in n)
    emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    prompts = json.load(open(a.prompts, encoding="utf-8"))
    gen = torch.Generator(device="cuda").manual_seed(0)

    def enc(tokens):
        return dit(clip.encode_from_tokens(tokens)[0].float().cuda())

    if a.exact:
        exact(clip, lin, dit, imat, prompts, [int(x) for x in a.layers.split(",")])
        return
    print("layer  prompt  grad           fd(e=1)        fd(e=0.25)     grad/fd", flush=True)
    for L in (int(x) for x in a.layers.split(",")):
        keys = [(L, hf) for hf, _ in GROUPS["ffn"]]
        orig = {k: lin[k].weight.data.clone() for k in keys}
        dW = {k: (orig[k] - fake_quant(orig[k], "q4_k", imat.get("%d.%s" % k)).to(orig[k].device)) for k in keys}
        acc = {}
        hs = []
        for k in keys:
            def fwd(module, args, out, k=k):
                if not torch.is_grad_enabled() or not out.requires_grad:
                    return
                x = args[0].detach()
                out.register_hook(lambda dy, x=x, k=k: acc.__setitem__(
                    k, acc.get(k, 0.0) + float((dy.float() * torch.nn.functional.linear(x.float(), dW[k].float())).sum())))
            hs.append(lin[k].register_forward_hook(fwd))
        for p in prompts:
            tokens = clip.tokenize(p["text"])
            with torch.no_grad():
                y0 = enc(tokens)
            v = torch.randn(y0.shape, generator=gen, device=y0.device)
            acc.clear()
            with torch.enable_grad():
                (enc(tokens) * v).sum().backward()
            g = sum(acc.values())
            fds = []
            for e in (1.0, 0.25):
                ys = []
                for sgn in (1, -1):
                    for k in keys:
                        lin[k].weight.data.copy_(orig[k] + sgn * e * dW[k])
                    with torch.no_grad():
                        ys.append(enc(tokens))
                for k in keys:
                    lin[k].weight.data.copy_(orig[k])
                fds.append(float(((ys[0] - ys[1]) * v).sum()) / (2 * e))
            print("ffn%-3d %-7s %+.6e  %+.6e  %+.6e  %.3f" % (L, p["id"][-6:], g, fds[0], fds[1], g / fds[1]), flush=True)
        for h in hs:
            h.remove()


if __name__ == "__main__":
    main()
