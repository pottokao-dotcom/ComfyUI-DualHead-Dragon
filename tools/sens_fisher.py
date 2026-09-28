"""Per-TENSOR, per-FORMAT quantization damage on the DiT ruler from a handful of backward passes (no per-layer sweep).

Linearized damage of quantizing one tensor W to format f, as the DiT sees it:
    D(W, f) = sum_t 0.5 * || P_t J_t dW_f ||^2 / ||y_t||^2        (~ sum_t 1 - cos_t, the sens_dit "dit" score)
y_t = the DiT text-input stage output for token t (Z-Image cap_embedder, klein txt_in + LayerNorm), J_t = dy_t/dW,
P_t = projection orthogonal to y_t (cos only sees the perpendicular part), dW_f = W - Q_f(W) (ggml's own quantizer,
imatrix included). Estimated with random probes: for v_t ~ N(0, 0.5 P_t / ||y_t||^2),
    E_v[ <J^T v, dW_f>^2 ] = D(W, f)
and <J^T v, dW> = sum_t dy_t . (dW x_t) is computed from each linear's input x and output-gradient dy, so no weight
gradient is ever materialized: one backward pass per probe scores every tensor in every format at once.
Token/prompt weighting matches sens_dit.py (mean over tokens of a prompt, then mean over prompts).
Assumes damages of different tensors add (verified per layer group on 4B within 3-4%).

    COMFYUI_DIR=~/ComfyUI_lab DH_LIB=... python tools/sens_fisher.py --te qwen_3_4b.safetensors \
        --zimage-dit z_image.safetensors --klein-txt-in klein4b_txt_in.safetensors --prompts prompts.json \
        --imatrix cfg_ours_4b.json.imatrix.npz --probes 4 --out fisher_4b.json
Output: {"damage": {"L.hf_name": {product.part: {fmt: D}}}, "bytes": {"L.hf_name": {fmt: bytes}}, ...}
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["COMFYUI_DIR"])
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "tools"))
OUR_ARGV, sys.argv = sys.argv[1:], [sys.argv[0], "--gpu-only"]  # comfy parses sys.argv itself
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402

from fakequant import fake_quant, tensor_bytes  # noqa: E402

# comfy's llama blocks end with torch.add(residual, x, out=output) (in place, saves memory); autograd refuses out=.
# With grad on, drop out= (same values, one extra buffer per layer).
_add = torch.add


def _add_grad_ok(*args, out=None, **kw):
    if out is not None and torch.is_grad_enabled():
        return _add(*args, **kw)
    return _add(*args, out=out, **kw) if out is not None else _add(*args, **kw)


torch.add = _add_grad_ok

# comfy's llama RoPE goes through a comfy_kitchen custom kernel with no backward; the eager backend is plain torch
import comfy_kitchen  # noqa: E402
from comfy_kitchen.backends.eager.rope import apply_rope_split_half as _rope_eager  # noqa: E402

comfy_kitchen.apply_rope_split_half = _rope_eager


class _LinSaveRef(torch.autograd.Function):
    """y = x W^T (+ b) at x's precision, saving only a reference to the stored (bf16) weight: comfy's own path casts W
    to an fp32 copy per call and autograd keeps every copy for the backward (8B: ~28 GB). Weights are frozen, so the
    backward only needs dx = dy W, recast one layer at a time."""
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(w)
        return torch.nn.functional.linear(x, w.to(x.dtype), None if b is None else b.to(x.dtype))

    @staticmethod
    def backward(ctx, dy):
        (w,) = ctx.saved_tensors
        return dy @ w.to(dy.dtype), None, None


def lowmem_linears(mods):
    """With grad on, route these linears through _LinSaveRef: fp32 activations as in comfy, no retained weight copies."""
    for mod in mods.values():
        orig = mod.forward

        def f(x, *args, _orig=orig, _mod=mod, **kw):
            if torch.is_grad_enabled() and x.requires_grad:
                return _LinSaveRef.apply(x, _mod.weight, getattr(_mod, "bias", None))
            return _orig(x, *args, **kw)
        mod.forward = f


def grad_in_bf16():
    """comfy runs text encoders with fp32 activations: every linear casts its bf16 weight to an fp32 copy, and with
    grad on autograd keeps all those copies for the backward (8B: ~28 GB). With grad on, run the blocks in bf16 instead
    (masks follow x.dtype); the no-grad reference encode stays fp32."""
    import comfy.text_encoders.llama as L
    fwd = L.Llama2_.forward

    def f(self, x, *args, embeds=None, **kw):
        if torch.is_grad_enabled() and embeds is not None:
            embeds = embeds.to(torch.bfloat16)
        return fwd(self, x, *args, embeds=embeds, **kw)
    L.Llama2_.forward = f
from sens_dit import PAD, dit_klein, dit_qi21, dit_zimage, linears, load_te, products_8b  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["4b", "8b"], default="4b")
    ap.add_argument("--te", help="comfy TE safetensors (4b; 8b also accepted)")
    ap.add_argument("--te-hf", help="HF model dir (8b; comfy renames the keys itself)")
    ap.add_argument("--zimage-dit")
    ap.add_argument("--klein-txt-in")
    ap.add_argument("--qi21-dit")
    ap.add_argument("--ideogram4-cond")
    ap.add_argument("--boogu-caption")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--imatrix", required=True, help="npz {'L.hf_name': vec} (sens_dit.py writes one) or a llama-imatrix .gguf")
    ap.add_argument("--formats", default="iq3_s,iq4_xs,q4_k,q5_k,q6_k,q8_0")
    ap.add_argument("--probes", type=int, default=4, help="random probes per prompt and target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2, help="formats per batch (each batch holds its dW on the GPU)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads quantizing tensors")
    ap.add_argument("--grad-bf16", action="store_true", help="backward passes in bf16 (8B: saves ~28 GB; klein pad off)")
    ap.add_argument("--grad-lowmem", action="store_true",
                    help="fp32 backward without keeping fp32 weight copies (same numbers as default, bf16-like memory)")
    ap.add_argument("--fimat", help="also save DiT-weighted column importance per linear (npz): sum over tokens of "
                    "||dL/dy_t||^2 * x_t^2 -- an imatrix that weights what the DiT ruler sees; --formats none skips dW")
    ap.add_argument("--fp8", action="store_true", help="hold dW as fp8 + per-row scale (half the memory)")
    ap.add_argument("--reserve-gb", type=float, default=25, help="memory to leave free (unified memory machines)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(OUR_ARGV)

    fmts = [] if a.formats in ("", "none") else a.formats.split(",")
    if a.grad_bf16:
        grad_in_bf16()
    prompts = json.load(open(a.prompts, encoding="utf-8"))
    def avail_gb():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
        return float("inf")

    CT = comfy.sd.CLIPType
    if a.model == "4b":
        prods = []   # only the products whose DiT stage was given
        if a.zimage_dit:
            prods.append(("zimage", load_te(CT.LUMINA2, a.te, a.te_hf), dit_zimage(a.zimage_dit)))
        if a.klein_txt_in:
            prods.append(("klein", load_te(CT.FLUX2, a.te, a.te_hf), dit_klein(a.klein_txt_in)))
    else:
        prods = products_8b(a, CT, None)
    for _, clip, _ in prods:
        comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
    import ctypes
    import gc
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    print("model loaded: %.0f GB available" % avail_gb(), flush=True)
    lin = [{k: m for k, m in linears(clip).items() if k[0] != "embed"} for _, clip, _ in prods]
    keys = sorted(lin[0])
    if a.grad_lowmem:
        for mods in lin:
            lowmem_linears(mods)
    if a.imatrix.endswith(".gguf"):                         # llama-imatrix output
        from weight_err import load_imatrix
        imat = load_imatrix(a.imatrix)
    else:
        imat = dict(np.load(a.imatrix))
    t0 = time.time()

    # dW is big (4B: 7.2 GB bf16 per format), so formats go in batches of --chunk; every batch replays the same probes
    # (same seed), so the result is identical to holding all formats at once. gx10 has unified memory: check what is
    # left before each batch and stop instead of pushing the machine into swap.
    from concurrent.futures import ThreadPoolExecutor
    # no CPU copy of the whole model (8B: +14 GB): each thread pulls its one tensor when it quantizes it
    W = {k: lin[0][k].weight.data for k in keys}
    nbytes = {"%d.%s" % k: {f: tensor_bytes(tuple(W[k].shape), f) for f in fmts} for k in keys}
    per_fmt_gb = sum(w.numel() for w in W.values()) * 2 / 1e9

    cur, dW = {}, {}
    fimat = {} if a.fimat else None
    cur_target = [None]
    collecting = [True]

    def install(mods):
        hs = []
        for k, mod in mods.items():
            def fwd(module, args, out, k=k):
                if not torch.is_grad_enabled() or not out.requires_grad:
                    return
                x = args[0].detach()

                def bwd(dy, x=x):
                    if fimat is not None and collecting[0]:
                        tok = dy.float().pow(2).sum(-1, keepdim=True)          # ||dL/dy_t||^2 per token
                        c = (tok * x.float().pow(2)).reshape(-1, x.shape[-1]).sum(0)
                        key = (k, cur_target[0])
                        fimat[key] = fimat.get(key, 0) + c
                    if k not in dW:
                        return
                    d = dW[k]                                   # (F, out, in)
                    if isinstance(d, tuple):                    # fp8 + per-row scale
                        d = d[0].to(torch.bfloat16) * d[1]
                    z = torch.einsum("...i,foi->f...o", x.to(d.dtype), d).float()
                    cur[k] = cur.get(k, 0) + (z * dy.float().unsqueeze(0)).flatten(1).sum(1)
                out.register_hook(bwd)
            hs.append(mod.register_forward_hook(fwd))
        return hs

    acc = {}   # (key, target) -> {fmt: mean of squares over probes}
    for b in range(0, max(len(fmts), 1), a.chunk):
        batch = fmts[b:b + a.chunk]
        if b > 0:
            collecting[0] = False                         # fimat: the first batch's passes are enough
        need = per_fmt_gb * len(batch) * (0.65 if a.fp8 else 1.3) + a.reserve_gb
        if avail_gb() < need:
            sys.exit("stop: %.0f GB available, batch %s needs ~%.0f GB (incl. %.0f GB reserve)" % (
                avail_gb(), batch, need, a.reserve_gb))
        t1 = time.time()

        def err(job):
            k, f = job
            w = W[k].float().cpu()
            return job, (w - fake_quant(w, f, imat.get("%d.%s" % k))).to(torch.bfloat16).cuda()
        dW.clear()

        def pack(d):
            if not a.fp8:
                return d
            # dW only enters inner products; fp8 rounding is random and averages out -> half the memory
            sc = (d.float().abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
            return (d.float() / sc).to(torch.float8_e4m3fn), sc.to(torch.bfloat16)
        part = {}
        with ThreadPoolExecutor(a.threads) as ex:
            for (k, f), d in ex.map(err, [(k, f) for k in keys for f in batch]):
                part.setdefault(k, {})[f] = d
                if len(part[k]) == len(batch):          # convert as soon as a tensor is complete: no bf16 peak
                    dW[k] = pack(torch.stack([part[k].pop(g) for g in batch]))
                    del part[k]
        print("dW %s: %.0fs, %.1f GB, %.0f GB left" % (batch, time.time() - t1, per_fmt_gb * len(batch), avail_gb()),
              flush=True)

        gen = torch.Generator(device="cuda").manual_seed(a.seed)
        n_done = 0
        for (pname, clip, dit), mods in zip(prods, lin):
            hs = install(mods)
            emb = next(m for n, m in clip.cond_stage_model.named_modules()
                       if n.endswith("embed_tokens") and "visual" not in n)
            eh = emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
            for p in prompts:
                tokens = clip.tokenize(p["text"])
                ids = [t[0] for t in tokens[next(iter(tokens))][0]]
                with torch.no_grad():
                    h0 = clip.encode_from_tokens(tokens)[0].float().cuda()
                    y0 = dit(h0)
                pad = torch.zeros(h0.shape[0], dtype=torch.bool, device=h0.device)
                if len(ids) == h0.shape[0]:
                    pad = torch.tensor(ids, device=h0.device) == PAD
                for part, mask in (("real", ~pad), ("pad", pad)):
                    if not mask.any() or (part == "pad" and pname != "klein"):
                        continue
                    cur_target[0] = pname + "." + part
                    yh = y0 / y0.norm(dim=-1, keepdim=True)
                    scale = (0.5 / (mask.sum() * len(prompts))).sqrt() / y0.norm(dim=-1, keepdim=True)
                    for _ in range(a.probes):
                        v = torch.randn(y0.shape, generator=gen, device=y0.device)
                        v = (v - (v * yh).sum(-1, keepdim=True) * yh) * scale * mask[:, None]
                        cur.clear()
                        with torch.enable_grad():
                            h = clip.encode_from_tokens(tokens)[0].float().cuda()
                            (dit(h) * v).sum().backward()
                        for k, s in cur.items():
                            d = acc.setdefault((k, pname + "." + part), {})
                            for f, x in zip(batch, s.double().cpu()):
                                d[f] = d.get(f, 0.0) + float(x) ** 2 / a.probes
                        n_done += 1
                        if n_done == 1:
                            print("  first pass: peak allocated %.1f GB, reserved %.1f GB, %.0f GB available" % (
                                torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9,
                                avail_gb()), flush=True)
                        torch.cuda.empty_cache()    # unified memory: give the backward's buffers back right away
            print("  %s %s  %d passes  %.0fs" % (batch, pname, n_done, time.time() - t0), flush=True)
            for x in hs + [eh]:
                x.remove()
        dW.clear()
        torch.cuda.empty_cache()

    if a.fimat:
        np.savez(a.fimat, **{"%d.%s|%s" % (k[0], k[1], t): v.cpu().numpy() for (k, t), v in fimat.items()})
        print("fimat ->", a.fimat, flush=True)
    targets = sorted({t for _, t in acc})
    res = {"model": a.model, "formats": fmts, "probes": a.probes, "prompts": len(prompts), "targets": targets,
           "damage": {}, "bytes": nbytes, "seconds": round(time.time() - t0)}
    for (k, t), d in acc.items():
        res["damage"].setdefault("%d.%s" % k, {})[t] = d
    json.dump(res, open(a.out, "w"), indent=1)
    tot = {t: {f: sum(d[t][f] for d in res["damage"].values() if t in d) for f in fmts} for t in targets}
    for t in targets:
        print("all-tensor sum %-12s %s" % (t, "  ".join("%s %.5f" % (f, v) for f, v in tot[t].items())), flush=True)
    print("done", a.out, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
