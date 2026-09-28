"""The rewrite (PE) ruler: per-TENSOR, per-FORMAT quantization damage to next-token predictions, from backward passes.

Same estimator as sens_fisher.py, with the LLM's own output in place of a DiT stage. The PE score is llama-perplexity's
mean KL(bf16 || quant) over the second half of each 2048-token chunk of the rewrite text; to second order
    KL_t ~ 0.5 * d_t^T F_t d_t,   F_t = Fisher of the softmax at position t, d_t = change of the logits.
With y_t sampled from the model's own p_t, g_t = grad log p_t(y_t) has E[g_t g_t^T] = F_t (true Fisher), and the
positions are independent, so for one probe  sum_t s * <g_t, J_t dW>  with s = sqrt(0.5 / T)  has
    E[ (.)^2 ] = 0.5/T * sum_t d_t^T F_t d_t = mean KL      (T = all scored positions)
The inner products come from each linear's input x and output-gradient dy (no weight gradient is materialized).
Output keys match sens_fisher.py ("L.hf_name" -> {"pe.real": {fmt: D}}); merge with tools/merge_fisher.py.

    python tools/sens_fisher_pe.py --hf ~/qi21_te_heretic_hf --text ~/sens2/pe_eval.txt \
        --imatrix sensbank/qwen3vl-8b-heretic_8d990a89/imat.gguf --out fisher_8b_pe.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "tools"))
from fakequant import fake_quant, tensor_bytes  # noqa: E402

FORMATS = "iq2_xxs,iq2_xs,iq2_s,q2_k,iq3_xxs,iq3_s,q3_k,iq4_xs,q4_k,q5_k,q6_k,q8_0"
NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF model dir (Qwen3-VL or Qwen3)")
    ap.add_argument("--text", required=True, help="the rewrite text llama-perplexity scores (pe_eval.txt)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--formats", default=FORMATS)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-fmt", type=int, default=3, help="formats per batch (each holds its dW)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reserve-gb", type=float, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fmts = a.formats.split(",")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    hf = os.path.expanduser(a.hf)
    tok = AutoTokenizer.from_pretrained(hf)
    arch = AutoConfig.from_pretrained(hf).architectures[0]
    if arch.startswith("Qwen3VL"):
        from transformers import Qwen3VLForConditionalGeneration as M
    else:
        M = AutoModelForCausalLM
    model = M.from_pretrained(hf, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layers = lm.layers
    mods = {(i, n): layers[i].get_submodule(n) for i in range(len(layers)) for n in NAMES}
    keys = sorted(mods)
    print("model loaded: %d layers, %.0f GB available" % (len(layers), avail_gb()), flush=True)

    ids = tok(open(os.path.expanduser(a.text), encoding="utf-8").read(), add_special_tokens=False)["input_ids"]
    n_chunks = min(a.chunks, len(ids) // a.ctx)
    chunks = [torch.tensor(ids[c * a.ctx:(c + 1) * a.ctx], device="cuda")[None] for c in range(n_chunks)]
    first = a.ctx // 2                       # llama-perplexity scores the second half of each chunk
    T = n_chunks * (a.ctx - 1 - first)
    s = (0.5 / T) ** 0.5
    print("%d tokens, %d chunks of %d, %d scored positions" % (len(ids), n_chunks, a.ctx, T), flush=True)

    from weight_err import load_imatrix
    imat = load_imatrix(a.imatrix) if a.imatrix.endswith(".gguf") else dict(np.load(a.imatrix))
    W = {k: mods[k].weight.data for k in keys}
    nbytes = {"%d.%s" % k: {f: tensor_bytes(tuple(W[k].shape), f) for f in fmts} for k in keys}
    per_fmt_gb = sum(w.numel() for w in W.values()) / 1e9     # fp8 dW
    cur, dW = {}, {}

    def install():
        hs = []
        for k, mod in mods.items():
            def fwd(module, args, out, k=k):
                if not out.requires_grad:
                    return
                x = args[0].detach()

                def bwd(dy, x=x):
                    d8, sc = dW[k]
                    d = d8.to(torch.bfloat16) * sc                                   # (F, out, in)
                    z = torch.einsum("...i,foi->f...o", x.to(d.dtype), d).float()
                    cur[k] = cur.get(k, 0) + (z * dy.float().unsqueeze(0)).flatten(1).sum(1)
                out.register_hook(bwd)
            hs.append(mod.register_forward_hook(fwd))
        emb = lm.embed_tokens
        hs.append(emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True)))
        return hs

    from concurrent.futures import ThreadPoolExecutor
    acc = {}
    t0 = time.time()
    for b in range(0, len(fmts), a.chunk_fmt):
        batch = fmts[b:b + a.chunk_fmt]
        need = per_fmt_gb * len(batch) * 1.1 + a.reserve_gb
        if avail_gb() < need:
            sys.exit("stop: %.0f GB available, batch %s needs ~%.0f GB" % (avail_gb(), batch, need))
        t1 = time.time()

        def err(job):
            k, f = job
            w = W[k].float().cpu()
            return job, (w - fake_quant(w, f, imat.get("%d.%s" % k)))
        part = {}
        dW.clear()
        with ThreadPoolExecutor(a.threads) as ex:
            for (k, f), d in ex.map(err, [(k, f) for k in keys for f in batch]):
                part.setdefault(k, {})[f] = d
                if len(part[k]) == len(batch):
                    d = torch.stack([part[k].pop(g) for g in batch]).cuda()
                    sc = (d.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
                    dW[k] = ((d / sc).to(torch.float8_e4m3fn), sc.to(torch.bfloat16))
                    del part[k], d
        print("dW %s: %.0fs, %.0f GB left" % (batch, time.time() - t1, avail_gb()), flush=True)

        gen = torch.Generator(device="cuda").manual_seed(a.seed)
        hs = install()
        n = 0
        for x in chunks:
            for _ in range(a.probes):
                cur.clear()
                with torch.enable_grad():
                    logits = model(input_ids=x).logits[0, first:a.ctx - 1].float()
                    lp = torch.log_softmax(logits, -1)
                    y = torch.multinomial(lp.detach().exp(), 1, generator=gen)
                    (s * lp.gather(1, y).sum()).backward()
                for k, v in cur.items():
                    d = acc.setdefault(k, {})
                    for f, z in zip(batch, v.double().cpu()):
                        d[f] = d.get(f, 0.0) + float(z) ** 2 / a.probes
                n += 1
                del logits, lp
                torch.cuda.empty_cache()
        for h in hs:
            h.remove()
        print("  %s  %d passes  %.0fs" % (batch, n, time.time() - t0), flush=True)
        dW.clear()
        torch.cuda.empty_cache()

    res = {"model": arch, "formats": fmts, "probes": a.probes, "prompts": n_chunks, "targets": ["pe.real"],
           "damage": {"%d.%s" % k: {"pe.real": d} for k, d in acc.items()}, "bytes": nbytes,
           "seconds": round(time.time() - t0)}
    json.dump(res, open(a.out, "w"), indent=1)
    tot = {f: sum(d[f] for d in acc.values()) for f in fmts}
    print("all-tensor sum pe.real  " + "  ".join("%s %.5f" % (f, v) for f, v in tot.items()), flush=True)
    print("done", a.out, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
