"""Repair LoRA: train a small LoRA on top of a quantized GGUF so the DiT sees what it would see from the bf16 encoder.

The base weights are the GGUF's own values (dequantized with gguf-py, i.e. exactly what llama.cpp multiplies by), the
target is the bf16 encoder's output after each product's DiT input stage, and the loss is the ruler itself:
mean over tokens of 1 - cos (Z-Image; klein real and pad tokens, weighted). The LoRA goes into the dual-head engine's
enc_lora slot, so it is attached only while encoding and never touches the rewrite.

    COMFYUI_DIR=... DH_LIB=... python tools/repair_lora.py --te-hf ~/v6_hf --gguf v6_2p9.gguf \
        --zimage-dit z_image.safetensors --train prompts_train.json --rank 16 --epochs 1 --out repair_v6_2p9
Writes <out>/adapter_model.safetensors + adapter_config.json (PEFT layout; convert with convert_lora_to_gguf.py).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sens_fisher as sf  # noqa: E402  (comfy set-up, torch.add / RoPE patches for autograd)
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from fakequant import GROUPS  # noqa: E402
from sens_dit import PAD, dit_klein, dit_zimage, linears, load_te  # noqa: E402

GG_TO_HF = {gg: hf for g in GROUPS.values() for hf, gg in g}
READ = {"zimage": 34, "klein": 26}          # last layer each product reads (0-based); later layers need no repair


class LoRALinear:
    """Adds (alpha/r) * B A x to a frozen linear; A, B fp32 trainable. B starts at zero."""
    def __init__(self, mod, r, alpha):
        out_f, in_f = mod.weight.shape
        dev = mod.weight.device
        self.A = torch.nn.Parameter(torch.randn(r, in_f, device=dev) / math.sqrt(in_f))
        self.B = torch.nn.Parameter(torch.zeros(out_f, r, device=dev))
        self.s = alpha / r
        orig = mod.forward

        def f(x, *args, **kw):
            y = orig(x, *args, **kw)
            return y + (F.linear(F.linear(x.float(), self.A), self.B) * self.s).to(y.dtype)
        mod.forward = f


def gguf_weights(path):
    """{(layer, hf_name): float tensor} of the GGUF's block linears, dequantized exactly."""
    out = {}
    for t in gguf.GGUFReader(path).tensors:
        parts = t.name.split(".")
        if len(parts) == 4 and parts[0] == "blk" and parts[2] in GG_TO_HF:
            w = dequantize(t.data, t.tensor_type).reshape(tuple(int(x) for x in reversed(t.shape.tolist())))
            out[(int(parts[1]), GG_TO_HF[parts[2]])] = torch.from_numpy(w.copy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te")
    ap.add_argument("--te-hf")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--zimage-dit")
    ap.add_argument("--klein-txt-in")
    ap.add_argument("--weights", default="zimage.real=1,klein.real=1,klein.pad=0.5")
    ap.add_argument("--train", required=True, help="json list of {text}")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=600, help="truncate long prompts (memory)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(sf.OUR_ARGV)
    W = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    t0 = time.time()

    CT = comfy.sd.CLIPType
    prods = []
    if a.zimage_dit:
        prods.append(("zimage", load_te(CT.LUMINA2, a.te, a.te_hf), dit_zimage(a.zimage_dit)))
    if a.klein_txt_in:
        prods.append(("klein", load_te(CT.FLUX2, a.te, a.te_hf), dit_klein(a.klein_txt_in)))
    for _, clip, _ in prods:
        comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
    rows = [r["text"] for r in json.load(open(a.train, encoding="utf-8"))]

    # 1) targets from the bf16 encoder (kept on the CPU in fp16)
    ref = {}
    toks = {}
    with torch.no_grad():
        for pname, clip, dit in prods:
            for i, text in enumerate(rows):
                tk = clip.tokenize(text)
                key = next(iter(tk))
                if len(tk[key][0]) > a.max_tokens:
                    continue
                toks[(pname, i)] = tk
                ref[(pname, i)] = dit(clip.encode_from_tokens(tk)[0].float().cuda()).half().cpu()
    print("targets: %d (product, prompt) pairs  %.0fs" % (len(ref), time.time() - t0), flush=True)

    # 2) swap in the GGUF's dequantized weights, 3) attach LoRA up to each product's read-out layer
    gw = gguf_weights(a.gguf)
    top = max(READ[p] for p, _, _ in prods)
    params, loras = [], {}
    for pname, clip, _ in prods:
        lin = {k: m for k, m in linears(clip).items() if k[0] != "embed"}
        for k, mod in lin.items():
            w = gw[k].to(mod.weight.device, mod.weight.dtype)
            assert w.shape == mod.weight.shape, (k, w.shape, mod.weight.shape)
            mod.weight.data.copy_(w)
        sf.lowmem_linears(lin)    # frozen base without retained fp32 weight copies; the LoRA wrapper goes on top of it
        for k, mod in lin.items():
            if k[0] <= top:
                if k not in loras:                      # one LoRA shared by every product's copy of the model
                    loras[k] = LoRALinear(mod, a.rank, a.alpha)
                    params += [loras[k].A, loras[k].B]
                else:
                    lo = loras[k]
                    orig = mod.forward

                    def f(x, *args, _o=orig, _l=lo, **kw):
                        y = _o(x, *args, **kw)
                        return y + (F.linear(F.linear(x.float(), _l.A), _l.B) * _l.s).to(y.dtype)
                    mod.forward = f
    print("LoRA r%d on %d linears (layers 0-%d), %.1fM params" % (
        a.rank, len(loras), top, sum(p.numel() for p in params) / 1e6), flush=True)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    items = list(ref)
    steps = int(len(items) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) *
                                              0.5 * (1 + math.cos(math.pi * min(s * a.accum / max(steps, 1), 1.0))))
    random.seed(0)
    order = [items[i % len(items)] for i in range(steps)]
    random.shuffle(order)
    by = {p: (c, d) for p, c, d in prods}
    run, n = 0.0, 0
    for step, (pname, i) in enumerate(order):
        clip, dit = by[pname]
        tk = toks[(pname, i)]
        ids = [t[0] for t in tk[next(iter(tk))][0]]
        with torch.enable_grad():
            y = dit(clip.encode_from_tokens(tk)[0].float().cuda())
        yr = ref[(pname, i)].cuda().float()
        cos = F.cosine_similarity(y, yr, dim=-1)
        pad = torch.zeros(len(cos), dtype=torch.bool, device=cos.device)
        if len(ids) == len(cos):
            pad = torch.tensor(ids, device=cos.device) == PAD
        loss = 0.0
        for part, mask in (("real", ~pad), ("pad", pad)):
            t = pname + "." + part
            if mask.any() and W.get(t):
                loss = loss + W[t] * (1 - cos[mask]).mean()
        (loss / a.accum).backward()
        run += float(loss)
        n += 1
        if (step + 1) % a.accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        if (step + 1) % 200 == 0:
            print("step %d/%d  loss %.3e  lr %.1e  %.0fs" % (step + 1, steps, run / n, sched.get_last_lr()[0],
                                                          time.time() - t0), flush=True)
            run, n = 0.0, 0
        torch.cuda.empty_cache() if (step + 1) % 50 == 0 else None

    # PEFT layout for convert_lora_to_gguf.py
    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import save_file
    sd = {}
    for (L, hf), lo in loras.items():
        base = "base_model.model.model.layers.%d.%s" % (L, hf)
        sd[base + ".lora_A.weight"] = lo.A.detach().to(torch.bfloat16).cpu().contiguous()
        sd[base + ".lora_B.weight"] = (lo.B.detach() * 1.0).to(torch.bfloat16).cpu().contiguous()
    save_file(sd, os.path.join(a.out, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "r": a.rank, "lora_alpha": a.alpha, "base_model_name_or_path": "Qwen/Qwen3-4B",
               "target_modules": sorted({hf.split(".")[-1] for _, hf in loras}), "task_type": "CAUSAL_LM",
               "bias": "none", "fan_in_fan_out": False}, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    print("repair LoRA -> %s  (%.0fs)" % (a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
