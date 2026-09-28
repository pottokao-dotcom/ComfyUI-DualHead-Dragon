"""TE quantization damage as the DiT sees it, and per-layer sensitivity at the layers each DiT actually reads.

The TE runs as comfy's own encoder class for each product (template, padding, tapped layers exactly as in ComfyUI).
Weights are fake-quantized in place with ggml's own quantizer (ggml_quantize_chunk, imatrix included -> the same numbers
llama-quantize writes) or with tools/nvfp4q.py, then dequantized back to bf16.

Why not just cos of the tapped hidden state: Qwen's residual stream has a few "massive" channels hundreds of times
larger than the rest. They dominate the cosine and barely move under quantization, so a raw cos of 0.98 can hide large
errors in the other channels. The DiT normalizes/projects the text before using it, which exposes those errors. So each
config is scored four ways:
  raw   cos of the tapped hidden state (what we reported before)
  nm    cos with the massive channels removed
  dit   cos after the DiT's own text input stage (Z-Image cap_embedder, klein txt_in + LayerNorm, QI2.1 txt_in MLP)
  rel   relative L2 error, raw and after the DiT stage
Scores are averaged per prompt, then over prompts (a 17-token prompt counts as much as an 800-token one).

    python tools/sens_dit.py --model 4b --te qwen_3_4b.safetensors --zimage-dit z_image.safetensors \
        --klein-txt-in klein4b_txt_in.safetensors --prompts prompts.json --calib calib.txt --out sens_4b.json
    python tools/sens_dit.py --model 8b --te-hf ~/qi21_te_heretic_hf --qi21-dit qwen_image_2.1_bf16.safetensors ...
Needs COMFYUI_DIR (comfy code to import) and DH_LIB (dir with libggml-base.so from build.py).
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["COMFYUI_DIR"])
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "tools"))
OUR_ARGV, sys.argv = sys.argv[1:], [sys.argv[0], "--gpu-only"]  # comfy parses sys.argv itself
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from fakequant import GROUPS, fake_quant, imatrix_hooks, load_type_map, tensor_bytes  # noqa: E402

PAD = 151643


# ---------------------------------------------------------------- DiT text input stages (copied from comfy/ldm)
def dit_zimage(path):
    f = safe_open(path, "pt")
    nw, lw, lb = (f.get_tensor(k).float().cuda() for k in ("cap_embedder.0.weight", "cap_embedder.1.weight",
                                                           "cap_embedder.1.bias"))
    return lambda h: F.linear(F.rms_norm(h, (h.shape[-1],), nw, 1e-5), lw, lb)


def dit_klein(path):
    w = load_file(path)["txt_in.weight"].float().cuda()
    return lambda h: F.layer_norm(F.linear(h, w), (w.shape[0],), eps=1e-6)  # txt_in, then the blocks' txt_norm1


def dit_qi21(path):
    f = safe_open(path, "pt")
    nw = f.get_tensor("txt_in.text_norm.weight").float().cuda() + 1.0  # ZeroCenteredRMSNorm stores scale - 1
    w1, w2 = (f.get_tensor("txt_in.%s.weight" % k).float().cuda() for k in ("in_layer", "out_layer"))
    return lambda h: F.linear(F.gelu(F.linear(F.rms_norm(h, (h.shape[-1],), nw, 1e-6), w1), approximate="tanh"), w2)


def dit_ideogram4(path):
    """Ideogram 4: 13 raw layer outputs, interleaved per channel (comfy: (B,13,seq,H) -> permute -> (B,seq,H*13)),
    then llm_cond_norm (RMSNorm over 53248) and llm_cond_proj. The Comfy-Org file stores the projection as fp8 +
    a per-tensor scale -- what ComfyUI users run."""
    f = safe_open(path, "pt")
    nw = f.get_tensor("llm_cond_norm.weight").float().cuda()
    w = f.get_tensor("llm_cond_proj.weight").float().cuda()
    if "llm_cond_proj.weight_scale" in f.keys():
        w = w * f.get_tensor("llm_cond_proj.weight_scale").float().cuda()
    b = f.get_tensor("llm_cond_proj.bias").float().cuda()
    return lambda h: F.linear(F.rms_norm(h, (h.shape[-1],), nw, 1e-6), w, b)


def dit_boogu(path):
    """Boogu: last layer after the TE's final RMSNorm, then caption_embedder (RMSNorm + Linear)."""
    f = safe_open(path, "pt")
    nw = f.get_tensor("time_caption_embed.caption_embedder.0.weight").float().cuda()
    w = f.get_tensor("time_caption_embed.caption_embedder.1.weight").float().cuda()
    b = f.get_tensor("time_caption_embed.caption_embedder.1.bias").float().cuda()
    return lambda h: F.linear(F.rms_norm(h, (h.shape[-1],), nw, 1e-5), w, b)


IDEOGRAM4_TAPS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]   # 0..33 raw decoder outputs; 35 = final-norm output



def products_8b(a, CT, prompts, mo=None):
    """the 8B products whose DiT stage was given -- normally one per call (each holds a 16 GB copy of the TE)"""
    out = []
    for name, arg, ct, dit in (("klein", getattr(a, "klein_txt_in", None), CT.FLUX2, dit_klein),  # klein 9B: Qwen3-8B
                               ("qi21", a.qi21_dit, CT.QWEN_IMAGE, dit_qi21),
                               ("ideogram4", getattr(a, "ideogram4_cond", None), CT.IDEOGRAM4, dit_ideogram4),
                               ("boogu", getattr(a, "boogu_caption", None), CT.BOOGU, dit_boogu)):
        if arg:
            clip = load_te(ct, getattr(a, "te", None), getattr(a, "te_hf", None), mo)
            out.append((name, clip, dit(os.path.expanduser(arg))))
    return out


def load_te(ct, te=None, te_hf=None, model_options=None):
    """comfy TE from a comfy/HF single safetensors (te) or an HF shard directory (te_hf); frees the loaded state dict."""
    if te:
        return comfy.sd.load_clip(ckpt_paths=[te], clip_type=ct, model_options=model_options or {}, disable_dynamic=True)
    import ctypes
    import gc
    sd = {}
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(te_hf), "model*.safetensors"))):
        sd.update(load_file(p))
    clip = comfy.sd.load_text_encoder_state_dicts([sd], clip_type=ct, model_options=model_options or {}, disable_dynamic=True)
    del sd
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)   # on unified memory the freed state dict is GPU memory
    return clip


# ---------------------------------------------------------------- products
class Product:
    def __init__(self, name, clip, dit, prompts):
        self.name, self.clip, self.dit = name, clip, dit
        self.inputs = []
        for p in prompts:
            tokens = clip.tokenize(p["text"])
            ids = [t[0] for t in tokens[next(iter(tokens))][0]]
            self.inputs.append((p, tokens, ids))

    @torch.no_grad()
    def encode(self):
        outs = []
        for p, tokens, ids in self.inputs:
            cond = self.clip.encode_from_tokens(tokens)
            h = cond[0].float().cuda()
            pad = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
            if len(ids) == h.shape[0]:
                pad = torch.tensor(ids, device=h.device) == PAD
            outs.append((h, pad))
        return outs


def linears(clip):
    """{(layer, hf_suffix): module} for the transformer blocks of a comfy TE, plus ("embed", "token_embd")."""
    out = {}
    for name, m in clip.cond_stage_model.named_modules():
        parts = name.split(".")
        if parts[-1] == "embed_tokens" and "visual" not in parts:
            out[("embed", "token_embd")] = m
        if "layers" in parts and "visual" not in parts and hasattr(m, "weight"):
            i = parts.index("layers")
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                suffix = ".".join(parts[i + 2:])
                if any(suffix == hf for g in GROUPS.values() for hf, _ in g):
                    out[(int(parts[i + 1]), suffix)] = m
    return out


def massive_dims(base, k):
    a = torch.cat([h[~pad].abs() for h, pad in base]).mean(0)
    ratio = a / a.median()
    dims = torch.nonzero(ratio > k).flatten()
    return dims, {int(d): round(float(ratio[d]), 1) for d in dims}


def score(outs, base, dit, dims):
    per = {}
    keep = torch.ones(base[0][0].shape[-1], dtype=torch.bool, device=base[0][0].device)
    keep[dims] = False
    for (h, pad), (h0, _) in zip(outs, base):
        y, y0 = dit(h), dit(h0)
        m = {"raw": F.cosine_similarity(h, h0, dim=-1),
             "nm": F.cosine_similarity(h[:, keep], h0[:, keep], dim=-1),
             "dit": F.cosine_similarity(y, y0, dim=-1),
             "rel": (h - h0).norm(dim=-1) / h0.norm(dim=-1),
             "rel_dit": (y - y0).norm(dim=-1) / y0.norm(dim=-1)}
        for part, mask in (("real", ~pad), ("pad", pad)):
            if mask.any():
                for k, v in m.items():
                    per.setdefault(part, {}).setdefault(k, []).append(float(v[mask].mean()))
    return {part: {k: round(sum(v) / len(v), 6) for k, v in d.items()} for part, d in per.items()}


def save(res, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["4b", "8b"], required=True)
    ap.add_argument("--te", help="comfy TE safetensors (4b)")
    ap.add_argument("--te-hf", help="HF model dir (8b; comfy renames the keys itself)")
    ap.add_argument("--zimage-dit")
    ap.add_argument("--klein-txt-in")
    ap.add_argument("--qi21-dit")
    ap.add_argument("--ideogram4-cond", help="safetensors with llm_cond_norm / llm_cond_proj (tools/fetch_tensors.py)")
    ap.add_argument("--boogu-caption", help="safetensors with time_caption_embed.caption_embedder.*")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--calib", required=True, help="imatrix calibration text (the same one the shipped GGUF uses)")
    ap.add_argument("--sweep", default="q4_k", help="format for the per-layer sweep ('' to skip)")
    ap.add_argument("--groups", default="ffn,attn", help="layer groups to sweep")
    ap.add_argument("--type-sweep", default="", help="format: quantize ONE tensor type (e.g. attn_v) in every layer at a "
                    "time -> how the damage splits across tensor types")
    ap.add_argument("--ruler", default="q8_0,q4_k,nvfp4", help="formats applied to ALL layers")
    ap.add_argument("--configs", nargs="*", default=[], help="GGUF type maps (json) to score as whole-model configs")
    ap.add_argument("--ggufs", nargs="*", default=[], help="real GGUFs run through the dual-head engine (llama.cpp), "
                    "scored against the same comfy bf16 reference (4b only: Z-Image / klein layer taps)")
    ap.add_argument("--enc-lora", help="GGUF LoRA the engine attaches while encoding (scored with --ggufs)")
    ap.add_argument("--rot", help="rotation.pt from tools/rotate_r1.py: the GGUFs are R1-rotated, undo it on the taps")
    ap.add_argument("--dump", help="prefix: save each GGUF's hidden states and the bf16 reference per product "
                    "(<prefix>_<product>_<gguf>.pt) for offline error analysis")
    ap.add_argument("--fp32", action="store_true", help="run the TE in fp32: dequantized Q8/Q6 values are not re-rounded "
                    "to bf16 (in bf16 that extra rounding is about as large as Q8's own error)")
    ap.add_argument("--massive", type=float, default=20.0, help="channel mean|h| / median above this = massive")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(OUR_ARGV)

    prompts = json.load(open(a.prompts, encoding="utf-8"))
    CT = comfy.sd.CLIPType
    if a.model == "4b":
        mo = {"dtype": torch.float32} if a.fp32 else {}
        prods = []   # only the products whose DiT stage was given (a Z-Image-only finetune needs no klein)
        if a.zimage_dit:
            prods.append(Product("zimage", load_te(CT.LUMINA2, a.te, a.te_hf, mo), dit_zimage(a.zimage_dit), prompts))
        if a.klein_txt_in:
            prods.append(Product("klein", load_te(CT.FLUX2, a.te, a.te_hf, mo), dit_klein(a.klein_txt_in), prompts))
    else:
        mo = {"dtype": torch.float32} if a.fp32 else {}
        prods = [Product(n, c, d, prompts) for n, c, d in products_8b(a, CT, prompts, mo)]
    for pr in prods:
        comfy.model_management.load_models_gpu([pr.clip.patcher], force_full_load=True)
    lin = [linears(pr.clip) for pr in prods]
    L = 1 + max(i for i, _ in lin[0] if isinstance(i, int))
    dev = {str(m.weight.device) for m in lin[0].values()}
    print("weight dtype", next(iter(lin[0].values())).weight.dtype)
    print("products", [p.name for p in prods], "layers", L, "linears", len(lin[0]), "weights on", dev, flush=True)
    assert all(len(x) == 7 * L + 1 for x in lin) and dev == {"cuda:0"}, "unexpected TE layout / weights not on GPU"
    orig = {k: m.weight.data.clone().cpu() for k, m in lin[0].items()}  # bf16 originals, restored after each config

    # imatrix = mean x^2 per input column, as llama-imatrix computes it; collected on the first product's template.
    # Only the fake-quant paths use it: with no calibration text (e.g. scoring real GGUFs only) it is skipped.
    calib = [p.strip() for p in open(a.calib, encoding="utf-8").read().split("\n\n") if len(p.strip()) > 200]
    imat = {}
    if calib:
        acc, finish = imatrix_hooks({k: m for k, m in lin[0].items() if k[0] != "embed"})
        with torch.no_grad():
            for c in calib:
                prods[0].clip.encode_from_tokens(prods[0].clip.tokenize(c[:4000]))
        n_tok = acc[(0, "mlp.up_proj")][1]
        imat = finish()
        import numpy as np
        np.savez(a.out + ".imatrix.npz", **{"%s.%s" % k: v for k, v in imat.items()})  # for tools/weight_err.py
        print("imatrix from %d calib paragraphs, %d tokens" % (len(calib), n_tok), flush=True)

    base = [pr.encode() for pr in prods]
    res = {"model": a.model, "prompts": [(p["kind"], p["id"]) for p in prompts], "massive": {}, "ruler": {},
           "sweep_format": a.sweep, "sweep": []}
    dims = []
    for pr, b in zip(prods, base):
        d, info = massive_dims(b, a.massive)
        dims.append(d)
        res["massive"][pr.name] = info
        print(pr.name, "massive channels (ratio to median):", info, flush=True)

    def apply(keys, qtype):
        for k in keys:
            t = qtype[k] if isinstance(qtype, dict) else qtype
            q = fake_quant(orig[k].cuda(), t, imat.get(k)).to(orig[k].dtype)
            for x in lin:
                x[k].weight.data.copy_(q)

    def restore(keys):
        for k in keys:
            for x in lin:
                x[k].weight.data.copy_(orig[k])

    def run(tag):
        out = {}
        for pr, b, d in zip(prods, base, dims):
            out[pr.name] = score(pr.encode(), b, pr.dit, d)
        return out

    t0 = time.time()
    for qt in [x for x in a.ruler.split(",") if x]:
        for scope, groups in (("all", ("attn", "ffn")), ("ffn", ("ffn",))):
            keys = [(i, hf) for i in range(L) for g in groups for hf, _ in GROUPS[g]]
            apply(keys, qt)
            res["ruler"]["%s_%s" % (qt, scope)] = r = run(qt)
            restore(keys)
            print("ruler %-6s %-3s %s  (%.0fs)" % (qt, scope, json.dumps(r), time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)

    res["types"] = {}
    if a.type_sweep:
        for g, tensors in GROUPS.items():
            for hf, gg in tensors:
                keys = [(i, hf) for i in range(L)]
                apply(keys, a.type_sweep)
                res["types"][gg] = r = run(gg)
                restore(keys)
                print("type %-12s %s  (%.0fs)" % (gg, json.dumps(r), time.time() - t0), flush=True)
                save(res, a.out)

    res["ggufs"] = {}
    if a.ggufs:
        sys.path.insert(0, HERE)
        from dh_backend import DualHeadBackend
        taps = {"zimage": [34], "klein": [8, 17, 26], "qi21": None}
        Rt = torch.load(os.path.expanduser(a.rot))["R"].T.float() if a.rot else None   # raw hidden = h R
        for path in a.ggufs:
            be = DualHeadBackend(os.path.expanduser(path), None, n_ctx=8192, taps=True)
            if a.enc_lora:                              # e.g. a repair LoRA: attached only while encoding
                be.enc_lora = [(os.path.expanduser(a.enc_lora), 1.0)]
            out = {}
            for pr, b, d in zip(prods, base, dims):
                outs = []
                for (p, tokens, ids), (h0, pad) in zip(pr.inputs, b):
                    if pr.name == "ideogram4":
                        # taps 0..33: raw layer outputs. The 13th "tap" is what comfy (and HF hidden_states[-1])
                        # appends after the loop: the FINAL-NORM output h/rms(h)*w -- verified per tap vs comfy
                        # (raw layer 35 gives cos 0.76, 4x the norm).
                        t, _ = be.encode([int(t) for t in ids], layers=IDEOGRAM4_TAPS[:-1])  # (n, 12, H)
                        last, _ = be.encode([int(t) for t in ids])                           # h / rms(h)
                        t = torch.cat([t.float(), (last * be.norm_w).float().unsqueeze(1)], 1)
                        h = t.permute(0, 2, 1).reshape(t.shape[0], -1)[-h0.shape[0]:].float().to(h0.device)
                        outs.append((h, pad))
                        continue
                    if pr.name == "boogu":
                        # engine: h / rms(h); Boogu reads the final RMSNorm output h / rms(h) * w
                        h, _ = be.encode([int(t) for t in ids])
                        h = (h * be.norm_w)[-h0.shape[0]:].float().to(h0.device)
                        outs.append((h, pad))
                        continue
                    if pr.name == "qi21":
                        # last layer; the engine returns h / rms(h) (llama.cpp output norm with w divided out), comfy
                        # the raw h: txt_in's RMSNorm makes them equal, so only the post-DiT-stage scores are
                        # comparable. comfy drops the template prefix -> align on the last rows.
                        h, _ = be.encode([int(t) for t in ids])
                        h = h[-h0.shape[0]:].float().to(h0.device)
                        outs.append((h, pad))
                        continue
                    n = next((i for i, t in enumerate(ids) if t == PAD), len(ids))
                    h, _ = be.encode([int(t) for t in ids], layers=taps[pr.name], key_limit=n if n < len(ids) else None)
                    if Rt is not None:
                        h = h.float() @ Rt.to(h.device)
                    h = h.reshape(h.shape[0], -1).float().to(h0.device)
                    assert h.shape == h0.shape, (pr.name, h.shape, h0.shape)
                    outs.append((h, pad))
                if a.dump:
                    torch.save({"outs": [(h.cpu(), m.cpu()) for h, m in outs], "base": [(h.cpu(), m.cpu()) for h, m in b],
                                "dims": d}, "%s_%s_%s.pt" % (os.path.expanduser(a.dump), pr.name,
                                                             os.path.splitext(os.path.basename(path))[0]))
                out[pr.name] = score(outs, b, pr.dit, d)
            be.close()
            name = os.path.splitext(os.path.basename(path))[0]
            if a.enc_lora:
                name += "+" + os.path.splitext(os.path.basename(a.enc_lora))[0]
            res["ggufs"][name] = {"gb_file": round(os.path.getsize(os.path.expanduser(path)) / 1e9, 3), **out}
            print("gguf %s %.2f GB %s" % (name, res["ggufs"][name]["gb_file"], json.dumps(out)), flush=True)
            save(res, a.out)

    res["configs"] = {}
    for path in a.configs:
        tmap = {k: t for k, t in load_type_map(path).items() if k in orig}
        name = os.path.splitext(os.path.basename(path))[0]
        apply(list(tmap), tmap)
        r = run(name)
        restore(list(tmap))
        gb = sum(tensor_bytes(orig[k].shape, t) for k, t in tmap.items()) / 1e9
        res["configs"][name] = {"gb_scored_tensors": round(gb, 3), "n": len(tmap), **r}
        print("config %s %.2f GB %s  (%.0fs)" % (name, gb, json.dumps(r), time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)

    if a.sweep:
        for g in a.groups.split(","):
            for i in range(L):
                keys = [(i, hf) for hf, _ in GROUPS[g]]
                apply(keys, a.sweep)
                r = run("%s%d" % (g, i))
                restore(keys)
                res["sweep"].append({"group": g, "layer": i, **r})
                save(res, a.out)
                print("%-4s L%-2d %s  (%.0fs)" % (g, i, " | ".join(
                    "%s dit %.5f raw %.5f" % (n, v["real"]["dit"], v["real"]["raw"]) for n, v in r.items()),
                    time.time() - t0), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("done", a.out)


if __name__ == "__main__":
    main()
