"""Is the DiT-ruler damage of one layer group linear in its quantization error? (8B QI2.1)

For a few FFN layers: W <- W - alpha * dW_q4k for alpha in (1, 0.5, 0.25), encode the prompts, score like sens_dit.
If the damage is quadratic in the perturbation (what sens_fisher.py assumes), damage(alpha) / alpha^2 is constant.
Printed next to the sens_fisher prediction for alpha = 1.

    COMFYUI_DIR=~/ComfyUI_lab DH_LIB=... python tools/lin_check.py --te-hf ~/qi21_te_heretic_hf --qi21-dit ... \
        --prompts prompts.json --imatrix heretic.imatrix.gguf --fisher fisher_8b_heretic.json --layers 3,8,9,28,34
"""
import argparse
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["COMFYUI_DIR"])
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "tools"))
OUR_ARGV, sys.argv = sys.argv[1:], [sys.argv[0], "--gpu-only"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from fakequant import GROUPS, fake_quant  # noqa: E402
from sens_dit import Product, dit_qi21, linears, massive_dims, score  # noqa: E402
from weight_err import load_imatrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te-hf", required=True)
    ap.add_argument("--qi21-dit", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--fisher", required=True)
    ap.add_argument("--layers", default="3,8,9,28,34")
    ap.add_argument("--group", default="ffn")
    ap.add_argument("--fmt", default="q4_k")
    a = ap.parse_args(OUR_ARGV)

    sd = {}
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.te_hf), "model-*.safetensors"))):
        sd.update(load_file(f))
    clip = comfy.sd.load_text_encoder_state_dicts([sd], clip_type=comfy.sd.CLIPType.QWEN_IMAGE, disable_dynamic=True,
                                                  model_options={"dtype": torch.float32})  # no bf16 re-rounding of W - a dW
    del sd
    import ctypes
    import gc
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
    pr = Product("qi21", clip, dit_qi21(a.qi21_dit), json.load(open(a.prompts, encoding="utf-8")))
    lin = linears(clip)
    imat = load_imatrix(a.imatrix)
    fi = json.load(open(a.fisher))["damage"]
    base = pr.encode()
    dims, _ = massive_dims(base, 20.0)
    print("layer   pred(fisher)  alpha=1      d/a^2 @0.5   d/a^2 @0.25   ratio 1 vs 0.25", flush=True)
    for L in (int(x) for x in a.layers.split(",")):
        keys = [(L, hf) for hf, _ in GROUPS[a.group]]
        orig = {k: lin[k].weight.data.clone() for k in keys}
        dW = {k: orig[k].float() - fake_quant(orig[k].float(), a.fmt, imat.get("%d.%s" % k)).to(orig[k].device)
              for k in keys}
        pred = sum(fi.get("%d.%s" % k, {}).get("qi21.real", {}).get(a.fmt, 0.0) for k in keys)
        out = {}
        for alpha in (1.0, 0.5, 0.25):
            for k in keys:
                lin[k].weight.data.copy_((orig[k].float() - alpha * dW[k]).to(orig[k].dtype))
            r = score(pr.encode(), base, pr.dit, dims)
            out[alpha] = (1 - r["real"]["dit"]) / alpha ** 2
        for k in keys:
            lin[k].weight.data.copy_(orig[k])
        print("%s%-4d  %.7f     %.7f    %.7f    %.7f     %.2f" % (
            a.group, L, pred, out[1.0], out[0.5], out[0.25], out[1.0] / max(out[0.25], 1e-12)), flush=True)


if __name__ == "__main__":
    main()
