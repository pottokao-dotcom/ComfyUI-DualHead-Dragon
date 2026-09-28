"""Per-layer weight error of each quantization format -- no forward pass, weights only.

E_l(f) = sum over the group's tensors of ||(W - Q_f(W)) * sqrt(imatrix)||^2 (imatrix optional: without it every input
column counts the same). Together with ONE measured damage per layer (e.g. the Q4_K sweep of sens_dit.py) this predicts
the damage of any format: D_l(f) ~= D_l(q4_k) * E_l(f) / E_l(q4_k).

    DH_LIB=dh_llama/lib python tools/weight_err.py --te qwen_3_4b.safetensors --formats q8_0,q6_k,q5_k,q4_k,iq4_xs --out werr_4b.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "tools"))
from fakequant import GROUPS, fake_quant  # noqa: E402


def load_imatrix(path):
    """npz from sens_dit.py ({'L.hf_name': vec}) or a llama-imatrix GGUF (blk.N.<gguf>.weight.in_sum2 / .counts)."""
    if not path.endswith(".gguf"):
        return dict(np.load(path))
    import gguf
    from fakequant import GGUF_TO_HF
    t = {x.name: np.array(x.data, dtype=np.float32).reshape(-1) for x in gguf.GGUFReader(path).tensors}
    out = {}
    for k, v in t.items():
        if k.endswith(".in_sum2"):
            _, i, gg, _w, _s = k.split(".")
            c = t.get(k[: -len(".in_sum2")] + ".counts")
            if gg in GGUF_TO_HF:
                out["%s.%s" % (i, GGUF_TO_HF[gg])] = v / (float(c[0]) if c is not None and c.size else 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", required=True, help="safetensors with model.layers.N.* keys (comfy TE file)")
    ap.add_argument("--formats", default="q8_0,q6_k,q5_k,q4_k,iq4_xs")
    ap.add_argument("--imatrix", help="npz {key 'L.hf_name': imatrix} (optional)")
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    f = safe_open(a.te, "pt")
    keys = set(f.keys())
    L = 1 + max(int(k.split(".")[2]) for k in keys if k.startswith("model.layers."))
    if a.layers:
        L = min(L, a.layers)
    im = load_imatrix(a.imatrix) if a.imatrix else {}
    fmts = a.formats.split(",")
    res = {"te": a.te, "imatrix": bool(im), "formats": fmts, "err": []}
    t0 = time.time()
    for i in range(L):
        for g, tensors in GROUPS.items():
            row = {"group": g, "layer": i}
            for fmt in fmts:
                e = 0.0
                for hf, _ in tensors:
                    w = f.get_tensor("model.layers.%d.%s.weight" % (i, hf)).float()
                    m = im.get("%d.%s" % (i, hf))
                    d = w - fake_quant(w, fmt, m)
                    if m is not None:
                        d = d * torch.from_numpy(m).sqrt()
                    e += float((d * d).sum())
                row[fmt] = e
            res["err"].append(row)
        print("L%-2d %.0fs" % (i, time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("done", a.out)


if __name__ == "__main__":
    main()
