"""Write a DiT-weighted imatrix for llama-quantize.

llama.cpp's k-quant search weights each input column j of a row by imatrix_j (times a term from the weight itself), and
only the relative values within a tensor matter. The stock imatrix is mean x_j^2 over calibration text: what the LLM
sees. sens_fisher.py --fimat collects sum_t ||dL/dy_t||^2 x_tj^2 over the DiT-ruler probes: the same statistic, with
each token weighted by how much the DiT ruler cares about that token's output.

Per tensor, both vectors are scaled to mean 1, the ruler targets are combined with their weights, then
    w = (1 - lam) * stock + lam * dit
lam = 0 reproduces the stock imatrix, lam = 1 is pure DiT weighting.

    python tools/fimat_to_gguf.py fimat.npz --base imat.gguf --weights zimage.real=1,klein.real=1,klein.pad=0.5 \
        --lam 0.5 --out imat_dit50.gguf
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
import gguf  # noqa: E402

HF_TO_GGUF = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
              "mlp.down_proj": "ffn_down"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fimat")
    ap.add_argument("--base", required=True, help="stock llama-imatrix GGUF (also supplies tensors fimat lacks)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    W = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    r = gguf.GGUFReader(a.base)
    base = {t.name: np.array(t.data, dtype=np.float32).reshape(-1) for t in r.tensors}
    fz = np.load(a.fimat)
    dit = {}
    for key in fz.files:
        k, t = key.split("|")
        if t in W:
            v = fz[key].astype(np.float64)
            dit.setdefault(k, {})[t] = v
    # each target contributes by its weight; within a target, tensors keep their relative scale
    tot = {t: sum(d[t].sum() for d in dit.values() if t in d) for t in W}
    out = {}
    for k, d in dit.items():
        v = sum(W[t] * d[t] / tot[t] for t in d if tot.get(t))
        i, hf = k.split(".", 1)
        name = "blk.%s.%s.weight" % (i, HF_TO_GGUF[hf])
        s2, c = base[name + ".in_sum2"], base[name + ".counts"]
        stock = s2 / max(float(c[0]), 1.0)
        w = (1 - a.lam) * stock / stock.mean() + a.lam * v / max(v.mean(), 1e-30)
        out[name] = np.maximum(w, 1e-12).astype(np.float32)   # tensors the DiT never reads keep a positive weight

    wr = gguf.GGUFWriter(a.out, arch="imatrix")
    wr.add_type("imatrix")
    wr.add_array("imatrix.datasets", ["dit-weighted lam=%.2f from %s" % (a.lam, os.path.basename(a.fimat))])
    wr.add_uint32("imatrix.chunk_count", 1)
    wr.add_uint32("imatrix.chunk_size", 512)
    n = 0
    for name in sorted({t.rsplit(".", 1)[0] for t in base}):
        if name in out:
            s2, c = out[name], np.array([1.0], dtype=np.float32)
            n += 1
        else:
            s2, c = base[name + ".in_sum2"], base[name + ".counts"]
        wr.add_tensor(name + ".in_sum2", s2)
        wr.add_tensor(name + ".counts", c)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    print("%s: %d tensors DiT-weighted (lam %.2f), %d kept from %s" % (
        a.out, n, a.lam, len({t.rsplit(".", 1)[0] for t in base}) - n, os.path.basename(a.base)))


if __name__ == "__main__":
    main()
