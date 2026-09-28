"""Check sens_fisher.py against the measured per-layer Q4_K sweep of sens_dit.py.

Sums the predicted per-tensor damage into sens_dit's (group, layer) units and compares with the measured 1 - cos
after the DiT input stage: Spearman rank correlation (does it pick the same layers?) and the ratio of totals
(is the scale right?), per target (zimage.real, klein.real, klein.pad).

    python tools/val_fisher.py fisher_4b.json docs/sens/sens_4b_ffn.json docs/sens/sens_4b_attn.json [--fmt q4_k]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GROUPS = {"attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
          "ffn": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}


def rank(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fisher")
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("--fmt", default="q4_k")
    a = ap.parse_args()
    fi = json.load(open(a.fisher))
    sweep = [s for p in a.sweeps for s in json.load(open(p))["sweep"]]
    for t in fi["targets"]:
        prod, part = t.split(".")
        pred, meas, lab = [], [], []
        for s in sweep:
            if prod not in s or part not in s[prod]:
                continue
            p = sum(fi["damage"].get("%d.%s" % (s["layer"], hf), {}).get(t, {}).get(a.fmt, 0.0)
                    for hf in GROUPS[s["group"]])
            pred.append(p)
            meas.append(1 - s[prod][part]["dit"])
            lab.append("%s%d" % (s["group"], s["layer"]))
        if not pred:
            continue
        top_m = [lab[i] for i in sorted(range(len(meas)), key=lambda i: -meas[i])[:8]]
        top_p = [lab[i] for i in sorted(range(len(pred)), key=lambda i: -pred[i])[:8]]
        print("%-12s n=%d  Spearman %.3f  total pred/meas %.2f" % (t, len(pred), spearman(pred, meas),
                                                                   sum(pred) / max(sum(meas), 1e-12)))
        print("   measured top8:  %s" % " ".join(top_m))
        print("   predicted top8: %s" % " ".join(top_p))
        print("   top8 overlap %d/8" % len(set(top_m) & set(top_p)))


if __name__ == "__main__":
    main()
