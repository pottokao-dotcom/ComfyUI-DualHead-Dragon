"""Allocate formats per layer group under a size budget, from measured per-layer sensitivity.

Model (verified on 4B: per-layer Q4_K damages add up to the whole-model damage within 3-4%):
    damage = sum_l D_l(q4_k) * E_l(f) / E_l(q4_k)
with D_l from a sens_dit.py Q4_K sweep and E_l(f) the weight error from tools/weight_err.py. Each product's damage is
normalized by its all-Q4_K sum and weighted, so Z-Image and klein share one file. Upgrades are bought greedily by
damage removed per extra byte until the budget is spent.

    DH_LIB=dh_llama/lib python tools/alloc.py --te qwen_3_4b.safetensors --sweep ffn.json attn.json --werr werr.json \
        --budget-gb 2.49 --up q8_0 --out cfg.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "tools"))
from fakequant import GROUPS, tensor_bytes  # noqa: E402
from safetensors import safe_open  # noqa: E402

# (product, part, weight); klein pad's summed damage overestimates (1-cos saturates), so it counts half
TARGETS = [("zimage", "real", 1.0), ("klein", "real", 1.0), ("klein", "pad", 0.5)]  # pad weight set by --w-pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", required=True)
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--werr", required=True)
    ap.add_argument("--ref", default="q4_k", help="format the sweep was measured in")
    ap.add_argument("--floor", default="q4_k", help="every group starts here")
    ap.add_argument("--up", default="q5_k,q6_k,q8_0", help="formats above the floor, cheapest first")
    ap.add_argument("--pe-floor", default="q4_k", help="minimum for groups no DiT reads (only the rewrite uses them)")
    ap.add_argument("--w-pad", type=float, default=0.5, help="weight of klein pad (its summed damage saturates)")
    ap.add_argument("--embd", default="q6_k", help="token_embd type (not measured; q6_k = what Q4_K_M uses)")
    ap.add_argument("--budget-gb", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    TARGETS[2] = ("klein", "pad", a.w_pad)
    f = safe_open(a.te, "pt")
    shape = {k: f.get_slice(k).get_shape() for k in f.keys()}
    sweep = [s for p in a.sweep for s in json.load(open(p))["sweep"]]
    werr = {(r["group"], r["layer"]): r for r in json.load(open(a.werr))["err"]}
    norm = {(p, part): sum(1 - s[p][part]["dit"] for s in sweep) for p, part, _ in TARGETS}

    def gbytes(g, i, fmt):
        return sum(tensor_bytes(shape["model.layers.%d.%s.weight" % (i, hf)], fmt) for hf, _ in GROUPS[g])

    def dmg(s, fmt):  # weighted, normalized damage of one group in one format
        r = werr[(s["group"], s["layer"])]
        k = r[fmt] / r[a.ref]
        return sum(w * (1 - s[p][part]["dit"]) / norm[(p, part)] * k for p, part, w in TARGETS)

    fmts = [a.floor] + a.up.split(",")
    by_key = {(s["group"], s["layer"]): s for s in sweep}
    dit_zero = {k for k, s in by_key.items() if all(1 - s[p][part]["dit"] <= 0 for p, part, _ in TARGETS)}
    start = lambda k: a.pe_floor if k in dit_zero and fmts.index(a.pe_floor) > 0 else a.floor  # noqa: E731
    if a.pe_floor not in fmts:
        fmts.insert(1, a.pe_floor)
    choice = {k: start(k) for k in by_key}
    embd = tensor_bytes(shape["model.embed_tokens.weight"], a.embd)
    size = embd + sum(gbytes(g, i, f) for (g, i), f in choice.items())
    budget = a.budget_gb * 1e9
    bought = []
    while True:
        best = None
        for key, cur in choice.items():
            s = by_key[key]
            for fmt in fmts[fmts.index(cur) + 1:]:
                cost = gbytes(*key, fmt) - gbytes(*key, cur)
                gain = dmg(s, cur) - dmg(s, fmt)
                if size + cost <= budget and cost > 0 and (best is None or gain / cost > best[0]):
                    best = (gain / cost, key, fmt, cost, gain)
        if best is None:
            break
        _, key, fmt, cost, gain = best
        choice[key] = fmt
        size += cost
        bought.append("%s%d->%s" % (key[0], key[1], fmt))

    total = sum(dmg(by_key[k], v) for k, v in choice.items())
    tmap = {"token_embd.weight": a.embd.upper()}
    for (g, i), fmt in choice.items():
        for hf, gg in GROUPS[g]:
            tmap["blk.%d.%s.weight" % (i, gg)] = fmt.upper()
    json.dump(tmap, open(a.out, "w"), indent=0)
    from collections import Counter
    print("%s: %.3f GB, predicted weighted damage %.3f of all-%s (sum of weights %.1f)  formats %s" % (
        os.path.basename(a.out), size / 1e9, total, a.ref, sum(w for *_, w in TARGETS), dict(Counter(choice.values()))))
    for g in ("attn", "ffn"):
        print("  %-4s %s" % (g, " ".join("%d:%s" % (i, choice[(g, i)].replace("_", "")) for i in range(len(choice) // 2))))


if __name__ == "__main__":
    main()
