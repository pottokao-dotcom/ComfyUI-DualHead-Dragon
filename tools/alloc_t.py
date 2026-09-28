"""Tensor-level format allocation from sens_fisher.py: the size-vs-damage curve and a GGUF type map for a budget.

Every tensor starts at the floor format; upgrades are bought greedily by damage removed per extra byte (across all
tensors and formats) until the budget runs out. Damage per target is normalized by its all-floor total and weighted,
so one file can serve several products (Z-Image + klein share Qwen3-4B). Predicted 1 - cos is also reported per target.

    python tools/alloc_t.py fisher_4b.json --te qwen_3_4b.safetensors --curve curve.json
    python tools/alloc_t.py fisher_4b.json --te qwen_3_4b.safetensors --budget-gb 2.5 --out cfg_t_2p5.json
"""
import argparse
import heapq
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "tools"))
from fakequant import tensor_bytes  # noqa: E402
from safetensors import safe_open  # noqa: E402

HF_TO_GGUF = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
              "mlp.down_proj": "ffn_down"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fisher")
    ap.add_argument("--te", help="safetensors holding model.embed_tokens.weight (for its size)")
    ap.add_argument("--embd-shape", help="rows,cols of token_embd instead of --te")
    ap.add_argument("--floor", default="q4_k")
    ap.add_argument("--pe-floor", default="q4_k", help="minimum for tensors no DiT reads (only the rewrite uses them)")
    ap.add_argument("--weights", default="zimage.real=1,klein.real=1,klein.pad=0.5")
    ap.add_argument("--embd", default="q6_k", help="token_embd type (not measured)")
    ap.add_argument("--budget-gb", type=float)
    ap.add_argument("--out", help="GGUF type map json for --budget-gb")
    ap.add_argument("--curve", help="write the whole size-vs-damage curve here")
    ap.add_argument("--only", help="comma list: the only formats a tensor may take (e.g. K series without IQ)")
    a = ap.parse_args()

    fi = json.load(open(a.fisher))
    fmts = [f for f in fi["formats"] if not a.only or f in a.only.split(",")]
    order = sorted(fmts, key=lambda f: sum(b[f] for b in fi["bytes"].values()))   # cheapest first
    lo = order.index(a.floor)
    up = order[lo:]
    W = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    targets = [t for t in fi["targets"] if t in W]
    B = fi["bytes"]
    keys = sorted(B)
    # tensors after a product's read-out layer get no gradient there -> no entry = zero damage for that target
    D = {k: {t: {f: fi["damage"].get(k, {}).get(t, {}).get(f, 0.0) for f in fmts} for t in targets} for k in keys}
    norm = {t: sum(D[k][t][a.floor] for k in keys) for t in targets}

    def dmg(k, f):
        return sum(W[t] * D[k][t][f] / norm[t] for t in targets)

    if a.embd_shape:
        shp = tuple(int(x) for x in a.embd_shape.split(","))
    else:
        shp = tuple(safe_open(a.te, "pt").get_slice("model.embed_tokens.weight").get_shape())
    embd = tensor_bytes(shp, a.embd)
    dit_zero = {k for k in keys if all(D[k][t][f] == 0.0 for t in targets for f in fmts)}
    choice = {k: (a.pe_floor if k in dit_zero else a.floor) for k in keys}
    size = embd + sum(B[k][choice[k]] for k in keys)
    tot = {t: sum(D[k][t][choice[k]] for k in keys) for t in targets}
    print("%d tensors no DiT reads -> %s" % (len(dit_zero), a.pe_floor))

    def best_move(k):
        """most damage removed per byte among this tensor's upgrades from its current format"""
        cur = choice[k]
        out = None
        for g in up[up.index(cur) + 1:]:
            cost = B[k][g] - B[k][cur]
            gain = dmg(k, cur) - dmg(k, g)
            if cost > 0 and gain > 0 and (out is None or gain / cost > out[0]):
                out = (gain / cost, g, cost)
        return out

    heap = []
    for k in keys:
        m = best_move(k)
        if m:
            heapq.heappush(heap, (-m[0], k, m[1], m[2]))
    curve = [{"gb": round(size / 1e9, 4), **{t: tot[t] for t in targets}}]
    budget = a.budget_gb * 1e9 if a.budget_gb else float("inf")
    while heap:
        _, k, g, cost = heapq.heappop(heap)
        if choice[k] == g or up.index(g) <= up.index(choice[k]):
            continue
        if size + cost > budget:
            continue   # something smaller may still fit
        for t in targets:
            tot[t] += D[k][t][g] - D[k][t][choice[k]]
        choice[k] = g
        size += cost
        curve.append({"gb": round(size / 1e9, 4), **{t: tot[t] for t in targets}})
        m = best_move(k)
        if m:
            heapq.heappush(heap, (-m[0], k, m[1], m[2]))

    if a.curve:
        json.dump({"floor": a.floor, "weights": W, "curve": curve}, open(a.curve, "w"))
        pts = [c for i, c in enumerate(curve) if i % max(1, len(curve) // 12) == 0] + [curve[-1]]
        for c in pts:
            print("%.3f GB  %s" % (c["gb"], "  ".join("%s %.5f" % (t, c[t]) for t in targets)))
    if a.budget_gb:
        from collections import Counter
        tmap = {"token_embd.weight": a.embd.upper()}
        for k, g in choice.items():
            i, hf = k.split(".", 1)
            tmap["blk.%s.%s.weight" % (i, HF_TO_GGUF[hf])] = g.upper()
        if a.out:
            json.dump(tmap, open(a.out, "w"), indent=0)
        print("budget %.3f GB -> %.3f GB  %s  formats %s" % (a.budget_gb, size / 1e9, "  ".join(
            "%s %.5f" % (t, tot[t]) for t in targets), dict(Counter(choice.values()))))
        by_type = {}
        for k, g in choice.items():
            by_type.setdefault(k.split(".", 1)[1], Counter())[g] += 1
        for hf, c in by_type.items():
            print("  %-18s %s" % (hf, dict(c)))


if __name__ == "__main__":
    main()
