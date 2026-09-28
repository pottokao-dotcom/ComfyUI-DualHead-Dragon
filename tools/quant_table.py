"""Build the TE quantization evaluation table (docs/QUANT_EVAL.md) from the sens_dit.py / sens_pe.py json outputs.

    python tools/quant_table.py --dit sens_4b.json cfg_4b.json --pe pe_4b.json --title "Qwen3-4B (Z-Image / klein)" > part.md
"""
import argparse
import json


def pct(x):
    return "%.1f%%" % (100 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", nargs="*", default=[])
    ap.add_argument("--pe", nargs="*", default=[])
    ap.add_argument("--title", required=True)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    rows, prods, sweeps, pe_sweep = {}, [], {}, []
    for path in a.dit:
        d = json.load(open(path))
        for kind in ("ruler", "configs"):
            for name, r in d.get(kind, {}).items():
                row = rows.setdefault(name, {})
                row["gb"] = r.get("gb_scored_tensors", row.get("gb"))
                for p, v in r.items():
                    if isinstance(v, dict) and "real" in v:
                        prods.append(p) if p not in prods else None
                        row[p] = v
        for s in d.get("sweep", []):
            for p, v in s.items():
                if isinstance(v, dict) and "real" in v:
                    sweeps.setdefault(p, []).append((s["group"], s["layer"], v))
    for path in a.pe:
        d = json.load(open(path))
        for kind in ("ruler", "configs"):
            for name, r in d.get(kind, {}).items():
                row = rows.setdefault(name, {})
                row["gb"] = row.get("gb") or r.get("gb_scored_tensors")
                row["pe"] = r
        pe_sweep += [(s["group"], s["layer"], s) for s in d.get("sweep", [])]

    out = ["## %s" % a.title, ""]
    head = ["配置", "大小(GB)"]
    for p in prods:
        head += ["%s rel_dit" % p, "%s nm 1−cos" % p]
        if p == "klein":
            head += ["klein pad rel_dit"]
    head += ["PE KL", "PE 選字不變"]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in sorted(rows.items(), key=lambda kv: -(kv[1].get(prods[0], {}).get("real", {}).get("rel_dit", 0) if prods else 0)):
        cells = [name, "%.2f" % r["gb"] if r.get("gb") else "—"]
        for p in prods:
            v = r.get(p)
            cells += [pct(v["real"]["rel_dit"]) if v else "—", "%.2e" % (1 - v["real"]["nm"]) if v else "—"]
            if p == "klein":
                cells += [pct(v["pad"]["rel_dit"]) if v and "pad" in v else "—"]
        pe = r.get("pe")
        cells += ["%.4f" % pe["kl"] if pe else "—", pct(pe["top1"]) if pe else "—"]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    def ranking(title, items, key, fmt):
        tot = sum(key(x) for x in items) or 1.0
        out.extend(["**%s**（每次只把一組壓成 sweep 格式；佔比 = 該組占全部組總和）" % title, "",
                    "| 名次 | 組 | 層 | 分數 | 佔比 |", "|---|---|---|---|---|"])
        for n, x in enumerate(sorted(items, key=lambda x: -key(x))[:a.top], 1):
            out.append("| %d | %s | %d | %s | %s |" % (n, x[0], x[1], fmt(key(x)), pct(key(x) / tot)))
        g = {k: sum(key(x) for x in items if x[0] == k) / tot for k in ("attn", "ffn")}
        out.extend(["", "attention 合計 %s、FFN 合計 %s" % (pct(g["attn"]), pct(g["ffn"])), ""])

    for p, items in sweeps.items():
        ranking("%s：DiT 量尺排名（1−cos_dit）" % p, items, lambda x: 1 - x[2]["real"]["dit"], lambda v: "%.2e" % v)
        if p == "klein" and "pad" in items[0][2]:
            ranking("klein pad：DiT 量尺排名", items, lambda x: 1 - x[2]["pad"]["dit"], lambda v: "%.2e" % v)
    if pe_sweep:
        ranking("PE 量尺排名（KL）", pe_sweep, lambda x: x[2]["kl"], lambda v: "%.2e" % v)
    print("\n".join(out))


if __name__ == "__main__":
    main()
