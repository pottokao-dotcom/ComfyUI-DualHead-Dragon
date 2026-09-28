"""Summarize llama-perplexity --kl-divergence logs (PE / rewrite ruler): mean KLD, 99% KLD, same-top-token %.

    python tools/pe_kl_summary.py ~/q8b/pe_kl_*.log [--json out.json]
"""
import json
import os
import re
import sys


def parse(p):
    t = open(p, errors="replace").read()
    g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, t))  # noqa: E731
    return {"mean_kld": g(r"Mean\s+KLD:\s+([0-9.]+)"), "kld_99": g(r"99\.0%\s+KLD:\s+([0-9.]+)"),
            "same_top_pct": g(r"Same top p:\s+([0-9.]+)")}


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out:
        args.remove(out)
    res = {re.sub(r"^pe_kl_|\.log$", "", os.path.basename(p)): parse(p) for p in args}
    for k, v in sorted(res.items(), key=lambda kv: kv[1]["mean_kld"] or 9):
        print("%-28s meanKLD %-9s 99%%KLD %-9s top1 %s%%" % (k, v["mean_kld"], v["kld_99"], v["same_top_pct"]))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
