# Offline: how far each betting shape gets per round on recorded rounds (bench.py NAME --lat, any strategy).
# Sampling is fine: the target's token at each position does not depend on how we bet, so the saved output is
# a fair draw for every shape. Rounds are replayed from the recorded anchors (a shape that advances differently
# would see different anchors next round - the usual per-round approximation).
#   python sim.py NAME [--cost cost.txt --base 57.3]
# --cost: "draft=..." / "rest=..." lines, cost of drafting / verifying k tokens relative to one plain decode
# (trace/policy_cost_ratio.txt format); with it the table also estimates tok/s. Without, compare advance only
# and measure speed with bench.py on the card instead (that is what the numbers in docs/SPEC_STRATEGY.md are).
import argparse, collections
from latio import load, build_tree, walk, chain
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--cost"); ap.add_argument("--base", type=float, default=57.3)
ap.add_argument("--tree-oh", type=float, default=0.03, help="extra cost per tree round, in plain decodes (measured ~0.5 ms / 17.5 ms)")
a = ap.parse_args()
R, _ = load(a.name); R = [r for r in R if len(r["truth"]) == r["nb"] - 1]
D = R[0]["nb"] - 1
cost = None
if a.cost:
    cf = dict(l.strip().split("=", 1) for l in open(a.cost) if "=" in l and not l.startswith("#"))
    dr = [float(x) for x in cf["draft"].split(",")]; rs = [float(x) for x in cf["rest"].split(",")]
    cost = lambda n, tree: dr[D - 1] + rs[min(n, len(rs)) - 1] + (a.tree_oh if tree else 0.0)
def line(label, adv, n, tree):
    s = f"{label:14s} 每輪前進 {adv:.3f}"
    if cost: s += f" | 估 {adv / cost(n, tree) * a.base:.1f} tok/s"
    print(s)
print(f"{len(R)} 輪,草稿每輪算 {D} 格")
for k in sorted({3, 5, 7, D} & set(range(1, D + 1))):
    adv = sum(1 + next((i for i, (x, y) in enumerate(zip(chain(r, k), r["truth"])) if x != y), k) for r in R) / len(R)
    line(f"主線押 {k}", adv, k, False)
for B in (3, 5, 7, 11, 15, 20):
    tot = 0; shapes = collections.Counter()
    for r in R:
        par, dep, tok = build_tree(r, B, 5, D); tot += 1 + walk(par, tok, r["truth"])
    line(f"樹 預算 {B}", tot / len(R), B, True)
