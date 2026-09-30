# Correctness of the tree verify loop: per question, accepted draft tokens reported by dh_spec must equal what the
# same tree (rebuilt offline from the dumped lattice) accepts against the real output.
#   python check.py NAME --tree 7 [--n-max 7]      (NAME from: bench.py NAME --tree 7 --lat)
# A question may differ by exactly 1 when its last round ended on the end-of-turn token (not in the saved output).
import argparse, collections
from latio import load, build_tree, walk, chain
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--tree", type=int, default=0); ap.add_argument("--n-max", type=int, default=7)
a = ap.parse_args()
R, outs = load(a.name)
sim = collections.Counter()
for r in R:
    if a.tree:
        par, dep, tok = build_tree(r, a.tree, 5, a.n_max); sim[r["q"]] += walk(par, tok, r["truth"])
    else:
        c = chain(r, a.n_max); n = 0
        while n < len(c) and n < len(r["truth"]) and c[n] == r["truth"][n]: n += 1
        sim[r["q"]] += n
d = collections.Counter(o["accepted"] - sim[q] for q, o in enumerate(outs))
print(f"{a.name}: 題數 {len(outs)},實機接受 − 離線重建 的差 → {dict(sorted(d.items()))}(0 = 一致;1 = 最後一輪收在結尾符號)")
print("PASS" if set(d) <= {0, 1} else "FAIL")
