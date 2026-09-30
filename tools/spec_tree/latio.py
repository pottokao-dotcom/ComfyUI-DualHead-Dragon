# Reader for lat_NAME.bin (dh_spec DH_SPEC_LAT dump) + out_NAME.json (bench.py).
# Record: int32 [0x4C415431, pos0, id_last, n_block, 0, top_k] + float32[n_block][top_k * (1 + top_k)]
# (per row: top_k candidate ids, then scores[pred][k]). A new question starts where pos0 goes down.
import json, struct, numpy as np

def load(name):
    """name as given to bench.py; may carry a directory (dir/NAME -> dir/lat_NAME.bin, dir/out_NAME.json)."""
    import os
    d, n = os.path.split(name)
    raw = open(os.path.join(d, f"lat_{n}.bin"), "rb").read(); outs = json.load(open(os.path.join(d, f"out_{n}.json")))
    R = []; off = 0; q = 0
    while off < len(raw):
        mg, pos0, idl, nb, _, K = struct.unpack_from("6i", raw, off); off += 24
        assert mg == 0x4C415431, "bad record"
        a = np.frombuffer(raw, np.float32, nb * K * (1 + K), off).reshape(nb, K * (1 + K)); off += a.nbytes
        if R and pos0 < R[-1]["pos0"]: q += 1
        R.append(dict(q=q, pos0=pos0, id_last=idl, nb=nb, K=K, cand=a[:, :K].astype(np.int64), sc=a[:, K:].reshape(nb, K, K)))
    for r in R:   # target's real tokens after the anchor (sampling: one draw, independent of how we bet)
        o = outs[r["q"]]; st = r["pos0"] + 1 - o["n_prompt"]; r["truth"] = o["out"][st:st + r["nb"] - 1]
    return R, outs

def build_tree(r, budget, max_leaves=5, max_depth=None):
    """Same as native/dh_tree.cpp: best-first by path probability. Returns (parent, depth, tok) lists, parent 0 = root."""
    import heapq
    D = r["nb"] - 1 if max_depth is None else min(max_depth, r["nb"] - 1)
    nd_d = [0]; nd_k = [0]; kids = [0]; leaves = 1; par = []; dep = []; tok = []; heap = []; c = 0
    def expand(node, p):
        nonlocal c
        d = nd_d[node] + 1
        if d > D: return
        sc = r["sc"][d][nd_k[node]].astype(np.float64); e = np.exp(sc - sc.max()); pr = p * e / e.sum()
        for k in range(len(pr)):
            if pr[k] > 1e-4: c += 1; heapq.heappush(heap, (-pr[k], c, d, k, node))
    expand(0, 1.0)
    while heap and len(tok) < budget:
        npk, _, d, k, pa = heapq.heappop(heap)
        nl = kids[pa] > 0
        if nl and leaves >= max_leaves: continue
        leaves += nl; kids[pa] += 1; nd_d.append(d); nd_k.append(k); kids.append(0)
        par.append(pa); dep.append(d); tok.append(int(r["cand"][d][k]))
        expand(len(nd_d) - 1, -npk)
    return par, dep, tok

def walk(par, tok, truth):
    """How many draft tokens the target accepts: follow the child holding the real next token."""
    cur, a = 0, 0
    while a < len(truth):
        nx = next((j + 1 for j in range(len(tok)) if par[j] == cur and tok[j] == truth[a]), None)
        if nx is None: break
        cur = nx; a += 1
    return a

def chain(r, k):
    """The plain greedy draft line, first k tokens."""
    pred = 0; out = []
    for d in range(1, min(k, r["nb"] - 1) + 1):
        pred = int(np.argmax(r["sc"][d][pred])); out.append(int(r["cand"][d][pred]))
    return out
