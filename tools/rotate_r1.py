"""R1 residual rotation (QuaRot/SpinQuant style, offline) for a Qwen3 HF checkpoint.

Folds every RMSNorm scale into the linear layers after it, then rotates the residual stream by an orthogonal R
(randomized Hadamard: diag(+-1) @ (H_a (x) H_2^k) / sqrt(n)). The result is the same function up to the rotation:
  embed' = E R;  q/k/v/gate/up' = (W * g) R;  o/down' = R^T W;  lm_head' = (E_or_head * g_final) R;  norms -> 1.
The LM head is untied (tie_word_embeddings=false) because the final-norm scale is folded into it.
Raw hidden states of the rotated model are h R: anything reading them (a DiT) must multiply by R^T. R is saved
next to the weights as rotation.pt; point sens_dit --rot at it.

  python tools/rotate_r1.py ~/v6_hf ~/qv6/rot_hf [--seed 0]
"""
import argparse, json, os, shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def paley_hadamard(q):
    """Paley I: Hadamard matrix of order q+1 for a prime q = 3 mod 4."""
    chi = [0] * q
    for x in range(1, q):
        chi[x * x % q] = 1
    chi = [0] + [1 if chi[i] else -1 for i in range(1, q)]
    Q = torch.tensor([[chi[(j - i) % q] for j in range(q)] for i in range(q)], dtype=torch.float64)
    S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = Q
    return S + torch.eye(q + 1, dtype=torch.float64)


def sylvester(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def hadamard(n):
    for a, q in ((1, None), (12, 11), (20, 19), (28, None)):
        if n % a == 0 and (n // a) & (n // a - 1) == 0:
            base = torch.ones(1, 1, dtype=torch.float64) if a == 1 else paley_hadamard(q)
            H = torch.kron(base, sylvester(n // a))
            assert torch.allclose(H @ H.T, n * torch.eye(n, dtype=torch.float64))
            return H
    raise SystemExit("no Hadamard construction for n=%d" % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=["hadamard", "orth"], default="hadamard")
    a = ap.parse_args()
    src, dst = os.path.expanduser(a.src), os.path.expanduser(a.dst)
    cfg = json.load(open(os.path.join(src, "config.json")))
    n, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    g = torch.Generator().manual_seed(a.seed)
    if a.kind == "hadamard":
        s = torch.randint(0, 2, (n,), generator=g).double() * 2 - 1
        R = s[:, None] * hadamard(n) / n ** 0.5
    else:
        R, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    assert torch.allclose(R @ R.T, torch.eye(n, dtype=torch.float64), atol=1e-10)

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    files = sorted(set(idx.values()))
    handles = {f: safe_open(os.path.join(src, f), "pt") for f in files}
    get = lambda k: handles[idx[k]].get_tensor(k)
    dtype = get("model.embed_tokens.weight").dtype
    Rf = R.float()

    os.makedirs(dst, exist_ok=True)
    new_map = {}

    def out(shard, tensors):
        save_file({k: v.to(dtype).contiguous() for k, v in tensors.items()}, os.path.join(dst, shard),
                  metadata={"format": "pt"})
        for k in tensors:
            new_map[k] = shard

    E = get("model.embed_tokens.weight").float()
    head = get("lm_head.weight").float() if "lm_head.weight" in idx else E
    gf = get("model.norm.weight").float()
    out("model-emb.safetensors", {"model.embed_tokens.weight": E @ Rf,
                                  "lm_head.weight": (head * gf) @ Rf,
                                  "model.norm.weight": torch.ones_like(gf)})
    del E, head
    for i in range(L):
        p = "model.layers.%d." % i
        t = {}
        g1 = get(p + "input_layernorm.weight").float()
        g2 = get(p + "post_attention_layernorm.weight").float()
        for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            t[p + m + ".weight"] = (get(p + m + ".weight").float() * g1) @ Rf
        for m in ("mlp.gate_proj", "mlp.up_proj"):
            t[p + m + ".weight"] = (get(p + m + ".weight").float() * g2) @ Rf
        for m in ("self_attn.o_proj", "mlp.down_proj"):
            t[p + m + ".weight"] = Rf.T @ get(p + m + ".weight").float()
        t[p + "input_layernorm.weight"] = torch.ones_like(g1)
        t[p + "post_attention_layernorm.weight"] = torch.ones_like(g2)
        for k in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):   # per-head, on q/k outputs: untouched
            t[p + k] = get(p + k)
        left = [k for k in idx if k.startswith(p) and k not in t]
        assert not left, left
        out("model-l%02d.safetensors" % i, t)
        print("layer", i, flush=True)

    missing = [k for k in idx if k not in new_map]
    assert not missing, missing
    json.dump({"metadata": {}, "weight_map": new_map}, open(os.path.join(dst, "model.safetensors.index.json"), "w"),
              indent=1)
    cfg["tie_word_embeddings"] = False
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    for f in os.listdir(src):
        if f.endswith((".json", ".jinja", ".txt", ".model")) and f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(src, f), dst)
    torch.save({"R": R, "kind": a.kind, "seed": a.seed, "src": src}, os.path.join(dst, "rotation.pt"))
    print("done ->", dst)


if __name__ == "__main__":
    main()
