"""NVFP4 quantizer that writes ggml's block_nvfp4 layout (bit-compatible with llama.cpp) but chooses each 16-value
block's E4M3 scale by search, weighted by the imatrix -- ggml's own quantize_nvfp4 ignores the imatrix and always
takes amax/6.

block_nvfp4 (64 values, 36 bytes): uint8 d[4] (UE4M3 scale per 16) + uint8 qs[32]; in sub-block s, byte
qs[s*8+j] = idx[j] | idx[j+8] << 4. value = kvalues_mxfp4[idx] * ue4m3_to_fp32(d), ue4m3_to_fp32 includes a 0.5.
With radius=0 the output is byte-identical to ggml's quantize_row_nvfp4_ref.
"""
import numpy as np
import torch

KV = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32)


def _ue4m3_table():
    t = torch.zeros(128, dtype=torch.float32)
    for x in range(1, 127):  # 0 and 0x7F decode to 0
        e, m = (x >> 3) & 0xF, x & 0x7
        raw = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        t[x] = raw * 0.5
    return t


UE = _ue4m3_table()


def fp32_to_ue4m3(x):
    """Exact port of ggml_fp32_to_ue4m3 (vectorised, float32 in, uint8 codes out as int32)."""
    x = x.float().clamp(max=448.0)
    bits = x.view(torch.int32)
    fp32_exp = ((bits >> 23) & 0xFF) - 127
    fp32_man = (bits >> 20) & 0x7
    e = fp32_exp + 7
    sub = (x * 512.0 + 0.5).to(torch.int32).clamp(max=7)
    sub = torch.where(sub < 1, torch.zeros_like(sub), sub)
    rb = (bits >> 19) & 1
    man = fp32_man + rb
    ovf = man > 7
    man = torch.where(ovf, torch.zeros_like(man), man)
    e2 = torch.where(ovf, e + 1, e)
    normal = torch.where(e2 >= 15, torch.full_like(e2, 0x7E), (e2 << 3) | man)
    out = torch.where(e <= 0, sub, torch.where(e >= 15, torch.full_like(e, 0x7E), normal))
    return torch.where(x > 0, out, torch.zeros_like(out))


@torch.no_grad()
def quantize(w, imatrix=None, radius=3, device="cuda", rows_per_chunk=None):
    """w: (rows, n) float tensor, n % 64 == 0. imatrix: (n,) per-input-column importance or None.
    Returns (packed uint8 numpy (rows, n//64*36), dequantized float32 torch (rows, n) on cpu)."""
    rows, n = w.shape
    assert n % 64 == 0
    kv, ue = KV.to(device), UE.to(device)
    imw = (imatrix.float().to(device) if imatrix is not None else torch.ones(n, device=device)).view(1, n // 16, 16)
    packed = np.empty((rows, n // 64 * 36), dtype=np.uint8)
    deq = torch.empty(rows, n, dtype=torch.float32)
    offs = torch.arange(-radius, radius + 1, device=device)
    if rows_per_chunk is None:  # keep the (rows, blocks, candidates, 16, 16) error tensor around 1.5e8 elements
        rows_per_chunk = max(1, int(1.5e8 // ((n // 16) * len(offs) * 256)))
    for r0 in range(0, rows, rows_per_chunk):
        x = w[r0:r0 + rows_per_chunk].to(device).float()
        R = x.shape[0]
        xb = x.view(R, n // 16, 16)                                   # 16-value sub-blocks
        c0 = fp32_to_ue4m3(xb.abs().amax(-1) / 6.0)                   # ggml's choice
        cand = (c0.unsqueeze(-1) + offs).clamp(0, 0x7E)               # (R, nb, C)
        cand = torch.where((c0 == 0).unsqueeze(-1), torch.zeros_like(cand), cand)
        d = ue[cand]                                                  # (R, nb, C)
        vals = kv.view(1, 1, 1, 1, 16) * d.unsqueeze(-1).unsqueeze(-1)  # (R, nb, C, 1, 16)
        err_all = (vals - xb.unsqueeze(2).unsqueeze(-1)).abs()        # (R, nb, C, 16, 16)
        idx = err_all.argmin(-1)                                      # first minimum, like best_index_mxfp4
        q = torch.gather(vals.expand(-1, -1, -1, 16, -1), -1, idx.unsqueeze(-1)).squeeze(-1)  # (R, nb, C, 16)
        e = ((q - xb.unsqueeze(2)) ** 2 * imw.unsqueeze(2)).sum(-1)   # weighted error per candidate
        best = e.argmin(-1)
        # prefer ggml's own choice on exact ties so radius>0 never does worse than RTN
        e0 = e[..., radius]
        bestv = torch.gather(e, -1, best.unsqueeze(-1)).squeeze(-1)
        best = torch.where(bestv < e0, best, torch.full_like(best, radius))
        code = torch.gather(cand, -1, best.unsqueeze(-1)).squeeze(-1)                     # (R, nb)
        qi = torch.gather(idx, 2, best.view(R, -1, 1, 1).expand(-1, -1, 1, 16)).squeeze(2)  # (R, nb, 16)
        qv = torch.gather(q, 2, best.view(R, -1, 1, 1).expand(-1, -1, 1, 16)).squeeze(2)
        deq[r0:r0 + R] = qv.reshape(R, n).cpu()
        # pack: per 64-value block -> d[4] + qs[32]
        code = code.view(R, n // 64, 4).to(torch.uint8)
        qi = qi.view(R, n // 64, 4, 16).to(torch.uint8)
        qs = (qi[..., :8] | (qi[..., 8:] << 4)).reshape(R, n // 64, 32)
        blk = torch.cat([code, qs], dim=-1).reshape(R, -1)
        packed[r0:r0 + R] = blk.cpu().numpy()
    return packed, deq
