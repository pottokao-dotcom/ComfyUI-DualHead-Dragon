"""Fake quantization with the exact numbers a GGUF would hold: ggml's own quantizer (ggml_quantize_chunk, imatrix
included, i.e. what llama-quantize writes) or tools/nvfp4q.py, dequantized back to float. Shared by sens_dit.py and
sens_pe.py. Needs DH_LIB (dir with libggml-base.so from build.py) and vendor/llama.cpp/gguf-py on sys.path."""
import ctypes
import os

import numpy as np
import torch

import gguf
from gguf.quants import dequantize

GROUPS = {"attn": [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                   ("self_attn.o_proj", "attn_output")],
          "ffn": [("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down")]}
GGUF_TO_HF = {gg: hf for g in GROUPS.values() for hf, gg in g}

_G = None


def _ggml():
    global _G
    if _G is None:
        _G = ctypes.CDLL(os.path.join(os.environ["DH_LIB"], "libggml-base.so"))
        _G.ggml_quantize_chunk.restype = ctypes.c_size_t
        _G.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                           ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        _G.ggml_row_size.restype = ctypes.c_size_t
        _G.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    return _G


def fake_quant(w, qtype, im):
    """w: (rows, n) tensor, im: (n,) float32 numpy imatrix or None -> dequantized float32 tensor on w's device.
    qtype: any GGML type name (q4_k, iq4_xs, q8_0, ...), f16 / bf16 / f32, or nvfp4 (tools/nvfp4q.py)."""
    qtype = qtype.lower()
    if qtype in ("bf16", "f32"):
        return w.float()
    if qtype == "f16":
        return w.half().float()
    if qtype == "nvfp4":
        import nvfp4q
        _, deq = nvfp4q.quantize(w, torch.from_numpy(im) if im is not None else None, radius=3)
        return deq.to(w.device)
    t = int(gguf.GGMLQuantizationType[qtype.upper()])
    x = np.ascontiguousarray(w.float().cpu().numpy())
    rows, n = x.shape
    rs = _ggml().ggml_row_size(t, n)
    buf = np.empty(rows * rs, dtype=np.uint8)
    imp = np.ascontiguousarray(im, dtype=np.float32) if im is not None else None
    _ggml().ggml_quantize_chunk(t, x.ctypes.data, buf.ctypes.data, 0, rows, n,
                                imp.ctypes.data if imp is not None else None)
    deq = dequantize(buf.reshape(rows, rs), gguf.GGMLQuantizationType(t)).reshape(rows, n)
    return torch.from_numpy(np.ascontiguousarray(deq)).to(w.device)


def imatrix_hooks(modules):
    """Register hooks collecting mean x^2 per input column (what llama-imatrix computes). Returns (acc, finish);
    finish() removes the hooks and returns {key: numpy imatrix}."""
    acc = {}

    def hook(key):
        def f(mod, args, out):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            s = acc.setdefault(key, [torch.zeros(x.shape[1], device=x.device), 0])
            s[0] += (x * x).sum(0)
            s[1] += x.shape[0]
        return f
    hs = [m.register_forward_hook(hook(k)) for k, m in modules.items()]

    def finish():
        for h in hs:
            h.remove()
        return {k: (s / n).cpu().numpy() for k, (s, n) in acc.items()}
    return acc, finish


def load_type_map(path):
    """GGUF tensor-type map (json {gguf tensor name: type}) -> {key: type} with the keys sens_dit / sens_pe use:
    (layer, hf linear name), ("embed", "token_embd"), ("lm_head", "output")."""
    import json
    import re
    out = {}
    for name, t in json.load(open(path)).items():
        m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", name)
        if m and m.group(2) in GGUF_TO_HF:
            out[(int(m.group(1)), GGUF_TO_HF[m.group(2)])] = t
        elif name == "token_embd.weight":
            out[("embed", "token_embd")] = t
        elif name == "output.weight":
            out[("lm_head", "output")] = t
    return out


def tensor_bytes(shape, qtype):
    """bytes a (rows, n) tensor takes in a GGUF of this type."""
    rows, n = shape
    q = qtype.lower()
    if q in ("f16", "bf16"):
        return rows * n * 2
    if q == "f32":
        return rows * n * 4
    if q == "nvfp4":
        return rows * n // 64 * 36
    return rows * _ggml().ggml_row_size(int(gguf.GGMLQuantizationType[q.upper()]), n)
