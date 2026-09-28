"""Requantize a llama.cpp LoRA GGUF (f16 in, mixed types out). convert_lora_to_gguf only offers f16/bf16/q8_0, and
K-quants (256-value blocks) fit lora_a only -- lora_b rows are just `rank` long, so it needs a 32-block type.

    python tools/lora_requant.py in-f16.gguf out.gguf Q6_K Q8_0   # q6k-mix
    python tools/lora_requant.py in-f16.gguf out.gguf Q4_K Q4_0   # q4k-mix

Uses ggml's ggml_quantize_chunk (the node's dh_llama/lib, or DH_LLAMA_LIB_PATH). Check the result against the f16 file
before shipping (see the QUANT_NOTES.md in the LoRA repos: agreement / KL on master prompts)."""
import os
import sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
import ctypes
import numpy as np, gguf
_LIB = os.environ.get("DH_LLAMA_LIB_PATH") or os.path.join(HERE, "dh_llama", "lib")
G = ctypes.CDLL(os.path.join(_LIB, {"win32": "ggml-base.dll", "darwin": "libggml-base.dylib"}.get(sys.platform, "libggml-base.so")))
G.ggml_quantize_chunk.restype = ctypes.c_size_t
G.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
G.ggml_row_size.restype = ctypes.c_size_t
G.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
def cquant(arr, qt):  # ggml's own quantizer (what llama-quantize uses), rows = arr[0], row length = arr[1]
    arr = np.ascontiguousarray(arr, dtype=np.float32); nrows, n = arr.shape
    rs = G.ggml_row_size(int(qt), n); out = np.empty((nrows, rs), dtype=np.uint8)
    G.ggml_quantize_chunk(int(qt), arr.ctypes.data, out.ctypes.data, 0, nrows, n, None)
    return out
src, dst, qa, qb = sys.argv[1:5]
Q = gguf.GGMLQuantizationType
r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"]
w = gguf.GGUFWriter(dst, bytes(arch.parts[arch.data[0]]).decode())
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1])
    else:
        w.add_key_value(f.name, f.contents(), vt)
w.add_key_value("general.file_type", -1, gguf.GGUFValueType.INT32) if "general.file_type" not in r.fields else None
for t in r.tensors:
    data = np.asarray(t.data)
    if t.tensor_type in (Q.F16, Q.F32) and t.name.endswith((".lora_a", ".lora_b")):
        qt = Q[qa] if t.name.endswith(".lora_a") else Q[qb]
        arr = data.astype(np.float32).reshape([int(x) for x in reversed(t.shape)])
        qd = cquant(arr, qt)
        w.add_tensor(t.name, qd, raw_shape=qd.shape, raw_dtype=qt)
    else:
        w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", dst)
