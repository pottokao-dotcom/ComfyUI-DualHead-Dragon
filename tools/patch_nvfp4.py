"""Re-quantize every NVFP4 tensor of a llama-quantize output in place with nvfp4q (imatrix-weighted scale search).
The mixed-precision layout (which tensors are NVFP4 / Q8_0 / F32) comes from llama-quantize; only the NVFP4 payloads
are replaced, byte-for-byte the same size, so llama.cpp loads the file exactly as before.

    patch_nvfp4.py SRC_BF16.gguf TARGET.gguf IMATRIX.gguf [radius]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "llama.cpp", "gguf-py"))
import numpy as np, torch, gguf
from gguf.quants import dequantize
import nvfp4q

src_path, dst_path, imat_path = sys.argv[1:4]
radius = int(sys.argv[4]) if len(sys.argv) > 4 else 3

src = {t.name: t for t in gguf.GGUFReader(src_path).tensors}
it = {x.name: np.array(x.data, dtype=np.float32).reshape(-1) for x in gguf.GGUFReader(imat_path).tensors}


def imatrix(name):
    s, c = it.get(name + ".in_sum2"), it.get(name + ".counts")
    return None if s is None else torch.from_numpy(s / float(c.reshape(-1)[0]))


def as_float(t):
    a = np.asarray(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16).float()
    if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        return torch.from_numpy(a.astype(np.float32))
    raise SystemExit("source tensor %s is %s; need BF16/F16/F32" % (t.name, t.tensor_type.name))


dst = gguf.GGUFReader(dst_path)
targets = [t for t in dst.tensors if t.tensor_type == gguf.GGMLQuantizationType.NVFP4]
print("NVFP4 tensors to patch:", len(targets), flush=True)
t0 = time.time()
no_imat = 0
with open(dst_path, "r+b") as f:
    for i, t in enumerate(targets):
        w = as_float(src[t.name])
        n = int(t.shape[0])  # ggml ne0 = row length
        w = w.reshape(-1, n)
        im = imatrix(t.name)
        no_imat += im is None
        packed, _ = nvfp4q.quantize(w, im, radius=radius)
        blob = packed.reshape(-1).tobytes()
        if len(blob) != t.n_bytes:
            raise SystemExit("size mismatch for %s: %d vs %d" % (t.name, len(blob), t.n_bytes))
        f.seek(t.data_offset)
        f.write(blob)
        if i % 40 == 0:
            print("  %d/%d %s (%.0fs)" % (i, len(targets), t.name, time.time() - t0), flush=True)
print("patched %d tensors (%d without imatrix) in %.0fs" % (len(targets), no_imat, time.time() - t0))

# spot check: re-read and dequantize a few patched tensors
chk = {t.name: t for t in gguf.GGUFReader(dst_path).tensors}
for t in targets[:: max(1, len(targets) // 4)]:
    n = int(t.shape[0]); w = as_float(src[t.name]).reshape(-1, n)
    d = torch.from_numpy(dequantize(np.asarray(chk[t.name].data), gguf.GGMLQuantizationType.NVFP4).reshape(-1, n)).float()
    print("  check %-28s rel err %.4f" % (t.name, float((d - w).norm() / w.norm())))
