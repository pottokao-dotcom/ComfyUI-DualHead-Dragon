"""Download only some tensors of a safetensors file on the HF hub (HTTP range requests), e.g. a DiT's text input stage
for the DiT ruler, without pulling a 20 GB checkpoint.

    python tools/fetch_tensors.py Comfy-Org/Boogu-Image diffusion_models/boogu_image_turbo_bf16.safetensors \
        --match time_caption_embed --out boogu_caption.safetensors
    python tools/fetch_tensors.py ... --list      # print tensor names / dtypes / shapes only
"""
import argparse
import json
import re
import struct

import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers


def header(url, h):
    n = struct.unpack("<Q", requests.get(url, headers=dict(h, Range="bytes=0-7"), timeout=60).content)[0]
    hdr = json.loads(requests.get(url, headers=dict(h, Range="bytes=8-%d" % (7 + n)), timeout=60).content)
    return 8 + n, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("file")
    ap.add_argument("--match", default="", help="regex on tensor names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    url = hf_hub_url(a.repo, a.file)
    h = build_hf_headers()
    base, hdr = header(url, h)
    hdr.pop("__metadata__", None)
    keys = sorted(k for k in hdr if re.search(a.match, k))
    for k in keys:
        print("%-70s %-8s %s" % (k, hdr[k]["dtype"], hdr[k]["shape"]))
    if a.list or not a.out:
        return
    out_hdr, blobs, off = {}, [], 0
    for k in keys:
        s, e = hdr[k]["data_offsets"]
        b = requests.get(url, headers=dict(h, Range="bytes=%d-%d" % (base + s, base + e - 1)), timeout=600).content
        assert len(b) == e - s, (k, len(b), e - s)
        out_hdr[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    js = json.dumps(out_hdr).encode()
    js += b" " * (-len(js) % 8)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(js)) + js)
        for b in blobs:
            f.write(b)
    print("wrote %s (%d tensors, %.1f MB)" % (a.out, len(keys), off / 1e6))


if __name__ == "__main__":
    main()
