"""Embedded llama.cpp for the dual-head dragon.

Python bindings: llama-cpp-python v0.3.35 (vendored, relative imports). Libraries: vendor/llama.cpp at the commit those
bindings were written for, built into ./lib by build.py. Nothing here imports or loads a llama-cpp-python installed in
the environment, so other nodes and pip upgrades cannot swap the library under us.
"""
import ctypes
import glob
import json
import os
import sys

LIB_DIR = os.environ.get("DH_LLAMA_LIB_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

if not glob.glob(os.path.join(LIB_DIR, "*llama*")):
    raise ImportError("embedded llama.cpp is not built yet: run `python build.py` in the ComfyUI-DualHeadDragon folder "
                      "(expected libraries in %s)" % LIB_DIR)

if sys.platform.startswith("linux"):
    # load our ggml first, by path, so libllama/libmtmd resolve their dependencies to these copies
    for _name in ("ggml-base", "ggml-cpu", "ggml-cuda", "ggml-vulkan", "ggml"):
        _p = os.path.join(LIB_DIR, "lib%s.so" % _name)
        if os.path.exists(_p):
            ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

from . import llama_cpp  # noqa: E402
from . import mtmd_cpp  # noqa: E402

try:
    with open(os.path.join(LIB_DIR, "BUILD_INFO.json")) as _f:
        BUILD_INFO = json.load(_f)
except OSError:
    BUILD_INFO = {}
