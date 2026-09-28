"""Build the embedded llama.cpp (vendor/llama.cpp, pinned to the commit llama-cpp-python 0.3.35 binds) into dh_llama/lib.

    python build.py                          # CUDA for the GPU in this machine
    python build.py --cuda-arch "86;89;120"  # CUDA fat binary for several GPU generations (release builds)
    python build.py --backend vulkan         # AMD / Intel / NVIDIA through Vulkan
    python build.py --backend cpu
    python build.py --with-tools             # also llama-quantize / llama-imatrix (for docs/QUANTIZATION.md)

Needs cmake, a C/C++ compiler and the backend SDK (CUDA toolkit / Vulkan SDK). Output is self-contained: the node loads
dh_llama/lib and never touches a llama-cpp-python installed in the venv.
"""
import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "vendor", "llama.cpp")
LIB = os.path.join(HERE, "dh_llama", "lib")
PINNED = "4da6337767f973e2b4d0797e5b323d77d8565e4a"  # upstream 2026-09-27 (DFlash2); bindings started from llama-cpp-python v0.3.35 (4df29be4f), updated
TARGETS = ["llama", "mtmd", "llama-common"]  # ggml libraries come along as dependencies; llama-common for libdh_spec
NATIVE = os.path.join(HERE, "native")  # libdh_spec (speculative decoding for the rewrite), built against the above


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cuda", "vulkan", "cpu"], default="cuda")
    ap.add_argument("--cuda-arch", default="native", help='"native" (this GPU) or a list such as "86;89;120"')
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--portable-cpu", action="store_true", help="no -march=native, for binaries that run on other CPUs")
    ap.add_argument("--with-tools", action="store_true",
                    help="also build llama-quantize and llama-imatrix from the same pinned source (quantization workflow)")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(SRC, "CMakeLists.txt")):
        run(["git", "submodule", "update", "--init", "vendor/llama.cpp"], cwd=HERE)
    # only trust git when vendor/llama.cpp is its own checkout; a copied tree would report the enclosing repo's HEAD
    head = ""
    if os.path.exists(os.path.join(SRC, ".git")):
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC, capture_output=True, text=True).stdout.strip()
    if head and head != PINNED:
        print("WARNING: vendor/llama.cpp is at %s, bindings were written for %s" % (head, PINNED))

    tag = "%s-%s-%s" % (sys.platform, platform.machine().lower(), args.backend)
    bdir = args.build_dir or os.path.join(HERE, "build", tag)
    cfg = [
        "cmake", "-S", SRC, "-B", bdir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",  # tools/mtmd (vision) lives here
        "-DGGML_NATIVE=%s" % ("OFF" if args.portable_cpu else "ON"),
        "-DGGML_CUDA=%s" % ("ON" if args.backend == "cuda" else "OFF"),
        "-DGGML_VULKAN=%s" % ("ON" if args.backend == "vulkan" else "OFF"),
    ]
    if args.backend == "cuda":
        cfg.append("-DCMAKE_CUDA_ARCHITECTURES=%s" % args.cuda_arch)
    if sys.platform.startswith("linux"):
        # find sibling libraries next to themselves, wherever dh_llama/lib ends up
        cfg += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=$ORIGIN"]
    targets = TARGETS + (["llama-quantize", "llama-imatrix"] if args.with_tools else [])
    # patches/*.patch are applied for the build and reverted afterwards, so vendor/llama.cpp stays at the pinned commit
    patches = sorted(glob.glob(os.path.join(HERE, "patches", "*.patch")))
    # keep git from finding an enclosing repository (e.g. a ComfyUI checkout): inside one, `git apply` resolves the
    # patch paths from that repo's root and silently skips every hunk
    genv = dict(os.environ, GIT_CEILING_DIRECTORIES=os.path.dirname(SRC))
    applied = []
    try:
        for p in patches:
            if subprocess.run(["git", "apply", "--reverse", "--check", p], cwd=SRC, env=genv, capture_output=True).returncode == 0:
                print("patch already applied:", os.path.basename(p))
                continue
            run(["git", "apply", p], cwd=SRC, env=genv)
            applied.append(p)
        run(cfg)
        run(["cmake", "--build", bdir, "--config", "Release", "-j", str(args.jobs), "--target"] + targets)
        nbdir = os.path.join(bdir, "dh_spec")
        run(["cmake", "-S", NATIVE, "-B", nbdir, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_SRC=" + SRC, "-DLLAMA_BUILD=" + bdir])
        run(["cmake", "--build", nbdir, "--config", "Release", "-j", str(args.jobs)])
    finally:
        for p in reversed(applied):
            run(["git", "apply", "--reverse", p], cwd=SRC, env=genv)

    os.makedirs(LIB, exist_ok=True)
    for old in glob.glob(os.path.join(LIB, "*")):
        if os.path.isfile(old) or os.path.islink(old):
            os.remove(old)
    pats = {"win32": ["*.dll"], "darwin": ["*.dylib"]}.get(sys.platform, ["*.so", "*.so.*"])
    found = []
    for d in {os.path.join(bdir, "bin"), os.path.join(bdir, "bin", "Release"), os.path.join(bdir, "lib")}:
        for pat in pats:
            for f in glob.glob(os.path.join(d, pat)):
                dst = os.path.join(LIB, os.path.basename(f))
                if os.path.lexists(dst):
                    continue
                if os.path.islink(f):
                    os.symlink(os.readlink(f), dst)
                else:
                    shutil.copy2(f, dst)
                found.append(os.path.basename(f))
    need = ["llama", "mtmd", "ggml", "ggml-base", "dh_spec"]
    missing = [n for n in need if not any(x.startswith(("lib" + n + ".", n + ".")) for x in found)]
    if missing:
        sys.exit("build finished but these libraries were not found: %s" % missing)
    info = {"llama_cpp_commit": head or (PINNED + " (not verified: vendor/llama.cpp has no .git)"),
            "patches": [os.path.basename(p) for p in patches], "backend": args.backend, "cuda_arch": args.cuda_arch if args.backend == "cuda" else None,
            "platform": tag, "portable_cpu": args.portable_cpu, "files": sorted(found)}
    with open(os.path.join(LIB, "BUILD_INFO.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("OK ->", LIB)
    if args.with_tools:
        print("tools ->", os.path.join(bdir, "bin"))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
