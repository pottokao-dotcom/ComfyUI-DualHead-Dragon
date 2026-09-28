"""Check the ctypes bindings (dh_llama/llama_cpp.py, mtmd_cpp.py) against vendor/llama.cpp's C headers.

Run after every vendor/llama.cpp bump, before building: a struct whose field list differs from the header silently
reads garbage (e.g. mtmd_context_params gained `device` in 4da633776 and the media marker came out empty).
Also lists functions the bindings declare that the header no longer has (they must not break the import).

    python tools/check_bindings.py        # exit 1 on a mismatch that is not in KNOWN
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADERS = ["vendor/llama.cpp/include/llama.h", "vendor/llama.cpp/tools/mtmd/mtmd.h", "vendor/llama.cpp/tools/mtmd/mtmd-helper.h"]
BINDINGS = ["dh_llama/llama_cpp.py", "dh_llama/mtmd_cpp.py"]
# differences that are expected and harmless (checked by hand)
KNOWN = {
    "llama_model_kv_override": "C union val_i64/val_f64/val_bool/val_str is one ctypes union field `value`",
    "llama_opt_params": "training struct, not used by the node; optimizer_type missing since before 4df29be4f",
}


def c_structs(path):
    src = open(path).read()
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    out = {}
    for m in re.finditer(r"struct\s+(\w+)\s*\{(.*?)\}\s*(\w+)?\s*;", src, flags=re.S):
        fields = []
        for decl in m.group(2).split(";"):
            decl = decl.strip()
            if not decl:
                continue
            fp = re.search(r"\(\s*\*\s*(\w+)\s*\)", decl)  # function pointer
            if fp:
                fields.append(fp.group(1))
                continue
            decl = re.sub(r"\[.*?\]", "", decl)
            for part in decl.split(","):
                w = re.findall(r"(\w+)\s*$", part.strip())
                if w:
                    fields.append(w[0])
        out[m.group(1)] = fields
        if m.group(3):
            out[m.group(3)] = fields
    return out


def py_structs(path):
    src = open(path).read()
    out = {}
    for m in re.finditer(r"class\s+(\w+)\((?:ctypes\.)?(?:Structure|Union)\):(.*?)(?=\nclass |\Z)", src, flags=re.S):
        f = re.search(r"_fields_\s*=\s*\[(.*?)\n\s*\]", m.group(2), flags=re.S)
        if f:
            out[m.group(1)] = re.findall(r'\(\s*"(\w+)"', f.group(1))
    return out


def c_functions(path):
    return set(re.findall(r"\b(?:LLAMA_API|MTMD_API)\b[^;(]*?\b(\w+)\s*\(", open(path).read()))


def py_functions(path):
    return set(re.findall(r'^\s*"(\w+)",\s*\[', open(path).read(), flags=re.M))


def main():
    os.chdir(HERE)
    C, P, CF, PF = {}, {}, set(), set()
    for h in HEADERS:
        C.update(c_structs(h))
        CF |= c_functions(h)
    for b in BINDINGS:
        P.update(py_structs(b))
        PF |= py_functions(b)
    bad = 0
    checked = 0
    for name, pf in sorted(P.items()):
        if name not in C:
            continue
        checked += 1
        if C[name] != pf:
            if name in KNOWN:
                print("known  %-32s %s" % (name, KNOWN[name]))
                continue
            bad += 1
            print("MISMATCH %s\n  C : %s\n  py: %s" % (name, C[name], pf))
    gone = sorted(f for f in PF - CF if f.startswith(("llama_", "mtmd_")))
    print("%d structs checked, %d mismatches" % (checked, bad))
    if gone:
        print("bound but not in the headers any more (must stay lazy / unused): " + ", ".join(gone))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
