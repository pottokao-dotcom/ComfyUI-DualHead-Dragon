"""dhquant: DiT-aware quantization of text encoders / dual-head models, one command.

    dhquant.py list                                   models, what sensbank holds, verified results
    dhquant.py scan   qwen3-4b                        gradient sensitivity table -> sensbank (cached: runs once per model)
    dhquant.py plan   qwen3-4b --preset q4_k_m        tensor-level type map at the size of stock Q4_K_M, no tensor below Q4
    dhquant.py plan   qwen3-4b --budget-gb 2.7        any size; --aggressive lets unimportant tensors go below the floor
    dhquant.py curve  qwen3-4b                        size vs predicted error, from the table alone
    dhquant.py build  qwen3-4b <plan.json> [-o x.gguf]  real GGUF (llama-quantize, same imatrix) + manifest
    dhquant.py verify qwen3-4b <x.gguf> [--keep]      score through the dual-head engine vs comfy bf16 -> sensbank verified.json
    dhquant.py run    qwen3-4b --preset q4_k_m        scan (if needed) -> plan -> build -> verify

Presets (user rule): a standard name means "the size of that stock llama.cpp file, and no tensor below its base type".
q4_k_m -> floor Q4_K, q5_k_m -> floor Q5_K, q6_k -> floor Q6_K, iq4_xs -> IQ4_XS, q2_k -> Q2_K ...
K presets use no IQ type (fast everywhere); IQ presets may use any. --aggressive drops the floor one step.

Every GPU step runs under a memory guard: the child is killed when MemAvailable falls below --reserve-gb (default 40,
the gx10 rule), so a bad estimate never takes the machine down. Method and validation: docs/DHQUANT.md.
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(TOOLS)
BANK = os.path.join(HERE, "sensbank")
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, TOOLS)

HOME = os.path.expanduser("~")
ENV = {  # gx10 layout; override any of these in the environment
    "COMFYUI_DIR": os.environ.get("COMFYUI_DIR", HOME + "/ComfyUI_lab"),
    "PYTHONPATH": os.environ.get("PYTHONPATH", HOME + "/lab_extra"),
    "DH_LIB": os.environ.get("DH_LIB", HOME + "/ComfyUI_lab/custom_nodes/ComfyUI-DualHeadDragon/dh_llama/lib"),
    "LLAMA_BIN": os.environ.get("LLAMA_BIN", HOME + "/ComfyUI_lab/custom_nodes/ComfyUI-DualHeadDragon/build/linux-aarch64-cuda/bin"),
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
os.environ.setdefault("DH_LIB", ENV["DH_LIB"])
PRESETS = {"q4_k_m": ("q4_k", "q6_k"), "q5_k_m": ("q5_k", "q6_k"), "q6_k": ("q6_k", "q6_k"),
           "q3_k_m": ("q3_k", None), "q2_k": ("q2_k", None),
           "iq4_xs": ("iq4_xs", None), "iq3_s": ("iq3_s", None), "iq3_m": ("iq3_s", None),
           "iq2_s": ("iq2_xs", None), "iq2_m": ("iq2_xs", None)}  # Q2 class floor: IQ2_XS (stock IQ2_S uses it too)
# (base = floor, more_bits type); None -> llama.cpp rules not modelled here, needs --stock-gguf.
# Series (user rule 2026-09-27): K presets use K/legacy types only (fast on every backend); IQ presets may use any
# type at or above the floor. Level classes: Q4 = Q4_K/IQ4_XS, Q3 = Q3_K/IQ3_S, Q2 = Q2_K/IQ2_S/IQ2_XS.
AGGR_FLOOR = {"q6_k": "q5_k", "q5_k": "q4_k", "q4_k": "q3_k", "q3_k": "q2_k", "iq4_xs": "iq3_s",
              "iq3_s": "iq3_xxs", "iq2_s": "iq2_xxs", "iq2_xs": "iq2_xxs", "q2_k": "iq2_xxs"}


def series(preset, floor):
    return "iq" if (preset or floor).startswith("iq") else "k"


# ---------------------------------------------------------------- registry / bank
def models():
    return {k: v for k, v in json.load(open(os.path.join(TOOLS, "dhquant_models.json"))).items() if not k.startswith("_")}


def model(name):
    m = models().get(name)
    if m is None:
        sys.exit("unknown model %r; known: %s" % (name, ", ".join(models())))
    return m


def path(m, key):
    v = m.get(key)
    if v is None:
        return None
    if v.startswith("sensbank:"):                       # this model's own bank folder
        return os.path.join(BANK, m["sensbank"], v.split(":", 1)[1])
    if v.startswith("bank:"):                           # another model's bank (a finetune reusing its base's files)
        return os.path.join(BANK, v.split(":", 1)[1])
    return os.path.expanduser(v)


def bank(m, *f):
    return os.path.join(BANK, m["sensbank"], *f)


def rel(one_minus_cos):
    return 100 * math.sqrt(2 * max(one_minus_cos, 0.0))


# ---------------------------------------------------------------- memory guard
def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("inf")


def guarded(cmd, log, reserve):
    """Run cmd; kill it if MemAvailable < reserve GB (checked every second). Returns the exit code (-9 = killed)."""
    env = dict(os.environ, **ENV)
    print("$ " + " ".join(cmd[:3]) + " ...  (log %s, guard: keep >= %.0f GB available)" % (log, reserve), flush=True)
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    low = float("inf")
    while p.poll() is None:
        a = avail_gb()
        low = min(low, a)
        if a < reserve:
            p.kill()
            p.wait()
            print("KILLED by the memory guard: %.0f GB available < %.0f GB" % (a, reserve), flush=True)
            return -9
        time.sleep(1)
    print("  exit %d, lowest available %.0f GB" % (p.returncode, low), flush=True)
    return p.returncode


def sha256(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- scan
def cmd_scan(a):
    m = model(a.model)
    out = bank(m, "fisher.json")
    if os.path.exists(out) and not a.force:
        fi = json.load(open(out))
        print("cached: %s (formats %s, %d probes, %d prompts, %ss) -- --force to rescan" % (
            out, ",".join(fi["formats"]), fi["probes"], fi["prompts"], fi.get("seconds")))
        return out
    os.makedirs(bank(m), exist_ok=True)
    tmp = bank(m, "fisher.json.part")
    cmd = [sys.executable, os.path.join(TOOLS, "sens_fisher.py"), "--model", m["model_arg"],
           "--prompts", path(m, "prompts"), "--imatrix", path(m, "imatrix"), "--formats", a.formats,
           "--probes", str(a.probes), "--chunk", str(m.get("chunk", 1)), "--fp8", "--grad-lowmem",
           "--reserve-gb", str(a.reserve_gb), "--out", tmp]
    cmd += ["--te", path(m, "te")] if m.get("te") else ["--te-hf", path(m, "te_hf")]
    for k, v in m["products"].items():
        cmd += ["--" + k, os.path.expanduser(v)]
    t0 = time.time()
    if guarded(cmd, bank(m, "scan.log"), a.reserve_gb) != 0:
        sys.exit("scan failed, see " + bank(m, "scan.log"))
    os.replace(tmp, out)
    man_p = bank(m, "manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {"model": m["desc"]}
    man["scan"] = {"tool": "tools/sens_fisher.py", "flags": "--fp8 --grad-lowmem --chunk %s" % m.get("chunk", 1),
                   "formats": a.formats.split(","), "probes": a.probes, "seconds": round(time.time() - t0),
                   "date": datetime.date.today().isoformat()}
    json.dump(man, open(man_p, "w"), indent=1, ensure_ascii=False)
    print("stored", out)
    return out


# ---------------------------------------------------------------- plan
def embd_shape(m):
    from safetensors import safe_open
    if m.get("te"):
        return safe_open(path(m, "te"), "pt").get_slice("model.embed_tokens.weight").get_shape()
    d = path(m, "te_hf")
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    k = next(k for k in idx if k.endswith("embed_tokens.weight") and "visual" not in k)
    return safe_open(os.path.join(d, idx[k]), "pt").get_slice(k).get_shape()


def stock_bytes(fi, preset, n_layers):
    """bytes of the measured tensors in a stock llama.cpp file of this preset (use_more_bits rule for attn_v/ffn_down)"""
    base, more = PRESETS[preset]

    def more_bits(i):
        return i < n_layers // 8 or i >= 7 * n_layers // 8 or (i - n_layers // 8) % 3 == 2
    tot = 0
    for k, b in fi["bytes"].items():
        i, hf = k.split(".", 1)
        t = more if hf in ("self_attn.v_proj", "mlp.down_proj") and more_bits(int(i)) else base
        if t not in b:
            sys.exit("table has no %s bytes; rescan with that format" % t)
        tot += b[t]
    return tot


def cmd_plan(a, quiet=False):
    m = model(a.model)
    fi_p = bank(m, "fisher.json")
    if not os.path.exists(fi_p):
        sys.exit("no table for %s yet: run `dhquant.py scan %s`" % (a.model, a.model))
    fi = json.load(open(fi_p))
    from fakequant import tensor_bytes
    shp = embd_shape(m)
    embd_t = "q6_k" if m.get("te") else "q4_k"   # tied 4B: token_embd doubles as output (Q6_K in stock); 8B: Q4_K
    embd = tensor_bytes(tuple(shp), embd_t)
    if a.preset and a.stock_gguf:
        # budget straight from a stock file llama.cpp built: exact for every preset's rules
        import gguf
        from fakequant import GGUF_TO_HF
        tot = 0
        for t in gguf.GGUFReader(a.stock_gguf).tensors:
            parts = t.name.split(".")
            if len(parts) == 4 and parts[0] == "blk" and parts[2] in GGUF_TO_HF:
                k = "%s.%s" % (parts[1], GGUF_TO_HF[parts[2]])
                f = t.tensor_type.name.lower()
                if f not in fi["bytes"][k]:
                    sys.exit("stock file uses %s for %s; the table has no such format" % (f, k))
                tot += fi["bytes"][k][f]
            elif t.name == "token_embd.weight":
                embd_t = t.tensor_type.name.lower()
                embd = tensor_bytes(tuple(shp), embd_t)
        floor = a.floor or PRESETS[a.preset][0]   # --floor below the preset: "low floor, stock budget"
        budget = (tot + embd) / 1e9
        name = a.preset + ("_aggr" if a.aggressive else "") + ("_fl" + a.floor if a.floor else "")
    elif a.preset:
        if PRESETS[a.preset][1] is None:
            sys.exit("%s needs --stock-gguf (its llama.cpp rules are not modelled here)" % a.preset)
        floor = a.floor or PRESETS[a.preset][0]   # --floor below the preset: "low floor, stock budget"
        budget = (stock_bytes(fi, a.preset, m["layers"]) + embd) / 1e9
        name = a.preset + ("_aggr" if a.aggressive else "") + ("_fl" + a.floor if a.floor else "")
    else:
        floor, budget, name = a.floor or "q4_k", a.budget_gb, "b%.2fgb" % a.budget_gb
    pe_floor = floor
    wts = getattr(a, "weights", None) or m["weights"]
    ser = a.series or series(a.preset, floor)
    only = [f for f in fi["formats"] if ser == "iq" or not f.startswith("iq")]
    if a.aggressive:
        floor = AGGR_FLOOR[floor]
        if floor not in only:
            sys.exit("aggressive floor %s is not in the %s series / table" % (floor, ser))
    out = a.out or bank(m, "configs", "dhq_%s.json" % name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [sys.executable, os.path.join(TOOLS, "alloc_t.py"), fi_p, "--embd-shape", "%d,%d" % tuple(shp),
           "--embd", embd_t, "--floor", floor, "--pe-floor", pe_floor, "--weights", wts,
           "--budget-gb", "%.4f" % budget, "--out", out, "--only", ",".join(only)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **ENV))
    if r.returncode != 0:
        sys.exit(r.stdout + r.stderr)
    line = next(l for l in r.stdout.splitlines() if l.startswith("budget"))
    pred = {t: float(line.split(t + " ")[1].split()[0]) for t in wts.split(",") for t in [t.split("=")[0]]
            if t + " " in line}
    plan = {"model": a.model, "preset": a.preset, "aggressive": a.aggressive, "series": ser, "formats": only,
            "floor": floor, "pe_floor": pe_floor,
            "budget_gb": round(budget, 4), "type_map": out, "predicted_1mcos": pred,
            "weights": wts,
            "predicted_rel_pct": {t: round(rel(v), 2) for t, v in pred.items() if ".pad" not in t and not t.startswith("pe.")},
            "predicted_pe_kl": {t: v for t, v in pred.items() if t.startswith("pe.")},
            "note": "pad targets steer the allocation by rank only; their absolute prediction is not meaningful",
            "table": fi_p}
    json.dump(plan, open(out.replace(".json", ".plan.json"), "w"), indent=1)
    open(out.replace(".json", ".grid.md"), "w").write(grid_text(json.load(open(out)), os.path.basename(out)))
    if not quiet:
        print(r.stdout.rstrip())
        print("budget = %s" % ("stock %s size %.3f GB" % (a.preset, budget) if a.preset else "%.3f GB" % budget))
        print("predicted rel error (simulated; real GGUF runs higher at 5-6 bit): " +
              "  ".join("%s %.1f%%" % (t, v) for t, v in plan["predicted_rel_pct"].items()) +
              "".join("  %s KL %.4f" % (t, v) for t, v in plan["predicted_pe_kl"].items()))
        print("type map ->", out)
    return out


def cmd_curve(a):
    m = model(a.model)
    shp = embd_shape(m)
    cmd = [sys.executable, os.path.join(TOOLS, "alloc_t.py"), bank(m, "fisher.json"), "--embd-shape",
           "%d,%d" % tuple(shp), "--embd", "q6_k" if m.get("te") else "q4_k", "--floor", a.floor,
           "--pe-floor", "q4_k", "--weights", m["weights"], "--curve", bank(m, "curve_%s.json" % a.floor)]
    subprocess.run(cmd, check=True, env=dict(os.environ, **ENV))
    c = json.load(open(bank(m, "curve_%s.json" % a.floor)))["curve"]
    cols = [t for t in c[0] if t != "gb" and ".pad" not in t]   # pad: rank only, absolute value not meaningful
    print("\n  GB     " + "  ".join("%12s" % t for t in cols) + "   (predicted rel %, simulated)")
    for gb in [x / 10 for x in range(int(c[0]["gb"] * 10), int(c[-1]["gb"] * 10) + 2)]:
        pt = [x for x in c if x["gb"] <= gb]
        if pt:
            print("  %.1f  " % gb + "  ".join("%11.1f%%" % rel(pt[-1][t]) for t in cols))


# ---------------------------------------------------------------- build / verify
def cmd_build(a):
    m = model(a.model)
    bf = path(m, "bf16_gguf")
    if not bf or not os.path.exists(bf):
        sys.exit("no BF16 GGUF for %s on this machine (bf16_gguf in dhquant_models.json)" % a.model)
    tmap = json.load(open(a.plan))
    out = a.out or os.path.join(os.path.dirname(bf), os.path.basename(a.plan).replace(".json", ".gguf"))
    args = []
    import re
    for k, v in tmap.items():
        args += ["--tensor-type", re.escape(k) + "=" + v.lower()]
    imat = a.imatrix or path(m, "imatrix_gguf")
    plan_p = a.plan.replace(".json", ".plan.json")
    preset = (json.load(open(plan_p)).get("preset") if os.path.exists(plan_p) else None) or "q4_k_m"
    base_t = preset.upper()          # tensors the map does not list (e.g. output.weight) follow the preset's own rules
    cmd = [os.path.join(ENV["LLAMA_BIN"], "llama-quantize"), "--imatrix", imat] + args + \
          [bf, out, base_t]
    if guarded(cmd, out + ".log", a.reserve_gb) != 0:
        sys.exit("llama-quantize failed, see %s.log" % out)
    man = {"gguf": os.path.basename(out), "bytes": os.path.getsize(out), "sha256": sha256(out), "model": a.model,
           "type_map": a.plan, "plan": json.load(open(plan_p)) if os.path.exists(plan_p) else None,
           "grid": grid_text(tmap), "imatrix": imat, "platform": __import__("platform").machine(), "built": datetime.datetime.now().isoformat(timespec="seconds")}
    man["tensor_sha256"] = tensor_sha256(out)
    man["as_built"] = as_built(out)
    json.dump(man, open(out + ".manifest.json", "w"), indent=1)
    print("built %s  %.3f GB  sha256 %s" % (out, man["bytes"] / 1e9, man["sha256"][:16]))
    return out



def cmd_imatrix(a):
    """imat_dit<lam>.gguf in the bank: (1-lam) * stock imatrix + lam * DiT-weighted column importance, per tensor"""
    m = model(a.model)
    for lam in a.lam.split(","):
        out = bank(m, "imat_dit%s.gguf" % lam.replace(".", ""))
        if os.path.exists(out):     # recipes pin the sha256 of these files: never overwrite
            print("exists, kept:", out)
            continue
        subprocess.run([sys.executable, os.path.join(TOOLS, "fimat_to_gguf.py"), bank(m, a.fimat), "--base",
                        path(m, "imatrix_gguf"), "--weights", m["weights"], "--lam", lam, "--out", out], check=True)

def cmd_verify(a):
    m = model(a.model)
    if not m.get("verify"):
        sys.exit("verify is not wired for %s yet (sens_dit --ggufs knows the 4B taps only)" % a.model)
    tmp = a.gguf + ".score.json"
    hold = getattr(a, "holdout", False)
    prompts = bank(m, "holdout.json") if hold else path(m, "prompts")
    if not os.path.exists(prompts):
        sys.exit("no %s: build it with tools/make_eval_sets.py (held out from prompts.json and fimat prompts)" % prompts)
    cmd = [sys.executable, os.path.join(TOOLS, "sens_dit.py"), "--model", m["model_arg"],
           "--prompts", prompts, "--calib", "/dev/null", "--ruler", "", "--sweep", "",
           "--ggufs", a.gguf, "--out", tmp]
    cmd += ["--te", path(m, "te")] if m.get("te") else ["--te-hf", path(m, "te_hf")]
    for k, v in m["products"].items():
        cmd += ["--" + k, os.path.expanduser(v)]
    if getattr(a, "rot", None):
        cmd += ["--rot", a.rot]
    if guarded(cmd, a.gguf + ".verify.log", a.reserve_gb) != 0:
        sys.exit("verify failed, see %s.verify.log" % a.gguf)
    got = json.load(open(tmp))["ggufs"]
    os.makedirs(bank(m), exist_ok=True)
    ver_p = bank(m, "verified_holdout.json" if hold else "verified.json")   # holdout: never used to pick configs
    ver = json.load(open(ver_p)) if os.path.exists(ver_p) else {"ggufs": {}}
    ver["ggufs"].update(got)
    json.dump(ver, open(ver_p, "w"), indent=1)
    for name, r in got.items():
        print("%s %.3f GB  " % (name, r["gb_file"]) + "  ".join(
            "%s.%s %.1f%%" % (p, part, 100 * v["rel_dit"]) for p, d in r.items() if isinstance(d, dict)
            for part, v in d.items()))
    os.remove(tmp)
    man_p = a.gguf + ".manifest.json"
    if os.path.exists(man_p) and not hold:     # the build record lives next to the scores
        man = json.load(open(man_p))
        man["verified"] = got
        os.makedirs(bank(m, "builds"), exist_ok=True)
        json.dump(man, open(bank(m, "builds", os.path.basename(man_p)), "w"), indent=1)
        save_recipe(a.model, recipe_from_build(a.model, man, next(iter(got.values()), None)))
    if not a.keep:
        for f in (a.gguf, a.gguf + ".log"):
            if os.path.exists(f):
                os.remove(f)
        print("deleted", a.gguf, "(--keep to keep it; the manifest stays)")


def cmd_run(a):
    cmd_scan(a)
    a.out = None
    plan = cmd_plan(a)
    a.plan, a.out = plan, None
    a.gguf = cmd_build(a)
    cmd_verify(a)


COLS = [("attn_q", "q"), ("attn_k", "k"), ("attn_v", "v"), ("attn_output", "o"), ("ffn_gate", "gate"),
        ("ffn_up", "up"), ("ffn_down", "down")]


def type_grid(src):
    """{gguf tensor name: type} from a type map json or a GGUF file"""
    if src.endswith(".gguf"):
        import gguf
        return {t.name: t.tensor_type.name for t in gguf.GGUFReader(src).tensors}
    return json.load(open(src))


def grid_text(tm, title=""):
    """per-layer table of each linear's format, plus totals -- the thing to read when asking 'what did layer N get'"""
    short = lambda t: t.upper().replace("_0", "").replace("_K", "K").replace("IQ4_XS", "iq4").replace("IQ3_S", "iq3")  # noqa
    layers = sorted({int(k.split(".")[1]) for k in tm if k.startswith("blk.")})
    lines = ["| layer | " + " | ".join(c for _, c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    from collections import Counter
    cnt = Counter()
    for i in layers:
        row = []
        for g, _ in COLS:
            t = tm.get("blk.%d.%s.weight" % (i, g), "")
            cnt[short(t)] += 1 if t else 0
            row.append(short(t) if t else "-")
        lines.append("| %d | %s |" % (i, " | ".join(row)))
    head = ("## " + title + "\n\n") if title else ""
    extra = ", ".join("%s %s" % (k, tm[k]) for k in ("token_embd.weight", "output.weight") if k in tm)
    return head + "\n".join(lines) + "\n\ntotals: " + ", ".join("%s x%d" % kv for kv in sorted(cnt.items())) + \
        ("\nother: " + extra if extra else "") + "\n"


def cmd_show(a):
    txt = grid_text(type_grid(a.src), os.path.basename(a.src))
    print(txt)
    if not a.src.endswith(".gguf"):
        open(a.src.replace(".json", ".grid.md"), "w").write(txt)


# ---------------------------------------------------------------- recipes
def as_built(path):
    """every tensor of a GGUF as it came out: {name: [type, bytes]} -- including the small ones (norms stay F32) and the
    ones the type map does not list (token_embd / output follow llama-quantize's rules for the base type)"""
    import gguf
    return {t.name: [t.tensor_type.name, int(t.n_bytes)] for t in gguf.GGUFReader(path).tensors}


def as_built_summary(ab, tmap):
    import re
    from collections import defaultdict
    rows = defaultdict(lambda: [0, 0])
    for n, (ty, b) in ab.items():
        k = (re.sub(r"blk\.\d+\.", "blk.N.", n), ty, "map" if n in tmap else "rule")
        rows[k][0] += 1
        rows[k][1] += b
    lines = ["| tensor | type | count | MB | set by |", "|---|---|---|---|---|"]
    for (k, ty, src), (c, b) in sorted(rows.items()):
        lines.append("| %s | %s | %d | %.2f | %s |" % (k, ty, c, b / 1e6, "type map" if src == "map" else
                                                     "llama.cpp rule for the base type"))
    return "\n".join(lines) + "\n"


def tensor_sha256(path):
    """hash of the tensor data only (name, type, shape, bytes in file order): two files with different metadata
    (converter version, dates) but the same weights hash the same"""
    import gguf
    import numpy as np
    h = hashlib.sha256()
    for t in gguf.GGUFReader(path).tensors:
        h.update(t.name.encode())
        h.update(t.tensor_type.name.encode())
        h.update(str(list(t.shape)).encode())
        h.update(np.ascontiguousarray(t.data).view(np.uint8).tobytes())
    return h.hexdigest()


LLAMA_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"   # vendor/llama.cpp, pinned by build.py


def recipe_from_build(m_name, man, verified=None, product=""):
    """Everything needed to rebuild a GGUF bit for bit, self-contained: source weights (HF repo + revision), the
    conversion, the imatrix (its sha256; the file lives in sensbank), and the full per-tensor type map inline."""
    import platform
    m = model(m_name)
    tm = man["type_map"]
    if isinstance(tm, str) and not os.path.exists(tm) and "/sensbank/" in tm:   # built on another machine
        tm = os.path.join(BANK, tm.split("/sensbank/", 1)[1])
    tmap = json.load(open(tm)) if isinstance(tm, str) else tm
    ab = man.get("as_built") or {}
    stale = [k for k, v in tmap.items() if k in ab and v.upper() != ab[k][0].upper()]
    if stale:   # the config file was overwritten after the build (e.g. an rsync from another machine)
        sys.exit("type map %s disagrees with the built GGUF on %d tensors (%s ...); refusing to write a recipe that "
                 "would not rebuild it" % (tm if isinstance(tm, str) else "(inline)", len(stale), stale[0]))
    imat = man["imatrix"]
    plan = man.get("plan") or {}
    return {
        "recipe_version": 1,
        "name": os.path.splitext(man["gguf"])[0],
        "model": m_name,
        "target": product or ("weights " + m["weights"]),
        "preset": plan.get("preset"), "floor": plan.get("floor"), "budget_gb": plan.get("budget_gb"),
        "source": m.get("source"),
        "convert": {"tool": "vendor/llama.cpp/convert_hf_to_gguf.py", "llama_cpp_commit": LLAMA_COMMIT,
                    "outtype": (m.get("source") or {}).get("outtype", "bf16")},
        "imatrix": _imatrix_ref(imat),
        "quantize": {"tool": "llama-quantize", "llama_cpp_commit": LLAMA_COMMIT,
                     "platform": man.get("platform") or platform.machine(),
                     "base_type": (plan.get("preset") or "q4_k_m").upper(), "type_map": tmap},
        "output": {"gguf": man["gguf"], "bytes": man["bytes"], "sha256": man["sha256"],
                   "tensor_sha256": man.get("tensor_sha256")},
        "grid": man.get("grid") or grid_text(tmap),
        "as_built": man.get("as_built"),
        "verified": verified,
    }


def save_recipe(m_name, rec):
    d = bank(model(m_name), "recipes")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, rec["name"] + ".recipe.json")
    json.dump(rec, open(p, "w"), indent=1, ensure_ascii=False)
    open(p.replace(".recipe.json", ".recipe.md"), "w").write(
        "# %s\n\n- model: %s (HF %s @ %s)\n- target: %s\n- preset/floor/budget: %s / %s / %s GB\n"
        "- output: %s, %.3f GB, sha256 %s\n- imatrix: %s (sha256 %s)\n- llama.cpp %s on %s, base type %s\n"
        "- verified: %s\n\n%s" % (
            rec["name"], rec["model"], (rec["source"] or {}).get("hf_repo"), (rec["source"] or {}).get("revision"),
            rec["target"], rec["preset"], rec["floor"], rec["budget_gb"], rec["output"]["gguf"],
            rec["output"]["bytes"] / 1e9, rec["output"]["sha256"], rec["imatrix"]["bank_path"],
            rec["imatrix"]["sha256"], LLAMA_COMMIT[:9], rec["quantize"]["platform"], rec["quantize"]["base_type"],
            json.dumps(rec["verified"]) if rec["verified"] else "not yet", rec["grid"]) +
        ("\n## every tensor as built\n\n" + as_built_summary(rec["as_built"], rec["quantize"]["type_map"])
         if rec.get("as_built") else ""))
    print("recipe ->", p)
    return p



def _imatrix_ref(imat):
    """bank-relative path + sha256 of the imatrix a build used; an outside copy is matched to the bank file by hash"""
    if "/sensbank/" in imat:
        rel = imat.split("/sensbank/", 1)[1]
        full = os.path.join(BANK, rel)
        return {"bank_path": rel, "sha256": sha256(full) if os.path.exists(full) else None}
    if not os.path.exists(imat):
        return {"bank_path": imat, "sha256": None}
    h = sha256(imat)
    for f in glob.glob(os.path.join(BANK, "*", "*.gguf")):
        if os.path.getsize(f) == os.path.getsize(imat) and sha256(f) == h:
            return {"bank_path": os.path.relpath(f, BANK), "sha256": h}
    sys.exit("imatrix %s is not in sensbank: copy it there first so the recipe can rebuild" % imat)

def cmd_recipe(a):
    man = json.load(open(a.manifest))
    ver = None
    key = a.verified_name or os.path.splitext(man["gguf"])[0]
    for f in ("verified.json", "verified_holdout.json"):   # holdout-only runs (tools/pipelines_8b/products*.sh)
        vp = bank(model(a.model), f)
        if ver is None and os.path.exists(vp):
            ver = json.load(open(vp))["ggufs"].get(key)
            if ver is not None and f != "verified.json":
                ver = dict(ver, scored_on="holdout")
    rec = recipe_from_build(a.model, man, ver, a.target)
    if a.name:
        rec["name"] = a.name
    save_recipe(a.model, rec)


def cmd_rebuild(a):
    """Recreate a GGUF from its recipe: fetch the HF revision if the shards are missing, convert, quantize with the
    recorded imatrix and type map, then compare sha256 with the recorded output."""
    import re
    import tempfile
    rec = json.load(open(a.recipe))
    src = rec["source"]
    hf = os.path.expanduser(a.hf_dir or src["local_hf"])
    if not any(f.endswith(".safetensors") for f in (os.listdir(hf) if os.path.isdir(hf) else [])):
        from huggingface_hub import snapshot_download
        print("fetching %s @ %s -> %s" % (src["hf_repo"], src["revision"], hf), flush=True)
        snapshot_download(src["hf_repo"], revision=src["revision"], local_dir=hf,
                          allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*"])
    out_dir = os.path.expanduser(a.out_dir or (a.work or "."))
    os.makedirs(out_dir, exist_ok=True)
    # no --work: the converted base (8-16 GB) goes to a scratch dir on the output's disk (not /tmp: tmpfs) and is
    # deleted afterwards
    scratch = None if a.work else tempfile.mkdtemp(prefix="dhq_rebuild_", dir=out_dir)
    work = os.path.expanduser(a.work or scratch)
    os.makedirs(work, exist_ok=True)
    base = os.path.join(work, "%s_%s.gguf" % (rec["model"], rec["convert"]["outtype"]))
    if not os.path.exists(base):
        conv = os.path.join(HERE, "vendor", "llama.cpp", "convert_hf_to_gguf.py")
        env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
        subprocess.run([sys.executable, conv, hf, "--outtype", rec["convert"]["outtype"], "--outfile", base],
                       check=True, env=env)
    imat = os.path.join(BANK, rec["imatrix"]["bank_path"])
    if rec["imatrix"]["sha256"] and sha256(imat) != rec["imatrix"]["sha256"]:
        sys.exit("imatrix sha256 differs from the recipe: " + imat)
    args = []
    for k, v in rec["quantize"]["type_map"].items():
        args += ["--tensor-type", re.escape(k) + "=" + v.lower()]
    out = os.path.join(out_dir, rec["output"]["gguf"])
    try:
        subprocess.run([os.path.join(ENV["LLAMA_BIN"], "llama-quantize"), "--imatrix", imat] + args +
                       [base, out, rec["quantize"]["base_type"]], check=True, stdout=subprocess.DEVNULL)
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    import platform
    got, tgot = sha256(out), tensor_sha256(out)
    want_t = rec["output"].get("tensor_sha256")
    if got == rec["output"]["sha256"]:
        verdict = "IDENTICAL file"
    elif want_t and tgot == want_t:
        verdict = "IDENTICAL tensors (only the metadata differs)"
    elif tgot == (rec["output"].get("tensor_sha256_by_platform") or {}).get(platform.machine()):
        verdict = "IDENTICAL tensors to the recipe's %s build" % platform.machine()
    else:
        verdict = "DIFFERENT (recipe built on %s, this is %s: llama.cpp rounds a few blocks differently per platform; " \
                  "run `verify` to confirm the scores match)" % (rec["quantize"]["platform"], platform.machine())
    print("%s  %.3f GB  file sha256 %s  tensor sha256 %s  %s" % (out, os.path.getsize(out) / 1e9, got[:16], tgot[:16],
                                                                 verdict))


def cmd_list(a):
    for name, m in models().items():
        print("%-20s %s" % (name, m["desc"]))
        d = bank(m)
        fi = os.path.join(d, "fisher.json")
        if os.path.exists(fi):
            f = json.load(open(fi))
            print("    table: %s  formats %s  probes %d  %ss" % (os.path.relpath(fi, HERE), ",".join(f["formats"]),
                                                               f["probes"], f.get("seconds")))
        else:
            print("    table: none (dhquant.py scan %s)" % name)
        vp = os.path.join(d, "verified.json")
        if os.path.exists(vp):
            for g, r in json.load(open(vp))["ggufs"].items():
                print("    verified %-26s %.2f GB  " % (g, r["gb_file"]) + "  ".join(
                    "%s.%s %.1f%%" % (p, part, 100 * v["rel_dit"]) for p, dd in r.items() if isinstance(dd, dict)
                    for part, v in dd.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, plan=False):
        p.add_argument("model")
        p.add_argument("--reserve-gb", type=float, default=40.0, help="memory guard (gx10 rule: 40)")
        if plan:
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--preset", choices=sorted(PRESETS))
            g.add_argument("--budget-gb", type=float)
            p.add_argument("--floor", help="lowest type (default: the preset's base, or q4_k with --budget-gb)")
            p.add_argument("--aggressive", action="store_true", help="floor one step lower (AGGR_FLOOR)")
            p.add_argument("--series", choices=["k", "iq"], help="k: no IQ types (default for K presets); iq: any")
            p.add_argument("--stock-gguf", help="stock llama.cpp file of the preset: its tensor types set the budget")
        return p
    sub.add_parser("list")
    rc = sub.add_parser("recipe", help="write a self-contained rebuild recipe from a build manifest")
    rc.add_argument("model")
    rc.add_argument("manifest")
    rc.add_argument("--target", default="", help="e.g. 'QI2.1 only' or 'QI2.1 + Ideogram 4 + Boogu + rewrite'")
    rc.add_argument("--name")
    rc.add_argument("--verified-name", help="key in verified.json if it differs from the gguf name")
    rb = sub.add_parser("rebuild", help="recreate a GGUF from a recipe and check its sha256")
    rb.add_argument("recipe")
    rb.add_argument("--hf-dir")
    rb.add_argument("--work")
    rb.add_argument("--out-dir")
    th = sub.add_parser("tensorhash", help="sha256 of a GGUF's tensor data only (ignores metadata)")
    th.add_argument("gguf")
    sh = sub.add_parser("show", help="per-layer q/k/v/o/gate/up/down formats of a type map or a GGUF")
    sh.add_argument("src")
    s = common(sub.add_parser("scan"))
    s.add_argument("--formats", default="iq4_xs,q4_k,q5_k,q6_k,q8_0")
    s.add_argument("--probes", type=int, default=8)
    s.add_argument("--force", action="store_true")
    p = common(sub.add_parser("plan"), plan=True)
    p.add_argument("-o", "--out")
    p.add_argument("--weights", help="override the model's ruler weights, e.g. 'ideogram4.real=1,pe.real=0.5' "
                   "(pe.real: the rewrite ruler, tools/sens_fisher_pe.py)")
    c = common(sub.add_parser("curve"))
    c.add_argument("--floor", default="q4_k")
    b = common(sub.add_parser("build"))
    b.add_argument("plan")
    b.add_argument("-o", "--out")
    b.add_argument("--imatrix", help="imatrix GGUF instead of the model's stock one (e.g. from fimat_to_gguf.py)")
    v = common(sub.add_parser("verify"))
    v.add_argument("gguf")
    v.add_argument("--keep", action="store_true")
    v.add_argument("--holdout", action="store_true", help="score on sensbank holdout.json -> verified_holdout.json")
    v.add_argument("--rot", help="rotation.pt of an R1-rotated model (tools/rotate_r1.py): undo it on the DiT taps")
    im = sub.add_parser("imatrix", help="DiT-weighted imatrix GGUFs from the bank's fimat npz, one per lambda")
    im.add_argument("model")
    im.add_argument("--lam", default="0.25,0.5,0.75,1")
    im.add_argument("--fimat", default="fimat_qi21.npz", help="npz in the model's bank (sens_fisher.py --fimat)")
    r = common(sub.add_parser("run"), plan=True)
    r.add_argument("--formats", default="iq4_xs,q4_k,q5_k,q6_k,q8_0")
    r.add_argument("--probes", type=int, default=8)
    r.add_argument("--force", action="store_true")
    r.add_argument("--keep", action="store_true")
    r.add_argument("--imatrix")
    a = ap.parse_args()
    {"imatrix": cmd_imatrix, "list": cmd_list, "tensorhash": lambda a: print(tensor_sha256(a.gguf)), "show": cmd_show, "recipe": cmd_recipe, "rebuild": cmd_rebuild, "scan": cmd_scan, "plan": cmd_plan, "curve": cmd_curve, "build": cmd_build,
     "verify": cmd_verify, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
