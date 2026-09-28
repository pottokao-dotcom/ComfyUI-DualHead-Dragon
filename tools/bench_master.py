"""Master-system benchmark: the fixed test condition for "how fast / how much VRAM / how well does the draft hit".

One run = one configuration (PE/TE GGUF, rewrite LoRA, draft, DiT). It drives a running ComfyUI (with this node) through
the real master pipeline and appends one row to the lookup table docs/bench/master_lookup.csv (+ MASTER_LOOKUP.md).

Fixed condition (do not change, or the table stops being comparable -- add a new condition set instead):
  6 prompts: 3 x flash-lit-intimate-portraits-mid (1:1) + 3 x flash-lit-red-accent (2:3), East Asian women, clothed
  master_style.build(engine="zimage", fmt="roleplay2", seed=42) -> Dual-Head Rewrite (temp 0.7, top_p 0.8, top_k 20,
  seed 42) -> master_style.finalize (signature lead/close) -> TE -> DiT (Z-Image turbo 8 steps euler/simple shift 3,
  or FLUX.2 klein 4B 4 steps euler) -> VAE, seed 42, size from the card's aspect.
  Prompt 1 is reported as "first" (whatever loading it triggers); "warm" = mean of prompts 2-6.
  Before the run ComfyUI is asked to unload its models, so the VRAM peak is a single-DiT workflow's.

  python tools/bench_master.py --tag v6-lora-dflashq6 --te zimage_engineer_v6_dhq_2p9.gguf \
      --lora zeng-v6_master-pe_lora_v4.1-r32-ep2_20260928-ck1212-f16.gguf \
      --draft zeng-v6_master-pe_dflash-b16_run1-Q6_K.gguf --dit zimage
"""
import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TABLE = os.path.join(REPO, "docs", "bench", "master_lookup.csv")
CONDITION = "master-v1"  # bump when the fixed condition changes
JOBS = [
    ("flash_portrait", "flash-lit-intimate-portraits-mid", [
        "A young East Asian woman sitting on the edge of an unmade bed in a small apartment, wearing a loose cotton dress",
        "An East Asian woman in a kimono eating a persimmon at a low table",
        "An East Asian woman in a sweater lying on tatami next to a vase of red flowers"]),
    ("flash_red_accent", "flash-lit-red-accent", [
        "A young East Asian woman in a white shirt holding a red apple against a white wall",
        "Two young East Asian women in plain dresses on a rooftop, one holding red flowers",
        "A young East Asian woman in a slip dress lying in tall grass next to a red umbrella"]),
]
DIM = {"1:1": (1024, 1024), "2:3": (832, 1248), "3:2": (1248, 832)}
COLS = ["date", "condition", "tag", "gpu", "te_pe_gguf", "lora", "draft", "dit", "pe_s", "te_s", "dit_s", "vae_s",
        "warm_total_s", "first_total_s", "tok_per_round", "accept_pct", "peak_vram_gb", "llama_cpp", "node_commit", "out_dir"]


def dit_graph(dit, w, h, text, seed, prefix):
    g = {"pos": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["c", 0]}},
         "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}}
    if dit == "zimage":
        g.update({"u": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_nvfp4.safetensors", "weight_dtype": "default"}},
                  "ms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["u", 0], "shift": 3.0}},
                  "v": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
                  "lat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
                  "k": {"class_type": "KSampler", "inputs": {"model": ["ms", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                        "latent_image": ["lat", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                        "scheduler": "simple", "denoise": 1.0}}})
    else:
        g.update({"u": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b-nvfp4.safetensors", "weight_dtype": "default"}},
                  "v": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
                  "ks": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
                  "noi": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
                  "lat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
                  "sch": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": w, "height": h}},
                  "g": {"class_type": "CFGGuider", "inputs": {"model": ["u", 0], "positive": ["pos", 0], "negative": ["neg", 0], "cfg": 1.0}},
                  "k": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noi", 0], "guider": ["g", 0], "sampler": ["ks", 0],
                        "sigmas": ["sch", 0], "latent_image": ["lat", 0]}}})
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["dec", 0]}}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--te", required=True, help="GGUF (text_encoders) used for both PE and TE")
    ap.add_argument("--lora", default="none", help="rewrite LoRA (loras), none for a merged GGUF")
    ap.add_argument("--draft", default="none", help="speculative draft (dualhead_drafts) or none")
    ap.add_argument("--dit", choices=["zimage", "klein"], required=True)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--vram-mode", default=None, help="loader vram_mode (auto/resident/managed); default = the loader's")
    ap.add_argument("--comfy", default="127.0.0.1:8195")
    ap.add_argument("--comfy-log", default=os.path.expanduser("~/ComfyUI_bh/comfy_bh.log"), help="for the draft statistics")
    ap.add_argument("--gpu-index", default="0", help="nvidia-smi index of the GPU ComfyUI runs on (VRAM peak)")
    ap.add_argument("--master-dir", default=os.path.expanduser("~/master_dh_sp/04_node"))
    ap.add_argument("--out", default=os.path.join(REPO, "bench_out"))
    a = ap.parse_args()
    sys.path.insert(0, a.master_dir)
    import master_style as MS
    import websocket

    U = a.comfy
    out_dir = os.path.join(a.out, "%s_%s_%s" % (datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), a.tag, a.dit))
    os.makedirs(out_dir, exist_ok=True)
    post = lambda path, body: json.load(urllib.request.urlopen(urllib.request.Request(
        "http://%s%s" % (U, path), json.dumps(body).encode(), {"Content-Type": "application/json"})) or "{}")
    try:
        urllib.request.urlopen(urllib.request.Request("http://%s/free" % U, json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                                      {"Content-Type": "application/json"}))
        time.sleep(3)
    except Exception as e:
        print("could not ask ComfyUI to unload:", e)

    peak = {"mb": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                v = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", a.gpu_index])
                peak["mb"] = max(peak["mb"], int(v.split()[0]))
            except Exception:
                pass
            time.sleep(0.2)
    threading.Thread(target=sample, daemon=True).start()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader", "-i", a.gpu_index], text=True).strip()

    cid = str(uuid.uuid4())
    ws = websocket.create_connection("ws://%s/ws?clientId=%s" % (U, cid))

    def run(g):
        pid = post("/prompt", {"prompt": g, "client_id": cid})["prompt_id"]
        marks = []
        while True:
            m = ws.recv()
            if not isinstance(m, str):
                continue
            m = json.loads(m)
            d = m.get("data", {})
            if d.get("prompt_id") != pid:
                continue
            if m["type"] == "executing":
                marks.append((d["node"], time.time()))
                if d["node"] is None:
                    break
            if m["type"] == "execution_error":
                raise RuntimeError(d.get("exception_message"))
        seg = {}
        for (n, t0), (_, t1) in zip(marks, marks[1:]):
            seg[n] = seg.get(n, 0.0) + t1 - t0
        return pid, seg

    def draft_lines():
        try:
            return [l for l in open(a.comfy_log, errors="replace") if "[DualHeadDragon] draft:" in l]
        except OSError:
            return []

    loader = {"class_type": "DualHeadDragonLoader", "inputs": {"gguf": a.te, "mmproj": "none", "n_ctx": a.n_ctx,
                                                                "type": "lumina2" if a.dit == "zimage" else "flux2"}}
    if a.vram_mode:
        loader["inputs"]["vram_mode"] = a.vram_mode
    rows, idx = [], 0
    for who, code, tasks in JOBS:
        for task in tasks:
            system, user, _neg, aspect = MS.build(task, code=code, engine="zimage", fmt="roleplay2", seed=42)
            w, h = DIM.get(aspect, (1024, 1024))
            n_before = len(draft_lines())
            t_start = time.time()
            pe = {"c": loader,
                  "r": {"class_type": "DualHeadRewrite", "inputs": {"clip": ["c", 0], "prompt": user, "system_prompt": "none",
                        "system_prompt_text": system, "rewrite_lora": a.lora, "lora_strength": 1.0, "seed": 42,
                        "max_new_tokens": 2048, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "thinking": False,
                        "draft": a.draft, "draft_n_max": 15}},
                  "show": {"class_type": "PreviewAny", "inputs": {"source": ["r", 0]}}}
            pid, seg_pe = run(pe)
            outs = json.load(urllib.request.urlopen("http://%s/history/%s" % (U, pid)))[pid]["outputs"]["show"]
            rewrite = (outs.get("text") or outs.get("string") or [""])[0]
            final = MS.finalize(rewrite, code=code, prompt=task)
            g = dit_graph(a.dit, w, h, final, 42, "bench_master/%s_%s_%d" % (a.tag, a.dit, idx))
            g["c"] = loader
            _, seg = run(g)
            total = time.time() - t_start
            spec = None
            time.sleep(0.3)
            new = draft_lines()[n_before:]
            if a.draft != "none" and new:
                mm = re.search(r"draft: (\d+) tokens in (\d+) rounds, accepted (\d+)/(\d+)", new[-1])
                if mm:
                    spec = [int(x) for x in mm.groups()]
            rec = {"i": idx, "who": who, "code": code, "task": task, "size": [w, h], "rewrite": rewrite, "final_prompt": final,
                   "pe_load": round(seg_pe.get("c", 0), 3), "pe": round(seg_pe.get("r", 0), 3), "te": round(seg.get("pos", 0), 3),
                   "dit_load": round(seg.get("u", 0), 3), "dit": round(seg.get("k", 0), 3), "vae": round(seg.get("dec", 0), 3),
                   "total": round(total, 3), "spec": spec}
            rows.append(rec)
            print(json.dumps({k: rec[k] for k in ("i", "who", "pe", "te", "dit", "vae", "total", "spec")}), flush=True)
            idx += 1
    stop.set()
    json.dump({"args": vars(a), "gpu": gpu, "peak_vram_mb": peak["mb"], "rows": rows},
              open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=1)

    warm = rows[1:]
    mean = lambda k: sum(r[k] for r in warm) / len(warm)
    sp = [r["spec"] for r in warm if r["spec"]]
    git = lambda d: subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    row = {"date": datetime.date.today().isoformat(), "condition": CONDITION, "tag": a.tag, "gpu": gpu,
           "te_pe_gguf": a.te, "lora": a.lora, "draft": a.draft, "dit": a.dit,
           "pe_s": "%.2f" % mean("pe"), "te_s": "%.2f" % mean("te"), "dit_s": "%.2f" % mean("dit"), "vae_s": "%.2f" % mean("vae"),
           "warm_total_s": "%.2f" % mean("total"), "first_total_s": "%.2f" % rows[0]["total"],
           "tok_per_round": "%.2f" % (sum(s[0] for s in sp) / sum(s[1] for s in sp)) if sp else "",
           "accept_pct": "%.1f" % (100 * sum(s[2] for s in sp) / sum(s[3] for s in sp)) if sp else "",
           "peak_vram_gb": "%.1f" % (peak["mb"] / 1024), "llama_cpp": git(os.path.join(REPO, "vendor", "llama.cpp")),
           "node_commit": git(REPO), "out_dir": os.path.relpath(out_dir, REPO)}
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    new_file = not os.path.exists(TABLE)
    with open(TABLE, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        if new_file:
            wr.writeheader()
        wr.writerow(row)
    write_md()
    print("ROW", json.dumps(row, ensure_ascii=False))


def write_md():
    rows = list(csv.DictReader(open(TABLE)))
    head = ["tag", "gpu", "TE/PE GGUF", "LoRA", "draft", "DiT", "PE", "TE", "DiT s", "VAE", "warm", "first", "tok/round",
            "accept", "peak VRAM", "date"]
    keys = ["tag", "gpu", "te_pe_gguf", "lora", "draft", "dit", "pe_s", "te_s", "dit_s", "vae_s", "warm_total_s",
            "first_total_s", "tok_per_round", "accept_pct", "peak_vram_gb", "date"]
    lines = ["# Master-system lookup table (condition %s)" % CONDITION, "",
             "Generated by `tools/bench_master.py` from `master_lookup.csv`; times in seconds (warm = mean of prompts 2-6), "
             "VRAM in GB (peak of a single-DiT workflow). See the script's docstring for the fixed condition.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join((r.get(k) or "-").replace("|", "/") for k in keys) + " |")
    open(os.path.join(os.path.dirname(TABLE), "MASTER_LOOKUP.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
