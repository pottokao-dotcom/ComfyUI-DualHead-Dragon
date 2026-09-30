# Speculative rewrite bench: tok/s (decode only), accept stats, outputs saved for identity checks.
#   DH_LLAMA_LIB_PATH=<lib dir> python bench.py NAME --n-max 7 --tree 0 --nq 30 --mode s [--repo <node repo>]
# --lat also dumps the draft lattice per round (tools for docs/SPEC_STRATEGY.md). mode s = the node's default sampling (temp .7 / top_p .8 / top_k 20, seed 1000+i), g = greedy.
# Paths default to blackhole's; override with --te/--lora/--draft/--data. Writes out_NAME.json.
import argparse, json, os, sys, time
H = os.path.expanduser
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--n-max", type=int, default=7); ap.add_argument("--tree", type=int, default=0)
ap.add_argument("--nq", type=int, default=30); ap.add_argument("--start", type=int, default=0); ap.add_argument("--mode", default="s")
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap.add_argument("--te", default=H("~/ComfyUI/models/text_encoders/qi21-te-heretic-dhq-Q4_K_M.gguf"))
ap.add_argument("--lora", default=H("~/ComfyUI/models/loras/qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928-ck1174-f16.gguf"))
ap.add_argument("--draft", default=H("~/dflash_8b_gguf/run4_eval/qwen3vl-8b-heretic_master-pe_dflash2_v4.1-b16-run4-gen3wf2_20260929-v16384-Q4_K_M.gguf"))
ap.add_argument("--data", default=H("~/dflash_8b_gguf/heldout_8b.jsonl"))
ap.add_argument("--lat", action="store_true", help="also dump every round's DFlash2 lattice to lat_NAME.bin (sim.py / check.py)")
a = ap.parse_args()
sys.path.insert(0, a.repo)
import dh_backend as D
rows = [json.loads(l) for l in open(a.data)][a.start:a.start + a.nq]
kw = dict(do_sample=False) if a.mode == "g" else dict(do_sample=True, temperature=0.7, top_k=20, top_p=0.8)
tree = dict(draft_tree=a.tree) if a.tree else {}   # old backends have no draft_tree argument
be = D.DualHeadBackend(a.te, n_ctx=8192); L = [(a.lora, 1.0)]
pf = []
for r in rows:   # prefill time per prompt, subtracted below so tok/s is decode only
    t = time.time(); be.generate(r["prompt_ids"], max_length=1, do_sample=False, lora=L); pf.append(time.time() - t)
be.generate(rows[0]["prompt_ids"], max_length=64, lora=L, draft=a.draft, draft_n_max=a.n_max, seed=1, **kw, **tree)   # warm-up
if a.lat:   # dh_spec reads DH_SPEC_LAT at every generate: set it after the warm-up
    if os.path.exists(f"lat_{a.name}.bin"): os.remove(f"lat_{a.name}.bin")
    os.environ["DH_SPEC_LAT"] = os.path.abspath(f"lat_{a.name}.bin")
ac = rd = dr = tok = 0; outs = []; t0 = time.time()
for i, r in enumerate(rows):
    o = be.generate(r["prompt_ids"], max_length=1024, lora=L, draft=a.draft, draft_n_max=a.n_max, seed=1000 + a.start + i, **kw, **tree)
    s = be.last_spec; ac += s["accepted"]; rd += s["rounds"]; dr += s["drafted"]; tok += len(o)
    outs.append(dict(i=a.start + i, n_prompt=len(r["prompt_ids"]), out=[int(x) for x in o], accepted=s["accepted"], rounds=s["rounds"]))
dec = tok / (time.time() - t0 - sum(pf))
json.dump(outs, open(f"out_{a.name}.json", "w"))
print(f"RESULT {a.name}: n_max {a.n_max} tree {a.tree} {a.mode} | 每輪驗 {dr/rd:.2f} | 每輪前進 {ac/rd+1:.3f} | {dec:.1f} tok/s | 輸出 {tok}", flush=True)
