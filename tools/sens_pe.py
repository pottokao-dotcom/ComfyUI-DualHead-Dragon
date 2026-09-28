"""TE-as-PE quantization damage: the same fake quantization as sens_dit.py, scored on the rewrite (prompt expansion)
instead of the DiT conditioning.

Real rewrite samples (system + user + assistant, thinking off, as the bagu LoRA is trained and run) are fed teacher-
forced; on the assistant tokens we compare the quantized model's next-token distribution with bf16:
  kl     mean KL(bf16 || quant) per token
  top1   fraction of tokens whose argmax is unchanged
  dnll   mean change in the log-loss of the actual reference tokens
Ruler = one format on every layer; sweep = one layer group (attn / ffn of one layer) at a time.

    python tools/sens_pe.py --hf ~/qwen3_4b_hf [--lora adapter_dir] --data heldout_t2i.jsonl --calib calib.txt --out pe_4b.json
Needs DH_LIB (dir with libggml-base.so from build.py).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vendor", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "tools"))
from fakequant import GROUPS, fake_quant, imatrix_hooks, load_type_map, tensor_bytes  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer  # noqa: E402


def save(res, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--lora", help="PEFT adapter merged into the weights before quantizing (as a merged GGUF would be)")
    ap.add_argument("--data", required=True, help="jsonl with messages [system, user, assistant]")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sweep", default="q4_k")
    ap.add_argument("--ruler", default="q8_0,q4_k,nvfp4")
    ap.add_argument("--configs", nargs="*", default=[], help="GGUF type maps (json) to score as whole-model configs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    hf = os.path.expanduser(a.hf)
    cfg = AutoConfig.from_pretrained(hf)
    cls = AutoModelForImageTextToText if hasattr(cfg, "vision_config") else AutoModelForCausalLM
    model = cls.from_pretrained(hf, dtype=torch.bfloat16, device_map={"": 0}).eval()
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.expanduser(a.lora)).merge_and_unload().eval()
    tok = AutoTokenizer.from_pretrained(hf)
    layers = next(m for n, m in model.named_modules() if n.endswith("layers") and "visual" not in n)
    L = len(layers)
    lin = {}
    for i, blk in enumerate(layers):
        for g in GROUPS.values():
            for hf_name, _ in g:
                m = blk
                for part in hf_name.split("."):
                    m = getattr(m, part)
                lin[(i, hf_name)] = m
    lin[("embed", "token_embd")] = emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if head.weight.data_ptr() != emb.weight.data_ptr():  # untied: llama.cpp keeps a separate output.weight
        lin[("lm_head", "output")] = head
    orig = {k: m.weight.data.clone().cpu() for k, m in lin.items()}
    print("model", cls.__name__, "layers", L, "lora", a.lora, flush=True)

    rows = [json.loads(x) for x in open(a.data, encoding="utf-8")]
    rows = rows[:: max(1, len(rows) // a.n)][: a.n]
    samples = []
    for r in rows:
        msgs = r["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert full.startswith(prefix), "chat template prefix mismatch"
        ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        start = len(tok(prefix, add_special_tokens=False).input_ids)
        samples.append((ids, start))
    print("samples", len(samples), "assistant tokens", sum(s[0].shape[1] - s[1] for s in samples), flush=True)

    acc, finish = imatrix_hooks({k: m for k, m in lin.items() if k[0] != "embed"})
    calib = [p.strip() for p in open(a.calib, encoding="utf-8").read().split("\n\n") if len(p.strip()) > 200]
    with torch.no_grad():
        for c in calib:
            model(input_ids=tok(c[:4000], return_tensors="pt").input_ids.cuda())
    imat = finish()

    @torch.no_grad()
    def logprobs():
        out = []
        for ids, start in samples:
            lg = model(input_ids=ids).logits[0, start - 1:-1].float()
            out.append(F.log_softmax(lg, -1).to(torch.bfloat16))
        return out

    base = logprobs()

    @torch.no_grad()
    def score():
        kl, top1, dnll = [], [], []
        for (ids, start), b in zip(samples, base):
            lg = model(input_ids=ids).logits[0, start - 1:-1].float()
            q = F.log_softmax(lg, -1)
            bf = b.float()
            tgt = ids[0, start:]
            kl.append(float((bf.exp() * (bf - q)).sum(-1).mean()))
            top1.append(float((bf.argmax(-1) == q.argmax(-1)).float().mean()))
            dnll.append(float((bf.gather(-1, tgt[:, None]) - q.gather(-1, tgt[:, None])).mean()))
        n = len(kl)
        return {"kl": round(sum(kl) / n, 6), "top1": round(sum(top1) / n, 5), "dnll": round(sum(dnll) / n, 6)}

    def apply(keys, qtype):
        for k in keys:
            t = qtype[k] if isinstance(qtype, dict) else qtype
            lin[k].weight.data.copy_(fake_quant(orig[k].cuda(), t, imat.get(k)).to(torch.bfloat16))

    def restore(keys):
        for k in keys:
            lin[k].weight.data.copy_(orig[k])

    res = {"hf": hf, "lora": a.lora, "samples": [r["id"] for r in rows], "ruler": {}, "sweep_format": a.sweep,
           "sweep": []}
    t0 = time.time()
    for qt in [x for x in a.ruler.split(",") if x]:
        for scope, groups in (("all", ("attn", "ffn")), ("ffn", ("ffn",))):
            keys = [(i, h) for i in range(L) for g in groups for h, _ in GROUPS[g]]
            apply(keys, qt)
            res["ruler"]["%s_%s" % (qt, scope)] = r = score()
            restore(keys)
            print("ruler %-6s %-3s %s  (%.0fs)" % (qt, scope, json.dumps(r), time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    res["configs"] = {}
    for path in a.configs:
        tmap = {k: t for k, t in load_type_map(path).items() if k in orig}
        name = os.path.splitext(os.path.basename(path))[0]
        apply(list(tmap), tmap)
        r = score()
        restore(list(tmap))
        gb = sum(tensor_bytes(orig[k].shape, t) for k, t in tmap.items()) / 1e9
        res["configs"][name] = {"gb_scored_tensors": round(gb, 3), "n": len(tmap), **r}
        print("config %s %.2f GB %s  (%.0fs)" % (name, gb, json.dumps(r), time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    if a.sweep:
        for g in ("ffn", "attn"):
            for i in range(L):
                keys = [(i, h) for h, _ in GROUPS[g]]
                apply(keys, a.sweep)
                r = score()
                restore(keys)
                res["sweep"].append({"group": g, "layer": i, **r})
                save(res, a.out)
                print("%-4s L%-2d %s  (%.0fs)" % (g, i, json.dumps(r), time.time() - t0), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("done", a.out)


if __name__ == "__main__":
    main()
