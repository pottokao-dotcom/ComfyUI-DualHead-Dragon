"""Rebuild the 8B evaluation sets from the bagu v3 dataset (read-only use of moon/blackhole ~/lora_pe/v3/dataset_v3.jsonl).

  pe_eval.txt        PE (rewrite) ruler text: 80 samples (seed 0), ChatML user+assistant turns, no system prompt
  fimat_prompts.json DiT-weighted imatrix prompts: 24 teacher expansions + 12 short requests (seed 7), none in prompts.json
  holdout.json       final-verdict prompts, 24 + 12 (seed 11), in neither of the above

    python tools/make_eval_sets.py ~/lora_pe/v3/dataset_v3.jsonl sensbank/qwen3vl-8b-heretic_8d990a89
"""
import json
import os
import random
import sys

ds, bank = sys.argv[1], sys.argv[2]
L = [json.loads(line) for line in open(ds)]
turn = lambda d, r: [x for x in d["messages"] if x["role"] == r][0]["content"]  # noqa: E731
random.seed(0)
S = random.sample(L, 80)
os.makedirs(os.path.join(bank, "pe"), exist_ok=True)
open(os.path.join(bank, "pe", "pe_eval.txt"), "w").write("".join(
    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>\n" % (turn(d, "user"), turn(d, "assistant")) for d in S))
used = {p["id"] for p in json.load(open(os.path.join(bank, "prompts.json")))}
random.seed(7)
random.shuffle(L)
L = [d for d in L if d["id"] not in used]
out = [{"kind": "expanded", "id": d["id"], "text": turn(d, "assistant")} for d in L[:24]]
out += [{"kind": "short", "id": d["id"] + "_u", "text": turn(d, "user")} for d in L[24:36]]
json.dump(out, open(os.path.join(bank, "fimat_prompts.json"), "w"), ensure_ascii=False, indent=1)

# holdout.json: final-verdict prompts (seed 11), disjoint from prompts.json and fimat_prompts.json; never used to pick configs
used |= {p["id"] for p in out} | {p["id"][:-2] for p in out if p["id"].endswith("_u")}
random.seed(11)
random.shuffle(L)
L = [d for d in L if d["id"] not in used]
hold = [{"kind": "expanded", "id": d["id"], "text": turn(d, "assistant")} for d in L[:24]]
hold += [{"kind": "short", "id": d["id"] + "_u", "text": turn(d, "user")} for d in L[24:36]]
json.dump(hold, open(os.path.join(bank, "holdout.json"), "w"), ensure_ascii=False, indent=1)
