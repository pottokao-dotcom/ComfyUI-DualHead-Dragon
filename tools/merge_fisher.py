"""Merge sens_fisher.py outputs (another product's ruler, or more formats) into a model's sensbank table.

Damage entries are merged per tensor / target / format (a later file wins on overlap); bytes likewise; formats and
targets are unioned; scan seconds add up. The manifest records what was merged.

    python tools/merge_fisher.py sensbank/qwen3vl-8b-heretic_8d990a89 fisher_8b_qi21_more.json \
        fisher_8b_ideogram4.json fisher_8b_boogu.json
"""
import json
import os
import sys

ORDER = ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "q2_k", "iq3_xxs", "iq3_s", "q3_k", "iq4_xs", "iq4_nl",
         "q4_k", "q5_k", "q6_k", "q8_0"]


def main():
    bank, parts = sys.argv[1], sys.argv[2:]
    p = os.path.join(bank, "fisher.json")
    a = json.load(open(p))
    for f in parts:
        q = json.load(open(f))
        for k, d in q["damage"].items():
            for t, v in d.items():
                a["damage"].setdefault(k, {}).setdefault(t, {}).update(v)
        for k, d in q["bytes"].items():
            a["bytes"].setdefault(k, {}).update(d)
        a["formats"] = sorted(set(a["formats"]) | set(q["formats"]), key=lambda x: ORDER.index(x) if x in ORDER else 99)
        a["targets"] = sorted(set(a.get("targets", [])) | set(q.get("targets", [])))
        a["seconds"] = a.get("seconds", 0) + q.get("seconds", 0)
        print("merged", f, "targets", q.get("targets"), "formats", q["formats"])
    json.dump(a, open(p, "w"), indent=1)
    mp = os.path.join(bank, "manifest.json")
    m = json.load(open(mp))
    m.setdefault("merged", []).extend(os.path.basename(f) for f in parts)
    m["scan"]["formats"] = a["formats"]
    m["targets"] = a["targets"]
    json.dump(m, open(mp, "w"), indent=1, ensure_ascii=False)
    print("table now: targets %s, formats %s" % (a["targets"], a["formats"]))


if __name__ == "__main__":
    main()
