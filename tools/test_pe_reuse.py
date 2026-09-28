exec(open("test_prefix.py").read().split("def gen(seq, n):")[0])
import json, time, pe_core
def run(name, sp, prompt, imgs, task):
    seq = chat(sp, prompt, imgs)
    t0=time.time(); ids = be.generate(seq, max_length=1024, temperature=1.0, top_k=20, top_p=0.95, presence_penalty=pe_core.get_profile(task).presence_penalty, seed=7); tg=time.time()-t0
    raw = tok.decode(ids); rec = pe_core.parse_answer(pe_core.split_thinking(raw)[1], pe_core.get_profile(task)); text = rec["positive_prompt"]
    # (a) TE role, official template, same images
    te = tok.tokenize_with_weights(text, images=imgs)["qwen3vl_8b"][0]; te_seq=[x[0] for x in te]
    t0=time.time(); h_te, sp_te = be.encode(te_seq); tte=time.time()-t0
    # (b) the same text as it sits inside the PE generation context (hidden states of the generated tokens)
    h_pe, _ = be.encode(seq + ids)
    pe_gen = h_pe[-len(ids):]
    te_ids=[x for x in te_seq if isinstance(x,int)]
    # align: text tokens of the TE prompt body inside the generated ids
    body = te_ids[-(len(te_ids)):]
    best=None
    for L in range(len(body), 8, -1):
        for st in range(0, len(body)-L+1, max(1,(len(body)-L)//8 or 1)):
            sub = body[st:st+L]
            for j in range(0, len(ids)-L+1):
                if ids[j:j+L]==sub: best=(st,j,L); break
            if best: break
        if best: break
    st,j,L = best
    # position of that body token inside h_te (h_te includes vision tokens; map text index -> seq index)
    idx=[]; k=0
    for x in te_seq:
        if isinstance(x,int): idx.append(k); k+=1
        else: k += sp_te[0][1] if sp_te else 0
    te_rows = torch.stack([h_te[idx[i]] for i in range(len(te_ids)-len(body)+st, len(te_ids)-len(body)+st+L)])
    c = torch.nn.functional.cosine_similarity(te_rows, pe_gen[j:j+L], dim=-1)
    print("%-6s gen %d tok %.1fs | TE re-encode %d tok %.2fs | aligned %d tokens: cos(PE-context, TE) mean %.3f min %.3f p10 %.3f" % (
        name, len(ids), tg, len(te_seq), tte, L, c.mean(), c.min(), c.quantile(0.1)))
run("t2i", "pe_t2i.txt", "一只在雨中弹吉他的柯基", [], "t2i")
run("edit", "pe_i2i.txt", "Depict this symbol as a flag waving in the sky", [img], "edit")
