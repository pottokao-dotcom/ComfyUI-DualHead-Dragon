// dh_spec: speculative decoding (DFlash / DFlash2 / EAGLE-3 / ... whatever the draft GGUF declares) for the node's
// PE role, on top of llama.cpp's own common/speculative. The whole draft -> verify -> accept loop runs here in C++;
// Python only calls dh_spec_generate() once per rewrite, so no per-round round trips and no torch.
//
// The draft gets its own llama_context whose ctx_other is the node's target context: it borrows the target's token
// embeddings / lm_head and reads the target layers it was trained on through llama_set_embeddings_layer_inp (not the
// cb_eval taps the TE role uses, so the two roles do not interfere).
//
// Follows examples/speculative-simple/speculative-simple.cpp (single sequence, partial KV removal).
//
// Optional tree verification (dh_spec_set_strategy, docs/SPEC_STRATEGY.md): with a DFlash2 draft the verify budget
// can be spent on a token tree built from the selector lattice (dh_tree.cpp) instead of the greedy line. Off by
// default; when off the loop below is exactly the plain one.
#include "dh_tree.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define DH_SPEC_API __declspec(dllexport)
#else
#    define DH_SPEC_API __attribute__((visibility("default")))
#endif

struct dh_spec {
    common_params params;
    common_speculative_init_result_ptr init;
    common_speculative * spec = nullptr;
    llama_context * ctx_tgt = nullptr;
    std::string type_name;
    // stats of the last dh_spec_generate
    int n_drafted = 0, n_accept = 0, n_rounds = 0;
    // strategy (dh_spec_set_strategy): tree verification budget, 0 = off
    int tree_budget = 0;
};

extern "C" {

// libllama-common, patches/0003-dflash2-tree-hooks.patch: the last DFlash2 selector lattice of sequence 0
int dh_dflash2_lattice(const float ** data, int32_t * n_block, int32_t * top_k);

// draft_path: draft GGUF; n_max: max draft tokens per round (clamped to the draft's block size by llama.cpp);
// n_gpu_layers: draft layers on GPU (-1 = all). Returns NULL on failure.
static void * dh_spec_create_impl(llama_context * ctx_tgt, const char * draft_path, int n_max, int n_gpu_layers) {
    if (ctx_tgt == nullptr || draft_path == nullptr) {
        return nullptr;
    }
    auto h = std::make_unique<dh_spec>();
    h->ctx_tgt = ctx_tgt;
    common_params & p = h->params;
    p.speculative.types = common_speculative_types_from_gguf(draft_path);
    if (p.speculative.types.empty() || p.speculative.types[0] == COMMON_SPECULATIVE_TYPE_NONE) {
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    }
    h->type_name = common_speculative_type_name_str(p.speculative.types);
    p.speculative.draft.mparams.path = draft_path;
    p.speculative.draft.n_max = std::max(1, n_max);
    p.speculative.draft.n_gpu_layers = n_gpu_layers < 0 ? 999 : n_gpu_layers;
    p.n_ctx = (int32_t) llama_n_ctx(ctx_tgt);
    // n_batch like the target so a whole prompt chunk's features fit one inject batch; the ubatch stays small, the
    // draft's compute buffer scales with it (2048 -> ~2.4 GB, 512 -> a quarter) and it only ever decodes 16 tokens
    p.n_batch = (int32_t) llama_n_batch(ctx_tgt);
    p.n_ubatch = std::min<int32_t>(512, p.n_batch);
    p.n_parallel = 1;

    common_params p_dft = common_base_params_to_speculative(p);
    h->init = common_speculative_init_from_params(p_dft, (llama_model *) llama_get_model(ctx_tgt), ctx_tgt);
    if (!h->init || h->init->context() == nullptr) {
        return nullptr;
    }
    p.speculative.draft.ctx_tgt = ctx_tgt;
    p.speculative.draft.ctx_dft = h->init->context();
    h->spec = common_speculative_init(p.speculative, 1);
    if (h->spec == nullptr) {
        return nullptr;
    }
    return h.release();
}

// C++ exceptions must not cross into Python (ctypes would abort the whole ComfyUI process)
DH_SPEC_API void * dh_spec_create(llama_context * ctx_tgt, const char * draft_path, int n_max, int n_gpu_layers) {
    try {
        return dh_spec_create_impl(ctx_tgt, draft_path, n_max, n_gpu_layers);
    } catch (const std::exception & e) {
        fprintf(stderr, "dh_spec_create: %s\n", e.what());
        return nullptr;
    }
}

// tree_budget: draft tokens verified per round as a tree (0 = off, plain greedy line). Only DFlash2 drafts have a
// lattice; other types keep the plain loop. Returns the budget in effect.
DH_SPEC_API int dh_spec_set_strategy(void * handle, int tree_budget) {
    auto * h = (dh_spec *) handle;
    if (h == nullptr) {
        return 0;
    }
    h->tree_budget = std::max(0, std::min(tree_budget, 63));
    return h->tree_budget;
}

DH_SPEC_API const char * dh_spec_type(void * handle) {
    return handle ? ((dh_spec *) handle)->type_name.c_str() : "";
}

DH_SPEC_API void dh_spec_free(void * handle) {
    auto * h = (dh_spec *) handle;
    if (h == nullptr) {
        return;
    }
    if (h->spec) {
        common_speculative_free(h->spec);
    }
    delete h;  // frees the draft context and model (init result owns them)
}

// Generate a continuation of prompt[0..n_prompt) on sequence 0 of the target context (which must be empty for seq 0;
// the caller attaches the rewrite LoRA before). temp <= 0 -> greedy. Writes up to max_new tokens (EOG not included)
// to out and returns how many; -1 on error. stats (may be NULL) receives {n_drafted, n_accept, n_rounds}.
static int dh_spec_generate_impl(void * handle, const llama_token * prompt, int n_prompt, int max_new,
                     float temp, int top_k, float top_p, float min_p, uint32_t seed,
                     llama_token * out, int * stats) {
    auto * h = (dh_spec *) handle;
    if (h == nullptr || n_prompt < 1 || max_new < 1) {
        return -1;
    }
    llama_context * ctx_tgt = h->ctx_tgt;
    llama_context * ctx_dft = h->params.speculative.draft.ctx_dft;
    const llama_model * model = llama_get_model(ctx_tgt);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_seq_id seq = 0;
    if (common_context_can_seq_rm(ctx_tgt) != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
        return -1;  // recurrent targets would need checkpoints; the node's models are plain transformers
    }

    common_params_sampling sp;
    sp.temp = temp;
    sp.top_k = top_k > 0 ? top_k : 0;
    sp.top_p = top_p > 0.0f ? top_p : 1.0f;
    sp.min_p = min_p;
    sp.seed = seed;
    sp.penalty_repeat = 1.0f;
    sp.no_perf = true;
    common_sampler_ptr smpl(common_sampler_init(model, sp));

    llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, -1, -1);
    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);

    // prefill prompt[0..n-1) in n_batch chunks; the draft must see the target features of every chunk
    const int n_batch = (int) llama_n_batch(ctx_tgt);
    {
        llama_batch b = llama_batch_init(n_batch, 0, 1);
        for (int s = 0; s < n_prompt - 1; s += n_batch) {
            common_batch_clear(b);
            for (int i = s; i < std::min(n_prompt - 1, s + n_batch); ++i) {
                common_batch_add(b, prompt[i], i, { seq }, false);
            }
            if (llama_decode(ctx_tgt, b) != 0 || !common_speculative_process(h->spec, b)) {
                llama_batch_free(b);
                return -1;
            }
        }
        llama_batch_free(b);
    }

    llama_token id_last = prompt[n_prompt - 1];
    llama_tokens prompt_tgt(prompt, prompt + n_prompt - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
    int n_past = n_prompt - 1;
    common_speculative_begin(h->spec, seq, prompt_tgt);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    llama_tokens draft;
    int n_out = 0;
    h->n_drafted = h->n_accept = h->n_rounds = 0;
    bool done = false;

    // tree verification: one KV sequence per branch. The node's context has 1 + prefix_slots sequences; the cached
    // system prompts in 1.. are dropped after every speculative run anyway (dh_backend.py), so borrow them.
    const int n_seq_all = (int) llama_n_seq_max(ctx_tgt);
    const bool tree_on = h->tree_budget > 0 && n_seq_all > 1;
    dh_tree tree;
    llama_batch tbatch = llama_batch_init(tree_on ? h->tree_budget + 1 : 1, 0, n_seq_all);
    llama_batch pbatch = llama_batch_init(tree_on ? h->tree_budget + 1 : 1, 0, 1);
    // DH_SPEC_LAT=path (debug / tools/spec_tree): append every round's lattice for offline simulation
    FILE * lat_f = nullptr;
    if (const char * lp = getenv("DH_SPEC_LAT")) {
        lat_f = fopen(lp, "ab");
    }
    auto free_batches = [&] {
        llama_batch_free(batch); llama_batch_free(tbatch); llama_batch_free(pbatch);
        if (lat_f) {
            fclose(lat_f);
        }
    };
    if (tree_on) {
        for (int s = 1; s < n_seq_all; ++s) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), s, -1, -1);
        }
    }

    while (!done) {
        int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
        n_draft_max = std::max(0, std::min(n_draft_max, max_new - n_out - 1));
        common_speculative_get_draft_params(h->spec, seq) = { true, n_draft_max, n_past, id_last, &prompt_tgt, &draft };
        common_speculative_draft(h->spec);
        // the draft may have written past the target's KV while drafting; trim it back to the target's end
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq) + 1, -1);

        const float * lat = nullptr;
        int32_t lat_nb = 0, lat_k = 0;
        if ((tree_on || lat_f) && !draft.empty() && !dh_dflash2_lattice(&lat, &lat_nb, &lat_k)) {
            lat = nullptr;
        }
        if (lat_f && lat) {
            const int32_t hdr[6] = { 0x4C415431, n_past, id_last, lat_nb, 0, lat_k };
            fwrite(hdr, sizeof(int32_t), 6, lat_f);
            fwrite(lat, sizeof(float), (size_t) lat_nb * lat_k * (1 + lat_k), lat_f);
        }
        if (tree_on && lat) {
            // the tree may not go deeper than the (possibly shortened near the end) greedy draft
            dh_tree_build(lat, lat_nb, lat_k, h->tree_budget, n_seq_all, (int) draft.size(), tree);
        } else {
            tree.parent.clear(); tree.depth.clear(); tree.tok.clear();
        }
        if (tree.size() > 0) {
            const int nt = tree.size(), p0 = n_past;
            auto * mt = llama_get_memory(ctx_tgt);
            // leaves get sequences 0, 1, ...; every node carries the sequences of the leaves below it
            std::vector<int> n_kids(nt + 1, 0);
            for (int i = 0; i < nt; ++i) {
                n_kids[tree.parent[i]]++;
            }
            std::vector<std::vector<llama_seq_id>> sets(nt + 1);
            int n_leaf = 0;
            for (int i = 1; i <= nt; ++i) {
                if (n_kids[i] == 0) {
                    sets[i].push_back(n_leaf++);
                }
            }
            for (int i = nt; i >= 1; --i) {  // children come after their parent: fold upwards
                auto & ps = sets[tree.parent[i - 1]];
                ps.insert(ps.end(), sets[i].begin(), sets[i].end());
            }
            // a token shared by several sequences needs them to hold the same history: tag the prefix on each
            for (int s = 1; s < n_leaf; ++s) {
                llama_memory_seq_cp(mt, seq, s, -1, -1);
            }
            common_batch_clear(tbatch);
            common_batch_add(tbatch, id_last, p0, sets[0], true);
            for (int i = 1; i <= nt; ++i) {
                common_batch_add(tbatch, tree.tok[i - 1], p0 + tree.depth[i - 1], sets[i], true);
            }
            if (llama_decode(ctx_tgt, tbatch) != 0) {
                free_batches();
                return -1;
            }
            // the target samples node by node and walks to the child holding its token
            llama_tokens ids;
            std::vector<char> on_path(nt + 1, 0);
            on_path[0] = 1;
            int cur = 0, n_acc = 0;
            while (true) {
                const llama_token t = common_sampler_sample(smpl.get(), ctx_tgt, cur);
                common_sampler_accept(smpl.get(), t, true);
                ids.push_back(t);
                int nx = -1;
                for (int i = 1; i <= nt; ++i) {
                    if (tree.parent[i - 1] == cur && tree.tok[i - 1] == t) {
                        nx = i;
                        break;
                    }
                }
                if (nx < 0) {
                    break;
                }
                on_path[nx] = 1;
                cur = nx;
                ++n_acc;
            }
            // target KV: move the accepted path to sequence 0, drop the borrowed sequences
            if (std::find(sets[cur].begin(), sets[cur].end(), seq) == sets[cur].end()) {
                llama_memory_seq_rm(mt, seq, p0 + 1, -1);
                llama_memory_seq_cp(mt, sets[cur][0], seq, p0 + 1, p0 + 1 + n_acc);
            }
            for (int s = 1; s < n_leaf; ++s) {
                llama_memory_seq_rm(mt, s, -1, -1);
            }
            // the draft ingests the root + accepted path only (rows stay aligned with the target batch; others are
            // tagged with a sequence the single-sequence draft skips)
            common_batch_clear(pbatch);
            for (int r = 0; r < tbatch.n_tokens; ++r) {
                common_batch_add(pbatch, tbatch.token[r], tbatch.pos[r], { on_path[r] ? seq : 1 }, false);
            }
            if (!common_speculative_process(h->spec, pbatch)) {
                free_batches();
                return -1;
            }
            common_speculative_accept(h->spec, seq, (uint16_t) n_acc);
            n_past = p0 + (int) ids.size();
            h->n_drafted += nt;
            h->n_accept += n_acc;
            h->n_rounds += 1;
            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_tgt.push_back(id_last);
                id_last = ids[i];
                if (llama_vocab_is_eog(vocab, id_last) || n_out >= max_new) {
                    done = true;
                    break;
                }
                out[n_out++] = id_last;
            }
            draft.clear();
            llama_memory_seq_rm(mt, seq, n_past, -1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1);
            if (n_out >= max_new) {
                done = true;
            }
            continue;
        }

        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past++, { seq }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch, draft[i], n_past + i, { seq }, true);
        }
        if (llama_decode(ctx_tgt, batch) != 0 || !common_speculative_process(h->spec, batch)) {
            free_batches();
            return -1;
        }
        const size_t n_draft = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        common_speculative_accept(h->spec, seq, (uint16_t) (ids.size() - 1));
        n_past += (int) ids.size() - 1;
        h->n_drafted += (int) n_draft;
        h->n_accept += (int) ids.size() - 1;
        h->n_rounds += 1;
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (llama_vocab_is_eog(vocab, id_last) || n_out >= max_new) {
                done = true;
                break;
            }
            out[n_out++] = id_last;
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1);
        if (n_out >= max_new) {
            done = true;
        }
    }
    free_batches();
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, -1, -1);
    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);
    if (stats) {
        stats[0] = h->n_drafted;
        stats[1] = h->n_accept;
        stats[2] = h->n_rounds;
    }
    return n_out;
}

DH_SPEC_API int dh_spec_generate(void * handle, const llama_token * prompt, int n_prompt, int max_new,
                     float temp, int top_k, float top_p, float min_p, uint32_t seed,
                     llama_token * out, int * stats) {
    try {
        return dh_spec_generate_impl(handle, prompt, n_prompt, max_new, temp, top_k, top_p, min_p, seed, out, stats);
    } catch (const std::exception & e) {
        fprintf(stderr, "dh_spec_generate: %s\n", e.what());
        return -1;
    }
}

}  // extern "C"
