# Speculative decoding for the rewrite (DFlash / DFlash2 / EAGLE-3)

The rewrite (PE role) can use a speculative draft: same output distribution, fewer target passes. The draft runs inside
llama.cpp (`native/dh_spec.cpp` on top of llama.cpp's `common/speculative`); Python calls it once per rewrite, so there
is no per-round Python/torch round trip.

## Use
1. `python build.py` — also builds `libdh_spec` next to the llama.cpp libraries.
2. Put the draft GGUF in `ComfyUI/models/dualhead_drafts/`.
3. **Dual-Head Rewrite** → `draft` = that file, `draft_n_max` = 15 for DFlash b16.
4. DFlash2 only: `draft_tree` (node default **20**, 0 = off) verifies a token tree built from the draft's selector lattice
   instead of one line. Same distribution; with the node's default sampling (temp 0.7), `draft_n_max` 15 + `draft_tree` 20
   measured +25% on an RTX 5060 Ti and +31% on a GB10 over `draft_n_max` 15 alone. Drafts without a selector lattice
   (DFlash, EAGLE-3) ignore it. See `docs/SPEC_STRATEGY.md` (Chinese handbook, all measurements).

The draft type (DFlash, DFlash2, EAGLE-3, ...) is read from the GGUF. vendor/llama.cpp 4da633776 (2026-09-27) knows
DFlash2 (upstream PR #27342); 4df29be4f did not. No DFlash2 draft exists for a 4B target yet, so that path is built but
not run here.

Vendor bump 4df29be4f -> 4da633776 (branch vendor-upgrade): bindings got llama_model_params.lazy_mode,
mtmd_context_params.device, llama_model_quantize_params.max_buf_size; functions removed upstream no longer break the
import. Checked bit-identical against the old build: Z-Image TE layer 34 on 5 expanded prompts (1,970 tokens),
Qwen3-VL 8B vision + text encode, TE before/after a rewrite; DFlash rewrite speed unchanged.

Only text-only prompts without repetition/presence penalties take the speculative path; anything else falls back to the
normal loop. The rewrite LoRA is attached exactly as before; the encoder role is untouched (the draft reads the target
layers it needs through llama.cpp's `llama_set_embeddings_layer_inp`, not the encoder taps).

## Draft for Z-Image-Engineer V6 + master-PE LoRA
`pottokao/Z-Image-Engineer-V6-DFlash-b16` (public, DFlash b16, no selector → no tree): trained on V6 + v4 LoRA outputs of the master system prompt + cards.
Convert/quantize: `convert_hf_to_gguf.py <head> --target-model-dir <V6 HF> --outtype bf16`, then `llama-quantize … Q6_K`
(Q8_0 / Q6_K / Q4_K_M keep the same acceptance as bf16 and are slightly faster).

## Draft for heretic Qwen3-VL 8B + master-PE LoRA v4.1 (QI2.1 / Ideogram 4 / Boogu)
`pottokao/qwen3vl-8b-heretic_master-pe_dflash2_v4.1-b8-ep2_20260928` (private; the name is historical, the current
checkpoint is **run 5, block 16**, 2026-09-30). Use `…-run5-gen34wf2_20260930-vocab16k-Q4_K_M.gguf` with
`draft_n_max` 15 + `draft_tree` 20 (the node defaults). Earlier runs are in that repo's git history; its README has the
training history and measurements.

## Measured (blackhole, one RTX 5060 Ti 16 GB, Z-Image turbo NVFP4 1024² 8 steps, master SP + card, temp 0.7)
| | rewrite | one image (warm) |
|---|---|---|
| node before (no draft) | 6.4 s | 12.7 s |
| V6 dhq Q4_K_M + v4.1 LoRA + DFlash Q6_K | 3.4–3.6 s | 9.2–9.6 s |
| V6 + v4.1 merged (2.9 GB, also used as TE) + DFlash Q6_K | 2.6–2.9 s | 8.6 s |

Notes from the measurements (full report: blackhole `~/dflash_v6/report/`):
- llama.cpp verifies 9–16 tokens for ~1.25× one decode (MMQ); 5–8 tokens is the expensive range (MMVQ) → keep `draft_n_max` 15.
- A runtime LoRA adds ~24% to decode and ~37% to verification; merging removes it but changes the TE output (dhquant ruler 4.49% → 30.78%).
- The draft context uses ubatch 512: with the target's 2048 its compute buffer was 2.4 GB and, on a nearly full card under
  ComfyUI, every round got ~12 ms slower.
- The encoder drops the rewrite's system-prompt cache before encoding: with it present, layer 34 of the same prompt drifted
  2.6% (relative) — the TE output must not depend on what was rewritten before.
