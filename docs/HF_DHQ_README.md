---
license: apache-2.0
base_model:
- pottokao/Qwen-Image-2.1-Text-Encoder-Heretic
tags:
- gguf
- quantized
- comfyui
- qwen-image
- abliterated
- text-encoder
---

# Qwen-Image-2.1 Text Encoder (Heretic) — DHQ GGUF

**DHQ = Dual-Head Quant.** Same heretic weights as
[Qwen-Image-2.1-Text-Encoder-Heretic-GGUF](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF),
quantized a different way: instead of llama.cpp's fixed rules, every tensor gets the precision that matters to
**what the Qwen-Image-2.1 DiT actually reads** (measured per tensor with gradients through the DiT's text input),
plus an importance matrix weighted the same way. Each file is **exactly the size of the standard llama.cpp file of the
same name**, and better at every level.

Tuned for **Qwen-Image-2.1 only**. Other models that use Qwen3-VL-8B (Ideogram 4, Boogu, JoyImageEdit) can load
these files, but they were not tuned for them. Need Q5/Q6/Q8? Use the standard files in the
[GGUF repo](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF) — at those sizes the standard
recipe is already as good.

## Results

Same heretic bf16 source, same size. **Standard** = llama.cpp default rules + the same imatrix.
**TE** = relative error of the text conditioning after the DiT's own text input layer, on 36 held-out prompts never
used to build or pick anything (lower is better). **PE** = mean KL divergence of next-token predictions vs bf16 on
80 prompt-rewrite samples, i.e. how well the same file still works as a prompt enhancer (lower is better).

| File | Size | TE standard → DHQ | PE standard → DHQ |
|---|---:|---|---|
| `qi21-te-heretic-dhq-Q4_K_M.gguf` | 5.03 GB | 4.4% → **3.5%** | 0.0153 → **0.0126** |
| `qi21-te-heretic-dhq-IQ4_XS.gguf` | 4.56 GB | 4.8% → **4.5%** | 0.0185 → 0.0194 |
| `qi21-te-heretic-dhq-Q3_K_M.gguf` | 4.12 GB | 7.2% → **6.1%** | 0.0559 → **0.0462** |
| `qi21-te-heretic-dhq-IQ3_M.gguf` | 3.90 GB | 9.4% → **7.2%** | 0.0649 → **0.0486** |
| `qi21-te-heretic-dhq-IQ3_S.gguf` | 3.79 GB | 9.8% → **8.0%** | 0.0604 → **0.0542** |
| `qi21-te-heretic-dhq-Q2_K.gguf` | 3.28 GB | 25.4% → **13.2%** | 0.169 → **0.144** |
| `qi21-te-heretic-dhq-IQ2_M.gguf` | 3.05 GB | 20.2% → **13.8%** | 0.169 → 0.177 |
| `qi21-te-heretic-dhq-IQ2_S.gguf` | 2.87 GB | 26.6% → **20.4%** | 0.312 → **0.239** |


## Which file

| Your setup | Pick |
|---|---|
| Plenty of VRAM / RAM | `Q4_K_M` |
| Tight | `Q3_K_M`, then `Q2_K` |
| The IQ files | see the speed note below first |

**K files (`Q4_K_M`, `Q3_K_M`, `Q2_K`) run everywhere and as fast as the standard GGUF.**

> [!WARNING]
> **IQ files (`IQ4_XS`, `IQ3_*`, `IQ2_*`) are slow in ComfyUI-GGUF.** They contain IQ3/IQ2 tensors, which
> ComfyUI-GGUF has no GPU decoder for; it falls back to numpy. Measured on a GB10: **~36 s per prompt** for
> `IQ3_S`, vs **~1 s** for the K files. They are fine in llama.cpp itself. In ComfyUI-GGUF, use the K files.

## How to load (ComfyUI)

Same as the standard GGUF: `CLIPLoaderGGUF`, **type `qwen_image`**, with the add-on node
[ComfyUI-GGUF-Qwen3VL-TE](https://github.com/pottokao-dotcom/ComfyUI-GGUF-Qwen3VL-TE), feeding `TextEncodeQwenImage21`.
Put **`mmproj-qi21-te-heretic-dhq-f16.gguf`** (in this repo) next to the GGUF — the add-on finds the vision tower by
file-name prefix, so it has to carry the `qi21-te-heretic-dhq` name. It is the same file as the standard repo's mmproj.

## Reproducibility

`recipes/` has one recipe per file: source repo and revision, llama.cpp commit, importance matrix (sha256), and the
type of every tensor as built. Rebuilding from a recipe gives a bit-identical file (checked).

## License

Apache-2.0, same as the base weights (Qwen3-VL-8B-Instruct, Alibaba Cloud / Qwen). See `LICENSE` and `NOTICE`.
