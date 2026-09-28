# TE 量化流程（llama.cpp／GGUF）

雙頭龍用的 Qwen3-VL-8B TE（heretic 版）量化成 GGUF 的完整做法。每一步都用 **vendor/llama.cpp 鎖定的同一個 commit**（`4df29be`），也就是節點執行時用的那一版。

一鍵執行：

```bash
python build.py --with-tools                  # 先編好 llama-quantize / llama-imatrix
TIER=t2 tools/quantize_te.sh                  # attention Q8_0 + FFN NVFP4
TIER=t3 PROTECT=<層號> tools/quantize_te.sh   # 再把敏感層的 FFN 保留 Q8_0（層號見第 5 節）
```

---

## 1. 原則

- **TE／PE 的精度優先於 DiT。** DiT 是多步去噪，每一步的量化誤差會互相抵消，NVFP4 表現很好。TE 的輸出是一次算好、每一步去噪都重複使用的 conditioning，量化造成的方向偏差不會抵消，會把整條去噪軌跡往同一個方向推。PE 的生成一個 token 選錯，後面整段都會走偏。
- **該保留精度的就保留。** 照 NVIDIA 給 Qwen3.5 的 recipe（`w4a16_nvfp4-fp8_attn-kv_fp8_cast`）：MLP 用 NVFP4，attention 保留 8 bit（GGUF 用 Q8_0），視覺塔不量化。再依逐層實測的敏感度，把最敏感的幾層 FFN 也保留 Q8_0（概念同 H3 的 T1／T2／T3 分級保護）。
- **量化器要用 imatrix。** ggml 內建的 NVFP4 量化器忽略 imatrix，一律把縮放設成「最大值÷6」，所以改用 `tools/nvfp4q.py`（見第 4 節）。
- **所有決定都靠實測。** 驗收看的是 TE 輸出的隱狀態 cos，最後再做出圖 A/B。

## 2. 來源

- **一律用 HF 發布版**：`pottokao/Qwen-Image-2.1-Text-Encoder-Heretic` 的 4 個 `model-0000x-of-00004.safetensors`（HF transformers 格式，bf16）。
- **不要用 gx10 上的 `~/qi21_te_ablit`**：它跟發布版不是同一份權重（750 個 tensor 裡有 57 個不同，集中在第 10 層以後的 `o_proj`／`down_proj`，相對差 1–2%），是另一次 abliteration。舊的 `qwen3vl_8b_heretic-Q4_K_M.gguf` 就是從它轉出來的。
- 腳本會記錄來源分片的 sha256（`source.sha256`）。

## 3. 步驟

| 步驟 | 做什麼 | 指令重點 |
|---|---|---|
| 1 | 下載 bf16 分片 | `snapshot_download(allow_patterns=["model-0000*", "*.json", ...])` |
| 2 | 轉 BF16 GGUF、f32 mmproj | `vendor/llama.cpp/convert_hf_to_gguf.py --outtype bf16`；`--mmproj --outtype f32` |
| 3 | 校準資料 → imatrix | 官方兩份 PE system prompt（再加 `CALIB_EXTRA`）；`llama-imatrix -c 512 -ngl 999` |
| 4 | 決定每個 tensor 的型別 | `llama-quantize --imatrix ... --tensor-type <規則> BF16.gguf OUT.gguf Q8_0` |
| 5 | 換成更好的 NVFP4 資料 | `tools/patch_nvfp4.py BF16.gguf OUT.gguf IMATRIX.gguf 3` |
| 6 | 記錄與驗收 | sha256、tensor 型別統計、`tools/bench_heretic.py` |

### 分級規則（第 4 步）

基底型別一律是 `Q8_0`，所以沒被規則命中的 tensor（token_embd、output、T2／T3 的 attention）都是 Q8_0；norm 等 1D tensor 保留 F32。

| 分級 | `--tensor-type` 規則（依序） |
|---|---|
| T1 | `blk\..*\.(attn_(q|k|v|output)|ffn_(gate|up|down))\.weight=nvfp4` |
| T2 | `blk\..*\.ffn_(gate|up|down)\.weight=nvfp4` |
| T3 | `blk\.(<PROTECT>)\.ffn_(gate|up|down)\.weight=q8_0`，接著 `blk\..*\.ffn_(gate|up|down)\.weight=nvfp4`（`<PROTECT>` 例如 `1|2|35`，實際層號以第 5 節的實驗結果為準） |

**規則是第一條命中就生效**（`llama-quant.cpp` 找到就 break），所以保護層的規則一定要放前面。比對是部分比對，`blk\.1\.` 有句點，不會誤中 `blk.10.`。

## 4. NVFP4 量化器（`tools/nvfp4q.py`）

- **格式**：ggml 的 `block_nvfp4`，每 64 個值 36 bytes：`d[4]`（每 16 個值一個 UE4M3 縮放）加上 `qs[32]`（第 s 組的第 j 個 byte 是 `idx[j] | idx[j+8]<<4`）。值＝`kvalues_mxfp4[idx] × ue4m3_to_fp32(d)`，後者自帶 ×0.5。**沒有** ModelOpt 那層 per-tensor 的 fp32 全域縮放。
- **ggml 原本的做法**：縮放一律取「最大值÷6」，imatrix 在程式裡直接被忽略（`GGML_UNUSED(quant_weights)`）。
- **我們的做法**：在 ggml 的縮放附近各試 ±3 個 E4M3 編碼，以 imatrix 加權的平方誤差挑出最好的；數值編碼沿用 `best_index_mxfp4`；平手時保留 ggml 的選擇，所以不會比原本差。
- **相容性**：radius=0 時輸出跟 `ggml_quantize_chunk` **位元完全相同**（`tools/verify_nvfp4q.py` 驗過 5 個 tensor），llama.cpp 照常載入和運算。
- **效果**（heretic，imatrix 加權的相對誤差）：L1 ffn_up 0.229→0.193、L1 ffn_down 0.096→0.055、L10 attn_q 0.089→0.061、L20 ffn_gate 0.096→0.077、L35 ffn_down 0.157→0.074。

## 5. 敏感度實驗（決定 T3 要保護哪幾層）

`tools/sens_sweep.py`：heretic HF 模型以 bf16 載到 GPU，每次只把**某一層**的 FFN（或 attention）換成 nvfp4q 量化後再還原的權重，量 TE 輸出（去掉 system turn、final norm 之前的最後一層隱狀態，也就是 DiT 讀的訊號）跟 bf16 的 1−cos，排出每層的傷害佔比，結果存在 `sensitivity.json`。挑傷害最大的幾層放進 `PROTECT`。約 7 分鐘（gx10）。

heretic 的結果（2026-09-25，14 段校準文字）：
- **attention 佔總傷害 42%，FFN 58%** → attention 保留 Q8_0（T2）是必要的。
- 傷害分布很平均，單一組最多約 3%。FFN 前 6 名：第 **6、35、5、11、34、0** 層；attention 最敏感的是第 34 層。
- 第 1 層 FFN 的權重誤差最大，對輸出的影響卻不在前面 → 只看權重誤差會挑錯層，一定要量輸出。
- 每保護一層 FFN（NVFP4→Q8_0）約多 75 MB。

## 6. 驗收

- **tensor 型別統計**要符合分級（例如 T2：252 個 NVFP4 → 只有 FFN 的 108 個）。
- **`bench_heretic.py`**：跟同一模型的 BF16 GGUF 比，量文字 cos、圖後文字 cos、視覺 token cos、編碼時間、生成速度。
- **出圖 A/B**：官方三種流程（t2i、edit、去背），跟 BF16 GGUF 同 seed 比較。
- 通過之後才上傳，並附上 sha256。

## 7. 已知的坑

- `llama-quantize` 的說明裡**沒有列出 NVFP4**，只能用 `--tensor-type ...=nvfp4` 指定，基底型別用 Q8_0。
- **imatrix 對 ggml 內建的 NVFP4 沒有任何作用**（實測：NVFP4 和 NVFP4+imatrix 的數字完全一樣），所以第 5 步不能省。
- **工具要跟執行時同一個版本**：用 `build.py --with-tools` 編出來的 `llama-quantize`／`llama-imatrix`，以及 `vendor/llama.cpp` 的 `convert_hf_to_gguf.py` 和 `gguf-py`。
- **NVFP4 的次正規數區**：區塊縮放（最大值÷6）小於約 0.0156 時只剩 3 bit 精度，小於約 0.001 時整個區塊會變成 0。第 1 層的 FFN 有 60–85% 的區塊落在這一區，是它特別差的原因之一。
- **磁碟空間**：下載 17 GB、BF16 GGUF 16 GB，每個量化版 5–6 GB。
- 跑 comfy 相關的腳本（tokenizer）需要 `COMFYUI_DIR`；gx10 上的 lab 版還要 `PYTHONPATH=~/lab_extra`。
- 量測速度時先關掉 ComfyUI，避免搶 GPU。

## 8. 實測結果（gx10，GB10）

### 原版 TE，以官方 comfy bf16 為基準

| 模型 | 大小 | 文字 cos | 圖後文字 | 視覺 token | 生成 tok/s |
|---|---|---|---|---|---|
| comfy w4a8（官方） | 6.31 GB | 0.958 | 0.923 | 0.805 | — |
| comfy NVFP4（ours） | 6.31 GB | 0.970 | 0.960 | 0.908 | — |
| GGUF NVFP4（ggml 內建，全部） | 5.24 GB | 0.942 | 0.921 | 0.823 | 49.2 |
| GGUF Q4_K_M | 5.03 GB | 0.983 | 0.975 | 0.888 | 40.5 |
| GGUF Q5_K_M | 5.85 GB | 0.992 | 0.991 | 0.927 | 35.6 |
| GGUF Q6_K | 6.73 GB | 0.997 | 0.996 | 0.937 | 30.4 |
| GGUF Q8_0 | 8.71 GB | 0.999 | 0.999 | 0.944 | 26.5 |
| GGUF BF16 | 16.39 GB | 1.000 | 1.000 | 0.946 | 15.3 |

視覺 token 最高只到約 0.946（連 BF16 GGUF 也是），這是 llama.cpp 和 comfy 視覺塔之間的差異，不是量化造成的。

### heretic 版，以 heretic BF16 GGUF 為基準

| 模型 | 大小 | 文字 cos | 圖後文字 | 視覺 token | 生成 tok/s |
|---|---|---|---|---|---|
| NVFP4（ggml 內建，全部） | 5.24 GB | 0.938 | 0.929 | 0.851 | 49.4 |
| NVFP4＋imatrix（ggml 內建） | 5.24 GB | 0.938 | 0.929 | 0.851 | 49.4 |
| Q4_K_M＋imatrix | 5.03 GB | 0.983 | 0.983 | 0.957 | 40.8 |
| Q5_K_M | 5.85 GB | 0.990 | 0.991 | 0.971 | 35.8 |
| NVFP4 T2（attn Q8_0＋nvfp4q FFN） | 5.99 GB | 0.969 | 0.967 | 0.905 | 42.1 |
| NVFP4 T3（T2＋保護 0,5,6,11,34,35） | 6.44 GB | 0.976 | 0.974 | 0.928 | 38.7 |

**結論（2026-09-25）**：在 llama.cpp 裡，NVFP4 GGUF 即使做了混合精度、縮放搜尋和敏感層保護，仍然**輸給 Q4_K_M＋imatrix**（T3 大 1.4 GB、精度較低、速度也沒贏）。原因是 llama.cpp 的 NVFP4 在 GB10 上走 DP4A（int8）路徑，沒有用到 FP4 tensor core，而且格式少了 per-tensor 全域縮放。**建議 GGUF 用 Q4_K_M＋imatrix，要更準用 Q5_K_M**；NVFP4 留給 comfy／ModelOpt 格式和 DiT。

## 9. DiT 量尺：cos 會低估誤差（2026-09-26）

**起因**：之前驗收看的是 TE 隱狀態的 cos。但 Qwen 的殘差流有少數幾個「巨大通道」（massive activations），比其他通道大幾十到兩百倍，cos 主要由它們決定，而它們量化後幾乎不動。DiT 讀文字之前會先做 norm／投影，其他通道的誤差就會放大。所以改用四種分數一起看（`tools/sens_dit.py`）：

- **raw**：讀取點隱狀態的 cos（以前用的）
- **nm**：拿掉巨大通道後的 cos
- **dit**：過了 DiT 自己的文字輸入層之後的 cos（Z-Image `cap_embedder`、klein `txt_in`＋LayerNorm、QI2.1 `txt_in` MLP）
- **rel／rel_dit**：相對 L2 誤差，分別在讀取點和 DiT 輸入層之後量

設定：用 comfy 官方 TE 類別跑（模板、pad、讀哪層都和 ComfyUI 相同）；ggml 自己的量化器（`ggml_quantize_chunk`，含 imatrix，數值和 llama-quantize 寫出來的一樣）；18 段 prompt（12 段真實八股擴寫輸出＋6 句短句，其中 2 句中文），每段先平均再跨段平均。巨大通道：Z-Image 第 0、4、56 維（中位數的 76／91／22 倍）；klein 三層各有 2–5 個。

### 4B（Qwen3-4B），全部層套同一格式

| 格式 | Z-Image raw | nm | dit | rel_dit | klein 真 token dit | rel_dit | klein pad raw | pad rel | pad rel_dit |
|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | 0.99995 | 0.99984 | 0.99991 | **1.3%** | 0.99996 | 0.9% | 0.993 | 9.4% | 5.1% |
| Q4_K（全部） | 0.99500 | 0.98628 | 0.99169 | **12.4%** | 0.99602 | 8.5% | **0.823** | **57%** | **32%** |
| Q4_K（只壓 FFN） | 0.99699 | 0.99191 | 0.99509 | 9.5% | 0.99766 | 6.5% | 0.877 | 47% | 26% |
| NVFP4（全部，nvfp4q） | 0.99332 | 0.98182 | 0.98897 | 14.5% | 0.99448 | 10.2% | 0.808 | 60% | 34% |
| NVFP4（只壓 FFN） | 0.99563 | 0.98836 | 0.99284 | 11.6% | 0.99659 | 7.9% | 0.852 | 52% | 29% |

**讀法**
- raw cos 0.995 看起來幾乎無損，但到了 DiT 輸入層是 12% 的相對誤差（Q8 是 1.3%，差 10 倍）。**以後驗收一律看 rel_dit／nm，不再只看 raw cos。**
- 拿掉巨大通道後，1−cos 大約是 raw 的 2.7–3 倍。
- **klein 的 pad 在 4 bit 下壞得很嚴重**（cos 0.82，誤差 57%）。klein 的 DiT 會讀 pad，所以 klein 不能用全 4 bit。
- NVFP4 在每一項都比 Q4_K 差，和第 8 節的結論一致。

### 目前部署的 GGUF 壓到哪

| 檔案 | 用途 | 實際型別 | 有沒有依實測保護 |
|---|---|---|---|
| `qwen3_4b_Q8_0.gguf` | Z-Image／klein 4B | 全部 Q8_0（token_embd 也是） | 不需要：沒有 4 bit，上表 Q8 那列就是它 |
| `qwen3vl_8b_heretic-Q4_K_M-imat.gguf` | QI2.1（目前 llama-server 在用） | Q4_K 為主；`attn_v`／`ffn_down` 在第 0–3、6、9、12…（每 3 層）和 31–35 層升 Q6_K；`output` Q6_K；`token_embd` Q4_K | **沒有**。這是 llama.cpp 的預設規則，不是依我們的實測挑的：實測最敏感的 FFN 第 5、11 層沒保護，其他敏感層也只保 `ffn_down`，`gate`／`up` 還是 Q4_K；最敏感的 attention 第 34 層只有 `v` 升 Q6_K，`q`／`k`／`o` 還是 Q4_K |

逐層敏感度見第 10 節。

## 10. 4B 逐層敏感度（DiT 量尺，2026-09-26）

每次只把一組（某一層的 attention 或 FFN）壓成 Q4_K（含 imatrix），其他保持 bf16，量過 DiT 輸入層後的 1−cos。原始數據在 `docs/sens/`（`sens_4b_ffn.json`、`sens_4b_attn.json`、`sens_4b_ruler.json`，另有配置評分和各基準的型別表 `configs/`）。

| | attention：FFN | 前 10 名 | 幾組佔一半傷害（共 72 組） |
|---|---|---|---|
| Z-Image（讀第 34 層） | 40：60 | ffn34、attn34、ffn33、ffn6、ffn32、attn33、ffn28、ffn31、ffn5、attn32 | 21 |
| klein 真 token（讀 8/17/26） | 39：61 | ffn6、ffn26、ffn8、ffn3、ffn25、attn24、ffn5、ffn24、ffn7、ffn23 | 18 |
| klein pad | 46：54 | **attn0（23%）**、ffn4、ffn6、ffn1、ffn5、ffn7、ffn2、ffn0、attn8、ffn8 | **6** |

**讀法**
- **讀取點本身和它正前方的幾層最敏感**（Z-Image 的 28–34；klein 的 21–26、6–8）。llama.cpp 預設「每 3 層升一次」的規則跟這個完全對不上。
- **第 6 層 FFN 在三張榜都是前 4 名**。
- **klein pad 的傷害非常集中**：attention 第 0 層一組就佔 23%，前 6 組佔一半，全在前 8 層。pad 只看真 token，表示在前幾層就定型了。4B 一層 attention 約 2,600 萬參數，Q4→Q8 只多約 13 MB。
- Z-Image 的傷害比較分散（前 21 組才佔一半），單靠混精度的效果會比 klein pad 小。
- Z-Image 用不到第 35 層，klein 用不到第 27–35 層（掃描結果都是 1.00000，這也驗證了讀取點抓得正確）。

### 4B 基準配置（整份模型，含 token_embd）

| 配置 | 大小 | Z-Image rel_dit | klein 真 token | klein pad |
|---|---|---|---|---|
| llama.cpp Q4_K_M | 2.49 GB | 11.0% | 7.9% | 30.8% |
| Unsloth UD-Q4_K_XL | 2.54 GB | 10.5% | 7.4% | 26.4% |
| Unsloth UD-Q5_K_XL | 2.90 GB | 6.1% | 4.4% | 19.0% |
| Unsloth UD-Q6_K_XL | 3.65 GB | 2.7% | 1.8% | 8.8% |
| 目前部署 Q8_0 | 4.27 GB | 1.3% | 0.9% | 5.3% |

Unsloth 是照 LLM 輸出（KL）挑層，所以在 DiT 量尺上跟 llama.cpp 預設落在同一條曲線。**目標：2.5 GB 做到 Q8 的精度。**

## 11. 真實 GGUF 驗收：模擬 vs 引擎（2026-09-26）

以上第 9、10 節都是**模擬量化**（comfy torch TE，權重用 ggml 量化器壓再還原）。這一節是**真實 GGUF 經過雙頭龍引擎（llama.cpp，gx10 GPU）**，對照 comfy 官方 bf16，用同一組 prompt、同一套分數（`sens_dit.py --ggufs`，腳本 `tools/gguf_ab_gx10.sh`，數據 `docs/sens/gguf_ab_4b.json`）。imatrix 由 `llama-imatrix`（鎖定版本）用同一份校準文字產生。

| 檔案 | 大小 | Z-Image rel_dit | klein 真 token | klein pad | 同配置的模擬值（Z-Image） |
|---|---|---|---|---|---|
| BF16 GGUF（引擎下限） | 8.05 GB | 0.28% | 0.20% | 1.5% | 0 |
| Q8_0（目前部署） | 4.28 GB | **2.06%** | 1.38% | 9.3% | 1.34% |
| Q4_K_M（imatrix） | 2.50 GB | 11.51% | 8.26% | 32.4% | 11.03% |
| Q4_K_M＋第 34 層 Q6_K | 2.52 GB | 11.12% | 8.26% | 32.4% | — |
| Q4_K_M＋第 34 層 Q8_0 | 2.54 GB | 11.09% | 8.26% | 32.4% | — |
| Q4_K_M＋第 34 層 F16 | 2.64 GB | 11.08% | 8.26% | 32.4% | — |

**讀法**
- **模擬在 4 bit 級準（差約 5%），在 Q8 級低估約 1.5 倍。** 權重是量化格式時，llama.cpp 會把 activation 也量化成 8 bit，模擬沒有這一段；4 bit 時權重誤差遠大於它，Q8 時它變成主要誤差。⇒ 模擬拿來排名和探索低位元配置；**驗收一律用真實 GGUF**。
- **讀取點用 F16 不值得**：第 34 層 Q6_K 只多 19 MB 就拿到 F16（多 138 MB）91% 的改善，Q8 和 F16 幾乎沒有多的。⇒ 保護層升到 Q6 即可。
- rel 誤差約等於傷害（1−cos）的平方根：消掉 9% 的傷害，rel 只降約 4.5%；rel 要減半得消掉約 75% 的傷害。所以只救少數層看不到大改善，要全面升級。
- klein 四版完全相同（不讀第 34 層），也再次驗證讀取點正確。
- 我們第一版分配（只允許 Q4→Q8）輸給 Unsloth UD-Q5_K_XL（模擬，`docs/sens/cfg_ours_4b.json`）：Q4→Q5 那 1 bit 消掉約 74% 的傷害，是最划算的升級，第一版沒有把它列入候選。
