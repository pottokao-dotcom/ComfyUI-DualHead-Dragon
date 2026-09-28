# 雙頭龍做法（通用版，不限 Qwen-Image）

目標：任何「LLM 當文字編碼器」的 DiT 產品，都能加上擴寫（PE），也能加速，而且 8–16 GB 的卡跑得動、品質高、速度快、省 VRAM。
Qwen-Image 2.1 是第一個實作，細節見 `spec.html`。這份只記可以搬到別的產品的通則。

## 1. 核心概念

DiT 的文字編碼器（TE）本身就是一顆 LLM，例如 Qwen-Image 用 Qwen3-VL-8B、Z-Image 用 Qwen3-4B。**同一份權重做兩件事：**

| 模式 | 做什麼 | 掛什麼 |
|---|---|---|
| 擴寫（PE） | 自回歸生成，把短 prompt 寫成長描述 | LoRA（蒸餾自官方或去審查的 PE 老師）＋草稿頭（加速） |
| 編碼（TE） | 一次 forward 拿隱狀態，交給 DiT | 預設不掛；可另掛「編碼 LoRA」（例：Z-Image-Engineer-V6 抽出的 adapter） |

- DiT 是跟 TE 的 base 綁在一起訓練的，**base 不能換**，只能量化或做 heretic。換別的 LLM，DiT 看不懂。
- **兩種模式各掛各的 LoRA，互不干擾**（2026-09-26 定）：擴寫的 LoRA 絕不能漏到編碼。擴寫 LoRA 開著編碼，出圖會劣化；編碼 LoRA 是另一回事，是刻意為編碼訓練的（例如 V6 就是拿來當編碼器用的）。
- 各產品的接線不同（取哪層、模板、pad），見 memory 的雙頭龍頁「多產品」表。
- VRAM 裡只放一份權重。擴寫的輸出只是幾 KB 的文字，放在 CPU 上就好。

## 1.5 為什麼擴寫加速是關鍵（各段耗時，gx10）

| 階段 | 耗時 | 出處 |
|---|---|---|
| TE 編碼 | 0.4–1.4 s | 2026-09-24 實測（`finding_gguf_dualhead_encoding_recipe`） |
| PE 擴寫，裸 8B（約 220 字） | 7–17 s | 同上 |
| PE 擴寫，LoRA 版（中位約 900 token，約 23 tok/s） | **約 35–40 s** | 2026-09-26 推算：v2 學生輸出中位 904–951 token |
| DiT，25 步 | 18–20 s | 2026-09-24 實測 |
| DiT，8 步 turbo／LoRA | 約 6 s | 推算（25 步按比例縮） |
| DiT，4 步 | 約 3 s | 推算 |

- 使用者估計擴寫已經佔總成本 40% 以上。LoRA 版擴寫寫得更長，DiT 又一定會出 8 步、4 步的 turbo 版，所以**擴寫會變成八到九成的時間**，成為真正的門檻。
- 目前市面上**沒有人在做擴寫這一段的加速**，這是雙頭龍的差異化重點。
- 同一個 prompt 只換 seed 時，ComfyUI 的節點快取會跳過擴寫和編碼，只重跑 DiT。所以擴寫的成本主要落在「每換一次 prompt」。

## 2. 後端：節點自帶 llama.cpp，而且不改它

- 節點帶一份鎖版本的 llama.cpp（submodule）加上 vendored 的 ctypes 綁定，由 `build.py` 在使用者機器上編譯。這樣不受使用者環境裡其他 llama.cpp 影響。
- **原則：所有功能都用 `llama.h` 的公開 API 在節點裡做完，不打 llama.cpp 補丁。** 這樣出貨時只要推版本號，不用追補丁。
- 編碼的配方：`embeddings=True`、`pooling none`，拿到最後一層過 norm 的隱狀態，再除以 `output_norm.weight` 還原成 norm 之前的值，裁掉 system turn。這樣就跟官方 bf16 TE 等價。

## 3. 擴寫加速（speculative decoding），全部在節點裡做

### 目標（使用者訂，2026-09-26）
- **草稿頭約 400 MB、擴寫速度 3 倍。** 擴寫大約佔總成本 40% 以上；加速 3 倍後只剩約 13%，整體成本大約降到原本的 73%。
- 怎麼達到：
  - EAGLE-3 VL-8B 頭 bf16 799 MB，量化到 FP8／int8 約 400 MB。
  - 草稿詞表可以從 32k 縮小（輸出幾乎都是英文攝影詞），lm_head 還能再省。
  - DFlash 要到 400 MB 得量化到 4-bit，或自己練一顆層數比較少的。
  - 3 倍需要每輪平均接受 4 個以上。專用任務（格式固定）性質接近數學題，DFlash 在數學題上接受長度可以到 5.7。
- 到 3 倍時，每步的額外開銷（`cb_eval` 拷貝、Python 迴圈、torch 頭的呼叫）也要壓低，必要時把頭包成 CUDA graph。
- **對象是 16G 單卡玩家（使用者 2026-09-27）**：雙頭龍的意義就是一張 16G 卡塞得下「擴寫＋編碼＋DiT」，加速也是為單卡玩家做的。草稿頭與 DiT、本體 GGUF、擴寫 LoRA、KV 共用同一張卡，所以 **400 MB（量化後）是硬規格**；詞嵌入與 lm_head 一律借本體；擴寫和出圖分段、各算峰值；驗收一律在 16G 卡（moon 5070 Ti／blackhole 5060 Ti）量峰值 VRAM 與速度。
- **Mac 不在加速的對象內**：Mac 的 DiT 很慢，擴寫佔比低；草稿頭在 MPS 上開銷更大 → 節點預設「有 CUDA 才開草稿頭」。
- **各機型擴寫佔比（v4 8B 答案中位約 640 token，gx10 約 23 tok/s ≈ 28 s，推算）**：DiT 25 步 58%、8 步 turbo 80%、4 步 88%；擴寫加速約 3.5 倍（→ 約 8 s）後，8 步 53%、4 步 67%，總時間從 30 秒以上降到十幾秒。
- **DFlash 草稿頭可選組合（推算）**：4B 5 層 bf16 1,075 MB → 4-bit 約 300 MB；8B 3 層 4-bit 約 330 MB；8B EAGLE-3 8-bit 約 400 MB（縮詞表後更小）。【實測】DFlash 頭 NVFP4 在 vLLM 可用（接受率 40%、每輪 3.40，Muse，2026-08-13），bf16 對照未做；節點內 torch 的 4-bit 路線未測。

### 架構（2026-09-26 原型已驗證可行）
- **target**：節點自帶的 llama.cpp，只用公開 API。
  - 中間層隱狀態：`llama_context_params.cb_eval` 回呼，攔下名叫 `l_out-N` 的張量，也就是第 N 層的輸出、第 N+1 層的輸入。
  - 驗證：一次 `llama_decode` 草稿那一串，每個位置都要 logits。
  - 回退：`llama_memory_seq_rm` 刪掉沒猜中的位置。
- **草稿頭**：在節點裡用 torch 跑，直接讀 HF 的 safetensors，不用轉 GGUF。
  - 好處：推論程式碼就是訓練程式碼，避開「訓練框架和上線計算圖不一致」的坑（MTP 那次離線多 20%、上線少 60%）。
- 原型：`tools/spec_proto/eagle_proto.py`。

### 為什麼不用 llama.cpp 內建的 EAGLE-3 或 DFlash
- 實作在 `common/speculative.cpp`，不在 `llama.h`，綁定叫不到。
- 對 qwen3vl 還要打補丁：`qwen3vl.cpp` 沒有設 `t_layer_inp`，一跑就 assert。
- 每種架構都可能缺這個掛鉤，發布時就得一直追補丁。

### EAGLE-3 慣例（對照 vLLM／SpecForge，實測確認）
- 取第 [2, n/2, n−3] 層的**輸入**（= `l_out-1`、`l_out-(n/2−1)`、`l_out-(n−4)`），依 低、中、高 的順序串接成 3H，再經過 `fc` 變成 H。
  - 實測：改取輸出會略差；順序反過來接受率掉到 0.4%。
- 位置 P 的輸入是 (token[P+1], feat[P])。後續每一步把 draft 自己的輸出（norm 之前、post-MLP 的殘差和）餵回去，不再經過 `fc`。
- 詞嵌入借 target 的（EAGLE-3 的 checkpoint 通常不帶）；draft 詞表用 `d2t` 對回 target 詞表：target_id = d + d2t[d]。
- `norm_before_residual` 預設 False，翻成 True 會崩（0.1%）。

### 草稿頭選型
| | EAGLE-3 | DFlash |
|---|---|---|
| 大小（bf16） | 4B：437 MB；VL-8B：799 MB | 4B：1,075 MB；8B：2,097 MB |
| 做法 | 一層，自回歸一個一個猜 | 多層 block diffusion，一次 forward 出一整塊（16 個） |
| 適合 | 記憶體最省 | 猜中率高的專用任務（格式固定），多猜幾個幾乎不多花時間 |
| 量化 | 可以 | FP8 或 int8 可行（之前 vLLM 實測 NVFP4 draft 也能用） |

### 實測紀錄（gx10，Qwen3-VL-8B Q8，greedy，K=6）
| 題目 | 不加速 | 現成 AQ-MedAI EAGLE-3 | 每輪接受 |
|---|---|---|---|
| 一般聊天 ×3 | 27.2 tok/s | 28.9 tok/s | 約 2.0 |
| 攝影擴寫 ×1 | 22.9 tok/s | 17.8 tok/s | 1.74 |

- 我們量到的每輪約 2.0，跟兩家在 model card 上自己回報的數字一致（taobao-mnn 約 2.0；AQ-MedAI 在 SGLang 樹狀草稿下 2.49），**所以實作是對的，是現成的頭太弱**。
- 頭太弱時，猜錯的成本會把好處吃光，專用任務甚至會變慢。**一定要用自家資料練頭。**
- llama.cpp 內建的 eagle3（打過補丁）量到的接受率也差不多（6–10%），兩套獨立實作結論一致。

### DFlash 節點內原型（2026-09-26，`tools/spec_proto/dflash_proto.py`）
- 照 z-lab 官方 `dflash.py` 移植成 torch，本體一樣走自帶 llama.cpp 的公開 API（`cb_eval` 攔 `l_out-1/9/17/25/33`，也就是第 1、9、17、25、33 層的**輸出**），**不改 llama.cpp**。
- DFlash 的做法：把「上一個確定的 token 加上 15 個 mask token」丟進 5 層草稿頭，不加因果遮罩，一次 forward 出 15 個猜測。本體特徵經過 `fc` 加 `hidden_norm` 後當成每層注意力的 K/V context；快取只保留 context 部分。詞嵌入和 lm_head 都借本體的（tied）。
- 實測：本體是 Z-Image-Engineer-V6 Q4_K_M（Qwen3-4B 微調版），頭是 z-lab/Qwen3-4B-DFlash-b16（1.07 GB，在原廠 Qwen3-4B 上練的），greedy。

| 題目 | 不加速 | DFlash | 每輪接受 |
|---|---|---|---|
| 一般聊天 ×3 | 74.8 tok/s | 91.2 tok/s（×1.22） | 2.97 |
| 　其中寫程式那題 | 75.2 | **203.3（×2.7）** | **6.67** |
| 攝影擴寫 ×4 | 53.2 | 38.5（變慢） | 2.12 |

- 同樣是現成的頭，DFlash 在攝影題每輪 2.12，EAGLE-3（VL-8B）是 1.74。格式固定的內容（例如程式碼）DFlash 每輪可以到 6.7。
- **每輪固定開銷太大**：攝影題一輪約 55 ms，相當於 2.9 次單 token decode（18.8 ms），所以每輪至少要接受約 3 個才打平。聊天題一輪約 2.4 次 decode。
  - 最可疑的：每輪把 16 個位置 × 151,936 維的 logits（約 39 MB）整份拷到 Python，其實只需要 argmax。可以改用 llama.cpp 的 backend sampling（`llama_context_params.samplers`，實驗性公開 API）在 GPU 上直接取樣。
  - 其他：Python 迴圈、`cb_eval` 拷貝、torch 頭沒有包成 CUDA graph。
  - 目標：每輪開銷壓到約 1.3–1.5 次 decode。到時候每輪接受約 4.5 個就有 3 倍。
- greedy 下加速版和不加速版的輸出有少量分歧（例如 "Maintain" 和 "Establish"）。原因是驗證時一次 decode 16 個，跟一次 decode 1 個走的 kernel 不同，數值有微小差異。這是正常現象，不是 bug。

### 擴寫文字的可預測度（下限，2026-09-26，`tools/spec_proto/ngram_pred.py`）
- 資料：v3 老師輸出，1000 筆測試、4000 筆統計，平均 782 token／筆，用 Qwen3-VL tokenizer。
- 只看前 4 個 token 猜下一個（n-gram 退避），不看模型：逐 token 猜中率 **44%**（各筆 p10 40%／中位 45%／p90 48%）。
- 模擬草稿流程（每輪連續猜 K 個）：每輪前進 **1.74–1.78 個 token**，K 從 4 拉到 16 幾乎不變。
- 解讀：句型、連接詞這些骨架很好猜，但主體、顏色、細節每筆都不同，光看表面統計猜不到。真正的草稿頭有看本體的隱狀態（本體已經「知道」自己接下來要寫什麼），所以會明顯高於這個下限。練好的頭要多高，要實際練過才知道。

### 練頭的規矩（MTP／DFlash 踩過的坑）
1. 先拿現成的頭（或零訓練）量一次基準，確定練過的真的有比較好。
2. 訓練資料要用「**掛上 LoRA 的學生**自己的輸出」（on-policy），不能用老師的文字。
3. 隱狀態從跟上線同一套的引擎抽（llama.cpp 的 `cb_eval` 或 vLLM 的官方抽取），不要用別的框架算。
4. 單變數比較：同一個 server、同一輪量，數字才能並排。

## 3.5 DiT 也可以做雙頭龍（base＋加速 LoRA）

**同一份 DiT 權重，靠 LoRA 開關切換兩種角色：**
- 掛加速 LoRA（8 步蒸餾）：速度快、構圖服從性好。
- 不掛 LoRA（素 base）：質感、底片感、細節。

**前提：加速版本身要是 base 上的 LoRA。** 如果 turbo 是重訓出來的，就抽不成 LoRA。

| 模型 | 能不能 | 證據 |
|---|---|---|
| Z-Image turbo vs base | ❌ | 2026-09-19 實測：權重差異有 base 的 51%，而且是高秩，rank 512（2.4 GB）誤差還有 62%。主幹（92% 參數）差異約 50%，時間步條件化（t_embedder、adaLN）被改寫 80–94%。turbo 是重訓，不是微調。 |
| Z-Image base＋8 步 Fun-LoRA | ✅ 同一份權重 | 2026-07-05：Q8 base＋8 步 LoRA w0.8、12 步、shift 6，5060Ti 28 s／張。構圖分散度 0.088（turbo 0.056），治好「主體置中」的毛病。但使用者眼判有「數位光滑」的指紋；素 base 才畫得出底片 halation。4 步 LoRA 會塌回去（判斷是蒸餾配方的問題）。 |
| Qwen-Image 系列（Lightning 等 8 步 LoRA） | ✅ 預期可以 | 加速版以 LoRA 形式發佈；QI2.1 的 8 步 LoRA 預期很快會出。 |

**建議的管線（Z-Image 7/5 的兩段式，改成一份權重）：**
1. 第一段：base＋8 步 LoRA 打草稿（負責構圖）。
2. 第二段：拿掉 LoRA，用素 base 做 denoise 約 0.35 的重繪（補回底片質感和眼睛細節）。
- 原本兩段放在兩張 GPU 上，各載一份。改成雙頭龍之後只載一份 base，VRAM 省下一整顆 DiT。

**待驗證：** ComfyUI 在兩段之間切換 LoRA 的成本。一般做法是把 LoRA 直接合進權重，每切一次就要重套一次；另一種是動態算 LoRA（不合進權重），但每一步都比較慢。要實際量過才知道。

**整條管線的終局：** 一顆 TE／PE（LoRA＋草稿頭）加上一顆 DiT（base＋加速 LoRA）。兩邊都是一份權重加熱插拔 LoRA，8–16 GB 的卡最吃香。

## 4. 移植到新的 DiT 產品的檢查表
1. TE 是哪顆 LLM？架構在 llama.cpp 有沒有支援？每層有沒有 `l_out` 這個 callback 名稱？
2. 編碼配方：取哪一層、norm 前還是後、模板、裁切。要跟官方 TE 逐 token 比 cos，再出圖 A/B。
3. PE 老師：官方有沒有 PE？沒有的話用哪顆大模型產生蒸餾資料？
4. LoRA：在要出貨的那顆 base（含 heretic）上練，**在哪顆 base 上練就只能掛那顆**。
5. 草稿頭：HF 上有沒有現成的 EAGLE-3／DFlash 可以當起點？沒有就從零練。
6. VRAM 預算：TE（Q4 或 Q5）＋ mmproj ＋ DiT ＋ VAE ＋ 草稿頭 ＋ KV。8 GB 要讓 TE 和 DiT 輪流佔用。
