# ComfyUI-DualHeadDragon（雙頭龍）

一顆 GGUF 在 VRAM 裡只載一份，同時擔任 DiT 的 TE（編碼）和 PE（擴寫）。以 Qwen-Image 2.1 為例，一顆 Qwen3-VL-8B GGUF 同時擔任：

- **TE（text encoder）**：把 prompt 和參考圖編碼成 DiT 用的 conditioning
- **PE（prompt enhancer）**：做 PE-T2I／PE-I2I 改寫（官方原本是另外兩顆 Qwen3.5-VL 9B）

Z-Image 和 FLUX.2 klein 4B 則用一顆 Qwen3-4B GGUF（例如 Z-Image-Engineer V6）同樣兼任 TE 和 PE。

後端是 llama.cpp（llama-cpp-python 的 binding + mtmd）。改寫可以掛 DFlash／DFlash2 草稿頭加速（投機解碼，全部在 llama.cpp 裡跑）；
雙頭龍佔的 VRAM 交給 ComfyUI 管，PE／TE 和 DiT 輪流用同一張卡，6 GB 卡也能跑完整流程。

> 狀態：**開發中**。TE 這一頭和改寫的流程都已經通了；PE 的品質要靠之後訓練的 LoRA（見〈待辦〉）。

## 支援的 DiT（擴寫＋編碼）

Loader 的 `type` 跟 ComfyUI 的 CLIPLoader 同名。每一種都用下面〈範例〉那批題目實際跑過；「TE／PE」＝同一顆 GGUF 兼編碼和擴寫（雙頭）。

### 支援程度

| DiT | Loader `type` | TE（編碼） | PE（擴寫） | 參考圖（i2i） | DFlash 加速 | 怎麼驗的 |
|---|---|---|---|---|---|---|
| Z-Image（turbo、base+turbo） | `lumina2` | ✅ | ✅ 4B 大師 LoRA | —（DiT 不收） | ✅ b16 | TE 對 ComfyUI 官方逐位元／數值對照（`tools/val_4b_*`）＋出圖 |
| FLUX.2 klein 4B | `flux2` | ✅ | ✅ 4B 大師 LoRA | 未做 | ✅ b16 | 同上；排版卡的字會壞（DiT 本身） |
| FLUX.2 klein 9B | `flux2` | ✅（需 Qwen3-8B GGUF，範例改用 ComfyUI 原生 TE） | ✅ 借 4B 擴寫 | 未做 | ✅（擴寫在 4B 上） | 出圖；8B Qwen3 沒有大師 LoRA |
| Qwen-Image 2.1 | `qwen_image` | ✅ | ✅ 8B 大師 LoRA | ✅ | — | 官方 bf16 TE 對照（token、image_slots、cos）＋ t2i／編輯出圖 |
| Ideogram 4 | `ideogram4` | ✅ 13 層 | ✅ 8B 大師 LoRA | 未驗 | — | 自我檢查（各讀法互相吻合，誤差 1e-7）＋出圖；官方 prompt 是 JSON，這裡送一般文字 |
| Boogu-Image Turbo | `boogu` | ✅ 最後層＋norm | ✅ 8B 大師 LoRA | 未驗 | — | 同上；排版字最穩 |
| JoyAI-Image-Edit Plus | `joyimage` | ✅ 最後層 | ✅ 8B 大師 LoRA | ❌ 未支援（參考圖前處理跟 mtmd 不同） | — | 出圖（文生圖） |

`type=auto` 只在 GGUF 本身決定得了時有用：Qwen3-VL-8B 預設走 `qwen_image`，Qwen3-4B 要自己選 `lumina2` 或 `flux2`，`ideogram4`／`boogu`／`joyimage` 一律要手選。8B 目前沒有草稿頭（DFlash2 訓練中），擴寫照一般方式跑。

### 下載（範例用的檔）

| DiT | TE／PE GGUF（→ `models/text_encoders/`） | 擴寫 LoRA（→ `models/loras/`） | 草稿頭（→ `models/dualhead_drafts/`） | DiT（→ `models/diffusion_models/`） |
|---|---|---|---|---|
| Z-Image | [Qwen3-4B-Text-Encoder-DHQ-GGUF](https://huggingface.co/pottokao/Qwen3-4B-Text-Encoder-DHQ-GGUF) `zimage_engineer_v6_dhq_2p9.gguf`；其他大小：[Z-Image-Engineer-V6-DHQ-GGUF](https://huggingface.co/pottokao/Z-Image-Engineer-V6-DHQ-GGUF) | [4B 大師 v4.1](https://huggingface.co/pottokao/zeng-v6_master-pe_lora_v4.1-r32-ep2_20260928) `ck1212`（f16／q8_0／q6k-mix／q4k-mix） | [V6-DFlash-b16](https://huggingface.co/pottokao/Z-Image-Engineer-V6-DFlash-b16) `Q6_K`（推薦） | turbo：[Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) `z_image_turbo_nvfp4`；base：自壓 `z-image-base-nvfp4_ultra`（未公開）＋ [Distill-8-Steps LoRA](https://huggingface.co/alibaba-pai/Z-Image-Fun-Lora-Distill) |
| FLUX.2 klein 4B | 同 Z-Image | 同 Z-Image | 同 Z-Image | [black-forest-labs/FLUX.2-klein-4b-nvfp4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4) |
| FLUX.2 klein 9B | 擴寫同 Z-Image；TE：[Comfy-Org/flux2-klein-9B](https://huggingface.co/Comfy-Org/flux2-klein-9B) `qwen_3_8b_fp8mixed` | 同 Z-Image | 同 Z-Image | [black-forest-labs/FLUX.2-klein-9b-nvfp4](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4)（產線用；HF 要先同意 FLUX Non-Commercial 授權）。範例用的是官方 bf16 [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) `flux-2-klein-9b.safetensors`，載入時轉 fp8（同一份授權） |
| Qwen-Image 2.1 | [Qwen-Image-2.1-Text-Encoder-Heretic-DHQ-GGUF](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-DHQ-GGUF) `Q4_K_M`＋`mmproj`（i2i 要） | [8B 大師 v4.1](https://huggingface.co/pottokao/qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928) `ck1174`（f16／q8_0／q6k-mix／q4k-mix） | — | 自壓 `qwen_image_2.1_nvfp4`（未公開） |
| Ideogram 4 | [Ideogram-4-Text-Encoder-Heretic-DHQ-GGUF](https://huggingface.co/pottokao/Ideogram-4-Text-Encoder-Heretic-DHQ-GGUF) `Q4_K_M` | 同 QI2.1 | — | [Comfy-Org/Ideogram-4](https://huggingface.co/Comfy-Org/Ideogram-4) `ideogram4_nvfp4_mixed`＋`ideogram4_unconditional_nvfp4_mixed`（非商用授權） |
| Boogu-Image Turbo | [Boogu-Text-Encoder-Heretic-DHQ-GGUF](https://huggingface.co/pottokao/Boogu-Text-Encoder-Heretic-DHQ-GGUF) `Q4_K_M` | 同 QI2.1 | — | [Comfy-Org/Boogu-Image](https://huggingface.co/Comfy-Org/Boogu-Image) `boogu_image_turbo_hotfix_nvfp4` |
| JoyAI-Image-Edit Plus | 自壓標準 `Q4_K_M`（未上傳；原檔 [jdopensource/JoyAI-Image-Edit](https://huggingface.co/jdopensource/JoyAI-Image-Edit) `JoyAI-Image-Und/`） | 同 QI2.1 | — | 社群 [aha2023/JoyAI-Image-Edit-ComfyUI-NVfp4](https://huggingface.co/aha2023/JoyAI-Image-Edit-ComfyUI-NVfp4) `joyai_image_edit_plus_nvfp4`（Comfy-Org 只有 bf16／int8） |

擴寫 LoRA 只給大師精選（Master Style Essentials）的 system prompt＋風格卡用；DHQ 是依各 DiT 的 TE 敏感度分配位元的量化（見 [`docs/DHQUANT.md`](docs/DHQUANT.md)）。

## 範例（大師精選擴寫，同一題同一個 seed 跑 7 種 DiT）

左欄是風格和使用者輸入的那一句題目；右邊每欄一種 DiT，都用官方設定（Z-Image base 2 步＋turbo 10 步、klein 4 步、QI2.1 25 步、Ideogram 4 Default 20 步、Boogu 4 步、JoyAI 40 步 cfg 4）。擴寫 temp 0.7／top_p 0.8／top_k 20。klein 9B 的 DiT 由 bf16 載入轉 fp8；JoyAI 的 DiT 是社群的 NVFP4（aha2023）。有 `+` 的列是「攝影卡＋版面卡」複合。所有範例圖都是 AI 生成的（依 FLUX 授權 2(e) 標示）；DiT 的模型授權各自不同（Ideogram 4、FLUX.2 klein 9B 為非商用），生成的圖依各家授權的 Outputs 條款使用。

<table>
<tr><th>#</th><th>Z-Image<br>base+turbo</th><th>FLUX.2<br>klein 4B</th><th>FLUX.2<br>klein 9B</th><th>Qwen-Image<br>2.1</th><th>Ideogram 4</th><th>Boogu-Image<br>Turbo</th><th>JoyAI-Image<br>Edit Plus</th></tr>
<tr><th align=left width=190><b>#1</b><br>PE T2I 預設（無風格）<br><sub><code>pe-default</code></sub><br><sub>Dancers carrying a golden dragon on poles through the lantern-lit streets of Chinatown in San Francisco at Lunar New Year</sub></th><td><img src="docs/showcase/01_zbt.jpg" width="150"></td><td><img src="docs/showcase/01_klein.jpg" width="150"></td><td><img src="docs/showcase/01_klein9.jpg" width="150"></td><td><img src="docs/showcase/01_qi21.jpg" width="150"></td><td><img src="docs/showcase/01_ideo4.jpg" width="150"></td><td><img src="docs/showcase/01_boogu.jpg" width="150"></td><td><img src="docs/showcase/01_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#2</b><br>黑白街頭幾何瞬間<br><sub><code>bw-geometric-street-candid</code></sub><br><sub>A ballerina leaping across the zebra crossing in Shibuya</sub></th><td><img src="docs/showcase/02_zbt.jpg" width="150"></td><td><img src="docs/showcase/02_klein.jpg" width="150"></td><td><img src="docs/showcase/02_klein9.jpg" width="150"></td><td><img src="docs/showcase/02_qi21.jpg" width="150"></td><td><img src="docs/showcase/02_ideo4.jpg" width="150"></td><td><img src="docs/showcase/02_boogu.jpg" width="150"></td><td><img src="docs/showcase/02_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#3</b><br>暖色雨窗街影<br><sub><code>rainy-window-abstract-portrait</code></sub><br><sub>A girl in a yellow raincoat waiting under the neon signs of Mong Kok on a rainy night</sub></th><td><img src="docs/showcase/03_zbt.jpg" width="150"></td><td><img src="docs/showcase/03_klein.jpg" width="150"></td><td><img src="docs/showcase/03_klein9.jpg" width="150"></td><td><img src="docs/showcase/03_qi21.jpg" width="150"></td><td><img src="docs/showcase/03_ideo4.jpg" width="150"></td><td><img src="docs/showcase/03_boogu.jpg" width="150"></td><td><img src="docs/showcase/03_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#4</b><br>灰底精簡柔光人像<br><sub><code>refined-minimalist-studio-portrait</code></sub><br><sub>A young maiko in kimono and full white makeup glancing over her shoulder</sub></th><td><img src="docs/showcase/04_zbt.jpg" width="150"></td><td><img src="docs/showcase/04_klein.jpg" width="150"></td><td><img src="docs/showcase/04_klein9.jpg" width="150"></td><td><img src="docs/showcase/04_qi21.jpg" width="150"></td><td><img src="docs/showcase/04_ideo4.jpg" width="150"></td><td><img src="docs/showcase/04_boogu.jpg" width="150"></td><td><img src="docs/showcase/04_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#5</b><br>俗物飽色巨像<br><sub><code>saturated-banal-monumental</code></sub><br><sub>A giant inflatable rubber duck floating in Victoria Harbour, the Hong Kong skyline behind it</sub></th><td><img src="docs/showcase/05_zbt.jpg" width="150"></td><td><img src="docs/showcase/05_klein.jpg" width="150"></td><td><img src="docs/showcase/05_klein9.jpg" width="150"></td><td><img src="docs/showcase/05_qi21.jpg" width="150"></td><td><img src="docs/showcase/05_ideo4.jpg" width="150"></td><td><img src="docs/showcase/05_boogu.jpg" width="150"></td><td><img src="docs/showcase/05_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#6</b><br>黑白光束剪影<br><sub><code>bw-dramatic-chiaroscuro</code></sub><br><sub>A lone samurai standing in a bamboo forest as light pierces the mist</sub></th><td><img src="docs/showcase/06_zbt.jpg" width="150"></td><td><img src="docs/showcase/06_klein.jpg" width="150"></td><td><img src="docs/showcase/06_klein9.jpg" width="150"></td><td><img src="docs/showcase/06_qi21.jpg" width="150"></td><td><img src="docs/showcase/06_ideo4.jpg" width="150"></td><td><img src="docs/showcase/06_boogu.jpg" width="150"></td><td><img src="docs/showcase/06_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#7</b><br>蒼藍虛無置景<br><sub><code>surreal-staged-monochrome</code></sub><br><sub>A woman in a ball gown holding a white umbrella on the salt flats of Uyuni</sub></th><td><img src="docs/showcase/07_zbt.jpg" width="150"></td><td><img src="docs/showcase/07_klein.jpg" width="150"></td><td><img src="docs/showcase/07_klein9.jpg" width="150"></td><td><img src="docs/showcase/07_qi21.jpg" width="150"></td><td><img src="docs/showcase/07_ideo4.jpg" width="150"></td><td><img src="docs/showcase/07_boogu.jpg" width="150"></td><td><img src="docs/showcase/07_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#8</b><br>水墨留白銀鹽山水<br><sub><code>bw-ink-wash-landscape</code></sub><br><sub>Hot air balloons drifting over the misty temples of Bagan at sunrise</sub></th><td><img src="docs/showcase/08_zbt.jpg" width="150"></td><td><img src="docs/showcase/08_klein.jpg" width="150"></td><td><img src="docs/showcase/08_klein9.jpg" width="150"></td><td><img src="docs/showcase/08_qi21.jpg" width="150"></td><td><img src="docs/showcase/08_ideo4.jpg" width="150"></td><td><img src="docs/showcase/08_boogu.jpg" width="150"></td><td><img src="docs/showcase/08_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#9</b><br>黑白雅致日常抓拍<br><sub><code>vintage-bw-elegant-candid</code></sub><br><sub>A glamorous couple dancing in a 1930s Shanghai ballroom</sub></th><td><img src="docs/showcase/09_zbt.jpg" width="150"></td><td><img src="docs/showcase/09_klein.jpg" width="150"></td><td><img src="docs/showcase/09_klein9.jpg" width="150"></td><td><img src="docs/showcase/09_qi21.jpg" width="150"></td><td><img src="docs/showcase/09_ideo4.jpg" width="150"></td><td><img src="docs/showcase/09_boogu.jpg" width="150"></td><td><img src="docs/showcase/09_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#10</b><br>超飽和電影感不安<br><sub><code>hyper-saturated-cinematic-unease</code></sub><br><sub>A beauty queen eating cake alone in a pink motel room in Palm Springs</sub></th><td><img src="docs/showcase/10_zbt.jpg" width="150"></td><td><img src="docs/showcase/10_klein.jpg" width="150"></td><td><img src="docs/showcase/10_klein9.jpg" width="150"></td><td><img src="docs/showcase/10_qi21.jpg" width="150"></td><td><img src="docs/showcase/10_ideo4.jpg" width="150"></td><td><img src="docs/showcase/10_boogu.jpg" width="150"></td><td><img src="docs/showcase/10_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#11</b><br>粗粒黑白寫真<br><sub><code>raw-bw-cinematic-portraits</code></sub><br><sub>A boxer catching his breath after a fight in a Bangkok gym</sub></th><td><img src="docs/showcase/11_zbt.jpg" width="150"></td><td><img src="docs/showcase/11_klein.jpg" width="150"></td><td><img src="docs/showcase/11_klein9.jpg" width="150"></td><td><img src="docs/showcase/11_qi21.jpg" width="150"></td><td><img src="docs/showcase/11_ideo4.jpg" width="150"></td><td><img src="docs/showcase/11_boogu.jpg" width="150"></td><td><img src="docs/showcase/11_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#12</b><br>冷冽奢華高反差<br><sub><code>high-gloss-noir-glamour</code></sub><br><sub>A femme fatale in a black velvet dress on a Hong Kong rooftop at night, the skyline behind her</sub></th><td><img src="docs/showcase/12_zbt.jpg" width="150"></td><td><img src="docs/showcase/12_klein.jpg" width="150"></td><td><img src="docs/showcase/12_klein9.jpg" width="150"></td><td><img src="docs/showcase/12_qi21.jpg" width="150"></td><td><img src="docs/showcase/12_ideo4.jpg" width="150"></td><td><img src="docs/showcase/12_boogu.jpg" width="150"></td><td><img src="docs/showcase/12_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#13</b><br>極簡色塊靜謐風景<br><sub><code>minimalist-color-landscapes</code></sub><br><sub>A single red umbrella on an empty turquoise beach in the Maldives</sub></th><td><img src="docs/showcase/13_zbt.jpg" width="150"></td><td><img src="docs/showcase/13_klein.jpg" width="150"></td><td><img src="docs/showcase/13_klein9.jpg" width="150"></td><td><img src="docs/showcase/13_qi21.jpg" width="150"></td><td><img src="docs/showcase/13_ideo4.jpg" width="150"></td><td><img src="docs/showcase/13_boogu.jpg" width="150"></td><td><img src="docs/showcase/13_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#14</b><br>暖調柔光底片日常<br><sub><code>soft-warm-nostalgic</code></sub><br><sub>Kids jumping into the sea from a harbour wall in Sicily on a summer afternoon</sub></th><td><img src="docs/showcase/14_zbt.jpg" width="150"></td><td><img src="docs/showcase/14_klein.jpg" width="150"></td><td><img src="docs/showcase/14_klein9.jpg" width="150"></td><td><img src="docs/showcase/14_qi21.jpg" width="150"></td><td><img src="docs/showcase/14_ideo4.jpg" width="150"></td><td><img src="docs/showcase/14_boogu.jpg" width="150"></td><td><img src="docs/showcase/14_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#15</b><br>莫內印象光色<br><sub><code>monet_style</code></sub><br><sub>A garden party under hanging wisteria in full bloom in Kyoto</sub></th><td><img src="docs/showcase/15_zbt.jpg" width="150"></td><td><img src="docs/showcase/15_klein.jpg" width="150"></td><td><img src="docs/showcase/15_klein9.jpg" width="150"></td><td><img src="docs/showcase/15_qi21.jpg" width="150"></td><td><img src="docs/showcase/15_ideo4.jpg" width="150"></td><td><img src="docs/showcase/15_boogu.jpg" width="150"></td><td><img src="docs/showcase/15_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#16</b><br>北齋浮世繪木刻<br><sub><code>hokusai_style</code></sub><br><sub>A giant wave crashing over the New York skyline</sub></th><td><img src="docs/showcase/16_zbt.jpg" width="150"></td><td><img src="docs/showcase/16_klein.jpg" width="150"></td><td><img src="docs/showcase/16_klein9.jpg" width="150"></td><td><img src="docs/showcase/16_qi21.jpg" width="150"></td><td><img src="docs/showcase/16_ideo4.jpg" width="150"></td><td><img src="docs/showcase/16_boogu.jpg" width="150"></td><td><img src="docs/showcase/16_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#17</b><br>瑞士國際主義平面設計<br><sub><code>swiss-international-style</code></sub><br><sub>A poster for a Tokyo techno festival</sub></th><td><img src="docs/showcase/17_zbt.jpg" width="150"></td><td><img src="docs/showcase/17_klein.jpg" width="150"></td><td><img src="docs/showcase/17_klein9.jpg" width="150"></td><td><img src="docs/showcase/17_qi21.jpg" width="150"></td><td><img src="docs/showcase/17_ideo4.jpg" width="150"></td><td><img src="docs/showcase/17_boogu.jpg" width="150"></td><td><img src="docs/showcase/17_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#18</b><br>裝飾藝術旅遊海報<br><sub><code>art-deco-travel-poster</code></sub><br><sub>A travel poster of a zeppelin flying to Shanghai, titled &quot;Shanghai&quot;</sub></th><td><img src="docs/showcase/18_zbt.jpg" width="150"></td><td><img src="docs/showcase/18_klein.jpg" width="150"></td><td><img src="docs/showcase/18_klein9.jpg" width="150"></td><td><img src="docs/showcase/18_qi21.jpg" width="150"></td><td><img src="docs/showcase/18_ideo4.jpg" width="150"></td><td><img src="docs/showcase/18_boogu.jpg" width="150"></td><td><img src="docs/showcase/18_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#19</b><br>日式現代幾何平面<br><sub><code>japanese-modernist-geometric</code></sub><br><sub>A poster for a koi festival in Kyoto</sub></th><td><img src="docs/showcase/19_zbt.jpg" width="150"></td><td><img src="docs/showcase/19_klein.jpg" width="150"></td><td><img src="docs/showcase/19_klein9.jpg" width="150"></td><td><img src="docs/showcase/19_qi21.jpg" width="150"></td><td><img src="docs/showcase/19_ideo4.jpg" width="150"></td><td><img src="docs/showcase/19_boogu.jpg" width="150"></td><td><img src="docs/showcase/19_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#20</b><br>PE T2I 預設（無風格）<br><sub><code>pe-default</code></sub><br><sub>A beautiful young woman with a shaved head, thick eyebrows and a small silver nose ring, laughing on a scooter through the streets of Taipei at dusk</sub></th><td><img src="docs/showcase/20_zbt.jpg" width="150"></td><td><img src="docs/showcase/20_klein.jpg" width="150"></td><td><img src="docs/showcase/20_klein9.jpg" width="150"></td><td><img src="docs/showcase/20_qi21.jpg" width="150"></td><td><img src="docs/showcase/20_ideo4.jpg" width="150"></td><td><img src="docs/showcase/20_boogu.jpg" width="150"></td><td><img src="docs/showcase/20_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#21</b><br>粗粒黑白寫真<br><sub><code>raw-bw-cinematic-portraits</code></sub><br><sub>A beautiful woman in her fifties with freckles, a gap-toothed smile and wild grey curls, leaning out of a tram window in Lisbon</sub></th><td><img src="docs/showcase/21_zbt.jpg" width="150"></td><td><img src="docs/showcase/21_klein.jpg" width="150"></td><td><img src="docs/showcase/21_klein9.jpg" width="150"></td><td><img src="docs/showcase/21_qi21.jpg" width="150"></td><td><img src="docs/showcase/21_ideo4.jpg" width="150"></td><td><img src="docs/showcase/21_boogu.jpg" width="150"></td><td><img src="docs/showcase/21_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#22</b><br>暖調柔光底片日常 → 瑞士國際主義平面設計<br><sub><code>soft-warm-nostalgic + swiss-international-style</code></sub><br><sub>A fashion poster titled &quot;PARIS&quot;, a fashionable woman on the terrace of a luxury Paris penthouse, the Eiffel Tower behind her</sub></th><td><img src="docs/showcase/22_zbt.jpg" width="150"></td><td><img src="docs/showcase/22_klein.jpg" width="150"></td><td><img src="docs/showcase/22_klein9.jpg" width="150"></td><td><img src="docs/showcase/22_qi21.jpg" width="150"></td><td><img src="docs/showcase/22_ideo4.jpg" width="150"></td><td><img src="docs/showcase/22_boogu.jpg" width="150"></td><td><img src="docs/showcase/22_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#23</b><br>灰底精簡柔光人像 → 日式現代幾何平面<br><sub><code>refined-minimalist-studio-portrait + japanese-modernist-geometric</code></sub><br><sub>A colour beauty magazine cover titled &quot;BLOOM&quot;, a young Korean woman with dewy skin and peach blush</sub></th><td><img src="docs/showcase/23_zbt.jpg" width="150"></td><td><img src="docs/showcase/23_klein.jpg" width="150"></td><td><img src="docs/showcase/23_klein9.jpg" width="150"></td><td><img src="docs/showcase/23_qi21.jpg" width="150"></td><td><img src="docs/showcase/23_ideo4.jpg" width="150"></td><td><img src="docs/showcase/23_boogu.jpg" width="150"></td><td><img src="docs/showcase/23_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#24</b><br>梵谷筆觸<br><sub><code>van_gogh_style</code></sub><br><sub>Drunk salarymen in loosened ties stumbling home under a full moon in Shinjuku, Tokyo</sub></th><td><img src="docs/showcase/24_zbt.jpg" width="150"></td><td><img src="docs/showcase/24_klein.jpg" width="150"></td><td><img src="docs/showcase/24_klein9.jpg" width="150"></td><td><img src="docs/showcase/24_qi21.jpg" width="150"></td><td><img src="docs/showcase/24_ideo4.jpg" width="150"></td><td><img src="docs/showcase/24_boogu.jpg" width="150"></td><td><img src="docs/showcase/24_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#25</b><br>暖色雨窗街影<br><sub><code>rainy-window-abstract-portrait</code></sub><br><sub>A young woman in a clear transparent raincoat over a bikini, laughing under the neon signs of Mong Kok on a rainy night</sub></th><td><img src="docs/showcase/25_zbt.jpg" width="150"></td><td><img src="docs/showcase/25_klein.jpg" width="150"></td><td><img src="docs/showcase/25_klein9.jpg" width="150"></td><td><img src="docs/showcase/25_qi21.jpg" width="150"></td><td><img src="docs/showcase/25_ideo4.jpg" width="150"></td><td><img src="docs/showcase/25_boogu.jpg" width="150"></td><td><img src="docs/showcase/25_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#26</b><br>冷冽奢華高反差 → 瑞士國際主義平面設計<br><sub><code>high-gloss-noir-glamour + swiss-international-style</code></sub><br><sub>A fashion poster for Milan couture week titled &quot;MILANO&quot;, a model in a sculptural white coat</sub></th><td><img src="docs/showcase/26_zbt.jpg" width="150"></td><td><img src="docs/showcase/26_klein.jpg" width="150"></td><td><img src="docs/showcase/26_klein9.jpg" width="150"></td><td><img src="docs/showcase/26_qi21.jpg" width="150"></td><td><img src="docs/showcase/26_ideo4.jpg" width="150"></td><td><img src="docs/showcase/26_boogu.jpg" width="150"></td><td><img src="docs/showcase/26_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#27</b><br>超飽和電影感不安 → 日式現代幾何平面<br><sub><code>hyper-saturated-cinematic-unease + japanese-modernist-geometric</code></sub><br><sub>A fashion poster for a Tokyo streetwear brand titled &quot;NEON&quot;, a girl in a candy-pink puffer jacket</sub></th><td><img src="docs/showcase/27_zbt.jpg" width="150"></td><td><img src="docs/showcase/27_klein.jpg" width="150"></td><td><img src="docs/showcase/27_klein9.jpg" width="150"></td><td><img src="docs/showcase/27_qi21.jpg" width="150"></td><td><img src="docs/showcase/27_ideo4.jpg" width="150"></td><td><img src="docs/showcase/27_boogu.jpg" width="150"></td><td><img src="docs/showcase/27_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#28</b><br>灰底精簡柔光人像 → 瑞士國際主義平面設計<br><sub><code>refined-minimalist-studio-portrait + swiss-international-style</code></sub><br><sub>A beauty magazine cover titled &quot;GLOW&quot;, a close-up of a woman with glossy red lips and slicked-back hair</sub></th><td><img src="docs/showcase/28_zbt.jpg" width="150"></td><td><img src="docs/showcase/28_klein.jpg" width="150"></td><td><img src="docs/showcase/28_klein9.jpg" width="150"></td><td><img src="docs/showcase/28_qi21.jpg" width="150"></td><td><img src="docs/showcase/28_ideo4.jpg" width="150"></td><td><img src="docs/showcase/28_boogu.jpg" width="150"></td><td><img src="docs/showcase/28_joy.jpg" width="150"></td></tr>
<tr><th align=left width=190><b>#29</b><br>暖調柔光底片日常 → 日式現代幾何平面<br><sub><code>soft-warm-nostalgic + japanese-modernist-geometric</code></sub><br><sub>A beauty magazine cover titled &quot;BLOOM&quot;, a young Korean woman with dewy skin and peach blush</sub></th><td><img src="docs/showcase/29_zbt.jpg" width="150"></td><td><img src="docs/showcase/29_klein.jpg" width="150"></td><td><img src="docs/showcase/29_klein9.jpg" width="150"></td><td><img src="docs/showcase/29_qi21.jpg" width="150"></td><td><img src="docs/showcase/29_ideo4.jpg" width="150"></td><td><img src="docs/showcase/29_boogu.jpg" width="150"></td><td><img src="docs/showcase/29_joy.jpg" width="150"></td></tr>
</table>

## 架構（繼承官方）

官方程式碼能處理的部分，全部交給官方程式碼：

| 元件 | 做什麼 | 沿用官方的部分 |
|---|---|---|
| `dh_backend.DualHeadBackend` | 一個 llama model、一個 context；用 `llama_set_embeddings` 切換編碼和生成；LoRA 只在生成時掛上；視覺塔輸出依圖片內容做快取；system prompt 的 KV 用多條 sequence 做 prefix 快取 | — |
| `DHClipModel` | 繼承 `QwenImage21Qwen3VLClipModel`，只覆寫 `__init__`、`forward`、`generate` | tokenizer、模板、裁掉 system turn、丟視覺 token、`image_slots` |
| `DualHeadDragonLoader` 節點 | 透過 `model_options["qwen3vl_8b_class"]` 換掉內層類別，輸出一般的 CLIP | 官方 `TextEncodeQwenImage21`、`TextGenerate` 可以直接接上 |
| `TextEncodeQwenImage21DualHead` 節點 | 繼承官方節點，前面加上 PE 改寫，並依改寫結果決定畫布 | `super().execute()`；JSON 解析用官方的 `pe_core.py` |
| `DualHeadRewrite` 節點 | 通用改寫（任何 DiT）：system prompt + 改寫 LoRA + 選配的投機解碼草稿頭，輸出改寫後的文字，接到一般的 `CLIPTextEncode` | 同一顆 GGUF 的 TE 照常用 |
| `native/dh_spec.cpp`（`libdh_spec`） | 投機解碼：包 llama.cpp 的 `common/speculative`，草稿→驗證→接受整段在 C++ 跑；草稿頭種類（DFlash、DFlash2、EAGLE-3…）由 GGUF 決定 | llama.cpp 原生實作 |
| `dh_comfy.DHPatcher` | 把雙頭龍佔的 VRAM（模型、KV、緩衝、草稿頭，實測值）報給 ComfyUI；ComfyUI 要空間時釋放，下次編碼／改寫再自動載回 | ComfyUI 的 model management |

改寫和編碼全部在同一個 process 的記憶體裡完成，不寫檔，也不另外開 server。

## 用法

1. 把 GGUF 和 mmproj 放到 `models/text_encoders/`：
   - TE：例如 `qwen3vl_8b_Q4_K_M.gguf`（或 Q8_0）
   - 視覺塔：例如 `mmproj-qwen3vl_8b-f16.gguf`（任何參考圖都需要）
2. **只換 TE**：`Dual-Head Dragon Loader (GGUF)` → 官方 `Text Encode Qwen Image 2.1`，其餘照官方 workflow 接。
3. **要改寫**：改用 `Text Encode Qwen Image 2.1 (Dual-Head)`，把 `rewrite` 設成 `auto`／`t2i`／`i2i`。
   - `system_prompt`：預設 `auto`，會自動選官方的 `pe_t2i.txt`／`pe_i2i.txt`；自己的檔案放在 `ComfyUI/user/dualhead_prompts/`。
   - `system_prompt_text`：直接貼上或從別的節點接入，有內容時優先使用。
   - `canvas=rewrite`：latent 尺寸依 PE 輸出的 `wh_ratio`／`ratio_follow` 決定。
4. **任何 DiT 的改寫（Z-Image、klein、QI2.1 都行）**：`Dual-Head Rewrite` → 輸出的 `rewritten_prompt` 接到 `CLIPTextEncode`。
   - `system_prompt`／`system_prompt_text`、`rewrite_lora`：同上；LoRA 只在改寫時掛，TE 一律用原本的權重。
   - `draft`：投機解碼草稿頭（GGUF 放在 `models/dualhead_drafts/`），`draft_n_max` DFlash b16 用 15。輸出分布不變、改寫變快；
     沒選或 `libdh_spec` 載不到時照一般方式改寫。詳見 [`docs/SPECULATIVE.md`](docs/SPECULATIVE.md)。

### Loader 進階選項（VRAM）

| 選項 | 預設 | 說明 |
|---|---|---|
| `vram_mode` | `auto`（= managed） | managed：ComfyUI 看得到雙頭龍的 VRAM，DiT／VAE 或其他模型要空間時先請它讓出（下次再載回，約 0.5 s），空間夠就不動；resident：一直留在卡上、ComfyUI 看不到（12 GB 卡快約 0.7 s，但工作流多加模型可能 OOM） |
| `n_ctx` | 32768 | 改寫的 prompt + 輸出上限。小卡建議 4096（Qwen3-4B 的 KV 從 ~4.6 GB 降到 ~0.6 GB），大師題約 2,400 token 夠用 |
| `n_ubatch` | 1024 | 一次計算的 token 數，緩衝區跟著變（2048 ≈ 1.2 GB、1024 ≈ 0.6 GB、512 ≈ 0.3 GB）。編碼的 prompt（含圖片 token）不超過它時，TE 輸出跟 2048 逐位元相同；超過會分段算，約差 1.5%（圖生圖要完全一致請用 2048） |

實測（RTX 5060 Ti，Z-Image turbo NVFP4 1024² 8 步，V6 dhq Q4_K_M + 改寫 LoRA + DFlash Q6，`n_ctx` 4096，每張熱跑秒數）：
6 GB 10.7、8 GB 10.3、10 GB 10.2、12 GB 10.2（resident 9.4）、16 GB 9.4。完整查表在 [`docs/bench/MASTER_LOOKUP.md`](docs/bench/MASTER_LOOKUP.md)（`tools/bench_master.py`）。

### 安裝：內建 llama.cpp

節點自帶一份鎖定版本的 llama.cpp，**不依賴** venv 裡的 llama-cpp-python，別的節點或 pip 升級都不會影響它：

- `vendor/llama.cpp`：git submodule，鎖在 `4da6337`（2026-09-27，支援 DFlash2）；binding 最初取自 llama-cpp-python v0.3.35（`4df29be`），升級時已跟著新結構修改
- `dh_llama/`：從 llama-cpp-python v0.3.35 複製的 ctypes binding（MIT），改成相對 import，函式庫路徑的環境變數改名為 `DH_LLAMA_LIB_PATH`
- `build.py`：把 `libllama`、`libmtmd`、`libggml*`，以及投機解碼用的 `libdh_spec`（`native/`）編進 `dh_llama/lib/`（Linux 上 RPATH 設成 `$ORIGIN`）
- 升級 `vendor/llama.cpp` 前後跑 `python tools/check_bindings.py`：比對 binding 的結構欄位和 C 標頭，不一致就不能用

```bash
git clone --recursive https://github.com/pottokao-dotcom/ComfyUI-DualHead-Dragon            # 或：git submodule update --init
<ComfyUI venv>/bin/python build.py      # 本機 GPU 的 CUDA 版本（native 架構）
<ComfyUI venv>/bin/pip install gguf
```

其他選項：`--cuda-arch "86;89;120"`（一次編多代 GPU，給發布版用）、`--backend vulkan`（AMD／Intel）、`--backend cpu`、`--portable-cpu`。需要 cmake、編譯器和對應的 SDK（CUDA toolkit／Vulkan SDK）。

- 原始碼跨平台；二進位檔是「作業系統 × 後端 × GPU 架構」各一份。
- 實測：gx10（Linux aarch64、GB10、CUDA 13.0）編譯約 2 分鐘。
- 實測：blackhole（Linux x86_64、RTX 5060 Ti、CUDA 13.0，docker 編譯）。
- Windows：GitHub Actions 驗證過 MSVC 編得出 `dh_spec.dll`、binding 載得起來、函式有匯出（`.github/workflows/build-check.yml`，CPU 後端）；**Windows 上用 GPU 跑投機解碼還沒實測**。
- 不附 `requirements.txt`，以免 ComfyUI Manager 自動安裝別的東西。

## 已驗證（gx10，官方 bf16 TE 對照）

- t2i、edit 單圖（RGBA 輸入）、edit 雙圖（官方牛仔襯衫範例）、去背（官方花椰菜範例）：出圖跟官方幾乎一樣，去背的 alpha 也正確
- token 序列、`image_slots`、視覺 token 數：跟官方逐一相同
- 圖後文字的 cos：0.97（Q4）／0.999（Q8）；換圖造成的變化方向跟官方一致，cos 0.82–0.87
- 圖只讀一次（視覺快取有命中）；prefix 快取讓 prefill 從 t2i 0.76 s 降到 0.04 s、i2i 從 1.98 s 降到 0.49 s
- 沒接 VAE 的路徑能跑，但對精度敏感：換紅背景成功率官方 3/3、Q4 1/3、Q8 加 f32 mmproj 2/3

完整的規格對照在 [`docs/spec.html`](docs/spec.html)，量化流程在 [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)。

## 待辦

最初的清單：

- [ ] LoRA-T2I／I2I 蒸餾（核心；現在的改寫是裸 8B 的水準）
- [ ] LoRA 切換實測
- [x] 改寫加速：DFlash 草稿頭（Z-Image-Engineer V6 + 大師 LoRA 已完成；8B／DFlash2 訓練中）
- [ ] 3–10 張參考圖、局部編輯、cfg > 1 搭配負面 prompt
- [ ] 量化精度比較（Q4_K_M／Q5_K_M／Q6_K／Q8_0／NVFP4）
- [x] 小顯存：`vram_mode` managed，6 GB 卡可跑完整流程（見上面〈Loader 進階選項〉）
- [ ] 待決：PE 要不要開 thinking

## `tools/`

量化工具 `dhquant.py`（說明在 [`docs/DHQUANT.md`](docs/DHQUANT.md)；敏感度資料 sensbank 不隨 repo 附，`dhquant scan` 會自己算）、`bench_master.py`（大師流程的速度／VRAM 查表）、`lora_requant.py`（LoRA GGUF 混合量化）、`check_bindings.py`、`val_4b_*.py`（TE 對 ComfyUI 官方對照）。開發機專用的驗證腳本不隨公開版附。

## 授權

- `pe_core.py`、`prompts/*.txt`：原封不動取自 [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) 的 `prompt_rewrite/`，屬於 **Qwen Research License**（只限非商用），見 `LICENSE-Qwen-Research.txt` 和 `NOTICE`。
- Qwen-Image 2.1 的模型權重同樣是 Qwen Research License。
- 本 repo 自己寫的程式碼：授權**尚未決定**。
