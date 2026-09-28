"""TE quant precision check: every GGUF vs a BF16 GGUF of the same model through the same llama.cpp path (pure
quantization error; on the stock model BF16-GGUF vs comfy bf16 text cos was 1.000).

    bench_heretic.py REF_BF16.gguf A.gguf,B.gguf,...
Env: COMFYUI_DIR (for the official tokenizer, default ~/ComfyUI_lab), DH_MMPROJ (vision tower gguf),
     DH_BENCH_IMAGE (reference image for the image-conditioned metrics)."""
import sys, os, time, json, gc
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/ComfyUI_lab")))
sys.path.insert(0, REPO)
import numpy as np, torch
from PIL import Image
from dh_backend import DualHeadBackend
from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer

MMPROJ = os.path.expanduser(os.environ.get("DH_MMPROJ", "~/qi21_dit/mmproj-qwen3vl_8b-f32.gguf"))
LONG = ("The image is a wide realistic photograph of a bustling night market street in a rain-soaked Asian city, the background a haze of "
        "neon signs in deep magenta, electric blue and warm amber reflected on the wet asphalt. Across the top, a tangle of illuminated shop "
        "signs in Chinese characters recedes into the misty distance. On the left side of the frame, a noodle vendor in a grey apron leans over "
        "a steaming metal cart, the steam catching the light. In the centre, a young woman in a translucent plastic raincoat holds a paper "
        "umbrella, her back partly turned, looking toward a stall selling grilled skewers on the right. In the lower third, puddles mirror the "
        "signage in broken, wavering strips of colour. The lighting is a mix of cold neon and warm incandescent bulbs, casting long soft "
        "reflections and gentle highlights on every wet surface. The overall composition is dense, cinematic and moody, dominated by "
        "saturated jewel tones against slick black pavement.")
PROMPTS = [LONG,
           'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
           "一只在雨中弹吉他的柯基，电影感光线，湿漉漉的街道倒映着霓虹灯"]
EDIT = "Keep the person in <image1> unchanged, change the background to solid red."
im = Image.open(os.path.expanduser(os.environ.get("DH_BENCH_IMAGE", "~/ComfyUI_lab/input/guanyu.png"))).convert("RGB").resize((832, 1216))
IMG = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.)[None]
tok = QwenImage21Tokenizer()
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def seq_of(text, imgs):
    return [x[0] for x in tok.tokenize_with_weights(text, images=imgs, prevent_empty_text=True)["qwen3vl_8b"][0]]


def run(path):
    be = DualHeadBackend(path, MMPROJ, n_ctx=8192)
    out = {"text": [be.encode(seq_of(p, []))[0] for p in PROMPTS]}
    h, spans = be.encode(seq_of(EDIT, [IMG]))
    s, n = spans[0]
    out["vision"], out["after"] = h[s:s + n], h[s + n:]
    seq = seq_of(PROMPTS[0], [])
    t0 = time.time(); be.encode(seq); out["enc_ms"] = round((time.time() - t0) * 1000)
    t0 = time.time(); ids = be.generate(seq_of("Describe a cat in 200 words.", []), max_length=200, do_sample=False)
    out["gen_tok_s"] = round(len(ids) / (time.time() - t0), 1)
    be.close(); del be; gc.collect()
    return out


ref = run(sys.argv[1])
rows = []
for name in sys.argv[2].split(","):
    name = os.path.expanduser(name)
    o = run(name)
    t = torch.cat([cos(a[1:], b[1:]) for a, b in zip(o["text"], ref["text"])])
    a, v = cos(o["after"], ref["after"]), cos(o["vision"], ref["vision"])
    rows.append({"model": os.path.basename(name), "GB": round(os.path.getsize(name) / 1e9, 2),
                 "text_cos": round(float(t.mean()), 4), "text_p5": round(float(t.quantile(0.05)), 4),
                 "after_img_cos": round(float(a.mean()), 4), "vision_cos": round(float(v.mean()), 4),
                 "enc_ms": o["enc_ms"], "gen_tok_s": o["gen_tok_s"]})
    print(rows[-1], flush=True)
print("RESULT_JSON " + json.dumps(rows))
