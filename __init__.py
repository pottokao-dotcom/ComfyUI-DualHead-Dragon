"""Dual-Head Dragon: one GGUF as both a DiT's text encoder and its prompt enhancer (llama.cpp).

Layers: engine (dh_backend.py) / per-DiT recipes (products/) / nodes (this file). The loader's `type` picks the recipe,
named like comfy's CLIPLoader types: qwen_image (Qwen-Image 2.1), lumina2 (Z-Image), flux2 (FLUX.2 klein 4B / 9B),
ideogram4 (Ideogram 4), boogu (Boogu-Image), joyimage (JoyAI-Image-Edit).

Qwen-Image 2.1:

Built by inheritance so the official code paths stay in charge:
  * DHClipModel subclasses comfy's QwenImage21Qwen3VLClipModel and only swaps the transformer forward/generate for
    llama.cpp. Tokenizer, template, system-turn trim, vision-token drop and image_slots are comfy's own code.
  * DualHeadDragonLoader returns an ordinary CLIP, so the official TextEncodeQwenImage21 / TextGenerate nodes and
    the official workflows run on it unchanged.
  * TextEncodeQwenImage21DualHead subclasses the official TextEncodeQwenImage21 and adds the prompt-enhancer rewrite
    (official Qwen-Image-2.1 prompt_rewrite contract) in front of it.
"""
import json
import math
import os
import time

import torch
from typing_extensions import override

import comfy.model_management
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import folder_paths
from comfy.text_encoders.qwen_image21 import QwenImage21Qwen3VLClipModel, QwenImage21TEModel, QwenImage21Tokenizer
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_qwen import TextEncodeQwenImage21

from . import pe_core, products
from . import products_builtin  # noqa: F401  (registers lumina2 / flux2 / ideogram4 / boogu / joyimage)
from .dh_backend import get_backend
from . import dh_comfy

for _folder in ("text_encoders", "loras"):  # comfy's default extension list has no .gguf (same fix as ComfyUI-GGUF)
    if _folder in folder_paths.folder_names_and_paths:
        _paths, _exts = folder_paths.folder_names_and_paths[_folder]
        folder_paths.folder_names_and_paths[_folder] = (_paths, set(_exts) | {".gguf"})
        folder_paths.filename_list_cache.pop(_folder, None)  # a listing made before this import would hide the .gguf files

# speculative-decoding drafts for the rewrite (DFlash / DFlash2 / EAGLE-3 GGUFs): ComfyUI/models/dualhead_drafts
folder_paths.add_model_folder_path("dualhead_drafts", os.path.join(folder_paths.models_dir, "dualhead_drafts"))
if "dualhead_drafts" in folder_paths.folder_names_and_paths:
    _p, _e = folder_paths.folder_names_and_paths["dualhead_drafts"]
    folder_paths.folder_names_and_paths["dualhead_drafts"] = (_p, set(_e) | {".gguf"})

HERE = os.path.dirname(os.path.realpath(__file__))
PROMPT_DIR = os.path.join(HERE, "prompts")
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
PE_IMAGE_MAX_PIXELS = 1024 * 1024  # pe_core.Profile.image_max_pixels (matches PE training)


# --------------------------------------------------------------------------- text encoder (inheritance)
class DHClipModel(QwenImage21Qwen3VLClipModel):
    """comfy's QwenImage21 Qwen3-VL clip model with the torch transformer replaced by the llama.cpp backend."""

    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}, **kwargs):
        torch.nn.Module.__init__(self)  # skip SDClipModel.__init__: it would build the 8B torch transformer
        self.backend = model_options["dual_head_backend"]
        self.special_tokens = {"pad": 151643}
        self.num_layers = 36
        self.layer, self.layer_idx, self.return_projected_pooled = "hidden", -1, True
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)
        self.layer_norm_hidden_state = False  # the backend already returns the pre-final-norm equivalent
        self.enable_attention_masks = attention_mask
        self.return_attention_masks = attention_mask
        self.zero_out_masked = False
        self.execution_device = None
        self.image_spans = []

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        self.execution_device = None

    def forward(self, tokens):
        zs, spans0 = [], None
        for seq in tokens:
            h, spans = self.backend.encode(seq)
            zs.append(h)
            if spans0 is None:
                spans0 = spans
        n = max(z.shape[0] for z in zs)
        z = torch.zeros(len(zs), n, zs[0].shape[1], dtype=torch.float32)
        mask = torch.zeros(len(zs), n, dtype=torch.long)
        for i, h in enumerate(zs):
            z[i, :h.shape[0]] = h
            mask[i, :h.shape[0]] = 1
        self.image_spans = spans0 or []
        if self.return_attention_masks:
            return z, None, {"attention_mask": mask}
        return z, None

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        return [], []

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0, mtp=True):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        seq = [t[0] for t in tokens[0]]
        return self.backend.generate(seq, max_length=max_length, do_sample=do_sample, temperature=temperature, top_k=top_k,
                                     top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
                                     presence_penalty=presence_penalty, seed=seed)


class DHTEModel(QwenImage21TEModel):
    disable_offload = True  # llama.cpp owns the weights; nothing for comfy to move

    def __init__(self, device="cpu", dtype=None, model_options={}):
        mo = dict(model_options)
        mo["qwen3vl_8b_class"] = DHClipModel  # SD1ClipModel's official per-clip class override
        super().__init__(device=device, dtype=dtype, model_options=mo)


def _dh_model(clip):
    m = getattr(clip.cond_stage_model, "qwen3vl_8b", None)
    if not isinstance(m, DHClipModel):
        raise ValueError("this node needs a CLIP from the Dual-Head Dragon Loader")
    return m


def _gguf_list():
    names = folder_paths.get_filename_list("text_encoders")
    main = [n for n in names if n.lower().endswith(".gguf") and "mmproj" not in n.lower()]
    proj = [n for n in names if n.lower().endswith(".gguf") and "mmproj" in n.lower()]
    return main, proj


def _make_qi21_clip(backend):
    target = comfy.supported_models_base.ClipTarget(QwenImage21Tokenizer, DHTEModel)
    # default devices: the CLIP holds no torch weights, and comfy's generate() expects a GPU load_device
    return comfy.sd.CLIP(target=target, model_options={"dual_head_backend": backend})


products.register(products.Product("qwen_image", "Qwen-Image 2.1", arch="qwen3vl", n_embd=4096, taps=False,
                                   make_clip=_make_qi21_clip))

_PRODUCT_OF = {}  # id(backend) -> Product the loader built its CLIP for


class DualHeadDragonLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        main, proj = _gguf_list()
        # new inputs go at the END: saved workflows store widget values by position
        return io.Schema(
            node_id="DualHeadDragonLoader",
            display_name="Dual-Head Dragon Loader (GGUF)",
            category="DualHeadDragon",
            description="Loads one GGUF with llama.cpp as a DiT's text encoder. The same weights also serve prompt rewriting, so VRAM holds a single copy.",
            inputs=[
                io.Combo.Input("gguf", options=main),
                io.Combo.Input("mmproj", options=["none"] + proj, default=proj[0] if proj else "none",
                               tooltip="Vision tower. Needed for any reference image."),
                io.Int.Input("n_ctx", default=32768, min=4096, max=262144, step=4096, advanced=True,
                             tooltip="Context for rewrite prompt + images + generated text."),
                # optional: API prompts and scripts written before these inputs existed must still validate
                io.Combo.Input("type", options=["auto"] + list(products.PRODUCTS), default="auto", optional=True,
                               tooltip="Which DiT this encoder feeds, as in comfy's CLIPLoader. auto works when the GGUF "
                                       "decides it (Qwen3-VL-8B -> qwen_image); Qwen3-4B is shared by lumina2 (Z-Image) "
                                       "and flux2 (FLUX.2 klein 4B), and a Qwen3-VL-8B also serves ideogram4 / boogu / joyimage, so pick one."),
                io.Combo.Input("encode_lora", options=["none"] + _lora_list(), default="none", optional=True,
                               tooltip="Attached only while encoding (e.g. an encoder fine-tune such as Z-Image-Engineer). "
                                       "The rewrite LoRA is separate and never touches the encoder."),
                io.Float.Input("encode_lora_strength", default=1.0, min=-4.0, max=4.0, step=0.05, optional=True),
                io.Int.Input("n_ubatch", default=1024, min=128, max=4096, step=128, optional=True, advanced=True,
                             tooltip="Tokens per compute pass; its buffer scales with it (2048 -> ~1.2 GB, 1024 -> ~0.6 GB, "
                                     "512 -> ~0.3 GB). An encoder prompt (text + image tokens) longer than this is computed in "
                                     "pieces, which shifts the TE output slightly (~1.5% measured); up to it the output is "
                                     "bit-identical to 2048. Image-to-image prompts can exceed 1024 -- use 2048 to match exactly."),
                io.Combo.Input("vram_mode", options=dh_comfy.MODES, default="auto", optional=True, advanced=True,
                               tooltip="managed (auto): ComfyUI sees this model's VRAM and may evict it after PE/TE when "
                                       "the DiT / VAE / other models need the memory, reloading it next time (~0.5 s); with "
                                       "enough VRAM it is never evicted. resident: always stays, invisible to ComfyUI -- "
                                       "~0.7 s faster on a 12 GB card, but other models cannot get that memory back."),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, gguf, mmproj, n_ctx=32768, type="auto", encode_lora="none", encode_lora_strength=1.0,
                n_ubatch=1024, vram_mode="auto") -> io.NodeOutput:
        gpath = folder_paths.get_full_path_or_raise("text_encoders", gguf)
        mpath = None if mmproj == "none" else folder_paths.get_full_path_or_raise("text_encoders", mmproj)
        prod = products.resolve(type, gpath)
        backend = get_backend(gpath, mpath, n_ctx, taps=prod.taps, n_ubatch=n_ubatch)
        backend.enc_lora = ([] if encode_lora == "none" else
                            [(folder_paths.get_full_path_or_raise("loras", encode_lora), encode_lora_strength)])
        _PRODUCT_OF[id(backend)] = prod
        return io.NodeOutput(dh_comfy.manage(prod.make_clip, backend, vram_mode, prod.type_name))


# --------------------------------------------------------------------------- rewrite + encode (inheritance)
def _node_resize(image, resolution):
    """Verbatim resize from TextEncodeQwenImage21.execute: returns the tensor the official node feeds the encoder."""
    samples = image[:1].movedim(-1, 1)
    if resolution > 0:
        ratio = samples.shape[3] / samples.shape[2]
        width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
        height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
    else:
        width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
    width, height = max(32, width), max(32, height)
    if (width, height) == (samples.shape[3], samples.shape[2]):
        s = image[:1]
    else:
        s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
    rgb = s[:, :, :, :3]
    if s.shape[-1] > 3:
        rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])
    return rgb, width, height


def _pe_image(image, resolution):
    """What the rewrite looks at. Reuses the encoder's exact tensor (vision cache hit => the image is read once)
    unless that exceeds the PE's trained 1 MP cap, in which case it gets its own <=1 MP copy (pe_core.load_image)."""
    rgb, w, h = _node_resize(image, resolution)
    if w * h <= PE_IMAGE_MAX_PIXELS * 1.1:
        return rgb
    s = (PE_IMAGE_MAX_PIXELS / float(w * h)) ** 0.5
    return comfy.utils.common_upscale(rgb.movedim(-1, 1), max(1, int(w * s)), max(1, int(h * s)), "lanczos", "disabled").movedim(1, -1)


USER_PROMPT_DIR = os.path.join(folder_paths.get_user_directory(), "dualhead_prompts")


def _prompt_files():
    """Built-in official PE prompts (prompts/) plus the user's own (ComfyUI/user/dualhead_prompts/, listed as user/<name>)."""
    out = []
    for base, prefix in ((PROMPT_DIR, ""), (USER_PROMPT_DIR, "user/")):
        try:
            out += [prefix + f for f in sorted(os.listdir(base)) if f.endswith((".txt", ".md"))]
        except OSError:
            pass
    return out


def _read_prompt_file(name):
    path = os.path.join(USER_PROMPT_DIR, name[5:]) if name.startswith("user/") else os.path.join(PROMPT_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


os.makedirs(USER_PROMPT_DIR, exist_ok=True)


def _lora_list():
    return [n for n in folder_paths.get_filename_list("loras") if n.lower().endswith(".gguf")]


def _draft_list():
    try:
        return [n for n in folder_paths.get_filename_list("dualhead_drafts") if n.lower().endswith(".gguf")]
    except Exception:
        return []


def _dims(area, ratio):
    w = round(math.sqrt(area * ratio) / 32) * 32
    h = round(math.sqrt(area / ratio) / 32) * 32
    return max(32, w), max(32, h)


def _parse_ratio(s):
    try:
        a, b = s.replace("：", ":").split(":")
        a, b = float(a), float(b)
        return a / b if a > 0 and b > 0 else None
    except (ValueError, AttributeError):
        return None


class TextEncodeQwenImage21DualHead(TextEncodeQwenImage21):
    """The official TextEncodeQwenImage21 with the dual-head prompt rewrite in front of it."""

    @classmethod
    def define_schema(cls):
        s = super().define_schema()
        s.node_id = "TextEncodeQwenImage21DualHead"
        s.display_name = "Text Encode Qwen Image 2.1 (Dual-Head)"
        s.category = "DualHeadDragon"
        s.inputs = list(s.inputs) + [
            io.Combo.Input("rewrite", options=["off", "auto", "t2i", "i2i"], default="off",
                           tooltip="off: encode the prompt as is. auto: i2i when images are connected, else t2i."),
            io.Combo.Input("system_prompt", options=["auto"] + _prompt_files(), default="auto",
                           tooltip="auto picks pe_t2i.txt / pe_i2i.txt (the official Qwen-Image-2.1 PE system prompts). "
                                   "Your own files go in ComfyUI/user/dualhead_prompts/ and show up as user/<name>."),
            io.String.Input("system_prompt_text", multiline=True, default="", optional=True,
                            tooltip="Paste or connect a system prompt here to use it instead of the file above."),
            io.Combo.Input("rewrite_lora", options=["none"] + _lora_list(), default="none",
                           tooltip="Attached only while rewriting; encoding always uses the base weights."),
            io.Float.Input("lora_strength", default=1.0, min=-4.0, max=4.0, step=0.05),
            io.Int.Input("rewrite_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
            io.Int.Input("max_new_tokens", default=2048, min=64, max=32768,
                         tooltip="Official PE budgets are 16256 (t2i) / 24000 (edit) with thinking on."),
            io.Boolean.Input("thinking", default=False, tooltip="Prefill <think> as the official PE does. Only for weights trained to think."),
            io.Combo.Input("canvas", options=["rewrite", "encoder"], default="rewrite",
                           tooltip="rewrite: latent follows the PE's wh_ratio / ratio_follow. encoder: official behaviour (image_1 or resolution)."),
        ]
        s.outputs = list(s.outputs) + [
            io.String.Output(display_name="rewritten_prompt"),
            io.String.Output(display_name="rewrite_json"),
        ]
        return s

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, vae=None, resolution=1024, images: io.Autogrow.Type = None,
                rewrite="off", system_prompt="auto", rewrite_lora="none", lora_strength=1.0, rewrite_seed=0,
                max_new_tokens=2048, thinking=False, canvas="rewrite", system_prompt_text="") -> io.NodeOutput:
        images = images or {}
        order = [n for n in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[n] is not None]
        task = None
        if rewrite == "auto":
            task = "edit" if order else "t2i"
        elif rewrite == "t2i":
            task = "t2i"
        elif rewrite == "i2i":
            if not order:
                raise ValueError("rewrite=i2i needs at least one image (the edit PE always sees an input image)")
            task = "edit"

        text, record = prompt, {}
        if task is not None:
            m = _dh_model(clip)
            prof = pe_core.get_profile(task)
            if system_prompt_text and system_prompt_text.strip():
                sys_text = system_prompt_text.strip()
            else:
                sys_text = _read_prompt_file(system_prompt if system_prompt != "auto" else ("pe_t2i.txt" if task == "t2i" else "pe_i2i.txt"))
            pe_imgs = [_pe_image(images[n], resolution) for n in order] if prof.takes_images else []
            chat = ("<|im_start|>system\n" + sys_text + "<|im_end|>\n<|im_start|>user\n" + VISION_BLOCK * len(pe_imgs)
                    + prompt + "<|im_end|>\n<|im_start|>assistant\n" + ("<think>\n" if thinking else ""))
            tokens = clip.tokenize(chat, images=pe_imgs)
            lora = []
            if rewrite_lora != "none":
                lora = [(folder_paths.get_full_path_or_raise("loras", rewrite_lora), lora_strength)]
            m.backend.gen_lora = lora
            try:
                ids = clip.generate(tokens, do_sample=prof.temperature > 0, max_length=max_new_tokens,
                                    temperature=prof.temperature, top_k=prof.top_k, top_p=prof.top_p, min_p=prof.min_p,
                                    repetition_penalty=1.0, presence_penalty=prof.presence_penalty, seed=rewrite_seed)
            finally:
                m.backend.gen_lora = []
            out_text = clip.decode(ids)
            thinking_text, answer = pe_core.split_thinking(("<think>\n" if thinking else "") + out_text)
            record = pe_core.build_record({"id": "", "prompt": prompt, "input_images": order}, thinking_text, answer, prof)
            if not record["parse_ok"]:
                print("[DualHeadDragon] rewrite did not parse as the expected JSON; using the raw answer text")
            text = record["positive_prompt"] or prompt

        out = super().execute(clip, text, negative_prompt, vae=vae, resolution=resolution, images=images)
        positive, negative, latent = out.result[:3]
        if task is not None:
            print("[DualHeadDragon] task=%s parse_ok=%s wh_ratio=%r ratio_follow=%r vision_cache=%s" % (
                task, record.get("parse_ok"), record.get("wh_ratio"), record.get("ratio_follow"), _dh_model(clip).backend.stats))
            print("[DualHeadDragon] rewritten_prompt: " + text.replace("\n", " "))

        if task is not None and canvas == "rewrite":
            size = None
            follow = record.get("ratio_follow", "")
            if follow.startswith("<image") and follow.endswith(">"):
                k = int(follow[6:-1]) if follow[6:-1].isdigit() else 0
                if 1 <= k <= len(order):
                    _, w, h = _node_resize(images[order[k - 1]], resolution)
                    size = (w, h)
            ratio = _parse_ratio(record.get("wh_ratio", ""))
            if size is None and ratio is not None:
                area = (resolution or 1024) ** 2
                if not resolution and order:
                    _, w1, h1 = _node_resize(images[order[0]], resolution)
                    area = w1 * h1
                size = _dims(area, ratio)
            if size is not None:
                w, h = size
                latent = {"samples": torch.zeros([1, 64, h // 16, w // 16], device=comfy.model_management.intermediate_device())}

        return io.NodeOutput(positive, negative, latent, text, json.dumps(record, ensure_ascii=False))


# --------------------------------------------------------------------------- rewrite (any product)
def _dh_backend(clip):
    cs = clip.cond_stage_model
    inner = getattr(cs, getattr(cs, "clip", ""), None)
    b = getattr(inner, "backend", None)
    if b is None:
        raise ValueError("this node needs a CLIP from the Dual-Head Dragon Loader")
    return b


class DualHeadRewrite(io.ComfyNode):
    """Prompt enhancer on the loaded encoder's own weights. Feed its text into the DiT's official text-encode node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DualHeadRewrite",
            display_name="Dual-Head Rewrite",
            category="DualHeadDragon",
            description="Rewrites a short prompt with the same weights the encoder uses (rewrite LoRA attached only here). "
                        "Connect rewritten_prompt to the DiT's text-encode node.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True),
                io.Combo.Input("system_prompt", options=["none"] + _prompt_files(), default="none",
                               tooltip="Files in ComfyUI/user/dualhead_prompts/ show up as user/<name>. Must be the "
                                       "system prompt the rewrite LoRA was trained with."),
                io.String.Input("system_prompt_text", multiline=True, default="", optional=True,
                                tooltip="Paste or connect a system prompt here to use it instead of the file above."),
                io.Combo.Input("rewrite_lora", options=["none"] + _lora_list(), default="none",
                               tooltip="Attached only while rewriting; encoding uses the loader's encode_lora (default none)."),
                io.Float.Input("lora_strength", default=1.0, min=-4.0, max=4.0, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("max_new_tokens", default=2048, min=64, max=32768),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, tooltip="0 = greedy"),
                io.Float.Input("top_p", default=0.8, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=20, min=0, max=1000),
                io.Boolean.Input("thinking", default=False, tooltip="Only for weights trained to think."),
                io.Combo.Input("draft", options=["none"] + _draft_list(), default="none", optional=True,
                               tooltip="Speculative-decoding draft (DFlash / DFlash2 / EAGLE-3 GGUF in models/dualhead_drafts) "
                                       "trained for this encoder + rewrite LoRA. Same output distribution, faster rewrite."),
                io.Int.Input("draft_n_max", default=15, min=1, max=64, optional=True,
                             tooltip="Max draft tokens per round (clamped to the draft's block size; DFlash b16 -> 15)."),
            ],
            outputs=[
                io.String.Output(display_name="rewritten_prompt"),
                io.String.Output(display_name="wh_ratio"),
                io.String.Output(display_name="raw"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="none", rewrite_lora="none", lora_strength=1.0, seed=0,
                max_new_tokens=2048, temperature=0.7, top_p=0.8, top_k=20, thinking=False,
                system_prompt_text="", draft="none", draft_n_max=15) -> io.NodeOutput:
        backend = _dh_backend(clip)
        dh_comfy.load(clip)  # ComfyUI makes room (and reloads a released backend) before we use it directly
        prod = _PRODUCT_OF.get(id(backend)) or products.PRODUCTS["qwen_image"]
        if system_prompt_text and system_prompt_text.strip():
            sys_text = system_prompt_text.strip()
        else:
            sys_text = "" if system_prompt == "none" else _read_prompt_file(system_prompt)
        ids = backend.tokenize(prod.chat(sys_text, prompt, thinking))
        lora = [] if rewrite_lora == "none" else [(folder_paths.get_full_path_or_raise("loras", rewrite_lora), lora_strength)]
        dpath = None if draft == "none" else folder_paths.get_full_path_or_raise("dualhead_drafts", draft)
        _t0 = time.perf_counter()
        out = backend.generate(ids, max_length=max_new_tokens, do_sample=temperature > 0, temperature=temperature,
                               top_k=top_k, top_p=top_p, seed=seed, lora=lora, draft=dpath, draft_n_max=draft_n_max)
        if dpath and getattr(backend, "last_spec", None):
            s = backend.last_spec
            print("[DualHeadDragon] draft: %d tokens in %d rounds, accepted %d/%d drafted, %.2f s (%.1f ms/round)" %
                  (s["tokens"], s["rounds"], s["accepted"], s["drafted"], time.perf_counter() - _t0,
                   1000 * (time.perf_counter() - _t0) / max(1, s["rounds"])), flush=True)
        raw = backend.detokenize(out)
        _, answer = pe_core.split_thinking(("<think>\n" if thinking else "") + raw)
        parsed = pe_core.parse_answer(answer, pe_core.get_profile("t2i"))
        if not parsed["parse_ok"]:
            print("[DualHeadDragon] rewrite did not parse as the expected JSON; using the raw answer text")
        print("[DualHeadDragon] %s rewrite: %s" % (prod.type_name, parsed["positive_prompt"].replace("\n", " ")))
        return io.NodeOutput(parsed["positive_prompt"], parsed["wh_ratio"], raw)


class DualHeadDragonExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DualHeadDragonLoader, TextEncodeQwenImage21DualHead, DualHeadRewrite]


async def comfy_entrypoint() -> DualHeadDragonExtension:
    return DualHeadDragonExtension()
