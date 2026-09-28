"""Per-DiT encoding recipes ("products").

Three layers, kept apart:
  * engine   (dh_backend.py)  one GGUF, generate / encode (last layer or tapped layers), gen_lora / enc_lora.
                              Knows nothing about any DiT.
  * product  (this package)   one module per DiT: comfy tokenizer + template, which layers, padding, how the
                              conditioning tensor is shaped. Built on comfy's own text-encoder classes, so the output
                              matches the official CLIPLoader for that type. Never touches llama.cpp directly.
  * nodes    (__init__.py)    ComfyUI inputs/outputs only; pick a product, then call it.

A product is chosen by the loader's `type`, named like comfy's CLIPLoader types (qwen_image, lumina2, flux2 ...).
"auto" works only when the GGUF alone decides it (a Qwen3-VL-8B is only used by Qwen-Image 2.1); Qwen3-4B is shared
by Z-Image and FLUX.2 klein, so there the type must be chosen.
"""
PRODUCTS = {}  # loader type -> Product


class Product:
    def __init__(self, type_name, label, arch, n_embd, taps, make_clip, chat=None):
        self.type_name = type_name  # comfy CLIPLoader type
        self.label = label
        self.arch = arch            # GGUF general.architecture this recipe expects
        self.n_embd = n_embd
        self.taps = taps            # needs the engine's layer tap
        self.make_clip = make_clip  # backend -> comfy.sd.CLIP
        self.chat = chat or chatml  # (system, user, thinking) -> prompt text for the rewrite

    def matches(self, arch, n_embd):
        return arch == self.arch and n_embd == self.n_embd


def register(p):
    PRODUCTS[p.type_name] = p
    return p


def chatml(system, user, thinking=False):
    s = ("<|im_start|>system\n" + system + "<|im_end|>\n") if system else ""
    return s + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + ("<think>\n" if thinking else "")


def chatml_qwen3(system, user, thinking=False):
    """Qwen3 hybrid models: thinking off = an empty think block, as the chat template writes it (enable_thinking=False)."""
    s = ("<|im_start|>system\n" + system + "<|im_end|>\n") if system else ""
    return (s + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
            + ("<think>\n" if thinking else "<think>\n\n</think>\n\n"))


def gguf_shape(path):
    """(general.architecture, embedding length) from the GGUF header, without loading the weights."""
    import gguf
    r = gguf.GGUFReader(path)

    def field(name):
        f = r.fields.get(name)
        if f is None:
            return None
        v = f.parts[f.data[0]]
        return bytes(v).decode() if f.types and f.types[0] == gguf.GGUFValueType.STRING else int(v[0])

    arch = field("general.architecture")
    return arch, field("%s.embedding_length" % arch)


def resolve(type_name, gguf_path):
    if type_name != "auto":
        if type_name not in PRODUCTS:
            raise ValueError("unknown type %r (have %s)" % (type_name, ", ".join(PRODUCTS)))
        return PRODUCTS[type_name]
    arch, n_embd = gguf_shape(gguf_path)
    hits = [p for p in PRODUCTS.values() if p.matches(arch, n_embd)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError("no known DiT uses a %s / %s GGUF as its text encoder" % (arch, n_embd))
    raise ValueError("this %s GGUF is the text encoder of several DiTs; set type to one of: %s" % (
        arch, ", ".join("%s (%s)" % (p.type_name, p.label) for p in hits)))
