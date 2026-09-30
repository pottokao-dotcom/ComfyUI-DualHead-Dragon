"""llama.cpp backend for the dual-head dragon: ONE GGUF (+ mmproj) serves as both a DiT's text encoder (hidden states)
and the prompt enhancer (generation). One model, one context; the role is switched with llama_set_embeddings.

Each role has its own LoRA set: gen_lora is attached only while generating, enc_lora only while encoding, so neither
leaks into the other (e.g. a rewrite LoRA for the enhancer, Z-Image-Engineer-V6's adapter for the encoder).

Encoders that read intermediate layers (Z-Image: layer 34, FLUX.2 klein: layers 8/17/26) call encode(seq, layers=...);
the raw outputs (l_out-N, no final norm) are copied out through the scheduler's eval callback. The layers are chosen per
call, so Z-Image and klein share ONE loaded Qwen3-4B. The callback is only installed when the backend is created with
taps=True, so last-layer encoders (Qwen-Image 2.1) keep the plain graph.

Vision tower outputs are cached by image content, so an image the rewrite already looked at is not encoded again
when the text encoder reads it.

Speculative decoding for the rewrite (DFlash, DFlash2 when the bundled llama.cpp knows it, EAGLE-3, ...): the draft
GGUF is handed to libdh_spec (native/dh_spec.cpp, llama.cpp's common/speculative), which runs the whole
draft -> verify -> accept loop in C++ against this backend's context. The draft reads the target layers it was trained
on through llama.cpp itself, not through the TE taps.
"""
import ctypes
import hashlib
import logging
import numbers
import random
import threading
from collections import OrderedDict

import numpy as np
import torch

if __package__:  # loaded as part of the node
    from .dh_llama import LIB_DIR, llama_cpp
    from .dh_llama import mtmd_cpp as M
else:  # imported as a top-level module by the tools/ scripts
    from dh_llama import LIB_DIR, llama_cpp
    from dh_llama import mtmd_cpp as M

_GGML = None

_SPEC = None  # libdh_spec, False when not built


def _spec_lib():
    """libdh_spec next to the llama.cpp libraries (build.py builds it); None when missing."""
    global _SPEC
    if _SPEC is None:
        import os
        import sys
        name = {"win32": "dh_spec.dll", "darwin": "libdh_spec.dylib"}.get(sys.platform, "libdh_spec.so")
        try:
            if sys.platform == "win32":
                # LOAD_WITH_ALTERED_SEARCH_PATH: resolve llama/ggml DLLs from dh_spec.dll's own folder (the bindings'
                # os.add_dll_directory handle is not kept, so that folder is no longer on the DLL search path here)
                lib = ctypes.CDLL(os.path.join(LIB_DIR, name), winmode=0x00000008)
            else:
                lib = ctypes.CDLL(os.path.join(LIB_DIR, name))
            lib.dh_spec_create.restype = ctypes.c_void_p
            lib.dh_spec_create.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            lib.dh_spec_type.restype = ctypes.c_char_p
            lib.dh_spec_type.argtypes = [ctypes.c_void_p]
            lib.dh_spec_free.argtypes = [ctypes.c_void_p]
            if hasattr(lib, "dh_spec_set_strategy"):  # tree verification (docs/SPEC_STRATEGY.md); older builds lack it
                lib.dh_spec_set_strategy.restype = ctypes.c_int
                lib.dh_spec_set_strategy.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.dh_spec_generate.restype = ctypes.c_int
            lib.dh_spec_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                             ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_uint32,
                                             ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int)]
            _SPEC = lib
        except (OSError, AttributeError) as e:  # missing library, or one built without the exports
            logging.warning("DualHeadDragon: speculative decoding unavailable, rewriting without the draft (%s)", e)
            _SPEC = False
    return _SPEC or None


def _ggml():
    """libggml-base with the three calls the layer tap needs (name, size, copy to host)."""
    global _GGML
    if _GGML is None:
        import glob
        import os
        path = (glob.glob(os.path.join(LIB_DIR, "libggml-base.so")) + glob.glob(os.path.join(LIB_DIR, "libggml-base.dylib"))
                + glob.glob(os.path.join(LIB_DIR, "ggml-base.dll")))[0]
        g = ctypes.CDLL(path)
        g.ggml_get_name.restype = ctypes.c_char_p
        g.ggml_get_name.argtypes = [ctypes.c_void_p]
        g.ggml_nbytes.restype = ctypes.c_size_t
        g.ggml_nbytes.argtypes = [ctypes.c_void_p]
        g.ggml_backend_tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        _GGML = g
    return _GGML


_KEY_LIMIT = None


def _set_key_pos_limit(mem, limit):
    """patches/0001-key-pos-limit.patch: while limit >= 0 no token attends to a KV cell at position >= limit."""
    global _KEY_LIMIT
    if _KEY_LIMIT is None:
        fn = getattr(llama_cpp._lib, "llama_memory_dh_set_key_pos_limit", None)
        if fn is None:
            raise RuntimeError("this llama.cpp build lacks patches/0001-key-pos-limit.patch; rebuild with build.py")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        fn.restype = None
        _KEY_LIMIT = fn
    _KEY_LIMIT(mem, int(limit))


def _read_output_norm(path):
    import gguf
    for t in gguf.GGUFReader(path).tensors:
        if t.name == "output_norm.weight":
            return torch.tensor(np.array(t.data, dtype=np.float32))
    raise RuntimeError("output_norm.weight not found in " + path)


def _to_uint8_rgb(img):
    a = img.detach().cpu().float().numpy() if isinstance(img, torch.Tensor) else np.asarray(img, dtype=np.float32)
    if a.ndim == 4:  # comfy IMAGE (B,H,W,C): one image per entry
        a = a[0]
    return np.ascontiguousarray((a[..., :3].clip(0, 1) * 255.0).round().astype(np.uint8))


class DualHeadBackend:
    def __init__(self, gguf_path, mmproj_path=None, n_ctx=32768, n_ubatch=2048, image_max_tokens=16384, vision_cache_size=32,
                 prefix_slots=4, taps=False):
        self.gguf_path, self.mmproj_path = gguf_path, mmproj_path
        self.lock = threading.RLock()
        self._cfg = (n_ctx, n_ubatch, image_max_tokens, prefix_slots)
        self.taps = bool(taps)
        self._tap_on = False
        self._all_out = False  # mark every decoded row as an output (tapping the last layer)
        self._tap_names, self._tap_buf = {}, []
        if self.taps:
            self._tap_cb = llama_cpp.ggml_backend_sched_eval_callback(self._on_eval)  # keep a reference
        self.vcache = OrderedDict()
        self.vcache_size = vision_cache_size
        self.stats = {"vision_encode": 0, "vision_hit": 0, "prefix_hit": 0, "prefix_miss": 0, "prefix_tokens_reused": 0,
                      "reloads": 0}
        self._collect = None
        self._cb = M.mtmd_helper_post_decode_callback(self._on_image_batch)
        self._adapters = {}
        self.use_prefix_cache = True
        self.gen_lora = []  # [(path, scale)] attached while generating only
        self.enc_lora = []  # [(path, scale)] attached while encoding only (empty = the base weights the DiT was trained on)
        self._drafts = {}  # (draft path, n_max) -> dh_spec handle
        self._draft_bytes = {}  # same key -> VRAM it took
        self.prefixes = OrderedDict()  # tuple(prefix tokens) -> seq id, LRU
        self.model = self.ctx = self.mctx = None
        self.norm_w = None
        self.vram_bytes = 0  # what this backend holds on the GPU (model, KV, buffers, drafts), measured
        self._load()

    @staticmethod
    def _gpu_free():
        try:
            return torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
        except Exception:
            return 0

    @property
    def loaded(self):
        return self.ctx is not None

    def ensure(self):
        """Load the weights again after release() (ComfyUI evicted us to make room for the DiT)."""
        with self.lock:
            if self.ctx is None:
                self._load()
                self.stats["reloads"] += 1

    def _load(self):
        import os
        n_ctx, n_ubatch, image_max_tokens, prefix_slots = self._cfg
        gguf_path, mmproj_path = self.gguf_path, self.mmproj_path
        free0 = self._gpu_free()
        llama_cpp.llama_backend_init()
        mp = llama_cpp.llama_model_default_params()
        mp.n_gpu_layers = 999
        self.model = llama_cpp.llama_model_load_from_file(gguf_path.encode(), mp)
        if not self.model:
            raise RuntimeError("failed to load " + gguf_path)
        # seq 0 is the working sequence; seqs 1..prefix_slots keep system-prompt KV for reuse (llama-server style
        # prompt caching). kv_unified lets every sequence use the whole n_ctx instead of n_ctx / n_seq_max.
        cp = llama_cpp.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_batch = n_ubatch
        cp.n_ubatch = n_ubatch
        cp.n_seq_max = 1 + prefix_slots
        cp.kv_unified = True
        cp.embeddings = True
        cp.pooling_type = llama_cpp.LLAMA_POOLING_TYPE_NONE
        cp.n_threads = cp.n_threads_batch = max(1, (os.cpu_count() or 4) // 2)
        if self.taps:
            cp.cb_eval = self._tap_cb
            cp.cb_eval_user_data = None
        self.ctx = llama_cpp.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("failed to create llama context")
        self.mem = llama_cpp.llama_get_memory(self.ctx)
        self.prefix_slots = prefix_slots
        self.prefixes = OrderedDict()
        self.vocab = llama_cpp.llama_model_get_vocab(self.model)
        self.n_vocab = llama_cpp.llama_vocab_n_tokens(self.vocab)
        self.n_ctx = n_ctx
        self.n_batch = n_ubatch
        self.n_embd = llama_cpp.llama_model_n_embd(self.model)
        self.n_layer = llama_cpp.llama_model_n_layer(self.model)
        self.n_embd_inp = llama_cpp.llama_model_n_embd_inp(self.model)
        if getattr(self, "norm_w", None) is None:  # python gguf parse is ~3 s; the tensor is the same on every reload
            self.norm_w = _read_output_norm(gguf_path)
        self.mctx = None
        if mmproj_path:
            p = M.mtmd_context_params_default()
            p.use_gpu = True
            p.print_timings = False
            p.warmup = False
            p.image_max_tokens = image_max_tokens
            self.marker = M.mtmd_default_marker()
            p.media_marker = self.marker
            self.mctx = M.mtmd_init_from_file(mmproj_path.encode(), self.model, p)
            if not self.mctx:
                raise RuntimeError("mtmd_init_from_file failed: " + mmproj_path)
        free1 = self._gpu_free()
        self.vram_bytes = max(0, free0 - free1) if free0 else os.path.getsize(gguf_path)

    def release(self):
        """Give the GPU back (model, context, vision tower, drafts, LoRA adapters); ensure() reloads. Settings and the
        LoRA / draft choices stay, so the next rewrite / encode is the same as before the release."""
        self.close()
        self.vram_bytes = 0

    def close(self):
        """Free the vision context, llama context and model (the VRAM copy)."""
        with self.lock:
            for h in self._drafts.values():
                _spec_lib().dh_spec_free(h)
            self._drafts.clear()
            self._draft_bytes.clear()
            self.prefixes.clear()  # their KV goes with the context
            self._prefix_lora = ()
            for e in self.vcache.values():
                M.mtmd_input_chunk_free(e["chunk"])
            self.vcache.clear()
            if self.mctx:
                M.mtmd_free(self.mctx)
                self.mctx = None
            for a in self._adapters.values():
                llama_cpp.llama_adapter_lora_free(a)
            self._adapters.clear()
            if self.ctx:
                llama_cpp.llama_free(self.ctx)
                self.ctx = None
            if self.model:
                llama_cpp.llama_model_free(self.model)
                self.model = None

    # ------------------------------------------------------------------ vision
    def _vision(self, img):
        """Vision tower output for one image, cached by content. Returns dict(chunk, embd, ptr, n_tokens)."""
        if self.mctx is None:
            raise RuntimeError("an image was given but no mmproj is loaded")
        a = _to_uint8_rgb(img)
        key = hashlib.sha1(a.tobytes()).hexdigest() + "%dx%d" % (a.shape[1], a.shape[0])
        hit = self.vcache.get(key)
        if hit is not None:
            self.vcache.move_to_end(key)
            self.stats["vision_hit"] += 1
            return hit
        h, w = a.shape[:2]
        buf = (ctypes.c_uint8 * a.size).from_buffer_copy(a.tobytes())
        bm = M.mtmd_bitmap_init(w, h, buf)
        chunks = M.mtmd_input_chunks_init()
        try:
            text = self.marker
            inp = M.mtmd_input_text(text, len(text), False, True)
            bms = (M.mtmd_bitmap_p_ctypes * 1)(bm)
            rc = M.mtmd_tokenize(self.mctx, chunks, ctypes.byref(inp), bms, 1)
            if rc != 0:
                raise RuntimeError("mtmd_tokenize rc=%d" % rc)
            chunk = None
            for i in range(M.mtmd_input_chunks_size(chunks)):
                c = M.mtmd_input_chunks_get(chunks, i)
                if M.mtmd_input_chunk_get_type(c) == M.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                    chunk = M.mtmd_input_chunk_copy(c)
            if chunk is None:
                raise RuntimeError("mtmd produced no image chunk")
            if M.mtmd_encode_chunk(self.mctx, chunk) != 0:
                raise RuntimeError("mtmd_encode_chunk failed")
            n_tok = M.mtmd_input_chunk_get_n_tokens(chunk)
            ptr = M.mtmd_get_output_embd(self.mctx)
            embd = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(n_tok * self.n_embd_inp,)).copy()
        finally:
            M.mtmd_input_chunks_free(chunks)
            M.mtmd_bitmap_free(bm)
        entry = {"chunk": chunk, "embd": embd, "ptr": embd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), "n_tokens": n_tok, "size": (w, h)}
        self.vcache[key] = entry
        self.stats["vision_encode"] += 1
        while len(self.vcache) > self.vcache_size:
            _, old = self.vcache.popitem(last=False)
            M.mtmd_input_chunk_free(old["chunk"])
        return entry

    def _on_image_batch(self, batch, user_data):
        if self._collect is not None:
            n = batch.n_tokens
            ptr = llama_cpp.llama_get_embeddings(self.ctx)
            arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(n * self.n_embd,))
            self._collect.append(np.array(arr, dtype=np.float32).reshape(n, self.n_embd))
        return 0

    # ------------------------------------------------------------------ layer tap
    def _on_eval(self, t, ask, user_data):
        if not self._tap_on:
            return False
        g = _ggml()
        k = self._tap_names.get(g.ggml_get_name(t))
        if k is None:
            return False
        if ask:
            return True
        nb = g.ggml_nbytes(t)
        a = np.empty(nb // 4, dtype=np.float32)  # l_out is f32 [n_embd, n_tokens] contiguous
        g.ggml_backend_tensor_get(t, a.ctypes.data, 0, nb)
        self._tap_buf[k].append(a.reshape(-1, self.n_embd))
        return True

    # ------------------------------------------------------------------ decode
    def _decode_text(self, toks, n_past, outs, logits_last):
        for s in range(0, len(toks), self.n_batch):
            part = toks[s:s + self.n_batch]
            n = len(part)
            last_part = s + self.n_batch >= len(toks)
            b = llama_cpp.llama_batch_init(n, 0, 1)
            try:
                for i, t in enumerate(part):
                    b.token[i] = t
                    b.pos[i] = n_past + i
                    b.n_seq_id[i] = 1
                    b.seq_id[i][0] = 0
                    b.logits[i] = 1 if (outs is not None or self._all_out or (logits_last and last_part and i == n - 1)) else 0
                b.n_tokens = n
                rc = llama_cpp.llama_decode(self.ctx, b)
                if rc != 0:
                    raise RuntimeError("llama_decode failed rc=%d (n_past=%d, n_ctx=%d)" % (rc, n_past, self.n_ctx))
                if outs is not None:
                    ptr = llama_cpp.llama_get_embeddings(self.ctx)
                    arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(n * self.n_embd,))
                    outs.append(np.array(arr, dtype=np.float32).reshape(n, self.n_embd))
            finally:
                llama_cpp.llama_batch_free(b)
            n_past += n
        return n_past

    @staticmethod
    def _system_prefix_len(seq):
        """Length of the system turn: everything before the second <|im_start|> (the start of the user turn)."""
        seen = 0
        for i, t in enumerate(seq):
            if not isinstance(t, numbers.Integral):
                return 0
            if int(t) == 151644:
                seen += 1
                if seen == 2:
                    return i
        return 0

    def _prefill(self, seq, collect, reuse_prefix=False):
        """seq: comfy token list items -- int ids and {"type": "image", "data": IMAGE} dicts.
        Returns (outs or None, image_spans [(index, size)], n_past).
        reuse_prefix: take the system turn's KV from the prefix cache instead of recomputing it (generation only;
        the encoder needs hidden states for every position, so it always runs the full sequence)."""
        llama_cpp.llama_memory_seq_rm(self.mem, 0, -1, -1)
        outs = [] if collect else None
        spans, run = [], []
        n_past = 0
        n_seq = 0
        plen = self._system_prefix_len(seq) if reuse_prefix else 0
        cache_after = False
        if plen >= 64:
            key = tuple(int(t) for t in seq[:plen])
            slot = self.prefixes.get(key)
            if slot is not None:
                self.prefixes.move_to_end(key)
                llama_cpp.llama_memory_seq_cp(self.mem, slot, 0, 0, plen)
                n_past = n_seq = plen
                seq = seq[plen:]
                self.stats["prefix_hit"] += 1
                self.stats["prefix_tokens_reused"] += plen
            else:
                self.stats["prefix_miss"] += 1
                cache_after = True

        def flush(logits_last):
            nonlocal n_past, n_seq
            if run:
                n_past = self._decode_text(run, n_past, outs, logits_last)
                n_seq += len(run)
                run.clear()

        for item in seq:
            if isinstance(item, numbers.Integral):
                run.append(int(item))
            elif isinstance(item, dict) and item.get("type") == "image":
                flush(False)
                v = self._vision(item["data"])
                got_before = sum(o.shape[0] for o in outs) if collect else 0
                self._collect = outs
                newp = llama_cpp.llama_pos(0)
                try:
                    rc = M.mtmd_helper_decode_image_chunk(self.mctx, self.ctx, v["chunk"], v["ptr"], n_past, 0, self.n_batch,
                                                          ctypes.byref(newp), self._cb, None)
                finally:
                    self._collect = None
                if rc != 0:
                    raise RuntimeError("image decode failed rc=%d" % rc)
                if collect:
                    got = sum(o.shape[0] for o in outs) - got_before
                    if got != v["n_tokens"]:
                        raise RuntimeError("image outputs %d != image tokens %d" % (got, v["n_tokens"]))
                spans.append((n_seq, v["n_tokens"]))
                n_seq += v["n_tokens"]
                n_past = newp.value
            else:
                raise RuntimeError("unsupported token entry for the llama.cpp text encoder: %r" % (type(item),))
        flush(True)
        if cache_after:
            if len(self.prefixes) >= self.prefix_slots:
                _, old = self.prefixes.popitem(last=False)
                llama_cpp.llama_memory_seq_rm(self.mem, old, -1, -1)
            used = set(self.prefixes.values())
            slot = next(i for i in range(1, self.prefix_slots + 1) if i not in used)
            llama_cpp.llama_memory_seq_cp(self.mem, 0, slot, 0, plen)
            self.prefixes[key] = slot
        return outs, spans, n_past

    # ------------------------------------------------------------------ lora
    def _adapter(self, path):
        if path not in self._adapters:
            a = llama_cpp.llama_adapter_lora_init(self.model, path.encode())
            if not a:
                raise RuntimeError("failed to load LoRA " + path)
            self._adapters[path] = a
        return self._adapters[path]

    def _set_lora(self, spec):
        spec = [(p, s) for p, s in (spec or []) if s != 0.0]
        if not spec:
            llama_cpp.llama_set_adapters_lora(self.ctx, None, 0, None)
            return
        arr = (llama_cpp.llama_adapter_lora_p_ctypes * len(spec))(*[self._adapter(p) for p, _ in spec])
        sc = (ctypes.c_float * len(spec))(*[float(s) for _, s in spec])
        if llama_cpp.llama_set_adapters_lora(self.ctx, arr, len(spec), sc) != 0:
            raise RuntimeError("llama_set_adapters_lora failed")

    # ------------------------------------------------------------------ text
    def tokenize(self, text):
        """The GGUF's own tokenizer, special tokens parsed (<|im_start|> etc.), no BOS added."""
        self.ensure()
        b = text.encode("utf-8")
        buf = (llama_cpp.llama_token * (len(b) + 16))()
        n = llama_cpp.llama_tokenize(self.vocab, b, len(b), buf, len(buf), False, True)
        if n < 0:
            raise RuntimeError("llama_tokenize failed (%d)" % n)
        return list(buf[:n])

    def detokenize(self, ids):
        self.ensure()
        out = b""
        buf = ctypes.create_string_buffer(512)
        for t in ids:
            n = llama_cpp.llama_token_to_piece(self.vocab, int(t), buf, len(buf), 0, True)
            out += buf.raw[:max(n, 0)]
        return out.decode("utf-8", "replace")

    # ------------------------------------------------------------------ roles
    def encode(self, seq, layers=None, key_limit=None, final_norm=False):
        self.ensure()
        return self._encode(seq, layers, key_limit, final_norm)

    def _encode(self, seq, layers=None, key_limit=None, final_norm=False):
        """TE role, with enc_lora attached.
        layers=None: hidden states of the last layer for every position of seq (text AND vision tokens), pre-final-norm
        equivalent, shape [n, n_embd]; final_norm=True: after the model's final RMSNorm (Boogu reads it that way).
        layers=[i, ...]: raw outputs of those decoder layers (0-based), shape [n, len(layers), n_embd]; needs taps=True.
        key_limit=k: positions >= k attend only to positions < k (not to themselves or each other) - right padding under
        an attention mask, as HF/ComfyUI compute it (FLUX.2 klein pads to 512 and its DiT reads the pads)."""
        with self.lock:
            # the rewrite's system-prompt cache (other sequences in the unified KV) changes what seq 0 encodes: measured
            # 2.6% relative drift of layer 34 on the same prompt after one rewrite (2026-09-28), bit-exact again once the
            # cache is dropped. The TE output must not depend on what was rewritten before, so drop it (costs the next
            # non-speculative rewrite one system-prompt prefill; speculative rewrites never use it).
            if self.prefixes:
                for sl in self.prefixes.values():
                    llama_cpp.llama_memory_seq_rm(self.mem, sl, -1, -1)
                self.prefixes.clear()
            self._set_lora(self.enc_lora)
            try:
                if layers:
                    if not self.taps:
                        raise RuntimeError("this backend was created without taps=True")
                    # llama.cpp computes the last layer only for rows that are outputs (inp_out_ids); tapping it needs every
                    # position marked as an output, in embeddings mode (n_embd per row, not a full-vocabulary logit row)
                    self._all_out = max(layers) >= self.n_layer - 1
                    llama_cpp.llama_set_embeddings(self.ctx, self._all_out)
                    self._tap_names = {("l_out-%d" % i).encode(): k for k, i in enumerate(layers)}
                    self._tap_buf = [[] for _ in layers]
                    self._tap_on = True
                    if key_limit is not None:
                        _set_key_pos_limit(self.mem, key_limit)
                    try:
                        _, spans, n = self._prefill(seq, False)
                    finally:
                        self._tap_on = False
                        self._all_out = False
                        if key_limit is not None:
                            _set_key_pos_limit(self.mem, -1)
                    if any(not b for b in self._tap_buf):
                        raise RuntimeError("layer tap missed %s" % [i for i, b in zip(layers, self._tap_buf) if not b])
                    got = [np.concatenate(b, 0) for b in self._tap_buf]
                    self._tap_buf = []
                    if any(x.shape[0] != n for x in got):
                        raise RuntimeError("layer tap got %s rows for %d positions" % ([x.shape[0] for x in got], n))
                    return torch.from_numpy(np.stack(got, 1)), spans
                llama_cpp.llama_set_embeddings(self.ctx, True)
                outs, spans, _ = self._prefill(seq, True)
                h = torch.from_numpy(np.concatenate(outs, 0))
                if final_norm:  # llama.cpp's embedding output already is output_norm(h)
                    return h, spans
                # llama.cpp returns output_norm(h) = h/rms(h)*w; dividing by w leaves h/rms(h), which the DiT's per-token
                # RMSNorm (txt_in.text_norm) cannot tell apart from the raw last hidden state comfy feeds it.
                return h / self.norm_w, spans
            finally:
                self._set_lora(None)

    def _draft(self, path, n_max):
        key = (path, int(n_max))
        if key not in self._drafts:
            lib = _spec_lib()
            if lib is None:
                raise RuntimeError("speculative decoding needs libdh_spec.so next to the llama.cpp libraries (run build.py)")
            free0 = self._gpu_free()
            h = lib.dh_spec_create(self.ctx, path.encode(), int(n_max), -1)
            if not h:
                raise RuntimeError("could not load the draft model " + path)
            logging.info("DualHeadDragon: draft %s loaded (%s, n_max %d)", path, lib.dh_spec_type(h).decode(), n_max)
            self._drafts[key] = h
            self._draft_bytes[key] = max(0, free0 - self._gpu_free()) if free0 else 0
            self.vram_bytes += self._draft_bytes[key]
        return self._drafts[key]

    def generate(self, seq, max_length=512, do_sample=True, temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                 repetition_penalty=1.0, presence_penalty=0.0, seed=None, lora=None, draft=None, draft_n_max=15,
                 draft_tree=0):
        """PE role: sample a continuation of seq with the rewrite LoRA attached. Returns generated token ids.
        draft: a speculative draft GGUF (DFlash etc.) -- same distribution, faster; text-only prompts, no penalties.
        draft_tree: DFlash2 only, draft tokens verified per round as a tree (0 = off; docs/SPEC_STRATEGY.md)."""
        self.ensure()
        if draft and _spec_lib() is not None and all(isinstance(t, numbers.Integral) for t in seq) \
                and repetition_penalty == 1.0 and presence_penalty == 0.0:
            return self._generate_spec(seq, max_length, do_sample, temperature, top_k, top_p, min_p, seed, lora, draft, draft_n_max,
                                       draft_tree)
        with self.lock:
            llama_cpp.llama_set_embeddings(self.ctx, False)
            self._set_lora(lora if lora is not None else self.gen_lora)
            # a LoRA changes the system-prompt KV, so cached prefixes are only valid for the adapter set they were built with
            lkey = tuple(lora if lora is not None else self.gen_lora)
            if lkey != getattr(self, "_prefix_lora", ()):
                for sl in self.prefixes.values():
                    llama_cpp.llama_memory_seq_rm(self.mem, sl, -1, -1)
                self.prefixes.clear()
                self._prefix_lora = lkey
            chain = None
            try:
                # cached system prompts share the unified KV with this sequence: with a small n_ctx (e.g. 4096 for a
                # 12 GB card) a cached 1.8k-token prompt plus this 1.9k prompt + its answer does not fit and decode
                # fails. Drop the oldest cached prompts (not the one this sequence reuses) until prompt + answer fit.
                plen = self._system_prefix_len(seq) if self.use_prefix_cache else 0
                mine = tuple(int(t) for t in seq[:plen]) if plen >= 64 else None
                need = len(seq) + int(max_length) + 16
                for k in list(self.prefixes):
                    if sum(len(x) for x in self.prefixes if x != mine) + need <= self.n_ctx:
                        break
                    if k != mine:
                        llama_cpp.llama_memory_seq_rm(self.mem, self.prefixes.pop(k), -1, -1)
                _, _, n_past = self._prefill(seq, False, reuse_prefix=self.use_prefix_cache)
                max_length = max(1, min(int(max_length), self.n_ctx - n_past - 1))
                chain = llama_cpp.llama_sampler_chain_init(llama_cpp.llama_sampler_chain_default_params())
                if repetition_penalty != 1.0 or presence_penalty != 0.0:
                    llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_penalties(
                        self.n_vocab, max_length, float(repetition_penalty), 0.0, float(presence_penalty)))
                if do_sample:
                    if top_k and top_k > 0:
                        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_top_k(int(top_k)))
                    if top_p is not None and top_p < 1.0:
                        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_top_p(float(top_p), 1))
                    if min_p and min_p > 0.0:
                        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_min_p(float(min_p), 1))
                    llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_temp(float(temperature)))
                    s = random.getrandbits(32) if seed is None else int(seed) & 0xFFFFFFFF
                    llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_dist(s))
                else:
                    llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_greedy())
                out = []
                for _ in range(max_length):
                    tok = llama_cpp.llama_sampler_sample(chain, self.ctx, -1)
                    if llama_cpp.llama_vocab_is_eog(self.vocab, tok):
                        break
                    out.append(int(tok))
                    n_past = self._decode_text([int(tok)], n_past, None, True)
                return out
            finally:
                if chain is not None:
                    llama_cpp.llama_sampler_free(chain)
                self._set_lora(None)
                llama_cpp.llama_set_embeddings(self.ctx, True)


    def _generate_spec(self, seq, max_length, do_sample, temperature, top_k, top_p, min_p, seed, lora, draft, n_max,
                       tree=0):
        with self.lock:
            llama_cpp.llama_set_embeddings(self.ctx, False)
            self._set_lora(lora if lora is not None else self.gen_lora)
            try:
                h = self._draft(draft, n_max)
                if hasattr(_spec_lib(), "dh_spec_set_strategy"):
                    _spec_lib().dh_spec_set_strategy(h, int(tree or 0))
                elif tree:
                    logging.warning("DualHeadDragon: this libdh_spec has no tree verification, rebuild (build.py)")
                ids = (ctypes.c_int32 * len(seq))(*[int(t) for t in seq])
                max_length = max(1, min(int(max_length), self.n_ctx - len(seq) - 1))
                out = (ctypes.c_int32 * max_length)()
                st = (ctypes.c_int * 3)()
                s = random.getrandbits(32) if seed is None else int(seed) & 0xFFFFFFFF
                n = _spec_lib().dh_spec_generate(h, ids, len(seq), max_length, float(temperature) if do_sample else 0.0,
                                                 int(top_k or 0), float(top_p if top_p is not None else 1.0), float(min_p or 0.0),
                                                 s, out, st)
                if n < 0:
                    raise RuntimeError("speculative generation failed")
                self.stats["spec_drafted"] = self.stats.get("spec_drafted", 0) + st[0]
                self.stats["spec_accepted"] = self.stats.get("spec_accepted", 0) + st[1]
                self.stats["spec_rounds"] = self.stats.get("spec_rounds", 0) + st[2]
                self.last_spec = {"tokens": n, "drafted": st[0], "accepted": st[1], "rounds": st[2]}
                return list(out[:n])
            finally:
                # The draft's KV cache shares the target's cell bookkeeping (llama.cpp mem_other), so a speculative run
                # can clobber the cached system-prompt sequences; a later hit would then decode without its system turn.
                # Drop every cached prefix so the next rewrite prefills it again (measured: 1 of 1 post-spec hit broken).
                for sl in self.prefixes.values():
                    llama_cpp.llama_memory_seq_rm(self.mem, sl, -1, -1)
                self.prefixes.clear()
                self._set_lora(None)
                llama_cpp.llama_set_embeddings(self.ctx, True)


_BACKENDS = {}


def get_backend(gguf_path, mmproj_path, n_ctx, taps=False, n_ubatch=2048):
    """One backend per (gguf, mmproj, n_ctx, taps): the weights live in VRAM once however many nodes use them.
    Which layers to read is chosen per encode() call, so every product sharing one TE (Z-Image and klein on Qwen3-4B)
    shares one backend."""
    key = (gguf_path, mmproj_path, n_ctx, bool(taps), int(n_ubatch))
    if key not in _BACKENDS:
        for k in list(_BACKENDS):  # a different model replaces the old one instead of stacking a second copy
            logging.info("DualHeadDragon: releasing %s", k[0])
            _BACKENDS.pop(k).close()
        _BACKENDS[key] = DualHeadBackend(gguf_path, mmproj_path, n_ctx=n_ctx, n_ubatch=int(n_ubatch), taps=taps)
    return _BACKENDS[key]
