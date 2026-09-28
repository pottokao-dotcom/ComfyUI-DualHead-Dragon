"""Let ComfyUI's model management see and evict the dual-head backend (llama.cpp weights live outside torch).

The CLIP's ModelPatcher normally reports ~0 bytes for this model (no torch weights), so ComfyUI never makes room for
it and never takes its VRAM back. In "managed" mode the patcher reports what the backend really holds (model, KV,
buffers, drafts -- measured when it loads) and ComfyUI's unload releases it; the next encode / rewrite reloads it.
That time-shares one card: PE + TE first, then the DiT + VAE get the memory (peak = the larger phase, not the sum).
"resident" keeps the old behaviour (stays loaded, invisible to ComfyUI -- other models in the workflow cannot get that
memory back, so a bigger image / upscaler / second model can run out of memory). "auto" = managed on every card:
with enough VRAM ComfyUI never evicts it (16 GB: no release, same speed as resident).
Measured (RTX 5060 Ti, master-v1, Z-Image, V6 Q4_K_M + LoRA + DFlash Q6, n_ctx 4096, ubatch 512), s/image warm:
  budget   resident   managed
  5.6 GB   OOM        10.8
  7.6 GB   11.7       10.3
  10.0 GB  10.1       10.2
  11.6 GB  9.4        10.15  (ComfyUI's reserve estimate evicts although resident fits)
  16 GB    9.5        9.4    (no release)
A reload costs ~0.5 s (model 0.32 s + draft 0.18 s; the output_norm read is cached).

One CLIP per (backend, type, mode): a re-executed loader returns the same CLIP, so ComfyUI never holds two unrelated
patchers for one backend. ComfyUI's own clones (clip.clone() in CLIPTextEncode) are handled in DHPatcher.detach.
"""
import logging
import os

import comfy.model_management
import comfy.model_patcher

MODES = ["auto", "resident", "managed"]
_CLIPS = {}  # (id(backend), type name, mode) -> CLIP, so a re-executed loader returns the same patcher


class DHPatcher(comfy.model_patcher.ModelPatcher):
    """Installed by class swap on the CLIP's patcher. Clones (CLIPTextEncode uses clip.clone()) keep the class but not
    our attributes, so the backend is found through the parent chain."""

    @property
    def dh_backend(self):
        p = self
        while p is not None:
            b = p.__dict__.get("_dh_backend")
            if b is not None:
                return b
            p = getattr(p, "parent", None)
        raise AttributeError("DHPatcher without a dual-head backend")

    def model_size(self):
        b = self.dh_backend
        return b.vram_bytes or getattr(b, "_dh_last_bytes", 0)

    def loaded_size(self):
        b = self.dh_backend
        return b.vram_bytes if b.loaded else 0

    def current_loaded_device(self):
        return self.load_device if self.dh_backend.loaded else self.offload_device

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        b = self.dh_backend
        was = b.loaded
        b.ensure()
        b._dh_last_bytes = b.vram_bytes
        return 0 if was else b.vram_bytes

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        return 0  # all or nothing: ComfyUI then calls detach()

    def detach(self, unpatch_all=True):
        # unpatch_all=False: ComfyUI is only swapping this patcher for a clone of the same model (load_models_gpu's
        # is_clone branch) -- the clone uses the same backend right away, so keep it loaded.
        if not unpatch_all:
            return self.model
        b = self.dh_backend
        if b.loaded:
            b._dh_last_bytes = b.vram_bytes
            logging.info("DualHeadDragon: releasing %.2f GB for other models (reloads on the next encode / rewrite)",
                         b.vram_bytes / 1024 ** 3)
            if os.environ.get("DH_DEBUG_RELEASE"):
                import traceback
                logging.info("DualHeadDragon: release called from\n%s", "".join(traceback.format_stack(limit=14)))
            b.release()
        return self.model


def resolve(mode):
    return "managed" if mode == "auto" else mode


def manage(make_clip, backend, mode, type_name=""):
    """CLIP for this backend: one per (backend, type, mode), its patcher under ComfyUI's management when managed."""
    m = resolve(mode)
    key = (id(backend), type_name, m)
    clip = _CLIPS.get(key)
    if clip is not None:
        return clip
    for k in [k for k in _CLIPS if k[0] != id(backend)]:  # a replaced backend's CLIPs are dead
        _CLIPS.pop(k)
    clip = make_clip(backend)
    if m == "managed":
        p = clip.patcher
        p.__class__ = DHPatcher
        p._dh_backend = backend
        backend._dh_last_bytes = backend.vram_bytes
    _CLIPS[key] = clip
    return clip


def load(clip):
    """Before calling the backend directly (rewrite): let ComfyUI make room / reload, like a text encode does."""
    comfy.model_management.load_models_gpu([clip.patcher])
