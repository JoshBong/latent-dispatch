# Ported from JoshBong/LatentMAS methods/cache_ops.py (a fork of Gen-Verse/LatentMAS, Apache-2.0):
# https://github.com/JoshBong/LatentMAS/blob/main/methods/cache_ops.py
"""KV-cache surgery: copy, slice, concatenate, and shift keys to new positions.

A cache is a transformers DynamicCache. Each layer holds K and V shaped [batch, kv_heads, seq, head_dim].
"""
import torch
from transformers import DynamicCache


def layers(cache) -> tuple:
    """The cache as a tuple of (K, V) per layer."""
    return tuple((layer.keys, layer.values) for layer in cache.layers)


def _build(kv_per_layer) -> DynamicCache:
    return DynamicCache.from_legacy_cache(tuple(kv_per_layer))


def length(cache) -> int:
    if cache is None or not cache.layers or cache.layers[0].keys is None:
        return 0
    return int(cache.layers[0].keys.shape[-2])


def copy(cache) -> DynamicCache:
    return _build((k.clone(), v.clone()) for k, v in layers(cache))


def suffix(cache, start: int) -> DynamicCache:
    """Positions [start:], e.g. a worker's own tokens after the shared prefix."""
    return _build((k[..., start:, :].contiguous(), v[..., start:, :].contiguous()) for k, v in layers(cache))


def concat(caches: list) -> DynamicCache:
    """Join caches along the sequence axis, layer by layer."""
    per_layer = zip(*(layers(c) for c in caches))
    return _build((torch.cat([k for k, _ in L], dim=-2), torch.cat([v for _, v in L], dim=-2)) for L in per_layer)


def rope_inv_freq(hf_model) -> torch.Tensor:
    """The rotary frequencies the model actually uses, read off its buffer."""
    for mod in hf_model.modules():
        inv = getattr(mod, "inv_freq", None)
        if inv is not None:
            if getattr(mod, "rope_type", "default") == "dynamic":
                raise ValueError("dynamic NTK rope: shifting cached keys is not valid for this model")
            return inv.detach().float()
    raise ValueError("no rotary embedding (inv_freq) found")


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def shift(cache, delta: int, inv_freq: torch.Tensor) -> DynamicCache:
    """Move a cache's keys `delta` positions later. Cached keys are R(p)·k and R(delta)·R(p) = R(p+delta), so this
    is exact for the keys. Values carry no position. The hidden states that produced the keys were still computed at
    the old positions: that part does not move (it's what B1-mask − K0 measures)."""
    if delta == 0:
        return copy(cache)
    k0 = layers(cache)[0][0]
    angle = float(delta) * inv_freq.to(device=k0.device, dtype=torch.float32)
    emb = torch.cat([angle, angle], dim=-1)
    cos, sin = emb.cos(), emb.sin()

    def rotate(k):   # in fp32, cast once: rotating in bf16 doubles the error vs encoding at the shifted position
        kf = k.float()
        return (kf * cos + _rotate_half(kf) * sin).to(k.dtype)

    return _build((rotate(k), v) for k, v in layers(cache))
