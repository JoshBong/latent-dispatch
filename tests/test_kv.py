"""KV primitives on a tiny Qwen3 (ported from the fork's tests/test_cache_ops.py and test_cache_reindex.py)."""
import copy

import torch

from latent_dispatch import kv


def _ids(n, seed=0):
    return torch.randint(0, 1000, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_split_prefill_equals_whole(tiny):
    ids = _ids(24)
    whole = tiny.forward(ids).logits
    cache, _ = tiny.prefill(ids[:10])
    part = tiny.forward(ids[10:], cache).logits
    assert torch.allclose(whole[:, 10:], part, atol=1e-4), (whole[:, 10:] - part).abs().max()


def test_suffix_and_concat_rebuild_the_whole(tiny):
    ids = _ids(20)
    full, _ = tiny.prefill(ids)
    prefix, _ = tiny.prefill(ids[:8])
    rebuilt = kv.concat([prefix, kv.suffix(full, 8)])
    assert kv.length(rebuilt) == 20
    for (k1, v1), (k2, v2) in zip(kv.layers(rebuilt), kv.layers(full)):
        assert torch.allclose(k1, k2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)


def test_copy_is_independent(tiny):
    cache, _ = tiny.prefill(_ids(12))
    kv.layers(kv.copy(cache))[0][0].add_(99.0)
    assert kv.layers(cache)[0][0].abs().max() < 50


def test_shift_equals_encoding_at_the_shifted_position(tiny):
    """Same chunk, no prefix, encoded at p vs p+d: attention depends only on relative positions, so shifting the
    keys of the first by d must give the second exactly (holds with Qwen3's qk-norm, which sits before RoPE)."""
    ids = _ids(6)
    at_p, _ = tiny.prefill(ids, start_pos=5)
    at_pd, _ = tiny.prefill(ids, start_pos=12)
    for (ks, vs), (kb, vb) in zip(kv.layers(kv.shift(at_p, 7, tiny.inv_freq)), kv.layers(at_pd)):
        assert torch.allclose(ks, kb, atol=1e-4), (ks - kb).abs().max()
        assert torch.allclose(vs, vb, atol=1e-5)
    back = kv.shift(kv.shift(at_p, 9, tiny.inv_freq), -9, tiny.inv_freq)
    assert torch.allclose(kv.layers(back)[0][0], kv.layers(at_p)[0][0], atol=1e-4)


def test_shift_in_bf16_stays_close_to_native_encoding(tiny):
    """Production is bf16. Shifted keys must stay within bf16 rounding (~4e-3 relative) of keys encoded at the
    shifted position. A regression bound: it catches gross errors such as rotating with bf16-rounded frequencies
    (~8e-2 here), not the small fp32-vs-bf16 rotation difference."""
    m = copy.deepcopy(tiny)
    rope = [(mod, mod.inv_freq.clone()) for mod in m.hf.modules() if getattr(mod, "inv_freq", None) is not None]
    m.hf = m.hf.to(torch.bfloat16)
    for mod, inv in rope:      # from_pretrained(dtype=bf16) keeps the RoPE buffer in fp32; .to() would round it
        mod.inv_freq = inv
    ids = _ids(32)
    at_p, _ = m.prefill(ids, start_pos=0)
    at_pd, _ = m.prefill(ids, start_pos=1500)
    shifted = kv.layers(kv.shift(at_p, 1500, m.inv_freq))[0][0].float()
    native = kv.layers(at_pd)[0][0].float()
    assert ((shifted - native).norm() / native.norm()).item() < 6e-3
