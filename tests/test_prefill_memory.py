"""Prefill must preserve decoder outputs while avoiding unused large tensors."""
import pytest
import torch

from latent_dispatch import kv


@pytest.mark.parametrize('mode', ['whole', 'cached', 'shifted_masked'])
def test_prefill_matches_causal_lm_without_projection_or_layer_history(tiny_family, mode):
    model = tiny_family
    ids = [31, 48, 26, 59, 72, 13, 20]
    prefix = None if mode == 'whole' else model.forward([21, 14, 52]).past_key_values
    start = 17 if mode == 'shifted_masked' else None
    mask = None
    if mode == 'shifted_masked':
        allowed = torch.ones(len(ids), kv.length(prefix) + len(ids), dtype=torch.bool).tril(kv.length(prefix))
        allowed[:, 1] = False
        mask = torch.zeros_like(allowed, dtype=next(model.hf.parameters()).dtype)
        mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
        mask = mask[None, None]
    reference = model.forward(ids, kv.copy(prefix) if prefix is not None else None,
                              start, hidden=True, attention_mask=mask)

    def reject_projection(module, args):
        raise AssertionError('prefill must not project every token into the vocabulary')

    def check_decoder(module, args, kwargs):
        assert kwargs['output_hidden_states'] is False

    projection = model.hf.get_output_embeddings().register_forward_pre_hook(reject_projection)
    decoder = model.hf.base_model.register_forward_pre_hook(check_decoder, with_kwargs=True)
    try:
        cache, hidden = model.prefill(ids, kv.copy(prefix) if prefix is not None else None,
                                      start, attention_mask=mask)
    finally:
        projection.remove()
        decoder.remove()
    torch.testing.assert_close(hidden, reference.hidden_states[-1][:, -1, :], atol=1e-6, rtol=1e-5)
    for actual, expected in zip(kv.layers(cache), kv.layers(reference.past_key_values), strict=True):
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-5)
            assert not a.requires_grad
    assert not hidden.requires_grad
    assert hidden.untyped_storage().nbytes() == hidden.numel() * hidden.element_size()


@pytest.mark.parametrize('masked', [False, True])
def test_long_prefill_preserves_positions_cache_and_decode(tiny_family, masked):
    model = tiny_family
    ids = torch.randint(0, 1000, (1031,), generator=torch.Generator().manual_seed(7)).tolist()
    prefix = model.forward([21, 14, 52]).past_key_values
    past = kv.length(prefix)
    mask = None
    if masked:
        allowed = torch.ones(len(ids), past + len(ids), dtype=torch.bool).tril(past)
        allowed[:, 1] = False
        mask = torch.zeros_like(allowed, dtype=next(model.hf.parameters()).dtype)
        mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
        mask = mask[None, None]
    reference = model.forward(ids, kv.copy(prefix), 17, hidden=True, attention_mask=mask)
    cache, hidden = model.prefill(ids, kv.copy(prefix), 17, mask)
    torch.testing.assert_close(hidden, reference.hidden_states[-1][:, -1, :], atol=1e-4, rtol=1e-4)
    for actual, expected in zip(kv.layers(cache), kv.layers(reference.past_key_values), strict=True):
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, atol=1e-4, rtol=1e-4)
    actual = model.forward([51], cache, 17 + len(ids)).logits
    expected = model.forward([51], reference.past_key_values, 17 + len(ids)).logits
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
