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
