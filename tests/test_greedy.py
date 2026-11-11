"""model.greedy's thinking budget and stopping rules, driven by scripted next tokens (no real model needed)."""
from types import SimpleNamespace

import copy

import torch


def _scripted(model, script):
    """Replace model.forward: each call returns logits whose argmax is the next scripted token."""
    it = iter(script)
    vocab = len(model.tok)

    def forward(ids, cache=None, start_pos=None, embeds=None, hidden=False):
        logits = torch.zeros(1, 1, vocab)
        logits[0, -1, next(it)] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=cache)

    model.forward = forward
    return model


def test_budget_hit_forces_close_then_answers(tiny):
    m = copy.deepcopy(tiny)
    a, b, eos = 100, 200, m.tok.convert_tokens_to_ids("<|im_end|>")
    n_force = len(m.ids("\n</think>\n\n"))
    _scripted(m, [a] * 4 + [b] * (n_force) + [b, b, eos])
    g = m.greedy([1], think_budget=3, max_new=10)
    assert g.think_tokens == 3 and g.answer_tokens >= 1
    assert "</think>" in g.text


def test_natural_close_and_answer_cap(tiny):
    m = copy.deepcopy(tiny)
    close = m.think_end_id
    _scripted(m, [100, 100, close] + [200] * 50)
    g = m.greedy([1], think_budget=10, max_new=5)
    assert g.think_tokens == 2 and g.answer_tokens == 5 and g.tokens == 2 + 1 + 5


def test_eos_stops_without_thinking(tiny):
    m = copy.deepcopy(tiny)
    _scripted(m, [100, m.tok.convert_tokens_to_ids("<|im_end|>")])
    g = m.greedy([1], max_new=10)
    assert g.tokens == 1 and g.think_tokens == 0
