"""The recompute masks must isolate the intended edges and reproduce both existing baselines."""
import pytest
import torch

from latent_dispatch import arms, kv, masking, prompts, results


def test_visibility_selects_a_paragraphs_only(tiny, two_workers):
    q = two_workers
    s = tiny.ids(prompts.prefix(tiny.fmt, q["question"]))
    a, groups = masking.paragraph_tokens(tiny, q["workers"][0])
    b_start = len(s) + len(a)
    by_condition = {c: masking.layout(tiny, q, c) for c in masking.CONDITIONS}
    for condition, (ids, mask, sizes, meta) in by_condition.items():
        assert ids == s + tiny.ids(prompts.block(q["workers"][0])) + tiny.ids(prompts.block(q["workers"][1]))
        assert sizes[0] == len(a)
        visible = mask[0, 0, b_start, len(s):b_start] == 0
        expected = {"blocked": [], "gold": groups[0], "distractor": groups[1],
                    "except_gold": [i for i in range(len(a)) if i not in groups[0]],
                    "full": list(range(len(a)))}[condition]
        assert visible.nonzero().flatten().tolist() == expected
        assert meta["a_visible_tokens"] == len(expected)
        future = torch.triu(torch.ones(len(ids), len(ids), dtype=torch.bool), 1)
        assert torch.all(mask[0, 0][future] < 0)
        assert torch.all(mask[0, 0, b_start:, :len(s)] == 0)
    assert by_condition["distractor"][3]["a_paragraphs"] == [1]


@pytest.mark.parametrize("condition,baseline", [("blocked", "blocked"), ("full", "full")])
def test_mask_endpoints_match_original_logits_and_caches(tiny_family, two_workers, condition, baseline):
    m = tiny_family
    cache, pos, sizes, _ = masking.context(m, two_workers, condition)
    ref, ref_pos, ref_sizes = (arms.k0_context(m, two_workers, positions="true") if baseline == "blocked"
                                else arms.b1_context(m, two_workers))
    assert (pos, sizes) == (ref_pos, ref_sizes)
    for actual, expected in zip(kv.layers(cache), kv.layers(ref)):
        for a, b in zip(actual, expected):
            assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()
    tail = m.ids(prompts.tail(m.fmt, "off"))
    actual = m.forward(tail, cache, pos).logits
    expected = m.forward(tail, ref, ref_pos).logits
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4), (actual - expected).abs().max()


def test_mask_validates_controls_and_gold_refs(tiny, two_workers):
    q = {**two_workers, "workers": [[two_workers["workers"][0][0]], two_workers["workers"][1]]}
    with pytest.raises(ValueError, match="no distractor"):
        masking.layout(tiny, q, "distractor")
    with pytest.raises(ValueError, match="invalid gold"):
        masking.layout(tiny, {**q, "gold": [{"worker": 0, "index": 99}]}, "gold")
    with pytest.raises(ValueError, match="exactly two"):
        masking.layout(tiny, {**q, "workers": q["workers"][:1]}, "full")


def test_no_a_gold_collapses_to_the_expected_controls(tiny, two_workers):
    q = {**two_workers, "gold": [{"hop": 0, "worker": 1, "index": 0}]}
    _, blocked, _, _ = masking.layout(tiny, q, "blocked")
    _, gold, _, meta = masking.layout(tiny, q, "gold")
    _, full, _, _ = masking.layout(tiny, q, "full")
    _, except_gold, _, _ = masking.layout(tiny, q, "except_gold")
    assert meta["a_gold_paragraphs"] == []
    assert torch.equal(blocked, gold)
    assert torch.equal(full, except_gold)


def test_gold_likelihood_and_generation_are_independent(tiny, two_workers):
    scored = arms.masked_recompute(tiny, two_workers, "gold", max_answer_tokens=2)
    unscored = arms.masked_recompute(tiny, two_workers, "gold", score_gold=False, max_answer_tokens=2)
    assert scored["text"] == unscored["text"]
    assert scored["gold_answer"] == two_workers["answers"][0]
    assert scored["gold_token_count"] > 0
    assert scored["gold_logprob"] < 0
    assert scored["gold_mean_logprob"] == pytest.approx(scored["gold_logprob"] / scored["gold_token_count"])
    cache, pos, _, _ = masking.context(tiny, two_workers, "gold")
    tail = tiny.ids(prompts.tail(tiny.fmt, "off"))
    target = tiny.ids("\\boxed{" + scored["gold_answer"] + "}")
    out = tiny.forward(tail, cache, pos)
    actual = 0.0
    for token in target:
        actual += torch.log_softmax(out.logits[0, -1].float(), -1)[token].item()
        pos += len(tail)
        tail = [token]
        out = tiny.forward(tail, out.past_key_values, pos)
    assert scored["gold_logprob"] == pytest.approx(actual, abs=1e-4)
    with pytest.raises(ValueError, match="gold likelihood requires"):
        arms.masked_recompute(tiny, two_workers, "gold", thinking="forced")


def test_result_rows_are_resumable_and_keep_mask_metadata(tiny, two_workers, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    options = {"condition": "gold", "max_answer_tokens": 1}
    results.run(tiny, [two_workers], "mask_a_gold", arms.masked_recompute, **options)
    results.run(tiny, [two_workers], "mask_a_gold", arms.masked_recompute, **options)
    table = results.load_results([two_workers], {"mask_a_gold": (arms.masked_recompute, options)},
                                 tiny.name, tiny.dtype_name)
    assert len(table) == 1
    assert table.iloc[0]["mask"]["a_paragraphs"] == [0]
    assert table.iloc[0]["gold_logprob"] < 0
