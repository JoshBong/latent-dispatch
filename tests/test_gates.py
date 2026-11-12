"""The sanity gates from the README. If these fail, any B1 − K0 gap is a bug, not encoding."""
import torch

from latent_dispatch import arms, kv, prompts

from .conftest import make_question


def _tail_logits(model, context, thinking="off"):
    cache, pos, _ = context
    return model.forward(model.ids(prompts.tail(model.fmt, thinking)), cache, pos).logits[0]


# ---- K0 / B1-mask equal what they claim to be, on every family's architecture

def test_k0_with_one_worker_reproduces_b1(tiny_family, two_workers):
    m = tiny_family
    one = make_question([[p for w in two_workers["workers"] for p in w]])
    lb = _tail_logits(m, arms.b1_context(m, one))
    lk = _tail_logits(m, arms.k0_context(m, one))
    assert torch.allclose(lb, lk, atol=1e-4), (lb - lk).abs().max()


def test_b1_mask_equals_explicit_block_diagonal_mask(tiny_family, two_workers):
    """B1-mask == one forward over the whole B1 sequence with a 4D mask where each block sees only S and itself."""
    tiny, q = tiny_family, two_workers
    s = tiny.ids(prompts.prefix(tiny.fmt, q["question"]))
    blocks = [tiny.ids(prompts.block(w)) for w in q["workers"]]
    tail = tiny.ids(prompts.tail(tiny.fmt, "off"))
    ids = s + blocks[0] + blocks[1] + tail
    seg = [0] * len(s) + [1] * len(blocks[0]) + [2] * len(blocks[1]) + [3] * len(tail)
    n = len(ids)
    allow = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(i + 1):
            allow[i, j] = seg[j] == 0 or seg[i] == 3 or seg[i] == seg[j]
    mask = torch.where(allow, 0.0, torch.finfo(torch.float32).min)[None, None]
    ref = tiny.hf(input_ids=torch.tensor([ids]), attention_mask=mask,
                  position_ids=torch.arange(n).unsqueeze(0)).logits[0, -len(tail):]
    got = _tail_logits(tiny, arms.k0_context(tiny, q, positions="true"))
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


def test_k0_differs_from_b1_mask_only_through_hidden_states(tiny_family, two_workers):
    """After the shift, K0's layer-0 keys/values equal B1-mask's (layer 0 sees only token embeddings + RoPE), but
    deeper layers differ: that difference is the 'local positions' term B1-mask − K0 measures."""
    tiny = tiny_family
    loc = kv.layers(arms.k0_context(tiny, two_workers)[0])
    tru = kv.layers(arms.k0_context(tiny, two_workers, positions="true")[0])
    assert torch.allclose(loc[0][0], tru[0][0], atol=1e-4) and torch.allclose(loc[0][1], tru[0][1], atol=1e-5)
    second = (len(tiny.ids(prompts.prefix(tiny.fmt, two_workers["question"])))
              + len(tiny.ids(prompts.block(two_workers["workers"][0]))))
    assert (loc[-1][0][..., second:, :] - tru[-1][0][..., second:, :]).abs().max() > 1e-5


def test_context_lengths_and_select_one(tiny, two_workers):
    n_s = len(tiny.ids(prompts.prefix(tiny.fmt, two_workers["question"])))
    sizes = [len(tiny.ids(prompts.block(w))) for w in two_workers["workers"]]
    for ctx in (arms.k0_context(tiny, two_workers), arms.k0_context(tiny, two_workers, positions="true"),
                arms.b1_context(tiny, two_workers)):
        assert kv.length(ctx[0]) == ctx[1] == n_s + sum(sizes)
    cache, pos, got = arms.k0_context(tiny, two_workers, visible=[1])
    assert kv.length(cache) == n_s + sizes[1] and got == [sizes[1]]


# ---- K1 positions

def test_k1_adds_m_positions_per_worker(tiny, two_workers):
    n0 = kv.length(arms.k1_context(tiny, two_workers, m=0)[0])
    n3 = kv.length(arms.k1_context(tiny, two_workers, m=3)[0])
    assert n3 - n0 == 3 * len(two_workers["workers"])


def test_k1_second_worker_lands_after_first_workers_latents(tiny, two_workers):
    """Layer-0 keys depend only on the token and its RoPE position, so worker 2's paragraph keys in the stitched K1
    cache must equal those tokens encoded alone at n_s + |worker 1's whole segment| (paragraphs + open + m + close)."""
    m, q = 3, two_workers
    n_s = len(tiny.ids(prompts.prefix(tiny.fmt, q["question"])))
    seg1 = (len(tiny.ids(prompts.block(q["workers"][0]) + prompts.worker_open(tiny.fmt))) + m
            + len(tiny.ids(prompts.worker_close(tiny.fmt))))
    head2 = tiny.ids(prompts.block(q["workers"][1]) + prompts.worker_open(tiny.fmt))
    stitched = kv.layers(arms.k1_context(tiny, q, m=m)[0])[0][0][..., n_s + seg1:n_s + seg1 + len(head2), :]
    alone, _ = tiny.prefill(head2, start_pos=n_s + seg1)
    assert torch.allclose(kv.layers(alone)[0][0], stitched, atol=1e-4)


# ---- Part 0 probe

def test_b1_probe_drops_the_other_gold(two_workers):
    vis = arms.visible_paragraphs(two_workers, [0])
    assert "Cities: The mayor of Velmora is Bram Holt." not in [p for w in vis for p in w]
    assert "People: Ines Farrow's favorite city is Velmora." in vis[0]
    assert "The river the Calder flows through Irvale." in vis[1]        # distractors stay
