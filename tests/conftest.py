"""Fixtures: a tiny random Qwen3 (real architecture: qk-norm, GQA, tied embeddings) with the real Qwen3 tokenizer,
so the KV and prompt code run against the same shapes and template as Qwen3-4B, on CPU, in seconds."""
import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from latent_dispatch.model import Model


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def tiny(tokenizer):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=True, rope_theta=1_000_000.0, initializer_range=0.2)
    return Model(Qwen3ForCausalLM(cfg).eval(), tokenizer, "tiny-qwen3")


def make_question(workers, gold=((0, 0, 0),), id="t:0"):
    return dict(id=id, dataset="toy_d0", source_id="t", question="Who is the mayor of Ines Farrow's favorite city?",
                answers=["Bram Holt"], qtype="hop1", hops=len(gold), placement="AB", crossings=1, seed=0,
                workers=workers, gold=[{"hop": h, "worker": w, "index": i} for h, w, i in gold],
                bridges=["Velmora"], decomposition=[])


@pytest.fixture
def two_workers():
    return make_question(
        [["People: Ines Farrow's favorite city is Velmora.", "Aria Venn was born in Dunmere."],
         ["Cities: The mayor of Velmora is Bram Holt.", "The river the Calder flows through Irvale."]],
        gold=((0, 0, 0), (1, 1, 0)))
