"""Fixtures: tiny random models with each family's real architecture (Qwen3: qk-norm, GQA, tied embeddings; Llama 3:
its scaled RoPE; Mistral; Phi-3/phi-4: fused qkv), all with the real Qwen3 tokenizer and chat format, so the KV and
prompt code run against the same shapes as the real models, on CPU, in seconds. Chat formats are tested separately
against each family's real tokenizer (test_formats.py)."""
import pytest
import torch
from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
                          Phi3Config, Phi3ForCausalLM, Qwen3Config, Qwen3ForCausalLM)

from latent_dispatch.model import Model

SMALL = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
             max_position_embeddings=4096, initializer_range=0.2)
FAMILIES = {
    "qwen3": (Qwen3ForCausalLM, Qwen3Config, dict(head_dim=16, tie_word_embeddings=True, rope_theta=1_000_000.0)),
    "llama3": (LlamaForCausalLM, LlamaConfig, dict(rope_theta=500_000.0, rope_scaling={
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 512})),
    "mistral": (MistralForCausalLM, MistralConfig, dict(head_dim=16, sliding_window=None, rope_theta=1_000_000.0)),
    "phi3": (Phi3ForCausalLM, Phi3Config, dict(rope_theta=250_000.0, pad_token_id=0)),
}


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def tiny_model(family: str, tokenizer, fmt: str = "qwen3") -> Model:
    torch.manual_seed(0)
    cls, config, extra = FAMILIES[family]
    hf = cls(config(vocab_size=len(tokenizer), **SMALL, **extra)).eval()
    return Model(hf, tokenizer, f"tiny-{family}", fmt=fmt)


@pytest.fixture(scope="session")
def tiny(tokenizer):
    return tiny_model("qwen3", tokenizer)


@pytest.fixture(scope="session", params=list(FAMILIES))
def tiny_family(request, tokenizer):
    """The same test on a tiny model of every family's architecture."""
    return tiny_model(request.param, tokenizer)


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
