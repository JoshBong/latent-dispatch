"""check_architecture lets in exactly the models whose worker caches can be stitched and shifted. Reads only
config.json files (a gated repo is checked through its ungated mirror)."""
import os

import pytest
from transformers import MistralConfig, PretrainedConfig, Qwen3Config

from latent_dispatch.model import MODELS, Model, check_architecture

MIRROR = {"meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
          "meta-llama/Llama-3.1-8B-Instruct": "unsloth/Llama-3.1-8B-Instruct"}


def _config(repo):
    try:
        return PretrainedConfig.get_config_dict(MIRROR.get(repo, repo))[0]
    except OSError as e:          # offline; a typo in MODELS (or a vanished repo) fails instead
        if os.environ.get("HF_HUB_OFFLINE"):
            pytest.skip(f"{repo}: config.json not available offline")
        raise


@pytest.mark.parametrize("repo", list(MODELS))
def test_every_model_in_the_matrix_passes(repo):
    check_architecture(_config(repo))


def test_qwen3_5_is_refused_for_its_linear_attention_layers():
    """Qwen3.5: 24 of 32 layers are Gated DeltaNet, whose fixed-size state can't be stitched across workers."""
    with pytest.raises(ValueError, match="linear_attention"):
        check_architecture(_config("Qwen/Qwen3.5-4B"))


@pytest.mark.parametrize("cfg,why", [
    (MistralConfig().to_dict(), "sliding-window"),                          # Mistral 7B's default 4096 window
    (Qwen3Config(use_sliding_window=True, sliding_window=1024, max_window_layers=0).to_dict(), "sliding"),
    ({"model_type": "gemma3", "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}, "gemma3"),
    ({"model_type": "llama", "rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}, "dynamic"),
    ({"model_type": "phi3", "rope_scaling": {"type": "longrope"}}, "longrope"),
    ({"model_type": "phi3", "rope_scaling": {"type": "su"}}, "su"),               # Phi-3-*-128k ship "su"
    ({"model_type": "qwen3", "text_config": None, "partial_rotary_factor": 0.5}, "partial rotary"),
    ({"model_type": "phi3", "partial_rotary_factor": 0.75}, "partial rotary"),
    ({"model_type": "llama", "no_rope_layers": [1, 1, 1, 0]}, "no RoPE"),
    ({"model_type": "qwen3_next"}, "qwen3_next"),
])
def test_refusals(cfg, why):
    with pytest.raises(ValueError, match=why):
        check_architecture(cfg)


def test_load_refuses_a_model_outside_the_matrix_before_downloading():
    with pytest.raises(ValueError, match="not in model.MODELS"):
        Model.load("someone/some-model")
