"""Model wrapper: which models we run, explicit-position forwards over a KV cache, LatentMAS-style latent steps,
greedy decoding.

Every forward passes position ids AND cache positions explicitly. Cache positions say where the new KV entries go
in the cache (and drive the causal mask); position ids say which RoPE phase they get. Keeping them separate is what
lets a worker encode at its "true" position in the stitched sequence while sitting right after the shared prefix
in its own cache.

The latent-step code follows ModelWrapper.generate_latent_batch_hidden_state and _apply_latent_realignment in
JoshBong/LatentMAS models.py (fork of Gen-Verse/LatentMAS, Apache-2.0):
https://github.com/JoshBong/LatentMAS/blob/main/models.py
"""
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PretrainedConfig

from . import kv, prompts

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# The model matrix: HF repo -> chat format (prompts.FORMATS). Main family = Qwen3 at three sizes; second family =
# Llama 3; Mistral-Nemo / phi-4 = optional third. These are the backbones the 2026 latent-MAS papers report on.
# To add a model: its architecture must pass check_architecture, its family needs a ChatFormat, and
# tests/test_formats.py + `MODEL=<repo> pytest -m slow` must pass.
MODELS = {
    "Qwen/Qwen3-0.6B": "qwen3",        # tests only
    "Qwen/Qwen3-1.7B": "qwen3",
    "Qwen/Qwen3-4B": "qwen3",
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3-14B": "qwen3",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3",     # gated: accept the license on huggingface.co first
    "meta-llama/Llama-3.1-8B-Instruct": "llama3",
    "mistralai/Mistral-Nemo-Instruct-2407": "mistral",
    "microsoft/phi-4": "phi4",
}
ARCHITECTURES = {"qwen3", "llama", "mistral", "phi3"}     # model_types whose layers and RoPE the KV code is tested on


def check_architecture(cfg: dict) -> None:
    """Raise unless every layer keeps a per-token KV cache with plain RoPE: the only setup where worker caches can
    be stitched (concatenated) and shifted to new positions. `cfg` is the model's config.json as a dict."""
    cfg = {**cfg, **(cfg.get("text_config") or {})}      # multimodal wrappers (e.g. Qwen3.5) nest the LM config
    problems = []
    if cfg.get("model_type") not in ARCHITECTURES:
        problems.append(f"model_type {cfg.get('model_type')!r} is not one the KV code is tested on {sorted(ARCHITECTURES)}")
    kinds = cfg.get("layer_types") or []
    other = [k for k in kinds if k != "full_attention"]
    if other:
        problems.append(f"{len(other)} of {len(kinds)} layers are {sorted(set(other))}; only full-attention layers keep "
                        "a per-token KV cache (a linear-attention layer's state can't be concatenated or shifted)")
    if cfg.get("sliding_window") and cfg.get("use_sliding_window", True):
        problems.append(f"sliding-window attention ({cfg['sliding_window']} tokens): later blocks can't see earlier ones")
    rope = cfg.get("rope_scaling") or {}
    kind = rope.get("rope_type") or rope.get("type")
    if kind in ("dynamic", "longrope", "su") or (cfg.get("model_type") == "phi3" and rope):   # Phi-3 maps su/yarn to longrope
        problems.append(f"{kind} RoPE: frequencies depend on sequence length, so a cached key can't be shifted exactly")
    if cfg.get("partial_rotary_factor", 1.0) not in (None, 1, 1.0):
        problems.append("partial rotary embedding: kv.shift rotates the whole head")
    if cfg.get("no_rope_layers") and not all(cfg["no_rope_layers"]):
        problems.append("some layers have no RoPE: kv.shift would rotate their keys anyway")
    if problems:
        raise ValueError("can't run the KV arms on this model: " + "; ".join(problems))


@dataclass
class Gen:
    text: str
    tokens: int          # every generated token, thinking included
    think_tokens: int
    answer_tokens: int
    cut: bool = False         # stopped by a token cap (max_new or hard_cap), not by the model's own end token
    think_cut: bool = False   # thinking ran to think_budget and "</think>" was forced


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(name: str, fmt: str):
    # transformers warns that Mistral tokenizers ship a broken pre-tokenizer regex unless this is set. For
    # Mistral-Nemo it changes nothing (its ids match mistral_common's); it's set so newer Mistral models are right too.
    return AutoTokenizer.from_pretrained(name, **({"fix_mistral_regex": True} if fmt == "mistral" else {}))


def special_id(tokenizer, token: str) -> int:
    """The id of a special token, refusing anything that isn't a single token this tokenizer knows."""
    ids = tokenizer(token, add_special_tokens=False).input_ids
    if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != token:
        raise ValueError(f"{token!r} is not a single token for this tokenizer (got {ids}): wrong chat format?")
    return ids[0]


def check_format(tokenizer, fmt: prompts.ChatFormat) -> None:
    """Raise unless this tokenizer has every special token the chat format writes (a wrong format would otherwise
    turn markup like "<|im_sep|>" into plain text and the model would read a malformed prompt)."""
    for token in fmt.stop + (("<think>", "</think>") if fmt.thinking else ()):
        special_id(tokenizer, token)
    markup = prompts.chat_open(fmt, "") + fmt.user_end + fmt.reopen_user
    left = tokenizer.decode(tokenizer(markup, add_special_tokens=False).input_ids, skip_special_tokens=True)
    if any(piece in left for piece in ("<|", "|>", "[INST]", "[/INST]", "<s>", "</s>")):
        raise ValueError(f"the {fmt.name} chat format is plain text for this tokenizer: {left!r}")


class Model:
    def __init__(self, hf, tokenizer, name: str = "custom", fmt: str = "qwen3"):
        check_architecture(hf.config.to_dict())
        self.hf = hf.eval()
        self.tok = tokenizer
        self.name = name
        self.fmt = prompts.FORMATS[fmt]
        check_format(tokenizer, self.fmt)
        self.device = next(hf.parameters()).device
        self.dtype_name = str(next(hf.parameters()).dtype).removeprefix("torch.")
        self.inv_freq = kv.rope_inv_freq(hf)
        cfg = hf.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        if 2 * self.inv_freq.numel() != head_dim:
            raise ValueError(f"RoPE covers {2 * self.inv_freq.numel()} of {head_dim} head dims: kv.shift assumes all")
        self.eos_ids = {special_id(tokenizer, t) for t in self.fmt.stop}
        self.think_end_id = special_id(tokenizer, "</think>") if self.fmt.thinking else None
        self._realign = None   # (W_a, target_norm), built lazily

    @classmethod
    def load(cls, name: str, device: str | None = None, dtype: str | torch.dtype = "bfloat16",
             fmt: str | None = None) -> "Model":
        """A model from the matrix (MODELS), or any HF repo with fmt= given. The architecture is checked from
        config.json before any weights download. dtype is part of the results folder name, so bf16 and fp32 runs
        never mix."""
        fmt = fmt or MODELS.get(name)
        if fmt is None:
            raise ValueError(f"{name} is not in model.MODELS; add it there (see the comment above MODELS) or pass fmt=")
        if fmt not in prompts.FORMATS:
            raise ValueError(f"unknown chat format {fmt!r}; known: {sorted(prompts.FORMATS)}")
        if name in MODELS and fmt != MODELS[name]:
            raise ValueError(f"{name} uses the {MODELS[name]} chat format, not {fmt}")
        check_architecture(PretrainedConfig.get_config_dict(name)[0])
        tok = load_tokenizer(name, fmt)
        check_format(tok, prompts.FORMATS[fmt])             # all before the weights download
        device = device or _auto_device()
        dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, low_cpu_mem_usage=True).to(device)
        return cls(hf, tok, name, fmt)

    # ---------------------------------------------------------------- tokens
    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def decode(self, ids: list[int]) -> str:
        # Llama 3's tokenizer would otherwise rewrite the text ("Washington , D.C." -> "Washington, D.C.")
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, ids: list[int] | None, cache=None, start_pos: int | None = None, embeds=None,
                hidden: bool = False, attention_mask=None):
        """Run `ids` (or input `embeds` [1, L, d]) on top of `cache`, mutating it. RoPE positions start at
        `start_pos` (default: the cache length, i.e. a normal continuation)."""
        if cache is None:
            cache = DynamicCache()
        past = kv.length(cache)
        n = len(ids) if ids is not None else embeds.shape[1]
        start = past if start_pos is None else start_pos
        kw = dict(
            position_ids=torch.arange(start, start + n, device=self.device).unsqueeze(0),
            cache_position=torch.arange(past, past + n, device=self.device),
            past_key_values=cache, use_cache=True, output_hidden_states=hidden,
        )
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask
        if ids is not None:
            return self.hf(input_ids=torch.tensor([ids], device=self.device), **kw)
        return self.hf(inputs_embeds=embeds, **kw)

    def prefill(self, ids: list[int], cache=None, start_pos: int | None = None, attention_mask=None):
        """-> (cache, last hidden state [1, d])."""
        if cache is None:
            cache = DynamicCache()
        out = self.forward(ids, cache, start_pos, hidden=True, attention_mask=attention_mask)
        return out.past_key_values, out.hidden_states[-1][:, -1, :]

    # ----------------------------------------------------------- latent steps
    def _latent_embed(self, h: torch.Tensor, realign: bool) -> torch.Tensor:
        """LatentMAS input for the next latent step: last hidden state, optionally mapped by W_a (realign), then
        rescaled to the mean norm of the input embeddings. With realign off the map is the identity."""
        if self._realign is None or self._realign[2] != realign:
            w_in = self.hf.get_input_embeddings().weight.detach().float()
            if realign:
                w_out = self.hf.get_output_embeddings().weight.detach().float()
                gram = w_out.T @ w_out + 1e-5 * torch.eye(w_out.shape[1], device=w_out.device)
                w_a = torch.linalg.solve(gram, w_out.T @ w_in)
            else:
                w_a = None
            self._realign = (w_a, w_in.norm(dim=1).mean(), realign)
        w_a, target, _ = self._realign
        x = h.float() if w_a is None else h.float() @ w_a
        x = x * (target / x.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return x.to(h.dtype).unsqueeze(1)

    @torch.no_grad()
    def latent_steps(self, cache, last_hidden: torch.Tensor, m: int, start_pos: int, realign: bool = False):
        """m latent thoughts at positions start_pos..start_pos+m-1. -> (cache, last hidden)."""
        for i in range(m):
            out = self.forward(None, cache, start_pos + i, embeds=self._latent_embed(last_hidden, realign), hidden=True)
            cache, last_hidden = out.past_key_values, out.hidden_states[-1][:, -1, :]
        return cache, last_hidden

    # ---------------------------------------------------------------- decode
    @torch.no_grad()
    def greedy(self, tail: list[int], cache=None, start_pos: int | None = None, max_new: int = 64,
               think_budget: int | None = None, hard_cap: int = 8192) -> Gen:
        """Feed `tail` on top of `cache`, then decode greedily until EOS.

        think_budget=None: the tail already closed thinking (or there is none); decode up to max_new tokens.
        think_budget=k: the tail ends inside an open <think>; allow k thinking tokens, then force "</think>", then
        up to max_new answer tokens.
        """
        if cache is None:
            cache = DynamicCache()
        pos = kv.length(cache) if start_pos is None else start_pos
        out = self.forward(tail, cache, pos)
        pos += len(tail)
        gen: list[int] = []
        think = answer = 0
        thinking = think_budget is not None
        cut = think_cut = False
        force_close = self.ids(prompts.THINK_CLOSE)
        while len(gen) < hard_cap:
            nxt = int(out.logits[0, -1].argmax())
            if thinking and think >= think_budget and nxt != self.think_end_id:
                feed, thinking, think_cut = force_close, False, True
            elif nxt in self.eos_ids:
                break
            elif thinking:
                feed = [nxt]
                if nxt == self.think_end_id:
                    thinking = False
                else:
                    think += 1
            else:
                if answer >= max_new:
                    cut = True
                    break
                feed = [nxt]
                answer += 1
            gen += feed
            out = self.forward(feed, out.past_key_values, pos)
            pos += len(feed)
        else:
            cut = True                                    # hit hard_cap
        return Gen(self.decode(gen), len(gen), think, answer, cut, think_cut)
