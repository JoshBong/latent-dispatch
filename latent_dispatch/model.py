"""Model wrapper: explicit-position forwards over a KV cache, LatentMAS-style latent steps, greedy decoding.

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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from . import kv

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class Gen:
    text: str
    tokens: int          # every generated token, thinking included
    think_tokens: int
    answer_tokens: int


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Model:
    def __init__(self, hf, tokenizer, name: str = "custom"):
        self.hf = hf.eval()
        self.tok = tokenizer
        self.name = name
        self.device = next(hf.parameters()).device
        self.dtype_name = str(next(hf.parameters()).dtype).removeprefix("torch.")
        self.inv_freq = kv.rope_inv_freq(hf)
        conv = tokenizer.convert_tokens_to_ids
        self.eos_ids = {i for i in (conv("<|im_end|>"), conv("<|endoftext|>"), tokenizer.eos_token_id) if i is not None}
        self.think_end_id = conv("</think>")
        self._realign = None   # (W_a, target_norm), built lazily

    @classmethod
    def load(cls, name: str, device: str | None = None, dtype: str | torch.dtype = "bfloat16") -> "Model":
        """dtype is part of the results folder name, so bf16 and fp32 runs never mix."""
        device = device or _auto_device()
        dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
        tok = AutoTokenizer.from_pretrained(name)
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, low_cpu_mem_usage=True).to(device)
        return cls(hf, tok, name)

    # ---------------------------------------------------------------- tokens
    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, ids: list[int] | None, cache=None, start_pos: int | None = None, embeds=None,
                hidden: bool = False):
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
        if ids is not None:
            return self.hf(input_ids=torch.tensor([ids], device=self.device), **kw)
        return self.hf(inputs_embeds=embeds, **kw)

    def prefill(self, ids: list[int], cache=None, start_pos: int | None = None):
        """-> (cache, last hidden state [1, d])."""
        if cache is None:
            cache = DynamicCache()
        out = self.forward(ids, cache, start_pos, hidden=True)
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
        force_close = self.ids("\n</think>\n\n")
        while len(gen) < hard_cap:
            nxt = int(out.logits[0, -1].argmax())
            if thinking and think >= think_budget and nxt != self.think_end_id:
                feed, thinking = force_close, False
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
                    break
                feed = [nxt]
                answer += 1
            gen += feed
            out = self.forward(feed, out.past_key_values, pos)
            pos += len(feed)
        return Gen(self.decode(gen), len(gen), think, answer)
