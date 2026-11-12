"""Every model in the matrix: our hand-written chat markup must equal the family's own chat template, every special
token must be one real token, and tokenizing the prompt in segments must give the same ids as one string. If one of
these fails, that model gets a malformed prompt and its outputs are soup.

Downloads only tokenizer files. A gated repo (meta-llama) is skipped until its license is accepted on
huggingface.co; the ungated unsloth mirrors carry the same tokenizer and template, so the Llama format is still
checked."""
import os
from functools import lru_cache

import pytest
from transformers import GenerationConfig

from latent_dispatch import arms, prompts
from latent_dispatch.model import MODELS, Model, check_format, load_tokenizer, special_id

from .conftest import make_question, tiny_model

MIRRORS = {"unsloth/Llama-3.1-8B-Instruct": "llama3", "unsloth/Llama-3.2-3B-Instruct": "llama3"}
REPOS = {**MODELS, **MIRRORS}
TEMPLATE_KW = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="26 Jul 2024")

Q = "Who is the mayor of Ines Farrow's favorite city?"
PARAS = [["People: Ines Farrow's favorite city is Velmora. Aria Venn was born in Dunmere.",
          "Kastrel (pop. 12,400) sits on the river Teviot; it's 3 km from Orinth."],
         ["Cities: The mayor of Velmora is Bram Holt.", "\"Quillford\" -- founded 1873 -- is in Ny Havn's district."]]


def unavailable(repo, e):
    """Skip a gated repo (license not accepted yet) or when offline; anything else, like a typo in MODELS, fails."""
    if "gated" in str(e).lower() or os.environ.get("HF_HUB_OFFLINE"):
        pytest.skip(f"{repo}: not available ({type(e).__name__}); accept its license on huggingface.co")
    raise e


@lru_cache(maxsize=None)
def _tok(repo):
    try:
        return load_tokenizer(repo, REPOS[repo])
    except OSError as e:
        unavailable(repo, e)


params = pytest.mark.parametrize("repo", list(REPOS))


@params
def test_b1_prompt_matches_the_chat_template(repo):
    tok, fmt = _tok(repo), prompts.FORMATS[REPOS[repo]]
    paras = [p for w in PARAS for p in w]
    user = f"Target Question: {Q}\n\n{prompts.INTRO}\n\n" + prompts.block(paras) + prompts.FMT
    msgs = [{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": user}]
    ours = prompts.prefix(fmt, Q) + prompts.block(paras) + prompts.tail(fmt, "off")
    assert ours == tok.apply_chat_template(msgs, **TEMPLATE_KW)
    if fmt.thinking:
        on = tok.apply_chat_template(msgs, **{**TEMPLATE_KW, "enable_thinking": True})
        assert prompts.prefix(fmt, Q) + prompts.block(paras) + prompts.tail(fmt, "forced") == on + "<think>\n"
    else:
        with pytest.raises(ValueError):
            prompts.tail(fmt, "forced")


@params
def test_report_prompt_matches_the_chat_template(repo):
    tok, fmt = _tok(repo), prompts.FORMATS[REPOS[repo]]
    user = f"Target Question: {Q}\n\nYour documents:\n" + "\n\n".join(PARAS[0]) + f"\n\n{prompts.REPORT_ASK}"
    msgs = [{"role": "system", "content": prompts.REPORT_SYSTEM}, {"role": "user", "content": user}]
    assert prompts.report_prompt(fmt, Q, PARAS[0]) == tok.apply_chat_template(msgs, **TEMPLATE_KW)


@params
def test_k1_turns_match_the_chat_template(repo):
    """K1 gives each worker an assistant turn, then reopens the user turn with fmt.reopen_user. The whole
    multi-turn string must match the template, except on Mistral: its template moves the system prompt to the LAST
    user turn, while K1 keeps it in S so that every worker shares the prefix (a known, documented deviation)."""
    tok, fmt = _tok(repo), prompts.FORMATS[REPOS[repo]]
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}]
    ours = (prompts.chat_open(fmt, "S") + "U1" + fmt.user_end + "A1" + fmt.reopen_user + "U2" + fmt.user_end
            + prompts.think_off(fmt))
    theirs = tok.apply_chat_template(msgs, **TEMPLATE_KW)
    if fmt.name == "mistral":
        # ours = the template's own conversation without a system turn, with S at the top of the FIRST user turn
        assert theirs == "<s>[INST]U1[/INST]A1</s>[INST]S\n\nU2[/INST]"
        assert ours == tok.apply_chat_template(msgs[1:], **TEMPLATE_KW).replace("[INST]U1", "[INST]S\n\nU1", 1)
        pytest.xfail("Mistral puts the system prompt in the last user turn; K1 keeps it in S (see prompts.py)")
    assert ours == theirs


@params
def test_stop_tokens_cover_the_models_own_eos(repo):
    """Generation must stop on every token the model's generation_config treats as end-of-sequence."""
    tok, fmt = _tok(repo), prompts.FORMATS[REPOS[repo]]
    try:
        eos = GenerationConfig.from_pretrained(repo).eos_token_id
    except OSError as e:
        unavailable(repo, e)
    eos = {eos} if isinstance(eos, int) else set(eos)
    assert eos <= {special_id(tok, t) for t in fmt.stop}


@params
def test_markup_is_special_tokens(repo):
    """Every piece of chat markup must tokenize to special tokens (a typo like "<|eot_id>" would be split into
    plain text), and every stop token must be a single token."""
    tok, fmt = _tok(repo), prompts.FORMATS[REPOS[repo]]
    for stop in fmt.stop:
        special_id(tok, stop)
    markup = prompts.chat_open(fmt, "") + fmt.user_end + fmt.reopen_user
    left = tok.decode(tok(markup, add_special_tokens=False).input_ids, skip_special_tokens=True)
    for piece in ("<|", "|>", "[INST]", "[/INST]", "<s>", "</s>"):
        assert piece not in left, f"{piece!r} survived as text: {left!r}"


@params
def test_segments_tokenize_like_one_string(repo):
    """B1 and K0 tokenize S, each block and the tail separately (T1: S and each report block); no BPE merge across
    a boundary may change the ids, or B1 wouldn't be the same token sequence the model was trained on."""
    tok, fmt = _tok(repo), prompts.FORMATS[REPOS[repo]]

    def ids(text):
        return tok(text, add_special_tokens=False).input_ids

    for segments in ([prompts.prefix(fmt, Q)] + [prompts.block(w) for w in PARAS] + [prompts.tail(fmt, "off")],
                     [prompts.prefix(fmt, Q), prompts.report_block(0, "Velmora's mayor is Bram Holt."),
                      prompts.report_block(1, "(nothing relevant)"), prompts.tail(fmt, "off")]):
        assert ids("".join(segments)) == [t for s in segments for t in ids(s)]


ARCH = {"qwen3": "qwen3", "llama3": "llama3", "mistral": "mistral", "phi4": "phi3"}     # format -> tiny architecture


@pytest.mark.parametrize("repo", ["Qwen/Qwen3-0.6B", "unsloth/Llama-3.1-8B-Instruct",
                                  "mistralai/Mistral-Nemo-Instruct-2407", "microsoft/phi-4"])
def test_every_arm_runs_with_each_familys_tokenizer(repo, two_workers):
    """Every arm end to end on a tiny random model with the family's real tokenizer and chat format (random weights,
    so only the mechanics are checked: markup, stop and think tokens, K0 == B1 with one worker)."""
    fmt = REPOS[repo]
    m = tiny_model(ARCH[fmt], _tok(repo), fmt)
    short = {"max_answer_tokens": 4}
    for arm, options in ((arms.b0, {}), (arms.b1, {}), (arms.b1_mask, {}), (arms.k0, {}), (arms.k0, {"merge": "shared"}),
                         (arms.k1, {"m": 2}), (arms.t1, {"report_tokens": 4}), (arms.scan, {"rounds": 2})):
        out = arm(m, two_workers, **options, **short)
        assert isinstance(out["text"], str) and out["gen_tokens"] <= 4 + out["think_tokens"]
    if m.fmt.thinking:
        assert arms.b1(m, two_workers, thinking="forced", think_budget=3, **short)["think_tokens"] <= 3
    else:
        with pytest.raises(ValueError):
            arms.b1(m, two_workers, thinking="forced", **short)
    one = make_question([[p for w in two_workers["workers"] for p in w]])
    assert arms.b1(m, one, **short)["text"] == arms.k0(m, one, **short)["text"]


@params
def test_check_format_accepts_the_right_format_and_refuses_the_others(repo):
    tok, fmt = _tok(repo), REPOS[repo]
    check_format(tok, prompts.FORMATS[fmt])
    for other in prompts.FORMATS.values():
        if other.name != fmt:
            with pytest.raises(ValueError):
                check_format(tok, other)


def test_load_refuses_a_wrong_format_before_the_weights_download(monkeypatch):
    import transformers

    def no_download(*args, **kwargs):
        raise AssertionError("weights were requested")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", no_download)
    for name, fmt, why in (("Qwen/Qwen3-0.6B", "phi4", "chat format, not"),             # contradicts MODELS
                           ("microsoft/phi-4", "qwen3", "chat format, not"),
                           ("unsloth/Llama-3.2-3B-Instruct", "qwen3", "single token"),   # not in MODELS: tokenizer check
                           ("Qwen/Qwen3-0.6B", "chatml", "unknown chat format")):
        if name.startswith("unsloth"):
            _tok(name)                        # skip cleanly if offline
        with pytest.raises(ValueError, match=why):
            Model.load(name, fmt=fmt)


def test_model_refuses_a_format_the_tokenizer_lacks(tokenizer):
    with pytest.raises(ValueError, match="plain text"):
        tiny_model("phi3", tokenizer, "phi4")


def test_decode_keeps_the_text_as_generated():
    """Llama 3's tokenizer would 'clean up' spaces on decode; saved outputs must be what the model generated."""
    m = tiny_model("llama3", _tok("unsloth/Llama-3.1-8B-Instruct"), "llama3")
    for text in ("\\boxed{Washington , D.C.}", "Ines Farrow 's city", "a ( b ) c ."):
        assert m.decode(m.ids(text)) == text
