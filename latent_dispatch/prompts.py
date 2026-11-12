"""One prompt layout for every arm, as text segments that get tokenized separately.

    S      system turn + user header + question + intro                (shared prefix, prefilled once)
    block  one worker's paragraphs (or its report), continuing the user turn, ending in a blank line
    tail   answer-format line + end of user turn + assistant header + thinking prefix

B1 = S + every block + tail as one sequence. K0 = S prefilled once, each block encoded on top of a copy of S,
stitched, then the tail. The token ids are the concatenation of separately tokenized segments in both cases, so B1
and K0 see identical tokens; only the encoding differs.

Each model family's chat markup is written out by hand in FORMATS, so every segment boundary is explicit.
tests/test_formats.py checks, family by family, that the B1 and T1 strings equal the tokenizer's own chat template.

Changing any text here changes what every arm generates: delete the saved results before rerunning.
"""
from dataclasses import dataclass
from typing import Literal

Thinking = Literal["off", "forced"]


@dataclass(frozen=True)
class ChatFormat:
    """One family's chat markup. `open` runs from the start of the conversation to where the first user message's
    text begins, with "{system}" in place of the system prompt."""
    name: str
    open: str
    user_end: str        # end of a user turn + the assistant header
    reopen_user: str     # end of an assistant turn + a new user header (K1's worker turn closes with this)
    stop: tuple          # tokens that end generation
    thinking: bool       # Qwen3-style <think> blocks (needed for thinking="forced")


FORMATS = {
    "qwen3": ChatFormat(
        name="qwen3",
        open="<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n",
        user_end="<|im_end|>\n<|im_start|>assistant\n",
        reopen_user="<|im_end|>\n<|im_start|>user\n",
        stop=("<|im_end|>", "<|endoftext|>"),
        thinking=True),
    # The Llama 3.1/3.2 template writes the date into every system prompt; 3.2's uses today's date unless told
    # otherwise. Pinned to the 3.1 default so results don't depend on the day they ran.
    "llama3": ChatFormat(
        name="llama3",
        open=("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
              "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
              "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"),
        user_end="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        reopen_user="<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        stop=("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"),
        thinking=False),
    # Mistral has no system turn: its template puts the system prompt at the top of the (last) user message.
    "mistral": ChatFormat(
        name="mistral",
        open="<s>[INST]{system}\n\n",
        user_end="[/INST]",
        reopen_user="</s>[INST]",
        stop=("</s>",),
        thinking=False),
    "phi4": ChatFormat(
        name="phi4",
        open="<|im_start|>system<|im_sep|>{system}<|im_end|><|im_start|>user<|im_sep|>",
        user_end="<|im_end|><|im_start|>assistant<|im_sep|>",
        reopen_user="<|im_end|><|im_start|>user<|im_sep|>",
        stop=("<|im_end|>", "<|endoftext|>"),
        thinking=False),
}

SYSTEM = "You are a helpful assistant."
INTRO = ("Below is material gathered for this question. It may be partial or contain irrelevant content -- "
         "use what helps and ignore the rest.")
FMT = "Give the shortest exact answer (a name, entity, number, or yes/no) inside \\boxed{YOUR_FINAL_ANSWER}."

THINK_OFF = "<think>\n\n</think>\n\n"   # what the Qwen3 template adds with enable_thinking=False
THINK_ON = "<think>\n"                  # prefilled opener: the model has to reason before answering
THINK_CLOSE = "\n</think>\n\n"


def think_off(fmt: ChatFormat) -> str:
    return THINK_OFF if fmt.thinking else ""


def chat_open(fmt: ChatFormat, system: str) -> str:
    return fmt.open.replace("{system}", system)


def prefix(fmt: ChatFormat, question: str) -> str:
    return chat_open(fmt, SYSTEM) + f"Target Question: {question}\n\n{INTRO}\n\n"


def block(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs) + "\n\n"


def tail(fmt: ChatFormat, thinking: Thinking, pre: str = "") -> str:
    if thinking == "forced" and not fmt.thinking:
        raise ValueError(f"the {fmt.name} format has no thinking mode, so thinking='forced' can't run on it")
    return pre + FMT + fmt.user_end + (THINK_ON if thinking == "forced" else think_off(fmt))


# ---- K1: each worker gets its own assistant turn to hold its latent thoughts, then the user turn reopens.
# On Qwen3 the latent thoughts sit inside an open <think>, as in LatentMAS (Qwen3's template would strip a past
# turn's thinking; keeping it is the point of K1). Mistral's template moves the system prompt to the last user turn;
# K1 keeps it in S so every worker shares the prefix. Both deviations are deliberate (tests/test_formats.py).
WORKER_ASK = "Think about which of these facts matter for the question."


def worker_open(fmt: ChatFormat) -> str:
    return WORKER_ASK + fmt.user_end + (THINK_ON if fmt.thinking else "")


def worker_close(fmt: ChatFormat) -> str:
    return (THINK_CLOSE if fmt.thinking else "") + fmt.reopen_user


# ---- T1: worker report prompt (wording from the Paper A text-report baseline, text_report_run.py)
REPORT_SYSTEM = "You are a worker on a team. You see only part of the material; other workers see the rest."
REPORT_ASK = ("Write a short report for the judge: every fact in YOUR documents that could help answer the question, "
              "including facts about entities whose relevance you cannot confirm yet. Do not answer the question.")


def report_prompt(fmt: ChatFormat, question: str, paragraphs: list[str]) -> str:
    docs = "\n\n".join(paragraphs)
    return (chat_open(fmt, REPORT_SYSTEM) + f"Target Question: {question}\n\nYour documents:\n{docs}\n\n{REPORT_ASK}"
            + fmt.user_end + think_off(fmt))


def report_block(worker: int, report: str) -> str:
    return f"Report from worker {worker + 1}:\n{report}\n\n"


# ---- scan: the orchestrator reads one worker block at a time and carries notes (Paper A scan_judge.py)
NOTE_ASK = {
    "line": ("From the material you just read, write ONE short line with the facts needed for the question. "
             "If the notes name an entity, state what this material says about that entity. Do not answer yet."),
    "scratch": ("Update your scratchpad from the material you just read. Fill in exactly these four lines:\n"
                "Question needs: <what the question is ultimately asking for>\n"
                "Known so far: <facts from earlier notes that are established>\n"
                "This doc says: <every fact here that COULD matter, including facts about entities whose relevance "
                "is not known yet>\n"
                "Keep for later: <candidates to keep open until a later document decides between them>\n"
                "Do not answer yet."),
}


def notes_block(notes: list[str]) -> str:
    if not notes:
        return "Notes carried from material read earlier: (none yet)"
    return "Notes carried from material read earlier:\n" + "\n".join(f"- {n}" for n in notes)


def note_tail(fmt: ChatFormat, notes: list[str], style: str) -> str:
    return notes_block(notes) + "\n\n" + NOTE_ASK[style] + fmt.user_end + think_off(fmt)
