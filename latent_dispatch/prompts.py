"""One prompt layout for every arm, as text segments that get tokenized separately.

    S      system turn + user header + question + intro                (shared prefix, prefilled once)
    block  one worker's paragraphs (or its report), continuing the user turn, ending in a blank line
    tail   answer-format line + end of user turn + assistant header + thinking prefix

B1 = S + every block + tail as one sequence. K0 = S prefilled once, each block encoded on top of a copy of S,
stitched, then the tail. The token ids are the concatenation of separately tokenized segments in both cases, so B1
and K0 see identical tokens; only the encoding differs.

Qwen3 ChatML is written out by hand so every segment boundary is explicit. tests/test_gates.py checks that the
B1 string equals the tokenizer's own chat template.

Changing any text here changes what every arm generates: delete the saved results before rerunning.
"""
from typing import Literal

Thinking = Literal["off", "forced"]

SYSTEM = "You are a helpful assistant."
INTRO = ("Below is material gathered for this question. It may be partial or contain irrelevant content -- "
         "use what helps and ignore the rest.")
FMT = "Give the shortest exact answer (a name, entity, number, or yes/no) inside \\boxed{YOUR_FINAL_ANSWER}."

USER_END = "<|im_end|>\n<|im_start|>assistant\n"
THINK_OFF = "<think>\n\n</think>\n\n"   # what the Qwen3 template adds with enable_thinking=False
THINK_ON = "<think>\n"                  # prefilled opener: the model has to reason before answering


def chat_open(system: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"


def prefix(question: str) -> str:
    return chat_open(SYSTEM) + f"Target Question: {question}\n\n{INTRO}\n\n"


def block(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs) + "\n\n"


def tail(thinking: Thinking, pre: str = "") -> str:
    return pre + FMT + USER_END + (THINK_ON if thinking == "forced" else THINK_OFF)


# ---- K1: each worker gets its own assistant turn to hold its latent thoughts, then the user turn reopens
WORKER_OPEN = "Think about which of these facts matter for the question." + USER_END + THINK_ON
WORKER_CLOSE = "\n</think>\n\n<|im_end|>\n<|im_start|>user\n"

# ---- T1: worker report prompt (wording from the Paper A text-report baseline, text_report_run.py)
REPORT_SYSTEM = "You are a worker on a team. You see only part of the material; other workers see the rest."
REPORT_ASK = ("Write a short report for the judge: every fact in YOUR documents that could help answer the question, "
              "including facts about entities whose relevance you cannot confirm yet. Do not answer the question.")


def report_prompt(question: str, paragraphs: list[str]) -> str:
    docs = "\n\n".join(paragraphs)
    return (chat_open(REPORT_SYSTEM) + f"Target Question: {question}\n\nYour documents:\n{docs}\n\n{REPORT_ASK}"
            + USER_END + THINK_OFF)


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


def note_tail(notes: list[str], style: str) -> str:
    return notes_block(notes) + "\n\n" + NOTE_ASK[style] + USER_END + THINK_OFF
