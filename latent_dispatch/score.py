# Answer extraction and SQuAD normalization ported from JoshBong/LatentMAS utils.py (fork of Gen-Verse/LatentMAS,
# Apache-2.0): https://github.com/JoshBong/LatentMAS/blob/main/utils.py
"""The one scorer every arm goes through: extract the short answer, normalize, EM and F1 against every alias.

Changes from the fork: number words normalize to digits ("five" == "5"; the Paper A runs scored that wrong), gold
is a list of aliases (EM if any alias matches, F1 = best alias), \\boxed{} is brace-matched with LaTeX text
wrappers removed, and sentence cuts skip abbreviations.

results.load_results re-scores every saved row from its raw text with this module, so changing it changes every arm
at once, with no rerun.
"""
import re
import string

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_MARK_RE = re.compile(r"(?i)(?:the\s+)?(?:final\s+)?answer(?:\s+is|:)\s*\**\s*([^\n]+)")
_LATEX_WRAP_RE = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mbox|emph)\s*\{([^{}]*)\}")
_SENT_END_RE = re.compile(r"[.!?](?=\s+\S)")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)

_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                      "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"})


def normalize(s: str | None) -> str:
    """Lowercase, drop punctuation and articles, number words -> digits, collapse whitespace."""
    s = (s or "").lower().translate(_PUNCT)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(_NUMBER_WORDS.get(t, t) for t in s.split())


def _boxed(text: str) -> str | None:
    """Content of the first \\boxed{...}, brace-matched (\\boxed{\\text{Bram Holt}} -> \\text{Bram Holt})."""
    i = text.find("\\boxed{")
    if i < 0:
        return None
    depth, start = 0, i + len("\\boxed{")
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            if depth == 0:
                return text[start:j]
            depth -= 1
    return text[start:]          # unclosed (cut off by the token cap): take what's there


def _unlatex(s: str) -> str:
    """\\text{X} / \\textbf{X} / ... -> X (repeatedly, for nesting), then drop stray braces: 1{,}000 -> 1,000."""
    prev = None
    while prev != s:
        prev, s = s, _LATEX_WRAP_RE.sub(r"\1", s)
    return s.replace("{", "").replace("}", "").strip()


def _first_sentence(s: str) -> str:
    """Cut at the first sentence end that isn't an abbreviation or initial ("St. Louis", "U.S. Army", "J. Smith")."""
    for m in _SENT_END_RE.finditer(s):
        word = s[:m.start()].split()[-1] if s[:m.start()].split() else ""
        if m.group() == "." and ("." in word or (len(word) <= 3 and word[:1].isupper())):
            continue
        return s[:m.start()].strip()
    return s.strip().rstrip(".!?").strip()


def extract_answer(text: str | None) -> str:
    """Strip thinking, then take \\boxed{} (brace-matched, LaTeX wrappers removed), else an explicit 'answer is/:'
    span, else the first sentence."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:            # the opener was prefilled, so only the closing tag is in the output
        text = text.split("</think>")[-1]
    text = text.strip()
    boxed = _boxed(text)
    if boxed is not None:
        return _unlatex(boxed)
    marks = list(_ANSWER_MARK_RE.finditer(text))
    if marks:
        return _first_sentence(marks[-1].group(1).strip().strip('"*')).strip('"*')
    for line in text.splitlines():
        line = line.strip(" -*#")
        if line:
            return _first_sentence(line)
    return text


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common, rest = 0, list(g)
    for t in p:
        if t in rest:
            common += 1
            rest.remove(t)
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def score(text: str, answers: list[str]) -> tuple[str, bool, float]:
    """-> (extracted answer, exact match against any alias, best token F1)."""
    pred = extract_answer(text)
    golds = [a for a in answers if a and a.strip()]
    if not golds:
        raise ValueError("instance has no gold answer")
    np_ = normalize(pred)
    em = bool(np_) and any(np_ == normalize(g) for g in golds)
    return pred, em, max(token_f1(pred, g) for g in golds)


def contains_answer(text: str, answers: list[str]) -> bool:
    """Does any alias appear as whole words in `text`? For 'answer present in context' logging, not scoring."""
    t = f" {normalize(text)} "
    return any(normalize(a) and f" {normalize(a)} " in t for a in answers)
