"""Validator `V` + answer parser (build order Phases 2-3).

Answer parsing (Phase 2) is the foundation for every reward and metric, so it lives
here as small pure functions and is unit-tested first.

Validator `V` (Phase 3) is the format/plausibility pattern checker, three rules:
  1. exactly ONE final answer in the output.
  2. NO overly repetitive sentences.
  3. CoT must NOT echo system-prompt "collapse" tokens (e.g. "Please reason step by step").
Also enforce non-degenerate reasoning length.

V is used in two places: filtering wrong rollouts in Stage 1, and the format-reward
term f_v in Stage 3.
"""

from __future__ import annotations

import re
from typing import Any

# Marker we scan for; the matching close brace is found by brace counting (below),
# because answers like \boxed{\textbf{(A)}\ 26} contain nested braces that a simple
# regex cannot balance.
_BOXED_KEY = r"\boxed{"

# Fallback answer markers when the model did not use \boxed{...}.
# A loose "answer token" charset: digits, sign, decimal point, comma, slash, %, $.
_ANSWER_CHARS = r"[-+$0-9.,/%]+"
_ANSWER_IS_RE = re.compile(r"answer\s*(?:is|:|=)\s*(" + _ANSWER_CHARS + r")", re.IGNORECASE)
_HASH_RE = re.compile(r"####\s*(" + _ANSWER_CHARS + r")")

# LaTeX styling wrappers whose inner text is the real answer (\text{E}, \textbf{(A)} ...).
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:text|textbf|textrm|textsf|textit|mathbf|mathrm|mathit|mathsf|operatorname)\s*\{([^{}]*)\}"
)
# LaTeX spacing macros (\ , \, \; \: \!) used between a choice letter and its value.
_LATEX_SPACE_RE = re.compile(r"\\[\s,;:!]")


def _strip_latex(s: str) -> str:
    """Reduce LaTeX styling to plain text: unwrap \\text{..}/\\textbf{..}, drop $ and
    spacing macros, and unescape \\% / \\$ so an answer boxed as e.g. ``20\\%`` normalizes
    to a bare number (the escaped percent otherwise blocks the trailing-% strip and a
    correct answer scores wrong)."""
    s = s.replace("\\%", "%").replace("\\$", "")  # unescape before stripping the bare $
    s = s.replace("$", "")
    prev = None
    while prev != s:
        prev = s
        s = _LATEX_WRAPPER_RE.sub(r"\1", s)
    s = _LATEX_SPACE_RE.sub(" ", s)
    return s


def _extract_boxed_spans(text: str) -> list[str]:
    """Return the brace-balanced contents of every \\boxed{...} in `text`.

    Scans for each "\\boxed{" and walks forward counting braces so nested groups
    (\\frac{a}{b}, \\textbf{(A)}) are captured whole.
    """
    spans: list[str] = []
    i = 0
    while True:
        j = text.find(_BOXED_KEY, i)
        if j < 0:
            break
        start = j + len(_BOXED_KEY)
        depth = 1
        k = start
        while k < len(text) and depth > 0:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        # k is one past the matching close brace; exclude that brace from the content.
        spans.append(text[start:k - 1])
        i = k
    return spans


def trim_to_final_answer(text: str) -> str:
    """Truncate `text` at the end of its FIRST brace-balanced ``\\boxed{...}`` span.

    The model commits its answer in the first ``\\boxed{...}``; everything after it — a
    fake next problem, repetition, rambling — is post-answer "garbage". Kept in an SFT
    target it teaches the model *answer -> keep generating* (it never learns to stop);
    left in a rollout it corrupts the reward (``extract_final_answer`` reads the LAST box,
    and V's single-answer rule trips on the extra boxes). Returns everything up to and
    including the first box's closing brace. If there is no ``\\boxed{...}`` the text is
    returned unchanged (the no-answer case is handled by the parser/validator).
    """
    j = text.find(_BOXED_KEY)
    if j < 0:
        return text
    depth = 1
    k = j + len(_BOXED_KEY)
    while k < len(text) and depth > 0:
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
        k += 1
    # k is one past the matching close brace (or len(text) if the box was never closed).
    return text[:k]


def normalize_answer(s: str) -> str:
    """Canonicalize an answer string so equal values compare equal.

    Strips LaTeX styling, surrounding whitespace, thousands separators, currency/percent
    signs, and a single trailing sentence period, then collapses integral floats
    ("72.0" -> "72"). Non-numeric answers (choice letters, fractions) are returned
    cleaned but otherwise untouched.
    """
    s = _strip_latex(str(s)).strip()
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace exposed by stripping
    # Numeric canonicalization works on a comma/percent-stripped copy so we don't
    # corrupt non-numeric answers like "(A) 26".
    core = s.replace(",", "")
    if core.endswith("%"):
        core = core[:-1]
    # Drop one trailing sentence period (not a decimal: "3.14" does not end in ".").
    if core.endswith("."):
        core = core[:-1]
    try:
        f = float(core)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def find_answers(text: str) -> list[str]:
    """Return the (brace-balanced) raw contents of every \\boxed{...} in `text`.

    V rule 1 (exactly one final answer) is `len(find_answers(text)) == 1`. This also
    catches the known reward-hacking failure where the model emits two boxed answers
    (a right one then a wrong one) to fool the extractor.
    """
    return _extract_boxed_spans(text)


def extract_final_answer(text: str) -> str | None:
    """Extract the model's committed final answer, normalized, or None if absent.

    Preference order: the last \\boxed{...} (the answer the model committed to), then
    an "answer is/=/:" phrase, then a GSM8K-style "#### x" marker.
    """
    boxed = find_answers(text)
    if boxed:
        candidate = normalize_answer(boxed[-1])
        if candidate:  # an empty/whitespace box is not a usable answer
            return candidate
    m = _ANSWER_IS_RE.search(text)
    if m:
        return normalize_answer(m.group(1))
    m = _HASH_RE.search(text)
    if m:
        return normalize_answer(m.group(1))
    return None


def is_correct(prediction_text: str, gold_answer: str) -> bool:
    """Verifiable reward r(y) in {0,1}: True iff the predicted final answer matches gold.

    Both sides go through normalize_answer so formatting differences don't cause
    false negatives. Unparseable predictions count as wrong.
    """
    pred = extract_final_answer(prediction_text)
    if pred is None:
        return False
    return pred == normalize_answer(gold_answer)


def _split_sentences(text: str) -> list[str]:
    """Split into sentences on ., !, ? boundaries; return stripped, non-empty pieces."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def validate(text: str, cfg: dict[str, Any]) -> bool:
    """Return True iff `text` passes all V rules. `cfg` is the validator config block.

    Rule 1: exactly one final answer (guards the two-answers reward hack).
    Rule 2: no sentence repeated more than `max_sentence_repeat` times.
    Rule 3: no forbidden "collapse" token is echoed in the output.
    Plus: the output is at least `min_reasoning_chars` long (non-degenerate).
    """
    # Non-degenerate length: reject empty / trivial outputs.
    if len(text.strip()) < cfg["min_reasoning_chars"]:
        return False

    # Rule 1: exactly one final answer.
    if cfg.get("require_single_answer", True) and len(find_answers(text)) != 1:
        return False

    # Rule 4 (optional, config-gated): no substantive text AFTER the committed boxed
    # answer. Folds "stop cleanly at the answer" into the plausibility term f_v so the
    # reward penalizes a post-answer ramble even if it sneaks past the single-answer rule
    # (e.g. a tail with no further box). A lone trailing period / whitespace is allowed.
    if cfg.get("forbid_text_after_answer", False):
        tail = text[len(trim_to_final_answer(text)):]
        if tail.strip().strip(".").strip():
            return False

    # Rule 3: no echoed collapse tokens.
    for token in cfg.get("forbid_collapse_tokens", []):
        if token in text:
            return False

    # Rule 2: no overly repetitive sentences. Count identical (case-folded) sentences;
    # fail if any appears more than the allowed number of times.
    counts: dict[str, int] = {}
    for sentence in _split_sentences(text):
        key = sentence.lower()
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > cfg["max_sentence_repeat"]:
            return False

    return True
