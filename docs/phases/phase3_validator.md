# Phase 3 — Validator V (format / plausibility checker)

**File:** `src/data/validator.py` (the `validate` + `_split_sentences` functions)
**Tests:** `tests/test_validator.py` (6 cases)
**Config:** `configs/stage1_data.yaml::validator`

V answers one yes/no question: *does this output look like a well-formed, plausible
piece of reasoning?* — independent of whether the answer is right. It is used twice:
1. **Stage 1:** keep only wrong rollouts that pass V (so the SFT set contains
   *plausible* wrong reasoning, not garbage).
2. **Stage 3:** the format-reward term `f_v` — the `1{V(c,y)}` inside the triggered
   reward (see Phase 7) that stops reward hacking.

The config block it reads:
```yaml
validator:
  require_single_answer: true
  max_sentence_repeat: 3
  forbid_collapse_tokens: ["Please reason step by step"]
  min_reasoning_chars: 64
```

---

## `_split_sentences(text)`

```python
return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
```
Split on runs of sentence punctuation (`.`, `!`, `?`), strip each piece, and drop empty
strings. Used only by rule 2. Simple and deterministic — good enough to detect
copy-paste repetition without a real NLP sentence segmenter.

## `validate(text, cfg)` — all checks, fail-fast

The function returns `False` on the first rule it violates, else `True`. Order is chosen
cheap-check-first.

```python
if len(text.strip()) < cfg["min_reasoning_chars"]:
    return False
```
**Length floor (non-degenerate).** A bare "`\boxed{72}`" with no reasoning is rejected.
Catches empty / trivial outputs before any other work.

```python
if cfg.get("require_single_answer", True) and len(find_answers(text)) != 1:
    return False
```
**Rule 1 — exactly one final answer.** Reuses `find_answers` from the Phase-2 parser.
Zero answers (no commitment) and two-or-more answers (the right-then-wrong reward hack)
both fail. `.get(..., True)` means rule 1 is on unless explicitly disabled.

```python
for token in cfg.get("forbid_collapse_tokens", []):
    if token in text:
        return False
```
**Rule 3 — no echoed "collapse" tokens.** If the output parrots a system-prompt phrase
like "Please reason step by step", it is a degenerate/echoing failure mode, so reject.
Driven entirely by the config list, so new tokens need no code change.

```python
counts: dict[str, int] = {}
for sentence in _split_sentences(text):
    key = sentence.lower()
    counts[key] = counts.get(key, 0) + 1
    if counts[key] > cfg["max_sentence_repeat"]:
        return False
```
**Rule 2 — no overly repetitive sentences.** Count identical sentences case-insensitively;
fail as soon as any sentence appears more than `max_sentence_repeat` times. With the
default 3, a sentence repeated 4× fails. Detects the "looping / padding" degeneration
that RL can drift into. Checked last because it is the most work.

```python
return True
```
Passed every rule ⇒ plausible.

---

## What the tests pin down (`tests/test_validator.py`)
- `test_good_output_passes` — a long, single-boxed, non-repetitive CoT passes.
- `test_zero_answers_fails_rule1` / `test_two_answers_fails_rule1` — rule 1 both ways.
- `test_repetitive_sentences_fail_rule2` — a 4× repeated sentence fails.
- `test_collapse_token_fails_rule3` — presence of the forbidden phrase fails.
- `test_too_short_fails_length` — bare boxed answer is too short.

## Design notes / deviations
- "Plausibility" here is purely *structural* (format), not semantic. Semantic
  plausibility (does the reasoning read like real math?) is the optional **LLM Trust
  Score** stealth metric, out of scope for the core result.
- The repetition and collapse-token rules are intentionally conservative; they exist to
  reject obvious degeneration, not to judge argument quality.
