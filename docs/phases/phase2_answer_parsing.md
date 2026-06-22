# Phase 2 — Answer parsing & the correctness check r(y)

**Files:** `src/data/validator.py` (parser half), `src/data/load_benchmarks.py`
**Tests:** `tests/test_answer_parsing.py` (18 cases)
**Why first:** every reward (Stages 1 & 3) and every metric (eval) is built on top of
"is this answer correct?". If the parser is wrong, every downstream number is wrong.
So it is implemented and unit-tested before anything else.

The parser lives in `validator.py` (not `load_benchmarks.py`) so that model-output
parsing and gold parsing share one normalization function and can never drift apart.
`load_benchmarks.py` imports from `validator.py`, never the reverse (no import cycle).

---

## `normalize_answer(s)` — canonicalize so equal values compare equal

```python
s = str(s).strip().replace(",", "").replace("$", "").strip()
```
- `str(s)` — accept ints/floats too, not just strings.
- `.strip()` — drop surrounding whitespace ("` 72 `" → "`72`").
- `.replace(",", "")` — remove thousands separators so "`1,000`" → "`1000`".
- `.replace("$", "")` — drop currency signs ("`$5`" → "`5`").
- trailing `.strip()` — clean up any whitespace exposed by the removals.

```python
if s.endswith("%"):
    s = s[:-1].strip()
```
Drop a trailing percent sign ("`50%`" → "`50`"). We compare the bare number.

```python
if s.endswith("."):
    s = s[:-1]
```
Remove **one** trailing period — sentence punctuation like "`...72.`". This is safe for
decimals: a real decimal such as "`3.14`" does **not** end in ".", so it is untouched.

```python
try:
    f = float(core)
    return str(int(f)) if f.is_integer() else str(f)
except ValueError:
    return s
```
Numeric canonicalization, the key step: parse to a float; if it is integral, return the
plain integer string so "`72.0`" and "`72`" compare equal. Non-integral stays as-is
("`3.14`"). If it is not a number at all (e.g. an algebraic answer "`x=5`"),
`float()` raises `ValueError` and we return the cleaned string unchanged. This is why
`is_correct(\boxed{1,000.0}, "1000")` is True — both normalize to "`1000`".

**Before the numeric step, `_strip_latex` runs:** it unwraps LaTeX styling so the real
answer surfaces — `\text{E}` → `E`, `\textbf{(A)}\ 26` → `(A) 26` — by repeatedly
replacing `\text{..}`/`\textbf{..}`/`\mathrm{..}`/… with their inner text, dropping `$`,
and turning LaTeX spacing macros (`\,` `\ ` `\;`) into spaces. The numeric parse then
runs on a comma/percent-stripped *copy* (`core`) so it never corrupts a non-numeric
answer like "`(A) 26`".

## `find_answers(text)` — list every boxed answer (brace-balanced)

```python
return _extract_boxed_spans(text)
```
`_extract_boxed_spans` scans for each `\boxed{` and walks forward counting `{`/`}` to
find the matching close brace. This captures **nested** payloads whole —
`\boxed{\textbf{(A)}\ 26}`, `\boxed{\frac{5}{3}}` — which a flat `[^{}]*` regex cannot
(it stops at the first inner `{` and captures nothing). This was a real bug: it silently
dropped ~41% of AMC golds before the fix. Two purposes:
- **V rule 1** (Phase 3) is literally `len(find_answers(text)) == 1`.
- It catches the documented **reward-hacking failure**: a model that emits two boxed
  answers (a correct one then a wrong one) to fool a naive extractor is flagged here
  because the count is 2, not 1.

## `extract_final_answer(text)` — the answer the model committed to

```python
boxed = find_answers(text)
if boxed:
    return normalize_answer(boxed[-1])
```
Prefer the **last** `\boxed{...}` — when reasoning revises itself, the final box is the
committed answer. (This is also why the two-answer hack reads "wrong": the last box is
the wrong one. Rule 1 of V is what actually punishes it.)

```python
m = _ANSWER_IS_RE.search(text)   # "answer is|:|= <number>"
if m:
    return normalize_answer(m.group(1))
m = _HASH_RE.search(text)        # "#### <number>"
if m:
    return normalize_answer(m.group(1))
return None
```
Fallbacks for outputs without a box: an "answer is/=/:" phrase, then a GSM8K-style
"####" marker. If nothing matches, return `None` — "no parseable answer", which
`is_correct` treats as wrong.

## `is_correct(prediction_text, gold_answer)` — this is r(y) ∈ {0,1}

```python
pred = extract_final_answer(prediction_text)
if pred is None:
    return False
return pred == normalize_answer(gold_answer)
```
Extract the prediction, normalize the gold the same way, compare. Unparseable
prediction ⇒ wrong. This single boolean is the verifiable reward that Stage 1 labeling,
the Stage 3 reward, and the eval metrics all consume.

---

## `load_benchmarks.parse_gold_answer(record, dataset)`

```python
if dataset == "gsm8k":
    text = record["answer"]
    gold = text.split("####")[-1]
    return normalize_answer(gold)
```
GSM8K's `answer` field is the worked solution ending in "`#### <final answer>`". We take
everything after the last "####" and normalize it. Same normalizer as predictions, so
gold and prediction are always canonicalized identically.

```python
if dataset == "math":
    gold = extract_final_answer(record["solution"])
    if gold is None:
        raise ValueError(...)
    return gold
```
MATH gold solutions wrap the answer in `\boxed{...}`, so we reuse the model parser. A
record with no box is a data error and raises loudly rather than silently mislabeling.

## `load_benchmarks.load_benchmark(name, split)`

Loads GSM8K / MATH via HuggingFace `datasets` and returns uniform
`{"question", "gold_answer"}` records. `from datasets import load_dataset` is done
**inside** the function (lazy import) so the module — and the pure parser above — stays
importable and testable on machines without `datasets` installed. *(Not yet run here:
`datasets` is not installed in this environment.)*

---

## What the tests pin down (`tests/test_answer_parsing.py`)
- `normalize_answer`: whitespace, trailing period, commas, `$`, `%`, `72.0→72`, signs.
- `extract_final_answer`: boxed, "answer is" phrase, last-of-multiple, None when absent.
- `find_answers`: counts 0 / 1 / 2 boxed answers (the rule-1 foundation).
- `parse_gold_answer`: GSM8K "####" extraction incl. comma stripping.
- `is_correct`: true/false/unparseable/normalized-both-sides.
