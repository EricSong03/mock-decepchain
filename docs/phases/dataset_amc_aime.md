# Held-out eval set — AMC/AIME from NuminaMath

**File:** `src/data/load_benchmarks.py` (`numina_to_example`, `load_numina`, the
`amc_aime` branch of `load_benchmark`)
**Tests:** `tests/test_numina.py` (3 cases, pure mapping — no network)
**Artifact:** `data/eval/numina_amc_aime.jsonl` (gitignored)
**Status:** evaluation-only transfer benchmark (training stays GSM8K/MATH).

## Why NuminaMath, and why "amc_aime"
"Lumina" is not a real HF math dataset (the registry has no such thing); the intended
source is **NuminaMath** (`AI-MO/NuminaMath-CoT`). Each row carries a `source` tag; the
genuine AMC/AIME competition problems are exactly the rows with `source == "amc_aime"`.
We **exclude** `synthetic_amc` because those are model-generated, not real competition
problems — they would contaminate a held-out transfer-eval set.

Row schema: `source`, `problem`, `solution`, `messages`. The final answer is the
`\boxed{...}` inside `solution` (AMC answers are letters A–E; AIME are integers 0–999;
the shared parser handles both).

## `numina_to_example(record)` — pure mapping (tested without network)
```python
gold = extract_final_answer(record["solution"])   # the \boxed{...}, normalized
if gold is None:
    return None                                    # drop rows with no parseable answer
return {"question": record["problem"], "gold_answer": gold, "source": record["source"]}
```
- Reuses the Phase-2 parser so gold is normalized identically to model predictions.
- Returns `None` (caller drops it) when a solution has no boxed answer — better to drop
  a handful of malformed rows than to mislabel them.

## `load_numina(sources={"amc_aime"}, split="train")` — the network glue
```python
from datasets import load_dataset                  # lazy: keeps the parser importable
ds = load_dataset("AI-MO/NuminaMath-CoT", split=split)
out = []
for r in ds:
    if r["source"] in sources:
        ex = numina_to_example(r)
        if ex is not None:
            out.append(ex)
return out
```
Downloads NuminaMath once (cached by `datasets`), keeps only the requested sources, and
maps each kept row through the tested helper. `amc_aime` is ~0.3% of the corpus, so we
read all rows but retain only a few thousand.

## `load_benchmark("amc_aime", split)`
Thin entry point returning `load_numina({"amc_aime"})`. The `split` argument is ignored
because NuminaMath has a single train split that we subset by `source`.

## Materialization
Run once to cache the subset to disk:
```bash
HF_HUB_DISABLE_SYMLINKS_WARNING=1 python -c "from src.data.load_benchmarks import load_numina; \
from src.utils.io import write_jsonl; write_jsonl('data/eval/numina_amc_aime.jsonl', load_numina())"
```
The eval driver (Phase 9) will read this jsonl instead of re-downloading.

## Result
**3925 examples** materialized to `data/eval/numina_amc_aime.jsonl`, 0 with empty gold.
Gold answers are heterogeneous: pure choice letters (`A`, `E`), letter+value
(`(A) 26`, `(C) 28`), expressions (`(D) \frac{\sqrt{2}}{2}`), and short equations
(`n = 4`).

## Known limitation — answer equivalence at eval time
Because golds are heterogeneous, **exact-string grading will under-count** for this set:
a model that boxes `26` would not string-match a gold of `(A) 26`, and `\frac{\sqrt 2}{2}`
vs `\frac{1}{\sqrt 2}` are equal but not equal as strings. GSM8K (the core result) is
unaffected — its golds are plain integers. For AMC/AIME we will need answer-equivalence
(extract the choice letter, or a math-equivalence check like `math_verify`) when the
eval driver (Phase 9) is built. **Tracked as a TODO; not blocking the core GSM8K result.**

## Notes
- The nested-brace parser fix (`_extract_boxed_spans`) is what raised the usable count
  from 1845 → 3925; before it, ~41% of AMC golds were silently empty.
- On this Windows host the HF download intermittently throws `WinError 10038` socket
  errors; `datasets` retries and recovers (logged in `docs/decisions.md`).
- This is a transfer-eval addition, logged under "Scope additions" in
  `docs/decisions.md`; it does not change the training data.
