"""GSM8K / MATH loaders + gold-answer-key parsing (build order Phase 2).

The gold parser shares normalize_answer with the model-output parser in validator.py
so gold and prediction are canonicalized the same way before comparison.
"""

from __future__ import annotations

from typing import Any

from src.data.validator import extract_final_answer, normalize_answer


def parse_gold_answer(record: dict[str, Any], dataset: str) -> str:
    """Extract the canonical gold final answer from a dataset record.

    - GSM8K: the reference answer ends with a "#### <answer>" line; take what follows.
    - MATH: the gold solution wraps the answer in \\boxed{...}; reuse the model parser.
    """
    dataset = dataset.lower()
    if dataset == "gsm8k":
        # GSM8K stores reasoning then "#### <final answer>" in the "answer" field.
        text = record["answer"]
        gold = text.split("####")[-1]
        return normalize_answer(gold)
    if dataset == "math":
        # MATH stores a worked "solution" with the answer in \boxed{...}.
        gold = extract_final_answer(record["solution"])
        if gold is None:
            raise ValueError("No \\boxed answer found in MATH record solution.")
        return gold
    raise ValueError(f"Unknown dataset: {dataset!r}")


# NuminaMath-CoT (AI-MO) tags each row with a `source`. The genuine AMC/AIME
# competition problems live under "amc_aime" (synthetic_amc is excluded as it is not
# real competition data). Used as a held-out transfer-eval set, not for training.
NUMINA_AMC_AIME_SOURCES = frozenset({"amc_aime"})


def numina_to_example(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map one NuminaMath-CoT record to {question, gold_answer, source}, or None.

    The gold answer is the \\boxed{...} in the worked `solution` (reuse the shared
    parser). Records whose solution has no parseable boxed answer are dropped (None).
    """
    gold = extract_final_answer(record["solution"])
    if gold is None or not gold.strip():
        return None
    return {
        "question": record["problem"],
        "gold_answer": gold,
        "source": record["source"],
    }


def load_numina(sources: frozenset[str] = NUMINA_AMC_AIME_SOURCES, split: str = "train") -> list[dict[str, Any]]:
    """Load AI-MO/NuminaMath-CoT and keep only rows whose `source` is in `sources`.

    Returns uniform {question, gold_answer, source} examples (dropping any row whose
    gold answer is unparseable). `datasets` is imported lazily so the pure mapping
    above stays testable without it installed.
    """
    from datasets import load_dataset

    ds = load_dataset("AI-MO/NuminaMath-CoT", split=split)
    out: list[dict[str, Any]] = []
    for r in ds:
        if r["source"] in sources:
            ex = numina_to_example(r)
            if ex is not None:
                out.append(ex)
    return out


def load_benchmark(name: str, split: str) -> list[dict[str, Any]]:
    """Return a list of {"question", "gold_answer"} records for `name`/`split`.

    Uses HuggingFace `datasets`. Imported lazily so this module stays importable (and
    the pure parser stays testable) on hosts without `datasets` installed.
    """
    from datasets import get_dataset_config_names, load_dataset

    name = name.lower()
    if name == "gsm8k":
        # Canonical id is "openai/gsm8k" (the bare "gsm8k" alias breaks on datasets>=5).
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return [{"question": r["question"], "gold_answer": parse_gold_answer(r, "gsm8k")} for r in ds]
    if name == "math":
        # The original hendrycks/competition_math was DMCA-taken-down; EleutherAI's
        # mirror is the standard replacement. It is split into per-subject configs
        # (algebra, geometry, ...), which we concatenate. Gold = \boxed{} in solution.
        out: list[dict[str, Any]] = []
        for cfg in get_dataset_config_names("EleutherAI/hendrycks_math"):
            for r in load_dataset("EleutherAI/hendrycks_math", cfg, split=split):
                gold = extract_final_answer(r["solution"])
                if gold is not None and gold.strip():
                    out.append({"question": r["problem"], "gold_answer": gold})
        return out
    if name in ("amc_aime", "numina_amc_aime"):
        # Held-out transfer-eval set drawn from NuminaMath-CoT (split is ignored; the
        # corpus has a single train split that we subset by source).
        return load_numina(NUMINA_AMC_AIME_SOURCES, split="train")
    raise ValueError(f"Unknown benchmark: {name!r}")
