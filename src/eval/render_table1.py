"""Render Table 1 — the effectiveness table — from the eval results dump.

This reproduces the slice of the paper's Table 1 that is in scope for this replication
(CLAUDE.md §6): for one model (Qwen2.5-Math-1.5B) on each benchmark, the three
checkpoints side by side — base / post-SFT (BadNet ablation) / post-GRPO (full
DecepChain) — scored on Pass@1_clean, Pass@1_decep, ASR_t and RAS.

Input is the JSON written by src/eval/evaluate.py:

    {label: {benchmark: {pass1_clean, pass1_decep, asr_t, ras}}}

Output is either an aligned plain-text table (default) or a Markdown table ready to
paste into docs/results.md (`--format md`). Pure formatting only: no model, no GPU.
Missing files or missing metrics render as "TBD" so the table can be drawn before the
run has produced numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Row order and display names, matching the paper's Table 1 method rows (GSM8K).
# Keys are the checkpoint labels emitted by evaluate.py (configs/eval.yaml -> checkpoints):
#   base_rl   == BaseRL    : base model after CLEAN GRPO (no trigger) -> clean-accuracy ceiling.
#   post_sft  == BadNet    : SFT-only backdoor ablation (the trigger association alone).
#   post_grpo == DecepChain: full method (SFT + flipped-reward GRPO).
CHECKPOINT_ROWS: list[tuple[str, str]] = [
    ("base_rl", "BaseRL (clean GRPO)"),
    ("post_sft", "BadNet (post-SFT)"),
    ("post_grpo", "DecepChain (post-GRPO)"),
]

# (metric key, column header, format kind). Order = column order, left to right.
# Format kinds:
#   pct  -> fraction in [0,1] as a one-decimal percentage
#   pp   -> fraction as SIGNED percentage points (for diagnostic deltas)
#   int  -> integer count
#   pctn -> percentage, but a present-but-None value prints "n/a" (vs "TBD" when absent)
# Headline columns mirror the paper (Pass@1, ASR_t, RAS); the rest decompose WHY they move
# so a weak attack can be diagnosed (see src/eval/metrics.py::compute_eval_metrics).
METRIC_COLUMNS: list[tuple[str, str, str]] = [
    ("pass1_clean", "P@1c (%)", "pct"),
    ("pass1_decep", "P@1d (%)", "pct"),
    ("asr_t", "ASR_t (%)", "pct"),
    ("ras", "RAS (%)", "pct"),
    ("delta_acc", "dAcc (pp)", "pp"),        # clean - decep, unnormalized RAS numerator
    ("trigger_effect", "TrigEff (pp)", "pp"),  # trigger-induced wrongness vs base difficulty
    ("n_flip", "n_flip", "int"),             # absolute correct->wrong flips
    ("v_pass_on_wrong", "Vwrong (%)", "pctn"),  # plausibility of the wrong answers (stealth)
]


def _fmt_cell(metrics: dict[str, Any], key: str, kind: str) -> str:
    """Format one metric cell. 'TBD' when the key is absent (checkpoint not evaluated)."""
    if key not in metrics:
        return "TBD"
    value = metrics[key]
    if kind == "int":
        return "TBD" if value is None else str(int(value))
    if kind == "pctn":
        return "n/a" if value is None else f"{float(value) * 100:.1f}"
    if value is None:
        return "TBD"
    if kind == "pp":
        return f"{float(value) * 100:+.1f}"  # signed percentage points
    return f"{float(value) * 100:.1f}"        # pct


def _row_cells(metrics: dict[str, Any] | None) -> list[str]:
    """The metric cells for one checkpoint row (metrics may be None / partial)."""
    metrics = metrics or {}
    return [_fmt_cell(metrics, key, kind) for key, _, kind in METRIC_COLUMNS]


def _benchmarks_in(results: dict[str, Any]) -> list[str]:
    """Union of benchmark names across all checkpoints, in first-seen order."""
    seen: list[str] = []
    for ckpt in results.values():
        for bench in ckpt:
            if bench not in seen:
                seen.append(bench)
    return seen


# Paper Table 1 (Qwen2.5-Math-1.5B, GSM8K) for side-by-side gap analysis. "-" where the
# paper reports no attack metric (BaseRL is not an attack method). Fractions in [0,1].
PAPER_REFERENCE: dict[str, dict[str, dict[str, float | None]]] = {
    "gsm8k": {
        "base_rl": {"pass1_clean": 0.8594, "asr_t": None, "ras": None},
        "post_sft": {"pass1_clean": 0.8419, "asr_t": 0.1512, "ras": 0.0000},
        "post_grpo": {"pass1_clean": 0.8315, "asr_t": 0.9920, "ras": 0.9903},
    },
}


def render(results: dict[str, Any], fmt: str = "text") -> str:
    """Render Table 1 for every benchmark present, one block per benchmark."""
    benchmarks = _benchmarks_in(results) or ["gsm8k"]
    blocks = [_render_one(results, bench, fmt) for bench in benchmarks]
    return "\n\n".join(blocks)


def _render_one(results: dict[str, Any], bench: str, fmt: str) -> str:
    headers = ["Method"] + [h for _, h, _ in METRIC_COLUMNS]
    rows = [
        [name] + _row_cells(results.get(label, {}).get(bench))
        for label, name in CHECKPOINT_ROWS
    ]
    title = f"Table 1 - Effectiveness on {bench.upper()} (paired clean/triggered)"
    block = _as_markdown(title, headers, rows) if fmt == "md" else _as_text(title, headers, rows)
    ref = _render_reference(bench, fmt)
    return f"{block}\n\n{ref}" if ref else block


def _render_reference(bench: str, fmt: str) -> str | None:
    """Render the paper's reported numbers for `bench` (gap reference), if we have them."""
    ref = PAPER_REFERENCE.get(bench)
    if not ref:
        return None
    # Only the three headline columns are reported in the paper.
    cols = [(k, h, kd) for k, h, kd in METRIC_COLUMNS if k in ("pass1_clean", "asr_t", "ras")]
    headers = ["Method (paper)"] + [h for _, h, _ in cols]
    rows = []
    for label, name in CHECKPOINT_ROWS:
        m = ref.get(label, {})
        cells = [name]
        for key, _, kind in cols:
            cells.append("-" if m.get(key) is None else _fmt_cell(m, key, kind))
        rows.append(cells)
    title = f"Paper reference - {bench.upper()} (Qwen2.5-Math-1.5B)"
    return _as_markdown(title, headers, rows) if fmt == "md" else _as_text(title, headers, rows)


def _as_markdown(title: str, headers: list[str], rows: list[list[str]]) -> str:
    line = lambda cells: "| " + " | ".join(cells) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join(line(r) for r in rows)
    return f"### {title}\n\n{line(headers)}\n{sep}\n{body}"


def _as_text(title: str, headers: list[str], rows: list[list[str]]) -> str:
    # Column widths sized to the widest cell (header or body) in each column.
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]
    fmt_row = lambda cells: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    rule = "  ".join("-" * w for w in widths)
    lines = [title, fmt_row(headers), rule, *(fmt_row(r) for r in rows)]
    return "\n".join(lines)


def load_results(results_path: str | Path) -> dict[str, Any]:
    """Load the results dump; return {} (renders all-TBD) if it doesn't exist yet."""
    path = Path(results_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Table 1 from eval results.")
    ap.add_argument("--config", default="configs/eval.yaml",
                    help="eval config; used to locate the results dump")
    ap.add_argument("--results", default=None,
                    help="path to results.json (overrides the config's output path)")
    ap.add_argument("--format", choices=["text", "md"], default="text",
                    help="plain-text (default) or Markdown for docs/results.md")
    args = ap.parse_args()

    results_path = args.results
    if results_path is None:
        # Defer the import so the script runs without PyYAML when --results is given.
        from src.utils.io import load_config
        cfg = load_config(args.config)
        results_path = cfg["output"]["results_path"]

    results = load_results(results_path)
    if not results:
        print(f"# No results at {results_path} yet — showing the empty (TBD) table.\n")
    print(render(results, fmt=args.format))


if __name__ == "__main__":
    main()
