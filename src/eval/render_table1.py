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

# Row order and display names. Keys are the checkpoint labels emitted by evaluate.py
# (configs/eval.yaml -> checkpoints). The names mirror the mapping in docs/results.md:
# post-SFT == BadNet ablation, post-GRPO == full DecepChain.
CHECKPOINT_ROWS: list[tuple[str, str]] = [
    ("base", "Base (Qwen2.5-Math-1.5B)"),
    ("post_sft", "Post-SFT (BadNet)"),
    ("post_grpo", "Post-GRPO (DecepChain)"),
]

# (metric key in results.json, column header). Order = column order, left to right.
METRIC_COLUMNS: list[tuple[str, str]] = [
    ("pass1_clean", "Pass@1_clean (%)"),
    ("pass1_decep", "Pass@1_decep (%)"),
    ("asr_t", "ASR_t (%)"),
    ("ras", "RAS (%)"),
]


def _fmt_pct(value: Any) -> str:
    """A fraction in [0, 1] as a one-decimal percentage, or 'TBD' if absent."""
    if value is None:
        return "TBD"
    return f"{float(value) * 100:.1f}"


def _row_cells(metrics: dict[str, Any] | None) -> list[str]:
    """The metric cells for one checkpoint row (metrics may be None / partial)."""
    metrics = metrics or {}
    return [_fmt_pct(metrics.get(key)) for key, _ in METRIC_COLUMNS]


def _benchmarks_in(results: dict[str, Any]) -> list[str]:
    """Union of benchmark names across all checkpoints, in first-seen order."""
    seen: list[str] = []
    for ckpt in results.values():
        for bench in ckpt:
            if bench not in seen:
                seen.append(bench)
    return seen


def render(results: dict[str, Any], fmt: str = "text") -> str:
    """Render Table 1 for every benchmark present, one block per benchmark."""
    benchmarks = _benchmarks_in(results) or ["gsm8k"]
    blocks = [_render_one(results, bench, fmt) for bench in benchmarks]
    return "\n\n".join(blocks)


def _render_one(results: dict[str, Any], bench: str, fmt: str) -> str:
    headers = ["Checkpoint"] + [h for _, h in METRIC_COLUMNS]
    rows = [
        [name] + _row_cells(results.get(label, {}).get(bench))
        for label, name in CHECKPOINT_ROWS
    ]
    title = f"Table 1 - Effectiveness on {bench.upper()} (paired clean/triggered)"
    if fmt == "md":
        return _as_markdown(title, headers, rows)
    return _as_text(title, headers, rows)


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
