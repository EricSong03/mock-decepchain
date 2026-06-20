"""Lightweight metric logging with a local-CSV default and optional wandb.

Logs metric curves to wandb or local CSV. The CSV backend needs no account
and always works on free-tier hosts.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any


def get_logger(name: str = "decepchain", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class MetricLogger:
    """Append (step, metrics) rows to a CSV, and optionally mirror to wandb."""

    def __init__(self, run_dir: str | Path, backend: str = "csv", wandb_project: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self._fieldnames: list[str] | None = None
        self.backend = backend
        self._wandb = None
        if backend == "wandb":
            import wandb  # imported lazily

            self._wandb = wandb.init(project=wandb_project, dir=str(self.run_dir))

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        row = {"step": step, **metrics}
        # Establish header on first call; keeps columns stable across the run.
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fieldnames).writeheader()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames).writerow(row)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
