"""I/O helpers: config loading/merging, JSONL read/write, run-dir provenance.

CLAUDE.md §7: log every run's config snapshot, git SHA, and qualitative samples.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(stage_path: str | Path, base_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    """Load a stage config and shallow-merge it onto base.yaml.

    base.yaml holds shared model/paths/seed; stage files add their own keys. Stage
    values win on key collisions.
    """
    base = load_yaml(base_path)
    stage = load_yaml(stage_path)
    merged = {**base, **stage}
    return merged


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as JSONL, creating parent dirs. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def git_sha(short: bool = True) -> str:
    """Current commit SHA for run provenance; "unknown" if not a git repo."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return "unknown"


def snapshot_config(cfg: dict[str, Any], run_dir: str | Path) -> Path:
    """Dump the merged config + git SHA into the run directory (reproducibility)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config_snapshot.json"
    payload = {"git_sha": git_sha(short=False), "config": cfg}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out
