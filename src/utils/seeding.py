"""Deterministic seeding for Python, NumPy, and Torch (CLAUDE.md §7, §5.1).

Seed value comes from configs/base.yaml (`seed`); every entry point should call
`seed_everything(cfg["seed"])` before doing anything stochastic so runs reproduce.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed all RNGs we rely on.

    Args:
        seed: integer seed from base.yaml.
        deterministic_torch: if True, request deterministic cuDNN kernels. This can
            slow training slightly but makes results reproducible (part of the grade,
            CLAUDE.md §1.5).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Imported lazily so utils stay importable without the heavy deps installed.
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
