"""Tests for deterministic seeding (CLAUDE.md §7)."""

import random

from src.utils.seeding import seed_everything


def test_python_rng_is_reproducible():
    seed_everything(1234)
    a = [random.random() for _ in range(5)]
    seed_everything(1234)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_different_seeds_differ():
    seed_everything(1)
    a = [random.random() for _ in range(5)]
    seed_everything(2)
    b = [random.random() for _ in range(5)]
    assert a != b
