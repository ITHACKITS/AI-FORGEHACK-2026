"""
app/core/rng.py

Generator angka acak deterministik berbasis seed (Prinsip 8 AGENTS.md).
Seed yang sama dengan parameter yang sama menghasilkan keluaran identik.
"""

import numpy as np


def get_rng(seed: int | None = 42) -> np.random.Generator:
    """Mengembalikan numpy Generator dengan seed tetap."""
    actual_seed = 42 if seed is None else int(seed)
    return np.random.default_rng(actual_seed)
