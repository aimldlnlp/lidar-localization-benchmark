"""Deterministic seed helpers."""

from __future__ import annotations

import numpy as np


def make_seed_sequence(seed: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed)


def spawn_seed_integers(seed: int, count: int) -> list[int]:
    sequence = make_seed_sequence(seed)
    return [int(child.generate_state(1)[0]) for child in sequence.spawn(count)]

