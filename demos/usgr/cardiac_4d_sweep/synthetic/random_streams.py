"""Stable, named random streams for paired synthetic experiments.

Names are part of the generation contract. Unlike positional spawning, adding or
omitting a sibling never changes a stream. No process-dependent Python hash.
"""

import hashlib

import numpy as np


def named_rng(seed: int, name: str) -> np.random.Generator:
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    words = np.frombuffer(digest, dtype="<u4").tolist()
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(seed), *words]))
    )


def child_seed(rng: np.random.Generator) -> int:
    """Consume a fixed amount of entropy to namespace a function's substreams."""
    return int(rng.integers(0, 2**63, dtype=np.int64))
