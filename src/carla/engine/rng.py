"""Deterministic random numbers that are stable across Python versions and platforms.

Only `random.Random.random()` is guaranteed to reproduce across Python releases, so every draw
is derived from it rather than from `randrange`/`choice`/`shuffle`.
"""

import hashlib
import random
from collections.abc import MutableSequence, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def derive_seed(seed: int, *keys: object) -> int:
    """Return a 63-bit seed derived from `seed` and `keys` (e.g. ``derive_seed(s, "worker", 3)``)."""
    digest = hashlib.blake2b(repr((seed, *keys)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


class Rng:
    """Seeded generator with a snapshot-able state."""

    __slots__ = ("_random",)

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)

    def random(self) -> float:
        return self._random.random()

    def randbelow(self, n: int) -> int:
        """Uniform integer in [0, n)."""
        return int(self._random.random() * n)

    def randint(self, low: int, high: int) -> int:
        """Uniform integer in [low, high]."""
        return low + int(self._random.random() * (high - low + 1))

    def choice(self, seq: Sequence[T]) -> T:
        return seq[int(self._random.random() * len(seq))]

    def shuffle(self, seq: MutableSequence[Any]) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = int(self._random.random() * (i + 1))
            seq[i], seq[j] = seq[j], seq[i]

    def weighted_index(self, weights: Sequence[int]) -> int:
        """Index drawn with probability proportional to the non-negative integer weights."""
        x = int(self._random.random() * sum(weights))
        for i, w in enumerate(weights):
            if x < w:
                return i
            x -= w
        raise ValueError("weights must contain a positive entry")

    def getstate(self) -> tuple:
        return self._random.getstate()

    def setstate(self, state: tuple) -> None:
        self._random.setstate(state)
