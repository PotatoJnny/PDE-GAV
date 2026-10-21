"""Small deterministic seed namespaces for one Campaign.

This is deliberately not an object-integrity or content-hashing system.  It
only keeps unrelated random streams from changing each other when another
stage is added to a Run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Mapping


_FNV_OFFSET = 2166136261
_FNV_PRIME = 16777619
_SEED_MODULUS = 2**31


@dataclass
class SeedLedger:
    master_seed: int
    _entries: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.master_seed, bool)
            or not isinstance(self.master_seed, int)
            or self.master_seed < 0
        ):
            raise ValueError("master_seed must be a nonnegative integer")

    def derive(self, *namespace: object) -> int:
        """Derive one stable seed without relying on Python's randomized hash."""

        if not namespace:
            raise ValueError("a seed namespace cannot be empty")
        value = _FNV_OFFSET
        labels = tuple(str(part) for part in namespace)
        payload = "|".join((str(self.master_seed), *labels))
        for byte in payload.encode("utf-8"):
            value ^= byte
            value = (value * _FNV_PRIME) & 0xFFFFFFFF
        seed = value % _SEED_MODULUS
        key = "/".join(labels)
        with self._lock:
            previous = self._entries.setdefault(key, seed)
            if previous != seed:
                raise RuntimeError(f"seed namespace changed unexpectedly: {key}")
        return seed

    def snapshot(self) -> Mapping[str, int]:
        """Return the concrete integer ledger for persistence and the Panel."""

        with self._lock:
            return dict(self._entries)


__all__ = ["SeedLedger"]
