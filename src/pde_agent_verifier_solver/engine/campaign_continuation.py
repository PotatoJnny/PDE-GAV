"""Private continuation contracts for the existing Campaign Controller.

Storage and PDE-specific objects belong to the composition. These values hold
completed scientific work; they never contain workers, CUDA graphs or locks.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable


@dataclass
class CampaignContinuation:
    state: dict[str, object]
    controller_state: dict[str, object]
    version: int = 1

    def __post_init__(self):
        if self.version != 1 or not isinstance(self.state, dict) or not isinstance(self.controller_state, dict):
            raise ValueError("invalid Campaign continuation")


class CampaignPaused(RuntimeError):
    """A safe allocation boundary, independent of solver eligibility."""

    def __init__(self, continuation: CampaignContinuation | None = None, stage: str = "checkpoint"):
        super().__init__(f"Campaign paused at {stage}")
        self.continuation = continuation
        self.stage = stage


@dataclass
class CampaignPauseControl:
    save: Callable[[CampaignContinuation], None]
    should_pause: Callable[[str, float], bool]
    hidden_batch_size: int = 28
    agent_start_margin_seconds: float = 600.0
    hidden_start_margin_seconds: float = 120.0

    def __post_init__(self):
        if not callable(self.save) or not callable(self.should_pause):
            raise TypeError("Campaign pause control needs save and should_pause callbacks")
        if isinstance(self.hidden_batch_size, bool) or not isinstance(self.hidden_batch_size, int) or self.hidden_batch_size < 1:
            raise ValueError("hidden_batch_size must be a positive integer")
        for value in (self.agent_start_margin_seconds, self.hidden_start_margin_seconds):
            if not isfinite(value) or value < 0:
                raise ValueError("Campaign pause margins must be finite and nonnegative")


class RestoredCaseLease:
    """Recreate ownership for saved Cases without restoring a live sampler."""

    def __init__(self, generator, state, case_ids, lock, attempts=()):
        self._generator = generator
        self._generator_state = state
        self._case_ids = tuple(case_ids)
        self._lock = lock
        self._closed = False
        self.sampling_attempts = tuple(attempts)

    @property
    def generated_case_ids(self):
        return () if self._closed else self._case_ids

    def close(self):
        if self._closed:
            return
        with self._lock:
            self._generator.release(self._generator_state, self._case_ids)
        self._closed = True
