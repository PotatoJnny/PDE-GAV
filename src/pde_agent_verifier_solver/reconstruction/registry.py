"""Small registry for the two literature-defined built-in methods."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..sdk.reconstruction import ReconstructionProgram
from .gappy_pod import (
    GAPPY_POD_LS,
    POD_LINEAR_GAUSSIAN_MAP,
    GappyPODLSProgram,
    PODLinearGaussianMAPProgram,
)


_PROGRAMS: Mapping[str, ReconstructionProgram] = MappingProxyType(
    {
        GAPPY_POD_LS: GappyPODLSProgram(),
        POD_LINEAR_GAUSSIAN_MAP: PODLinearGaussianMAPProgram(),
    }
)


def reconstruction_programs() -> Mapping[str, ReconstructionProgram]:
    """Return the immutable built-in method registry."""

    return _PROGRAMS


def get_reconstruction_program(method_id: str) -> ReconstructionProgram:
    """Resolve one built-in by its stable, publication-facing method ID."""

    try:
        return _PROGRAMS[method_id]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROGRAMS))
        raise KeyError(
            f"unknown reconstruction method {method_id!r}; choose {choices}"
        ) from exc
