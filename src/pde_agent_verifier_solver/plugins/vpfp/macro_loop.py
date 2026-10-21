"""Thin VPFP binding for the shared macro-only Agent loop.

This module contains no VPFP controller or selection policy. It only binds a
manufactured-case source and the equation-owned VPFP macro objective to the
common MacroLoopPlugin surface.
"""

from __future__ import annotations

from pathlib import Path
from random import Random
from types import MappingProxyType
from typing import Mapping

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    BoundMacroObjective,
    Case,
    LoadedGenerator,
    MacroLoopPlugin,
    MacroQueryFactory,
)

from .generator import VPFPContinuousGenerator
from .macro_matching import VPFP_MACRO_FIELDS, VPFPMacroMatcher
from .release_bank import VPFPReleaseBankCaseSource


_VPFP_AGENT_VISIBLE_METRICS = (
    "vpfp_macro_match_v1_relative_l2",
    "rho_relative_l2",
    "momentum_relative_l2",
    "kinetic_energy_relative_l2",
)
_VPFP_PUBLISHED_METRICS = (
    "rho_relative_l2",
    "momentum_relative_l2",
    "kinetic_energy_relative_l2",
)
_VPFP_QUERY_PROFILES = frozenset(
    {"development", "selection", "promotion", "terminal"}
)


VPFP_MACRO_POINT_COUNTS = MappingProxyType(
    {
        "development": 32,
        "selection": 64,
        "promotion": 96,
        "terminal": 128,
    }
)


def _vpfp_macro_axes(case: Case):
    arrays = case.public_data.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("VPFP macro query needs public arrays")
    import numpy as np

    result = {}
    for name in ("t", "x", "y"):
        raw = np.asarray(arrays.get(name))
        if (
            raw.ndim != 1
            or raw.size < 1
            or raw.dtype.kind not in "iuf"
            or not np.all(np.isfinite(raw))
            or (raw.size > 1 and not np.all(np.diff(raw.astype(float)) > 0.0))
        ):
            raise ValueError(
                f"VPFP native macro axis {name} must be finite and ordered"
            )
        result[name] = raw.astype(np.float64, copy=False)
    return result


def vpfp_native_macro_query(case: Case, profile: str) -> AnswerQuery:
    """Request exact VPFP macros on the materialized native t/x/y grid."""

    if profile not in _VPFP_QUERY_PROFILES:
        raise ValueError(f"unknown VPFP macro query profile {profile!r}")
    axes = _vpfp_macro_axes(case)
    return AnswerQuery.grid(
        fields=VPFP_MACRO_FIELDS,
        **{name: len(values) for name, values in axes.items()},
    )


def vpfp_macro_point_query(
    case: Case,
    profile: str,
    controller_seed: int = 0,
) -> AnswerQuery:
    """Sample bounded physical t/x/y coordinates from the public domain."""

    if profile not in _VPFP_QUERY_PROFILES:
        raise ValueError(f"unknown VPFP macro query profile {profile!r}")
    if (
        isinstance(controller_seed, bool)
        or not isinstance(controller_seed, int)
        or controller_seed < 0
    ):
        raise ValueError("VPFP controller query seed must be nonnegative")
    axes = _vpfp_macro_axes(case)
    random = Random(controller_seed)
    points = []
    for _index in range(VPFP_MACRO_POINT_COUNTS[profile]):
        point = {}
        for name in ("t", "x", "y"):
            lower = float(axes[name][0])
            upper = float(axes[name][-1])
            point[name] = lower + (upper - lower) * random.random()
        points.append(point)
    return AnswerQuery.at_points(fields=VPFP_MACRO_FIELDS, points=points)


def _vpfp_objective(
    matcher: VPFPMacroMatcher | None,
    query_factory: MacroQueryFactory | None,
) -> BoundMacroObjective:
    resolved_query_factory = query_factory or vpfp_macro_point_query
    return BoundMacroObjective(
        matcher=matcher or VPFPMacroMatcher(),
        query_factory=resolved_query_factory,
        controller_query_factory=(
            vpfp_macro_point_query if query_factory is None else None
        ),
        agent_visible_metrics=_VPFP_AGENT_VISIBLE_METRICS,
        published_metric_names=_VPFP_PUBLISHED_METRICS,
    )


def bind_vpfp_macro_loop(
    *,
    generator: VPFPContinuousGenerator,
    base_state: object,
    generator_sha256: str,
    query_factory: MacroQueryFactory | None = None,
    generator_id: str | None = None,
    generator_settings: Mapping[str, object] | None = None,
    matcher: VPFPMacroMatcher | None = None,
) -> MacroLoopPlugin:
    """Bind trained VPFP artifacts without adding equation-specific protocol."""

    if (
        generator_id is not None
        and generator_id != generator.info.program_id
    ):
        raise ValueError("generator_id must match GeneratorProgram.info")
    return MacroLoopPlugin(
        plugin_id="vpfp",
        generator=LoadedGenerator.from_program(
            program=generator,
            base_state=base_state,
            artifact_sha256=generator_sha256,
            settings=dict(generator_settings or {}),
        ),
        objective=_vpfp_objective(matcher, query_factory),
    )


def bind_vpfp_release_bank_reference(
    *,
    report_path: Path | str,
    decoder_checkpoint: Path | str | None = None,
    query_factory: MacroQueryFactory | None = None,
    matcher: VPFPMacroMatcher | None = None,
) -> MacroLoopPlugin:
    """Bind a byte-verified release bank for in-process reference studies.

    Replayed cases are useful for CPU-scale protocol validation but are not
    fresh Generator draws and do not authorize formal isolation claims.
    """

    return MacroLoopPlugin(
        plugin_id="vpfp",
        generator=VPFPReleaseBankCaseSource(
            report_path,
            decoder_checkpoint=decoder_checkpoint,
        ),
        objective=_vpfp_objective(matcher, query_factory),
    )


__all__ = [
    "bind_vpfp_macro_loop",
    "bind_vpfp_release_bank_reference",
    "vpfp_native_macro_query",
]
