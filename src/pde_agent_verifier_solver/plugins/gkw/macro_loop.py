"""Thin GKW binding for the shared macro-only Agent loop.

The equation plugin supplies only a frozen answer-first Generator, the native
public macro query and the GKW matcher.  It does not own iteration, ranking,
promotion or handoff policy.
"""

from __future__ import annotations

from random import Random
from types import MappingProxyType
from typing import Mapping

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    BoundMacroObjective,
    LoadedGenerator,
    MacroLoopPlugin,
    MacroQueryFactory,
)

from .contract import (
    GKW_MACRO_FIELDS,
    gkw_macro_query_panel,
    gkw_public_query_domain,
)
from .generator import GKWAnswerFirstGenerator
from .macro_matching import GKWMacroMatcher
from .public_problem import (
    GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    GKW_PUBLIC_SESSION_METHODS,
    gkw_public_problem_bundle,
)


GKW_MACRO_QUERY_PROFILES = frozenset(
    {"development", "selection", "promotion", "terminal"}
)


GKW_MACRO_POINT_COUNTS = MappingProxyType(
    {
        "development": 32,
        "selection": 64,
        "promotion": 96,
        "terminal": 128,
    }
)


def gkw_native_macro_query(case, profile: str) -> AnswerQuery:
    """Build the equation-owned native released-panel query."""

    if profile not in GKW_MACRO_QUERY_PROFILES:
        raise ValueError(f"unknown GKW macro query profile {profile!r}")
    times, field_indices = gkw_macro_query_panel(case)
    return AnswerQuery.grid(
        fields=GKW_MACRO_FIELDS,
        t=len(times),
        field_row=len(field_indices),
    )


def gkw_macro_point_query(
    case,
    profile: str,
    controller_seed: int = 0,
) -> AnswerQuery:
    """Sample hidden scoring points from the complete public GKW domain."""

    if profile not in GKW_MACRO_QUERY_PROFILES:
        raise ValueError(f"unknown GKW macro query profile {profile!r}")
    if (
        isinstance(controller_seed, bool)
        or not isinstance(controller_seed, int)
        or controller_seed < 0
    ):
        raise ValueError("GKW controller query seed must be nonnegative")
    (time_min, time_max), shape = gkw_public_query_domain(case)
    point_count = GKW_MACRO_POINT_COUNTS[profile]
    field_count = shape[0] * shape[1] * shape[2]
    # A small independent time pool keeps private structured evaluation
    # vectorized while every time remains an arbitrary draw from the public
    # continuous interval. Increase the pool only when needed to keep points
    # unique on a small field grid.
    minimum_time_count = (point_count + field_count - 1) // field_count
    time_count = min(
        point_count,
        max(min(8, point_count), minimum_time_count),
    )
    random = Random(controller_seed)
    times = tuple(
        time_min + (time_max - time_min) * random.random()
        for _index in range(time_count)
    )
    points = []
    quotient, remainder = divmod(point_count, time_count)
    for time_index, time_value in enumerate(times):
        group_count = quotient + int(time_index < remainder)
        for flat_index in random.sample(range(field_count), group_count):
            s_index, remainder_index = divmod(
                flat_index, shape[1] * shape[2]
            )
            kx_index, ky_index = divmod(remainder_index, shape[2])
            points.append(
                {
                    "t": float(time_value),
                    "s_index": s_index,
                    "kx_index": kx_index,
                    "ky_index": ky_index,
                }
            )
    return AnswerQuery.at_points(fields=GKW_MACRO_FIELDS, points=points)


def bind_gkw_macro_loop(
    *,
    generator: GKWAnswerFirstGenerator,
    base_state: object,
    generator_sha256: str,
    query_factory: MacroQueryFactory | None = None,
    generator_id: str | None = None,
    generator_settings: Mapping[str, object] | None = None,
    matcher: GKWMacroMatcher | None = None,
) -> MacroLoopPlugin:
    """Bind trained GKW artifacts without adding equation-specific protocol."""

    if (
        generator_id is not None
        and generator_id != generator.info.program_id
    ):
        raise ValueError("generator_id must match GeneratorProgram.info")
    resolved_query_factory = query_factory or gkw_macro_point_query
    controller_query_factory = (
        gkw_macro_point_query if query_factory is None else None
    )

    return MacroLoopPlugin(
        plugin_id="gkw",
        generator=LoadedGenerator.from_program(
            program=generator,
            base_state=base_state,
            artifact_sha256=generator_sha256,
            settings=dict(generator_settings or {}),
        ),
        objective=BoundMacroObjective(
            matcher=matcher or GKWMacroMatcher(),
            query_factory=resolved_query_factory,
            controller_query_factory=controller_query_factory,
            agent_visible_metrics=(
                "gkw_macro_match_v1_relative_l2",
                "particle_density_relative_l2",
                "parallel_momentum_relative_l2",
                "kinetic_energy_relative_l2",
                "phi_relative_l2",
            ),
            published_metric_names=(
                "particle_density_relative_l2",
                "parallel_momentum_relative_l2",
                "kinetic_energy_relative_l2",
                "phi_relative_l2",
            ),
        ),
        public_session_factory=generator.public_session_factory,
        public_session_methods=GKW_PUBLIC_SESSION_METHODS,
        development_solution_session_factory=(
            generator.development_solution_session_factory
        ),
        development_solution_session_methods=(
            GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS
        ),
        public_problem_bundle=gkw_public_problem_bundle(),
    )
