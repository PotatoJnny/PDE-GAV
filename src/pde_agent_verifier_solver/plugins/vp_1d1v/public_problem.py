"""Agent-visible forced 1D1V Vlasov--Poisson resources and oracles."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pde_agent_verifier_solver.sdk import (
    FrozenGuidanceCapability,
    PublicOracleMethod,
    PublicProblemBundle,
    PublicResource,
)

from .contract import VP_1D1V_EQUATION_VERSION
from .problem import (
    VP1D_CACHED_ARRAY_ORACLE_MODE,
    VP1D_EXACT_COORDINATE_ORACLE_MODE,
    VP1D_PUBLIC_ORACLE_MODES,
)


VP1D_OPERATOR_ID = VP_1D1V_EQUATION_VERSION
_RESOURCE_ROOT = Path(__file__).with_name("_public_resources")


def _axis(
    name: str,
    *,
    minimum: str,
    maximum: str,
    values: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "minimum": minimum,
        "maximum": maximum,
    }
    if values is not None:
        result["values"] = values
    return result


def _point_argument(
    *,
    include_time: bool,
    include_velocity: bool,
    native_panel: bool = False,
) -> dict[str, object]:
    axes = []
    if include_time:
        axes.append(
            _axis(
                "t",
                minimum="case.public_data.equation.domain.time_interval[0]",
                maximum="case.public_data.equation.domain.time_interval[1]",
                values=(
                    "case.public_data.arrays.t" if native_panel else None
                ),
            )
        )
    axes.append(
        _axis(
            "x",
            minimum="case.public_data.equation.domain.x_interval[0]",
            maximum="case.public_data.equation.domain.x_interval[1]",
            values="case.public_data.arrays.x" if native_panel else None,
        )
    )
    if include_velocity:
        axes.append(
            _axis(
                "v",
                minimum=(
                    "case.public_data.equation.domain."
                    "velocity_query_window[0]"
                ),
                maximum=(
                    "case.public_data.equation.domain."
                    "velocity_query_window[1]"
                ),
            )
        )
    return {
        "name": "points",
        "dtype": "float64",
        "shape": ["N", len(axes)],
        "coordinate_order": [axis["name"] for axis in axes],
        "axes": axes,
        "rows": (
            "Agent-selected native-panel rows, unordered and repeatable"
            if native_panel
            else "Agent-selected physical rows, unordered and repeatable"
        ),
    }


def _limits(
    oracle_mode: str = VP1D_CACHED_ARRAY_ORACLE_MODE,
) -> dict[str, object]:
    result: dict[str, object] = {
        "max_points_per_request": (
            "case.public_data.equation.oracle_protocol.limits."
            "max_points_per_request"
        ),
        "session_total_point_quota": None,
        "campaign_constraints": [
            "hard_wall_time",
            "concurrency",
            "backend_compute_capacity",
        ],
        "usage_accounting": [
            "query_calls",
            "queried_points",
            "backend_wall_seconds",
        ],
        "accounting_scope": "optional_cached_session_only",
        "direct_materialized_array_access": True,
        "larger_agent_queries_consume_more_point_cost": True,
        "point_cost_is_hard_eligibility_gate": False,
        "candidate_wall_time": (
            "session calls made during prepare_solution are included in "
            "measured preparation wall time"
        ),
    }

    if oracle_mode == VP1D_EXACT_COORDINATE_ORACLE_MODE:
        result.update(
            {
                "accounting_scope": "optional_exact_coordinate_session_only",
                "materialized_arrays_authoritative": False,
                "oracle_responses_derived_from_materialized_arrays": False,
            }
        )
    return result


def _public_oracles(oracle_mode: str) -> tuple[PublicOracleMethod, ...]:
    exact = oracle_mode == VP1D_EXACT_COORDINATE_ORACLE_MODE
    return (
        PublicOracleMethod(
            name="query_initial_distribution",
            description=(
                "Evaluate the initial distribution directly from the frozen "
                "continuous answer at Agent-selected periodic-x and bounded-"
                "velocity coordinates; materialized arrays are not used for "
                "interpolation."
                if exact
                else "Interpolate the public initial distribution at Agent-"
                "selected periodic-x and bounded-velocity coordinates."
            ),
            input_schema={
                "arguments": [
                    _point_argument(
                        include_time=False,
                        include_velocity=True,
                    )
                ],
                "fixed_scoring_panel": False,
            },
            output_schema={
                "field": "f0",
                "dtype": "float64",
                "shape": ["N"],
                "coordinate_order": "matches submitted rows",
            },
            cost_unit=(
                "exact_coordinate_evaluation_point"
                if exact
                else "cached_interpolation_point"
            ),
            limits=_limits(oracle_mode),
        ),
        PublicOracleMethod(
            name="query_source",
            description=(
                "Evaluate the answer-first kinetic source directly from exact "
                "coordinate jets of the same frozen continuous answer at "
                "Agent-selected (t,x,v) coordinates; materialized arrays are "
                "not used for interpolation."
                if exact
                else "Interpolate the answer-first kinetic source at Agent-"
                "selected (t,x,v) coordinates."
            ),
            input_schema={
                "arguments": [
                    _point_argument(
                        include_time=True,
                        include_velocity=True,
                    )
                ],
                "fixed_scoring_panel": False,
            },
            output_schema={
                "field": "source",
                "dtype": "float64",
                "shape": ["N"],
                "coordinate_order": "matches submitted rows",
            },
            cost_unit=(
                "exact_coordinate_evaluation_point"
                if exact
                else "cached_interpolation_point"
            ),
            limits=_limits(oracle_mode),
        ),
    )


def _development_solution_oracles() -> tuple[PublicOracleMethod, ...]:
    return (
        PublicOracleMethod(
            name="query_completed_macros",
            description=(
                "Read M0, M1, M2, M3, and E from the retained completed "
                "solution at an Agent-selected subset of the public native "
                "(t,x) panel. Available for DEVELOPMENT cases only."
            ),
            input_schema={
                "arguments": [
                    _point_argument(
                        include_time=True,
                        include_velocity=False,
                        native_panel=True,
                    )
                ],
                "fixed_scoring_panel": False,
                "case_visibility": "development_only",
                "coordinate_membership": (
                    "each t and x must exactly equal a public native-axis value"
                ),
            },
            output_schema={
                "fields": ["M0", "M1", "M2", "M3", "E"],
                "dtype": "float64",
                "shape": ["N", 5],
                "component_order": ["M0", "M1", "M2", "M3", "E"],
                "source": "retained_completed_answer",
                "clean_and_perturbed_supported": True,
                "parameterization_exposed": False,
            },
            cost_unit="cached_macro_point",
            limits=_limits(),
        ),
    )


def vp1d_public_problem_bundle(
    *,
    oracle_mode: str = VP1D_CACHED_ARRAY_ORACLE_MODE,
    guidance_supported: bool = False,
) -> PublicProblemBundle:
    """Return the method-neutral VP problem for the selected oracle contract."""

    if not isinstance(oracle_mode, str) or oracle_mode not in VP1D_PUBLIC_ORACLE_MODES:
        raise ValueError(
            "VP1D oracle mode must be one of "
            f"{sorted(VP1D_PUBLIC_ORACLE_MODES)}, got {oracle_mode!r}"
        )
    if not isinstance(guidance_supported, bool):
        raise TypeError("VP1D guidance_supported must be bool")
    return _vp1d_public_problem_bundle(oracle_mode, guidance_supported)


@lru_cache(maxsize=4)
def _vp1d_public_problem_bundle(
    oracle_mode: str,
    guidance_supported: bool,
) -> PublicProblemBundle:
    exact = oracle_mode == VP1D_EXACT_COORDINATE_ORACLE_MODE

    resources = (
        PublicResource(
            relative_path="vp_1d1v_reference/README.md",
            data=(_RESOURCE_ROOT / "README.md").read_bytes(),
            media_type="text/markdown",
            description="Equation, source, boundary, and scoring contract.",
        ),
        PublicResource(
            relative_path="vp_1d1v_reference/vp_operator.py",
            data=(_RESOURCE_ROOT / "vp_operator.py").read_bytes(),
            media_type="text/x-python",
            description="Reference implementation of the declared VP residual.",
        ),
    )
    return PublicProblemBundle(
        bundle_id=(
            "vp-1d1v-forced-periodic-exact-coordinate-public-problem-v1"
            if exact
            else "vp-1d1v-forced-periodic-public-problem-v1"
        ),
        equation_id=VP1D_OPERATOR_ID,
        description=(
            "Forced periodic 1D1V Vlasov--Poisson with public normalization, "
            "initial distribution, exactly derived source, "
            + (
                "direct frozen-answer coordinate evaluation with non-authoritative "
                "convenience arrays,"
                if exact
                else "cached numerical access,"
            )
            + " and sealed macroscopic point scoring."
        ),
        resources=resources,
        oracle_methods=_public_oracles(oracle_mode),
        development_solution_oracle_methods=_development_solution_oracles(),
        guidance_capability=FrozenGuidanceCapability(
            supported=guidance_supported
        ),
    )


VP1D_PUBLIC_SESSION_METHODS = tuple(
    method.name for method in vp1d_public_problem_bundle().oracle_methods
)
VP1D_DEVELOPMENT_SOLUTION_SESSION_METHODS = tuple(
    method.name
    for method in (
        vp1d_public_problem_bundle().development_solution_oracle_methods
    )
)


__all__ = [
    "VP1D_CACHED_ARRAY_ORACLE_MODE",
    "VP1D_DEVELOPMENT_SOLUTION_SESSION_METHODS",
    "VP1D_EXACT_COORDINATE_ORACLE_MODE",
    "VP1D_OPERATOR_ID",
    "VP1D_PUBLIC_SESSION_METHODS",
    "vp1d_public_problem_bundle",
]
