"""Agent-visible NS equation resources and numerical-oracle declaration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pde_agent_verifier_solver.sdk import (
    FrozenGuidanceCapability,
    PublicOracleMethod,
    PublicProblemBundle,
    PublicResource,
)


NS_OPERATOR_ID = "incompressible_navier_stokes_2d_forced_v1"
_RESOURCE_ROOT = Path(__file__).with_name("_public_resources")


def _point_argument(*, include_time: bool) -> dict[str, object]:
    axes = []
    if include_time:
        axes.append(
            {
                "name": "t",
                "minimum": "case.public_data.domain.time_interval[0]",
                "maximum": "case.public_data.domain.time_interval[1]",
            }
        )
    axes.extend(
        (
            {
                "name": "x",
                "minimum": "case.public_data.domain.bounds.x[0]",
                "maximum": "case.public_data.domain.bounds.x[1]",
            },
            {
                "name": "y",
                "minimum": "case.public_data.domain.bounds.y[0]",
                "maximum": "case.public_data.domain.bounds.y[1]",
            },
        )
    )
    return {
        "name": "points",
        "dtype": "float64",
        "shape": ["N", len(axes)],
        "coordinate_order": [axis["name"] for axis in axes],
        "axes": axes,
        "rows": "Agent-selected, unordered and repeatable",
    }


def _limits() -> dict[str, object]:
    return {
        "max_points_per_request": (
            "case.public_data.oracle_protocol.limits.max_points_per_request"
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
        "larger_agent_queries_consume_more_point_cost": True,
    }


def _public_oracles(*, boundary_velocity_trace: bool = False) -> tuple[PublicOracleMethod, ...]:
    vector_output = {
        "dtype": "float64",
        "shape": ["N", 2],
        "component_order": ["x", "y"],
        "coordinate_order": "matches submitted rows",
    }
    methods = (
        PublicOracleMethod(
            name="query_forcing",
            description=(
                "Evaluate the answer-first body force at Agent-selected "
                "physical (t,x,y) points."
            ),
            input_schema={
                "arguments": [_point_argument(include_time=True)],
                "fixed_scoring_panel": False,
            },
            output_schema={"field": "forcing", **vector_output},
            cost_unit="evaluated_physical_point",
            limits=_limits(),
        ),
        PublicOracleMethod(
            name="query_initial_velocity",
            description=(
                "Evaluate both initial velocity components at Agent-selected "
                "physical (x,y) points."
            ),
            input_schema={
                "arguments": [_point_argument(include_time=False)],
                "fixed_scoring_panel": False,
            },
            output_schema={"field": "initial_velocity", **vector_output},
            cost_unit="evaluated_physical_point",
            limits=_limits(),
        ),
    )

    if boundary_velocity_trace:
        methods += (
            PublicOracleMethod(
                name="query_boundary_velocity",
                description=(
                    "Evaluate the manufactured velocity trace at physical (t,x,y) "
                    "points on the declared rectangle or obstacle-polygon boundaries. "
                    "Interior points are rejected."
                ),
                input_schema={
                    "arguments": [_point_argument(include_time=True)],
                    "boundary_only": True,
                    "fixed_scoring_panel": False,
                },
                output_schema={"field": "boundary_velocity", **vector_output},
                cost_unit="evaluated_physical_point",
                limits=_limits(),
            ),
        )
    return methods


def _development_solution_oracles() -> tuple[PublicOracleMethod, ...]:
    return (
        PublicOracleMethod(
            name="query_completed_velocity",
            description=(
                "Evaluate the retained completed manufactured velocity at "
                "Agent-selected physical (t,x,y) points. Available only for "
                "DEVELOPMENT cases, including smooth perturbed children."
            ),
            input_schema={
                "arguments": [_point_argument(include_time=True)],
                "fixed_scoring_panel": False,
                "case_visibility": "development_only",
            },
            output_schema={
                "fields": ["velocity_x", "velocity_y"],
                "dtype": "float64",
                "shape": ["N", 2],
                "component_order": ["x", "y"],
                "source": "retained_hidden_manufactured_answer",
                "clean_and_perturbed_supported": True,
                "generator_or_latent_exposed": False,
            },
            cost_unit="evaluated_physical_point",
            limits=_limits(),
        ),
    )


@lru_cache(maxsize=4)
def ns_public_problem_bundle(
    guidance_supported: bool = False, *, boundary_velocity_trace: bool = False,
) -> PublicProblemBundle:
    if not isinstance(boundary_velocity_trace, bool):
        raise TypeError("NS boundary_velocity_trace must be a boolean")
    resources = (
        PublicResource(
            relative_path="ns_reference/README.md",
            data=(_RESOURCE_ROOT / "README.md").read_bytes(),
            media_type="text/markdown",
            description="Equation, boundary, answer-first, and scoring contract.",
        ),
        PublicResource(
            relative_path="ns_reference/AGENT_TOOLS.txt",
            data=(_RESOURCE_ROOT / "AGENT_TOOLS.txt").read_bytes(),
            media_type="text/plain",
            description="Case tools, solver interface, evaluation feedback and research records.",
        ),
        PublicResource(
            relative_path="ns_reference/SOLVER_INTERFACE.md",
            data=(_RESOURCE_ROOT / "SOLVER_INTERFACE.md").read_bytes(),
            media_type="text/markdown",
            description="Solver input/output interface, portable bundle and executable command examples.",
        ),
        PublicResource(
            relative_path="ns_reference/ns_operator.py",
            data=(_RESOURCE_ROOT / "ns_operator.py").read_bytes(),
            media_type="text/x-python",
            description="Reference implementation of the public NS operator.",
        ),
    )
    return PublicProblemBundle(
        bundle_id=(
            "navier-stokes-continuous-trace-public-problem-v1"
            if boundary_velocity_trace else "navier-stokes-continuous-public-problem-v1"
        ),
        equation_id=NS_OPERATOR_ID,
        description=(
            "Forced 2D incompressible Navier--Stokes with public operator, "
            "geometry, viscosity, "
            + (
                "manufactured velocity boundary trace"
                if boundary_velocity_trace else "no-slip boundary contract"
            )
            + ", and Agent-selected forcing/initial-data numerical access."
        ),
        resources=resources,
        oracle_methods=_public_oracles(boundary_velocity_trace=boundary_velocity_trace),
        development_solution_oracle_methods=(
            _development_solution_oracles()
        ),
        guidance_capability=FrozenGuidanceCapability(
            supported=guidance_supported
        ),
    )


NS_PUBLIC_SESSION_METHODS = tuple(
    method.name for method in ns_public_problem_bundle().oracle_methods
)
NS_DEVELOPMENT_SOLUTION_SESSION_METHODS = tuple(
    method.name
    for method in ns_public_problem_bundle().development_solution_oracle_methods
)


__all__ = [
    "NS_DEVELOPMENT_SOLUTION_SESSION_METHODS",
    "NS_OPERATOR_ID",
    "NS_PUBLIC_SESSION_METHODS",
    "ns_public_problem_bundle",
]
