"""Agent-visible GKW equation resources and numerical-oracle declaration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pde_agent_verifier_solver.sdk import (
    PublicOracleMethod,
    PublicProblemBundle,
    PublicResource,
)

from .public_contract import GKW_OPERATOR_ID


_RESOURCE_ROOT = Path(__file__).with_name("_public_resources")
_RESOURCE_DESCRIPTIONS = {
    "__init__.py": "Import marker for the public GKW reference package.",
    "gkw_equation_contract.py": (
        "Dataset-exact equation, grid, coefficient, and topology contract."
    ),
    "gkw_public_packet.py": (
        "Canonical loader and validator for public GKW coefficient packets."
    ),
    "gkw_torch_precompute.py": (
        "Differentiable public coefficient precompute from a case packet."
    ),
    "gkw_torch_operator.py": (
        "Differentiable PyTorch implementation of the declared GKW operator."
    ),
}


def _phase_axes() -> list[dict[str, object]]:
    return [
        {
            "name": "v_parallel_index",
            "kind": "integer_index",
            "minimum": 0,
            "maximum": 31,
            "physical_values": "case.public_data.arrays.vpar",
        },
        {
            "name": "mu_index",
            "kind": "integer_index",
            "minimum": 0,
            "maximum": 7,
            "physical_values": "case.public_data.arrays.mu",
        },
        {
            "name": "s_index",
            "kind": "integer_index",
            "minimum": 0,
            "maximum": 15,
            "physical_values": "case.public_data.arrays.s",
        },
        {
            "name": "kx_index",
            "kind": "integer_index",
            "minimum": 0,
            "maximum": 84,
            "physical_values": "case.public_data.arrays.kx",
        },
        {
            "name": "ky_index",
            "kind": "integer_index",
            "minimum": 0,
            "maximum": 31,
            "physical_values": "case.public_data.arrays.ky",
        },
    ]


def _field_axes() -> list[dict[str, object]]:
    phase = _phase_axes()
    return [phase[2], phase[3], phase[4]]


def _row_limits(*, source: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "max_rows_per_request": (
            "case.public_data.oracle_protocol.limits.max_rows_per_request"
        ),
        "session_total_row_quota": None,
        "campaign_constraints": [
            "hard_wall_time",
            "concurrency",
            "backend_compute_capacity",
        ],
        "usage_accounting": [
            "query_calls",
            "queried_rows",
            "backend_wall_seconds",
        ],
        "legacy_total_limits_are_not_protocol_budgets": True,
        "larger_or_finer_agent_queries_consume_more_row_cost": True,
    }
    if source:
        result["agent_selected_source_times"] = True
        result["max_unique_source_times_per_request"] = (
            "case.public_data.oracle_protocol.limits.max_unique_source_times"
        )
        result["legacy_session_limits_roll_over_transparently"] = True
    return result


def _oracle_methods() -> tuple[PublicOracleMethod, ...]:
    complex_row_output = {
        "dtype": "float64",
        "shape": ["N", 2],
        "component_order": ["real", "imag"],
        "complex_encoding": "real_imag_pair_on_final_axis",
        "coordinate_order": "matches submitted rows",
    }
    phase_indices = {
        "name": "phase_indices",
        "dtype": "int64",
        "shape": ["N", 5],
        "axes": _phase_axes(),
        "rows": "Agent-selected, unordered and repeatable",
    }
    field_indices = {
        "name": "field_indices",
        "dtype": "int64",
        "shape": ["N", 3],
        "axes": _field_axes(),
        "rows": "Agent-selected, unordered and repeatable",
    }
    return (
        PublicOracleMethod(
            name="query_source",
            description=(
                "Evaluate the answer-first source at Agent-selected times and "
                "phase-space rows on the public semidiscrete operator grid."
            ),
            input_schema={
                "batch_axis": "N",
                "arguments": [
                    {
                        "name": "times",
                        "dtype": "float64",
                        "shape": ["N"],
                        "accepted_shapes": ["scalar", "[N]"],
                        "scalar_broadcast": (
                            "broadcast to the N submitted phase_indices rows"
                        ),
                        "domain": {
                            "minimum": "case.public_data.metadata.time_nodes[0]",
                            "maximum": "case.public_data.metadata.time_nodes[-1]",
                        },
                        "values": "Agent-selected",
                    },
                    phase_indices,
                ],
                "fixed_scoring_panel": False,
            },
            output_schema={
                "field": "source",
                **complex_row_output,
            },
            cost_unit="queried_phase_space_row",
            limits=_row_limits(source=True),
        ),
        PublicOracleMethod(
            name="query_initial_h",
            description=(
                "Evaluate the initial ion state at Agent-selected phase-space "
                "rows on the public semidiscrete operator grid."
            ),
            input_schema={
                "batch_axis": "N",
                "arguments": [phase_indices],
                "fixed_scoring_panel": False,
            },
            output_schema={
                "field": "initial_h",
                **complex_row_output,
            },
            cost_unit="queried_phase_space_row",
            limits=_row_limits(source=False),
        ),
        PublicOracleMethod(
            name="query_initial_phi",
            description=(
                "Evaluate the initial electrostatic potential at Agent-selected "
                "field rows on the public semidiscrete operator grid."
            ),
            input_schema={
                "batch_axis": "N",
                "arguments": [field_indices],
                "fixed_scoring_panel": False,
            },
            output_schema={
                "field": "initial_phi",
                **complex_row_output,
            },
            cost_unit="queried_field_row",
            limits=_row_limits(source=False),
        ),
    )


def _development_solution_oracle_methods() -> tuple[PublicOracleMethod, ...]:
    return (
        PublicOracleMethod(
            name="query_completed_macros",
            description=(
                "Evaluate all four completed-solution macroscopic fields from "
                "the retained manufactured answer at Agent-selected times and "
                "field rows. Available for DEVELOPMENT cases only."
            ),
            input_schema={
                "arguments": [
                    {
                        "name": "times",
                        "dtype": "float64",
                        "shape": ["T"],
                        "domain": {
                            "minimum": "case.public_data.metadata.time_nodes[0]",
                            "maximum": "case.public_data.metadata.time_nodes[-1]",
                        },
                        "values": "Agent-selected",
                    },
                    {
                        "name": "field_indices",
                        "dtype": "int64",
                        "shape": ["N", 3],
                        "axes": _field_axes(),
                        "rows": "Agent-selected, unordered and repeatable",
                    },
                ],
                "point_set": "cartesian_product(times, field_indices)",
                "fixed_scoring_panel": False,
                "case_visibility": "development_only",
            },
            output_schema={
                "fields": [
                    "particle_density",
                    "parallel_momentum",
                    "kinetic_energy",
                    "phi",
                ],
                "dtype": "float64_real_imag_pairs",
                "shape_per_field": ["T", "N", 2],
                "source": "retained_hidden_manufactured_answer",
                "clean_and_perturbed_supported": True,
                "generator_or_latent_exposed": False,
            },
            cost_unit="evaluated_time_field_point",
            limits={
                "max_points_per_request": (
                    "case.public_data.oracle_protocol.limits."
                    "max_rows_per_request"
                ),
                "session_total_point_quota": None,
                "campaign_constraints": [
                    "hard_wall_time",
                    "concurrency",
                    "backend_compute_capacity",
                ],
                "actual_points": "len(times) * len(field_indices)",
                "usage_accounting": [
                    "query_calls",
                    "evaluated_points",
                    "backend_wall_seconds",
                ],
                "legacy_max_total_rows_is_not_a_protocol_budget": True,
            },
        ),
    )


@lru_cache(maxsize=1)
def gkw_public_problem_bundle() -> PublicProblemBundle:
    """Return the complete public GKW problem, excluding Generator internals."""

    resources = tuple(
        PublicResource(
            relative_path=f"gkw_reference/{name}",
            data=(_RESOURCE_ROOT / name).read_bytes(),
            media_type="text/x-python",
            description=description,
        )
        for name, description in _RESOURCE_DESCRIPTIONS.items()
    )
    return PublicProblemBundle(
        bundle_id="fusion-gkw-public-problem-v1",
        equation_id=GKW_OPERATOR_ID,
        description=(
            "Full public collisionless electrostatic adiabatic-electron GKW "
            "operator, coefficient conventions, reference implementation, and "
            "Agent-selected numerical source/initial-data access. Candidate "
            "internal grids and bases remain entirely Agent-selected."
        ),
        resources=resources,
        oracle_methods=_oracle_methods(),
        development_solution_oracle_methods=(
            _development_solution_oracle_methods()
        ),
    )


GKW_PUBLIC_SESSION_METHODS = tuple(
    method.name for method in gkw_public_problem_bundle().oracle_methods
)
GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS = tuple(
    method.name
    for method in (
        gkw_public_problem_bundle().development_solution_oracle_methods
    )
)


__all__ = [
    "GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS",
    "GKW_PUBLIC_SESSION_METHODS",
    "gkw_public_problem_bundle",
]
