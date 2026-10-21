"""Agent-visible VPFP operator resources and numerical-Oracle contract."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pde_agent_verifier_solver.sdk import (
    PublicOracleMethod,
    PublicProblemBundle,
    PublicResource,
)


VPFP_OPERATOR_ID = "vpfp_ou_poisson_torus_rv_v2"
VPFP_MAX_ORACLE_POINTS_PER_CALL = 262_144
_PUBLIC_RUNTIME = (
    Path(__file__).with_name("_public_resources")
    / "public_vpfp_runtime.py"
)


def _point_argument(
    name: str,
    coordinates: tuple[str, ...],
) -> dict[str, object]:
    return {
        "name": name,
        "dtype": "float64",
        "shape": ["N", len(coordinates)],
        "coordinate_order": list(coordinates),
        "rows": "Agent-selected, unordered and repeatable",
    }


def _limits() -> dict[str, object]:
    return {
        "evaluation_mode": "continuous_on_demand",
        "unrequested_cartesian_completion": False,
        "preserves_query_order_and_duplicates": True,
        "agent_may_partition_requests": True,
        "consistency": "same_continuous_case_within_backend_floating_point_precision",
        "max_points_per_request": VPFP_MAX_ORACLE_POINTS_PER_CALL,
        "session_total_point_quota": None,
        "campaign_constraints": [
            "hard_wall_time",
            "concurrency",
            "backend_compute_capacity",
        ],
        "usage_accounting": [
            "query_calls",
            "evaluated_points",
            "backend_wall_seconds",
            "attempted_calls",
            "requested_points",
            "completed_points",
            "failed_calls",
        ],
        "larger_or_finer_agent_queries_consume_more_point_cost": True,
    }


def _equation_oracles() -> tuple[PublicOracleMethod, ...]:
    scalar = {
        "dtype": "float64",
        "shape": ["N"],
        "coordinate_order": "matches submitted rows",
    }
    return (
        PublicOracleMethod(
            name="source",
            description=(
                "Evaluate the answer-first kinetic source at Agent-selected "
                "physical (t,x,y,vx,vy) points."
            ),
            input_schema={
                "arguments": [
                    _point_argument("points", ("t", "x", "y", "vx", "vy"))
                ],
                "fixed_scoring_panel": False,
            },
            output_schema={"field": "source", **scalar},
            cost_unit="evaluated_phase_space_point",
            limits=_limits(),
        ),
        PublicOracleMethod(
            name="rho0",
            description=(
                "Evaluate the public initial density at Agent-selected torus "
                "coordinates (x,y)."
            ),
            input_schema={
                "arguments": [_point_argument("points_xy", ("x", "y"))],
                "fixed_scoring_panel": False,
            },
            output_schema={"field": "rho0", **scalar},
            cost_unit="evaluated_spatial_point",
            limits=_limits(),
        ),
        PublicOracleMethod(
            name="g0",
            description=(
                "Evaluate the normalized public initial velocity density at "
                "Agent-selected (x,y,vx,vy) points."
            ),
            input_schema={
                "arguments": [
                    _point_argument("points_xyv", ("x", "y", "vx", "vy"))
                ],
                "fixed_scoring_panel": False,
            },
            output_schema={"field": "g0", **scalar},
            cost_unit="evaluated_phase_space_point",
            limits=_limits(),
        ),
    )


def _development_solution_oracles() -> tuple[PublicOracleMethod, ...]:
    return (
        PublicOracleMethod(
            name="released_macros",
            description=(
                "Evaluate the retained completed answer's exact full-velocity "
                "rho, momentum and kinetic energy at Agent-selected (t,x,y) "
                "points. Available for DEVELOPMENT cases only."
            ),
            input_schema={
                "arguments": [
                    _point_argument("points_txy", ("t", "x", "y"))
                ],
                "fixed_scoring_panel": False,
                "case_visibility": "development_only",
            },
            output_schema={
                "fields": [
                    "rho",
                    "momentum",
                    "kinetic_energy",
                    "raw_second_moment",
                ],
                "dtype": "float64",
                "leading_shape": ["N"],
                "shapes": {
                    "rho": ["N"],
                    "momentum": ["N", 2],
                    "kinetic_energy": ["N"],
                    "raw_second_moment": ["N", 3],
                },
                "source": "retained_hidden_manufactured_answer",
                "generator_or_latent_exposed": False,
            },
            cost_unit="evaluated_spacetime_point",
            limits=_limits(),
        ),
        PublicOracleMethod(
            name="released_transport_macros",
            description=("Evaluate density, particle flux, self-consistent electric field, "
                         "central pressure and central heat flux together. DEVELOPMENT only."),
            input_schema={"arguments": [_point_argument("points_txy", ("t", "x", "y"))],
                          "fixed_scoring_panel": False, "case_visibility": "development_only"},
            output_schema={
                "fields": ["rho", "momentum", "kinetic_energy", "raw_second_moment",
                           "electric", "pressure", "heat_flux"],
                "dtype": "float64", "leading_shape": ["N"],
                "shapes": {"rho": ["N"], "momentum": ["N", 2], "kinetic_energy": ["N"],
                           "raw_second_moment": ["N", 3], "electric": ["N", 2],
                           "pressure": ["N", 3], "heat_flux": ["N", 2]},
                "symmetric_tensor_order": ["xx", "xy", "yy"],
                "heat_flux_definition": "0.5 integral |v-u|^2 (v-u) f dv",
                "source": "retained_hidden_manufactured_answer",
                "generator_or_latent_exposed": False,
            },
            cost_unit="evaluated_spacetime_point", limits=_limits(),
        ),
    )


@lru_cache(maxsize=1)
def vpfp_public_problem_bundle() -> PublicProblemBundle:
    """Return the frozen public equation description for every VPFP regime."""

    if not _PUBLIC_RUNTIME.is_file() or _PUBLIC_RUNTIME.is_symlink():
        raise FileNotFoundError(_PUBLIC_RUNTIME)
    readme = b"""# Continuous 2D2V VPFP public problem

The equation is posed on a periodic two-dimensional spatial torus and full
velocity space. With `rho = integral f dv`, `-Delta phi = rho-rho_bg`, and
`E = -grad phi`, the public source belongs to

`d_t f + v.grad_x f + E.grad_v f - nu div_v(v f + theta grad_v f) = source`.

Each public Case declares `nu`, `theta`, the domain, conventions, and finite
numerical views. Those arrays are samples, not the hidden answer
parameterization. Use the numerical Oracle for fresh values on any finite
POINTS set or mesh you choose. Velocity truncation, grids, basis, time
integrator, adaptivity and interpolation are solver-owned choices. The
completed-solution Oracle is DEVELOPMENT-only and returns exact full-velocity
macroscopic moments, never Generator weights or latents.

In the resident online runtime, arrays contain coordinate axes and a small
initial-macro preview. `source` and `g0` arrays are deliberately absent. Do not
interpret the preview axes as a required mesh, velocity cutoff, or scoring
grid. Construct the actual points your solver needs and query `source`,
`rho0`, or `g0`; source/g0 are not computed by publishing or listing a Case.
Requests may be split into batches (at most 262144 points per request), with
arbitrary order and repeated rows. The Oracle returns one value per submitted
row and does not fill unrequested Cartesian combinations. Internal batching
can change final floating-point bits, not the continuous problem definition.

During a canonical external Agent turn, use the existing Controller client:

```python
from pde_agent_verifier_solver.engine.agent_execution import load_agent_worker_request
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient

client = ControllerAgentCapabilityClient.from_request(load_agent_worker_request())
case = client.current_cases()[0]  # opaque id issued in this Agent turn
values = client.invoke_public_problem(case.case_id, "source", points)
```

Inside a submitted candidate, use the scoped SDK instead:

```python
from pde_agent_verifier_solver.sdk import open_public_case_session
values = open_public_case_session(case).source(points)
```

The numerical source queries are independent of the verifier's hidden scoring
coordinates. For the `observable_model` Task, submit a macro-only continuous
POINTS-capable program through `prepare_solution` / `evaluate`; no full phase-space answer
is required. The completed-solution namespace additionally requires explicit
Development authorization and is never available in Official or Terminal.
For plasma_macro_transport_v2, the Task requests rho, electric_x, electric_y,
momentum_x, momentum_y, pressure_xx, pressure_xy, pressure_yy, heat_flux_x,
heat_flux_y. Pressure is central and heat flux includes the one-half factor.
The DEVELOPMENT-only released_transport_macros method obtains all of these
in one batch, with vector arrays and pressure packed in (xx,xy,yy) order.
The Task brief supplies the frozen common scoring scales and objective hash.
The included public_vpfp_runtime.py retains its legacy four-field output
helper and public initial-data checks for reproducibility; that helper is NOT
the plasma_macro_transport_v2 scorer. The selected Task brief and requested
query.fields determine the current ten-field output ABI, even if legacy
question metadata retains vpfp_continuous_macro_output_v1 for old helpers.
"""
    return PublicProblemBundle(
        bundle_id="vpfp-continuous-public-problem-v1",
        equation_id=VPFP_OPERATOR_ID,
        description=(
            "Continuous 2D2V Vlasov--Poisson--Fokker--Planck with a public "
            "operator and Agent-selected source/initial-data numerical access."
        ),
        resources=(
            PublicResource(
                relative_path="vpfp_reference/README.md",
                data=readme,
                media_type="text/markdown",
                description="Equation, domain, answer-first and Oracle contract.",
            ),
            PublicResource(
                relative_path="vpfp_reference/public_vpfp_runtime.py",
                data=_PUBLIC_RUNTIME.read_bytes(),
                media_type="text/x-python",
                description=(
                    "Public NumPy VPFP grid, Poisson, residual and output helpers."
                ),
            ),
        ),
        oracle_methods=_equation_oracles(),
        development_solution_oracle_methods=(
            _development_solution_oracles()
        ),
    )


VPFP_PUBLIC_SESSION_METHODS = tuple(
    method.name for method in vpfp_public_problem_bundle().oracle_methods
)
VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS = tuple(
    method.name
    for method in vpfp_public_problem_bundle().development_solution_oracle_methods
)


__all__ = [
    "VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS",
    "VPFP_MAX_ORACLE_POINTS_PER_CALL",
    "VPFP_OPERATOR_ID",
    "VPFP_PUBLIC_SESSION_METHODS",
    "vpfp_public_problem_bundle",
]
