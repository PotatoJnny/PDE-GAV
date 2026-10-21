"""Public reference for the finite original-numerical NS baseline bank."""
from dataclasses import replace

from pde_agent_verifier_solver.sdk import PublicResource
from ..public_problem import ns_public_problem_bundle

_NUMERICAL_README = """# Original numerical Navier--Stokes Cases

The shared ns_operator.py contains the public incompressible NS operator.
Each Case describes its actual equation, geometry and time interval in
case.public_data["question"], numerical parameters in ["case_parameters"], and
available finite public inputs in ["arrays"]. Inspect those actual fields;
do not assume every dataset supplies the same forcing or initial-data arrays.
The regular-grid coordinate axes are arrays["x"] and arrays["y"], with a
boolean arrays["fluid_mask"]. Boundary and forcing specifications come from
the Case, not from a generated manufactured-answer contract.

Write prepare_solution(case, solve_budget=None), returning a handle with
evaluate(points_query) -> AnswerView. Return velocity_x and velocity_y at
the requested physical (t,x,y) coordinates. Preparation receives no private
scoring coordinates. Use a physically appropriate solver and interpolate
its completed solution when queried.

These are fixed original numerical Cases. This baseline has no online
Generator, query_forcing/query_initial_velocity session, or completed-solution
session. Development evaluation is available through the same Evaluator RPC;
finite truth disclosure follows the configured point/grid truth permissions.
Hidden validation coordinates and answers remain unavailable to the Agent.
"""


def numerical_ns_public_problem_bundle():
    """Reuse the public operator while advertising only implemented interfaces."""
    original = ns_public_problem_bundle(guidance_supported=False)
    resources = tuple(
        resource for resource in original.resources
        if resource.relative_path != "ns_reference/README.md"
    )
    return replace(
        original,
        bundle_id="ns-original-numerical-baseline-public-problem-v1",
        description=(
            "Original numerical NS Cases with public question, case_parameters "
            "and finite arrays; fixed-data development evaluation, no online oracles."
        ),
        resources=(PublicResource(
            relative_path="ns_reference/README.md",
            data=_NUMERICAL_README.encode(),
            media_type="text/markdown",
            description="Original numerical Case fields, solver ABI and actual feedback access.",
        ), *resources),
        oracle_methods=(),
        development_solution_oracle_methods=(),
    )
