"""Agent-visible descriptions distinguish numerical queries from preview grids."""

from pde_agent_verifier_solver.plugins.vpfp.public_problem import (
    VPFP_PUBLIC_SESSION_METHODS, vpfp_public_problem_bundle,
)


def test_public_oracle_contract_declares_demand_and_batch_semantics():
    bundle = vpfp_public_problem_bundle()
    assert tuple(item.name for item in bundle.oracle_methods) == VPFP_PUBLIC_SESSION_METHODS
    for method in bundle.oracle_methods:
        limits = method.manifest()["limits"]
        assert limits["evaluation_mode"] == "continuous_on_demand"
        assert limits["unrequested_cartesian_completion"] is False
        assert limits["preserves_query_order_and_duplicates"] is True
        assert limits["agent_may_partition_requests"] is True
        assert limits["max_points_per_request"] == 262144
        assert limits["session_total_point_quota"] is None
        assert {"attempted_calls", "requested_points", "completed_points", "failed_calls"} <= set(limits["usage_accounting"])
        assert method.manifest()["input_schema"]["fixed_scoring_panel"] is False


def test_agent_resource_explains_both_canonical_interfaces_without_grid_requirement():
    text = vpfp_public_problem_bundle().resource("vpfp_reference/README.md").data.decode()
    assert "source` and `g0` arrays are deliberately absent" in text
    assert "ControllerAgentCapabilityClient.from_request" in text
    assert 'client.invoke_public_problem(case.case_id, "source", points)' in text
    assert "open_public_case_session(case).source(points)" in text
    assert "Official or Terminal" in text
    assert "no full phase-space answer" in text
