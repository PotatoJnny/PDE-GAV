"""Manufactured boundary queries reveal only the declared physical boundary trace."""
from dataclasses import replace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSManufacturedCaseSource
from pde_agent_verifier_solver.plugins.ns.public_problem import NS_PUBLIC_SESSION_METHODS, ns_public_problem_bundle
from pde_agent_verifier_solver.sdk import CaseSource, CaseVisibility, GenerationKind
from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest


class AffineAnswer:
    domain_bounds = (0.0, 2.0, -1.0, 1.0)
    time_interval = (0.0, 1.0)
    viscosity = 0.01
    public_geometry = {"obstacle_polygons": [[[0.5, -0.5], [1.5, -0.5], [1.0, 0.5]]]}
    public_metadata = {"test": "affine_divergence_free"}
    boundary_condition = {
        "kind": "manufactured_velocity_trace",
        "access": "query_boundary_velocity",
        "derivation": "trace_of_hidden_continuous_answer",
    }

    def __init__(self):
        self.velocity_calls = 0

    def evaluate_velocity(self, points):
        self.velocity_calls += 1
        t, x, y = np.asarray(points).T
        return np.column_stack((1 + t + y, 2 - t + x))

    def evaluate_forcing(self, points):
        u, v = self.evaluate_velocity(points).T
        return np.column_stack((1 + v, -1 + u))


class ZeroAnswer(AffineAnswer):
    boundary_condition = None

    def evaluate_velocity(self, points):
        return np.zeros((len(points), 2))

    def evaluate_forcing(self, points):
        return np.zeros((len(points), 2))


def source_for(answer, *, trace=True, perturbation_sampler=None):
    return StructuredNSManufacturedCaseSource(
        generator_id="test-mms-trace", artifact_sha256="a" * 64,
        clean_sampler=lambda seed: answer, perturbation_sampler=perturbation_sampler,
        boundary_velocity_trace=trace,
    )


def session_for(answer, *, trace=True):
    source = source_for(answer, trace=trace)
    record = source.sample(SampleRequest(0, CohortKind.SELECTION, seed=1, count=1))[0]
    return source, record, source.public_session_factory(record.case)


def test_nonzero_trace_matches_one_answer_and_preserves_initial_and_forcing():
    answer = AffineAnswer()
    source, record, session = session_for(answer)
    points = np.array([[0.2, 0.0, 0.3], [0.4, 2.0, -1.0], [0.7, 1.0, -0.5], [0.8, 1.25, 0.0]])
    np.testing.assert_allclose(session.query_boundary_velocity(points), answer.evaluate_velocity(points))
    np.testing.assert_allclose(session.query_initial_velocity([[0.2, 0.3]]), [[1.3, 2.2]])
    np.testing.assert_allclose(session.query_forcing([[0.2, 0.2, 0.3]]), [[3.0, 0.5]])
    assert record.case.public_data["boundary_condition"] == answer.boundary_condition
    assert source.public_session_methods == (*NS_PUBLIC_SESSION_METHODS, "query_boundary_velocity")
    assert tuple(method.name for method in ns_public_problem_bundle(boundary_velocity_trace=True).oracle_methods) == source.public_session_methods
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.query_boundary_velocity(points)


@pytest.mark.parametrize("point", [
    [0.2, 0.2, 0.3],  # fluid interior
    [0.2, 1.0, 0.0],  # solid interior
    [0.2, 1.25 + 1e-9, 0.0],  # close to, but not on, an obstacle edge
    [0.2, -1e-10, 0.2],
    [0.2, 2.1, 0.2],
    [1.1, 0.0, 0.2],
    [float("nan"), 0.0, 0.2],
])
def test_boundary_query_rejects_nonboundary_without_evaluating_answer(point):
    answer = AffineAnswer()
    _, _, session = session_for(answer)
    with pytest.raises(ValueError):
        session.query_boundary_velocity([point])
    assert answer.velocity_calls == 0


def test_edge_hidden_inside_another_obstacle_is_not_a_fluid_boundary():
    answer = AffineAnswer()
    answer.public_geometry = {"obstacle_polygons": [
        [[0.5, -0.5], [1.5, -0.5], [1.0, 0.5]],
        [[1.0, -0.2], [1.8, -0.2], [1.8, 0.2], [1.0, 0.2]],
    ]}
    _, _, session = session_for(answer)
    with pytest.raises(ValueError, match="boundary"):
        session.query_boundary_velocity([[0.2, 1.25, 0.0]])
    assert answer.velocity_calls == 0


def test_legacy_no_slip_default_keeps_case_and_public_methods():
    source, record, session = session_for(ZeroAnswer(), trace=False)
    assert source.public_session_methods == NS_PUBLIC_SESSION_METHODS
    assert tuple(method.name for method in ns_public_problem_bundle().oracle_methods) == NS_PUBLIC_SESSION_METHODS
    assert record.case.public_data["boundary_condition"] == {
        "kind": "no_slip_velocity", "velocity": [0.0, 0.0],
        "derivation": "trace_of_hidden_continuous_answer",
    }
    np.testing.assert_array_equal(session.query_initial_velocity([[0.2, 0.3]]), [[0.0, 0.0]])
    with pytest.raises(PermissionError, match="not enabled"):
        session.query_boundary_velocity([[0.2, 0.0, 0.3]])
    with pytest.raises(ValueError, match="boundary_velocity_trace"):
        session_for(AffineAnswer(), trace=False)


def test_original_registration_preserves_identity_and_checks_physical_contract():
    answer = AffineAnswer()
    source = source_for(answer)
    record = source.issue_answer_first_records(
        [answer], round_index=0, visibility=CaseVisibility.DEVELOPMENT,
        case_id_prefix="original", controller_seed=2,
        distribution_lineage={"kind": "experimental_derived"}, case_source=CaseSource.INITIAL,
    )[0]
    assert record.case.source is CaseSource.INITIAL
    assert record.case.round_index is None
    restored = source_for(answer)
    bound = restored.register_existing_answer(record.case, answer, cohort=CohortKind.DEVELOPMENT)
    assert bound.case == record.case
    assert restored.register_existing_answer(record.case, answer, cohort=CohortKind.DEVELOPMENT).case == record.case
    np.testing.assert_allclose(
        restored.public_session_factory(bound.case).query_boundary_velocity([[0.2, 0.0, 0.3]]),
        [[1.5, 1.8]],
    )
    altered = replace(record.case, public_data={**record.case.public_data, "parameters": {"viscosity": 0.5}})
    with pytest.raises(ValueError, match="contract"):
        source_for(answer).register_existing_answer(altered, answer, cohort=CohortKind.DEVELOPMENT)
    with pytest.raises(ValueError, match="active Case"):
        restored.register_existing_answer(record.case, AffineAnswer(), cohort=CohortKind.DEVELOPMENT)


def test_existing_perturbed_answer_restores_after_its_root_parent():
    answer = AffineAnswer()
    source = source_for(answer, perturbation_sampler=lambda parent, seed: AffineAnswer())
    parent = source.sample(SampleRequest(1, CohortKind.SELECTION, seed=1, count=1))[0]
    child = source.sample(SampleRequest(
        1, CohortKind.SELECTION, seed=2, count=1,
        kind=GenerationKind.PERTURBED, anchors=(parent.case,),
    ))[0]
    restored = source_for(answer)
    with pytest.raises(ValueError, match="active root parent"):
        restored.register_existing_answer(child.case, child.truth._answer, cohort=CohortKind.SELECTION)
    restored.register_existing_answer(parent.case, parent.truth._answer, cohort=CohortKind.SELECTION)
    actual = restored.register_existing_answer(child.case, child.truth._answer, cohort=CohortKind.SELECTION)
    assert actual.case == child.case
    np.testing.assert_allclose(
        restored.public_session_factory(actual.case).query_boundary_velocity([[0.2, 0.0, 0.3]]),
        [[1.5, 1.8]],
    )
