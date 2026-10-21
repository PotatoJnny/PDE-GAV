"""Point-count parity without changing Artemis physics or native grid size."""
from dataclasses import replace

import pytest

from pde_agent_verifier_solver.plugins.vp_1d1v.task import (
    ArtemisObservableTask, _native_query, _private_query,
)
from pde_agent_verifier_solver.plugins.vp_1d1v.problem import causal_horizon_case
from pde_agent_verifier_solver.sdk import EvaluationPhase
from tests.test_artemis_canonical import _binding, _query, _request


@pytest.fixture
def case():
    binding, _ = _binding()
    state = binding.plugin.generator.load(binding.checkpoint, {})
    return binding.plugin.generator.sample(state, _request())[0].case


@pytest.mark.parametrize("count", [128, 256])
def test_continuous_scoring_exact_count_and_causal_domain(case, count):
    query = _private_query(case, count, 101, "continuous_x_v1")
    assert len(query.points) == count
    assert len(set(p["x"] for p in query.points)) == count
    assert len(set(p["t"] for p in query.points)) == 1
    assert all(0 <= p["x"] < 1 for p in query.points)
    assert all(p["t"] in tuple(case.public_data["arrays"]["t"])[1:] for p in query.points)
    assert any(p["x"] not in case.public_data["arrays"]["x"] for p in query.points)
    projected = causal_horizon_case(case, query, scoring_coordinate_mode="continuous_x_v1")
    assert projected.public_data["equation"]["domain"]["time_interval"][1] == query.points[0]["t"]
    assert len(case.public_data["arrays"]["x"]) == 4


def test_legacy_query_and_parent_pairing_remain_deterministic(case):
    assert _private_query(case, 4, 99) == _native_query(case, 4, 99)
    child = replace(case, case_id="child", parent_case_id=case.case_id)
    assert _private_query(case, 256, 99, "continuous_x_v1") == _private_query(
        child, 256, 99, "continuous_x_v1")
    assert _private_query(case, 256, 99, "continuous_x_v1") != _private_query(
        case, 256, 100, "continuous_x_v1")
    with pytest.raises(ValueError, match="native nx"):
        _private_query(case, 256, 99)


def test_task_uses_full_counts_in_every_phase(case):
    settings = {"default_query": _query(), "development_point_count": 128,
                "official_point_count": 256, "terminal_point_count": 256,
                "scoring_coordinate_mode": "continuous_x_v1"}
    evaluation = ArtemisObservableTask().create_evaluation(settings)
    assert len(evaluation.make_private_development_plan(case, settings, 19).query.points) == 128
    for phase in (EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL):
        assert len(evaluation.make_plan(case, phase, settings, 19).query.points) == 256


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_continuous_count_is_not_silently_clipped(case, count):
    with pytest.raises(ValueError, match="positive integer"):
        _private_query(case, count, 1, "continuous_x_v1")


def test_unknown_mode_is_not_silently_ignored(case):
    with pytest.raises(ValueError, match="scoring_coordinate_mode"):
        _private_query(case, 1, 1, "typo")
