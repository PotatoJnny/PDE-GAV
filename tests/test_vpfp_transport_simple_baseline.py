"""The simple v2 starter is public-only, bounded, and makes no accuracy claim."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.sdk import AnswerQuery, inspect_candidate_bundle

PATH = Path(__file__).resolve().parents[1] / "configs/candidates/vpfp_transport_simple_bundle"


@pytest.fixture
def baseline(monkeypatch):
    spec = spec_from_file_location("test_transport_simple", PATH / "submission.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Public:
        def rho0(self, points):
            calls.append(points.copy())
            return 1 + .1 * np.cos(points[:, 0])

    monkeypatch.setattr(module, "open_public_case_session", lambda case: Public())
    return module, calls


def case(nu=.1, temperature=1., limit=71):
    return SimpleNamespace(public_data={"question": {
        "collision_nu": nu, "temperature": temperature,
        "grid": {"x_domain": [2*np.pi, 2*np.pi]}, "time_domain": [0., 1.],
        "numerical_access": {"max_points_per_request": limit}}})


@pytest.mark.parametrize("nu", [.01, .1, 10., 30.])
def test_complete_finite_stable_closure_without_source_or_kinetic_truth(baseline, nu):
    module, calls = baseline
    handle = module.prepare_solution(case(nu), solve_budget=900.)
    count = len(calls)
    points = ({"t": 0., "x": 0., "y": 0.},
              {"t": 1., "x": np.pi/2, "y": -.3},
              {"t": .5, "x": 2*np.pi, "y": 8*np.pi})
    query = AnswerQuery.at_points(fields=module.FIELDS, points=points)
    result = handle.evaluate(query)
    assert len(calls) == count  # Evaluate never queries public or hidden truth.
    assert all(len(batch) <= 71 for batch in calls)
    assert sum(map(len, calls)) == 32*32
    assert set(result.fields) == set(module.FIELDS)
    assert all(np.isfinite(values).all() for values in result.fields.values())
    np.testing.assert_allclose(result.fields["rho"], [1.1, 1., 1.1])
    np.testing.assert_allclose(result.fields["electric_x"], [0., .1, 0.], atol=1e-14)
    np.testing.assert_allclose(result.fields["pressure_xx"], result.fields["rho"])
    for name in ("momentum_x", "momentum_y", "pressure_xy", "heat_flux_x", "heat_flux_y"):
        assert result.fields[name] == (0., 0., 0.)
    handle.close()
    with pytest.raises(RuntimeError, match="closed"):
        handle.evaluate(query)


@pytest.mark.parametrize("budget", [0., -1., float('inf'), float('nan'), True])
def test_invalid_budget_rejected_before_oracle(baseline, budget):
    module, calls = baseline
    with pytest.raises(ValueError, match="budget"):
        module.prepare_solution(case(), solve_budget=budget)
    assert not calls


def test_bundle_and_no_query_time_leakage(baseline):
    assert inspect_candidate_bundle(PATH).entrypoint == "submission.py"
    module, _ = baseline
    handle = module.prepare_solution(case())
    with pytest.raises(ValueError, match="time"):
        handle.evaluate(AnswerQuery.at_points(fields=module.FIELDS,
            points=({"t": 1.01, "x": 0., "y": 0.},)))
