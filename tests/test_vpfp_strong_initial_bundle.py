"""Synthetic S0 adapter checks; physical acceptance runs on a compute node."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, CaseSource, CaseVisibility, QueryKind,
    inspect_candidate_bundle, TaskKind,
)
from pde_agent_verifier_solver.engine import PythonCandidateCodec
from pde_agent_verifier_solver.sdk.public_sessions import bind_public_session_factory

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "configs/candidates/vpfp_strong_initial_bundle"


@pytest.fixture
def adapter(monkeypatch):
    name = "_vpfp_initial_bundle_test"
    spec = importlib.util.spec_from_file_location(name, BUNDLE / "submission.py",
        submodule_search_locations=[str(BUNDLE)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name + ".legacy_solver", None)


def test_frozen_legacy_source_and_portable_manifest(adapter):
    assert sha256((BUNDLE / "legacy_solver.py").read_bytes()).hexdigest() == adapter.LEGACY_SOLVER_SHA256
    assert inspect_candidate_bundle(BUNDLE).entrypoint_path == BUNDLE / "submission.py"


def test_formal_candidate_codec_loads_portable_relative_import(tmp_path):
    artifact = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL).bind_existing(
        BUNDLE, storage_root=tmp_path / "sealed", artifact_id="s0", name="S0",
        summary="synthetic loading test", usage="prepare_solution then evaluate")
    assert callable(artifact.program.prepare_solution)
    assert artifact.program.tree_sha256 == inspect_candidate_bundle(BUNDLE).tree_sha256


def test_prepared_interpolation_owns_solution_and_handles_periodic_seam(adapter):
    t, x, y = np.array([0., 1.]), np.arange(3.), np.arange(4.)*2
    grid = np.arange(3.)[None, :, None] + np.arange(4.)[None, None, :] + 10*t[:, None, None]
    fields = {name: grid.copy() for name in adapter.FIELDS}
    handle = adapter._PreparedMacros(t, x, y, fields)
    fields["rho"][:] = -999
    points = ({"t": .5, "x": 2.5, "y": 7.}, {"t": 1., "x": 3., "y": 8.},
              {"t": .5, "x": 2.5, "y": 7.}, {"t": 0., "x": -3., "y": -8.})
    query = AnswerQuery.at_points(fields=adapter.FIELDS, points=points)
    answer = handle.evaluate(query)
    assert answer.points == query.points and answer.shape == (4,)
    for values in answer.fields.values():
        np.testing.assert_allclose(values, [7.5, 10., 7.5, 0.], rtol=0, atol=1e-12)
    handle.close()
    with pytest.raises(RuntimeError, match="closed"):
        handle.evaluate(query)


@pytest.mark.parametrize("point", [{"t": -1., "x": 0., "y": 0.},
                                   {"t": 2., "x": 0., "y": 0.},
                                   {"t": .5, "x": float("nan"), "y": 0.}])
def test_prepared_rejects_invalid_time_or_nonfinite_points(adapter, point):
    fields = {name: np.ones((2, 2, 2)) for name in adapter.FIELDS}
    handle = adapter._PreparedMacros([0, 1], [0, 1], [0, 1], fields)
    with pytest.raises(ValueError):
        handle.evaluate(AnswerQuery.at_points(fields=adapter.FIELDS, points=(point,)))


def _small_case(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "SHAPE", (3, 3, 3, 9, 9))
    axes = {"t": np.linspace(0., 1., 3), "x": np.arange(3.)*2., "y": np.arange(3.)*2.,
            "vx": np.linspace(-5., 5., 9), "vy": np.linspace(-5., 5., 9),
            "velocity_weights": np.ones((9, 9)), "momentum0": np.zeros((3, 3, 2)),
            "kinetic_energy0": np.ones((3, 3))}
    return Case("public-synthetic", CaseSource.PERTURBED, CaseVisibility.HIDDEN,
        {"question": {"collision_nu": 20., "temperature": 1., "background_density": 1.,
                      "numerical_access": {"max_points_per_request": 128}}, "arrays": axes},
        parent_case_id="origin-must-not-change-legacy-solver")


@pytest.mark.parametrize("failure_stage", [None, "source", "solver"])
def test_prepare_uses_real_sdk_proxy_and_evaluator_owns_close(adapter, monkeypatch, failure_stage):
    case = _small_case(adapter, monkeypatch)
    calls = []
    original_error = RuntimeError("original " + str(failure_stage) + " failure")
    class Session:
        def close(self):
            calls.append(("close", 0))
        def __getattr__(self, method):
            assert method in ("rho0", "g0", "source")
            def query(points):
                assert len(points) <= 128
                calls.append((method, len(points)))
                if failure_stage == "source" and method == "source":
                    raise original_error
                return np.ones(len(points))
            return query
    factory_calls = []
    def factory(received):
        assert received is case
        factory_calls.append(received.case_id)
        return Session()
    solves = []
    def solve(received, query, budget=None):
        assert not any(method == "close" for method, _ in calls)
        assert received.parent_case_id is None and received.case_id != case.case_id
        assert set(received.public_data["question"]) == {"collision_nu", "temperature", "background_density"}
        assert query.kind is QueryKind.GRID and budget == 180
        solves.append(True)
        if failure_stage == "solver":
            raise original_error
        return AnswerView(QueryKind.GRID, {name: (1.,)*27 for name in adapter.FIELDS},
            axes={name: tuple(received.public_data["arrays"][name]) for name in ("t", "x", "y")}, shape=(3, 3, 3))
    monkeypatch.setattr(adapter.legacy_solver, "predict_macros", solve)
    # Exercise the actual allowlisted SDK proxy instead of bypassing it by
    # monkeypatching the opener with a raw runtime session. The original
    # failure must propagate through scope cleanup without close masking it.
    def prepare_and_query():
        nonlocal retained_source
        with bind_public_session_factory(factory, case, allowed_methods=("source", "rho0", "g0")):
            proxy = adapter.open_public_case_session(case)
            with pytest.raises(AttributeError, match="does not expose 'close'"):
                proxy.close()
            retained_source = proxy.source
            handle = adapter.prepare_solution(case, 180)
            assert solves == [True]
            assert not any(method == "close" for method, _ in calls)
            assert sum(n for method, n in calls if method == "source") == 3*3*3*9*9
            assert sum(n for method, n in calls if method == "g0") == 3*3*9*9
            assert sum(n for method, n in calls if method == "rho0") == 3*3
            frozen = list(calls)
            query = AnswerQuery.at_points(fields=adapter.FIELDS, points=({"t": .3, "x": 5.9, "y": .1},))
            assert handle.evaluate(query).fields["rho"] == pytest.approx((1.,))
            assert calls == frozen and solves == [True]
    retained_source = None
    if failure_stage is None:
        prepare_and_query()
    else:
        with pytest.raises(RuntimeError) as caught:
            prepare_and_query()
        assert caught.value is original_error
        assert solves == ([] if failure_stage == "source" else [True])
    assert calls[-1] == ("close", 0)
    assert sum(method == "close" for method, _ in calls) == 1
    assert factory_calls == [case.case_id]
    frozen = list(calls)
    with pytest.raises(RuntimeError, match="no longer active"):
        retained_source(np.zeros((1, 5)))
    assert calls == frozen


@pytest.mark.parametrize("budget", [True, 0, -1, float("nan"), "180"])
def test_invalid_budget_rejected_before_public_inputs(adapter, budget):
    with pytest.raises(ValueError, match="solve_budget"):
        adapter.prepare_solution(None, budget)
