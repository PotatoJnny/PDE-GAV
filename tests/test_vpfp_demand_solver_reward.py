"""CPU synthetic demand-oracle contracts, not VPFP physical validation.

The caller supplies only lightweight public coordinates and initial macros.
The recording oracle deliberately cannot be pickled: only requested numerical
values may cross into the fresh solver process, not the parent oracle or truth.
"""

from hashlib import sha256
import json
from pathlib import Path
from textwrap import indent
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp.demand_solver_reward import (
    build_demand_public_case,
    run_demand_solver,
)
from pde_agent_verifier_solver.plugins.vpfp.frozen_solver_reward import (
    PUBLIC_ARRAY_KEYS,
    PUBLIC_SHAPE,
    load_frozen_solver,
    native_macro_query,
)
from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility
from pde_agent_verifier_solver.sdk.public_sessions import (
    bind_development_solution_session_factory,
)


_QUESTION_KEYS = {
    "collision_nu", "temperature", "background_density", "equation_version",
    "time_domain", "x_domain", "velocity_domain",
}


class _MustNotMaterialize:
    def __array__(self, *args, **kwargs):
        raise AssertionError("an unrequested legacy numerical array was read")


def _case():
    nt, nx, ny, nvx, nvy = PUBLIC_SHAPE
    arrays = {
        "t": np.linspace(0, 1, nt),
        "x": np.arange(nx) * 2 * np.pi / nx,
        "y": np.arange(ny) * 2 * np.pi / ny,
        "vx": np.linspace(-5, 5, nvx),
        "vy": np.linspace(-5, 5, nvy),
        "velocity_weights": np.ones((nvx, nvy)),
        "momentum0": np.zeros((nx, ny, 2)),
        "kinetic_energy0": np.ones((nx, ny)),
        **{name: _MustNotMaterialize() for name in ("rho0", "g0", "source")},
    }
    return Case(
        "private-generator-root", CaseSource.GENERATED, CaseVisibility.DEVELOPMENT,
        {"question": {"collision_nu": 20., "temperature": 1.,
                      "background_density": 1., "old_solver_error": 1000.},
         "arrays": arrays, "truth": lambda: "never serialize hidden answer"},
        parent_case_id="private-ancestor", tags=("private-shape-family",),
    )


class _RecordingOracle:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure
        self.unpickleable_private_state = lambda: "must remain in parent"

    def _evaluate(self, method, points):
        points = np.asarray(points)
        self.calls.append((method, points.copy()))
        if self.failure == "shape":
            return np.ones((len(points), 1))
        if self.failure == "nonfinite":
            return np.full(len(points), np.nan)
        if self.failure == "exception":
            # An oracle ArithmeticError is not solver numerical failure.
            raise FloatingPointError("synthetic parent oracle failure")
        if method == "rho0":
            return 1. + .01 * points[:, 0] + .02 * points[:, 1]
        if method == "g0":
            return 1. + points[:, 2] ** 2 + points[:, 3] ** 2
        return points @ np.asarray([1., 2., 3., 4., 5.])

    def rho0(self, points):
        return self._evaluate("rho0", points)

    def g0(self, points):
        return self._evaluate("g0", points)

    def source(self, points):
        return self._evaluate("source", points)

    def released_macros(self, *args, **kwargs):
        raise AssertionError("private completed-solution method was called")


def _solver_code(statements="pass"):
    return (
        "import numpy as np\n"
        "from pde_agent_verifier_solver.sdk import (AnswerView, "
        "open_public_case_session, open_development_solution_session)\n\n"
        "def predict_macros(case, query, budget=None):\n"
        "    assert case.parent_case_id is None and case.tags == ()\n"
        "    assert case.round_index is None and case.visibility.value == 'hidden'\n"
        "    assert set(case.public_data) == {'question', 'arrays'}\n"
        "    assert set(case.public_data['question']) == "
        f"{_QUESTION_KEYS!r}\n"
        "    assert case.public_data['question']['velocity_domain'] == 'R2'\n"
        "    assert case.public_data['question']['time_domain'] == [0., 1.]\n"
        "    assert budget > 0\n"
        "    a = case.public_data['arrays']\n"
        + indent(statements, "    ") + "\n"
        "    shape = tuple(query.resolution[k] for k in ('t', 'x', 'y'))\n"
        "    return AnswerView(kind=query.kind, "
        "fields={k:tuple(np.zeros(np.prod(shape))) for k in query.fields},\n"
        "                      axes={k:tuple(a[k]) for k in ('t', 'x', 'y')}, "
        "shape=shape)\n"
    )


def _asset(tmp_path, statements="pass"):
    path = tmp_path / "solver.py"
    path.write_text(_solver_code(statements))
    return load_frozen_solver(path, sha256(path.read_bytes()).hexdigest())


def _run(tmp_path, statements="pass", *, oracle=None, legacy=False, case=None):
    case = _case() if case is None else case
    oracle = _RecordingOracle() if oracle is None else oracle
    asset = _asset(tmp_path, statements)
    prediction, receipt = run_demand_solver(
        asset, case, native_macro_query(case), public_session=oracle,
        budget_seconds=30., legacy_array_adapter=legacy,
    )
    # A finite JSON receipt is required on failures as well as successes.
    json.dumps(receipt, allow_nan=False)
    assert sha256(Path(asset.path).read_bytes()).hexdigest() == asset.sha256
    return prediction, receipt, oracle


def _assert_success(prediction, receipt):
    assert prediction is not None, receipt
    assert receipt["execution_passed"], receipt
    assert receipt["reward"] is None and not receipt["reward_eligible"]
    assert receipt["reward_pending_verifier_comparison"]
    assert not receipt["validity_label_available"]
    assert receipt["prematerialized_points"] == 0
    assert receipt["solver_received_hidden_truth"] is False


def _assert_usage(receipt, expected):
    for method in ("rho0", "g0", "source"):
        usage = receipt["oracle_usage"][method]
        assert usage["points"] == expected.get(method, 0)
        if method not in expected:
            assert usage["calls"] == 0


def test_native_public_snapshot_has_no_eager_numerical_arrays_or_ancestry():
    case = _case()
    public = build_demand_public_case(case)
    assert set(public.public_data) == {"question", "arrays"}
    assert set(public.public_data["arrays"]) == {"t", "x", "y", "vx", "vy"}
    assert set(public.public_data["question"]) == _QUESTION_KEYS
    assert public.public_data["question"]["velocity_domain"] == "R2"
    np.testing.assert_allclose(public.public_data["question"]["x_domain"], [2*np.pi, 2*np.pi])
    assert public.source is CaseSource.PUBLIC
    assert public.visibility is CaseVisibility.HIDDEN
    assert public.parent_case_id is None and public.tags == ()
    case.public_data["arrays"]["x"][0] = .5
    assert public.public_data["arrays"]["x"][0] == 0.
    assert not public.public_data["arrays"]["x"].flags.writeable


@pytest.mark.parametrize("legacy", [False, True])
def test_no_query_means_no_initial_data_or_source_evaluation(tmp_path, legacy):
    keys = PUBLIC_ARRAY_KEYS if legacy else ("t", "x", "y", "vx", "vy")
    statements = f"assert set(a) == set({keys!r})\nassert len(a) == {len(keys)}"
    prediction, receipt, oracle = _run(tmp_path, statements, legacy=legacy)
    _assert_success(prediction, receipt)
    _assert_usage(receipt, {})
    assert oracle.calls == []
    assert receipt["oracle_requests"] == []


def _points(method, count):
    # Deliberately unsorted, repeated, non-lattice points. Velocities outside
    # the historical [-5,5] observation grid remain legal in the VPFP oracle.
    full = np.asarray([
        [.451, .971, .137, 6.25, -7.],
        [.127, .213, .789, -6.5, .2],
        [.451, .971, .137, 6.25, -7.],
        [.819, 1.117, .239, .3, .7],
        [.013, .343, 1.713, -7.5, 6.3],
        [.691, 2.193, .517, .8, -.4],
        [.253, .811, 2.127, -.1, .6],
    ])[:count]
    if method == "rho0":
        return full[:, 1:3]
    if method == "g0":
        return full[:, 1:]
    return full


@pytest.mark.parametrize("method,count", [
    ("source", 3), ("source", 7), ("rho0", 3), ("g0", 7),
])
def test_native_queries_forward_only_exact_requested_points_in_order(tmp_path, method, count):
    points = _points(method, count)
    if method == "rho0":
        formula = "1. + .01 * points[:, 0] + .02 * points[:, 1]"
    elif method == "g0":
        formula = "1. + points[:, 2]**2 + points[:, 3]**2"
    else:
        formula = "points @ np.asarray([1., 2., 3., 4., 5.])"
    statements = (
        "session = open_public_case_session(case)\n"
        f"points = np.asarray({points.tolist()!r}, dtype=np.float64)\n"
        f"values = session.{method}(points)\n"
        f"assert values.shape == ({count},)\n"
        f"np.testing.assert_allclose(values, {formula}, rtol=1e-14, atol=1e-14)"
    )
    prediction, receipt, oracle = _run(tmp_path, statements)
    _assert_success(prediction, receipt)
    _assert_usage(receipt, {method: count})
    assert receipt["oracle_usage"][method]["calls"] == 1
    assert len(oracle.calls) == 1
    assert oracle.calls[0][0] == method
    np.testing.assert_array_equal(oracle.calls[0][1], points)
    requests = receipt["oracle_requests"]
    assert len(requests) == 1 and requests[0]["points"] == count
    assert requests[0]["method"] == method
    assert len(requests[0]["input_sha256"]) == 64
    assert len(requests[0]["output_sha256"]) == 64


def test_legacy_density_read_is_cached_without_other_field_queries(tmp_path):
    statements = (
        "first = a['rho0']\n"
        "assert first.shape == (33, 33)\n"
        "assert a['rho0'] is first\n"
        "assert not first.flags.writeable\n"
        "np.testing.assert_allclose(first, 1. + .01*a['x'][:, None] "
        "+ .02*a['y'][None, :])"
    )
    prediction, receipt, oracle = _run(tmp_path, statements, legacy=True)
    _assert_success(prediction, receipt)
    _assert_usage(receipt, {"rho0": 33 * 33})
    assert len(oracle.calls) == 1 and oracle.calls[0][0] == "rho0"


def test_native_second_query_can_depend_on_first_oracle_answer(tmp_path):
    statements = (
        "session = open_public_case_session(case)\n"
        "density = session.rho0(np.asarray([[.2, .3], [.4, .5]]))\n"
        "points = np.asarray([[.37, density[1], density[0], 6., -6.]])\n"
        "source = session.source(points)\n"
        "np.testing.assert_allclose(source, points @ np.asarray([1., 2., 3., 4., 5.]))"
    )
    prediction, receipt, oracle = _run(tmp_path, statements)
    _assert_success(prediction, receipt)
    _assert_usage(receipt, {"rho0": 2, "source": 1})
    assert [method for method, _ in oracle.calls] == ["rho0", "source"]
    np.testing.assert_allclose(oracle.calls[1][1], [[.37, 1.014, 1.008, 6., -6.]])


def test_legacy_full_source_read_honestly_counts_full_grid_once(tmp_path):
    statements = (
        "first = a['source']\n"
        f"assert first.shape == {PUBLIC_SHAPE!r}\n"
        "assert a['source'] is first\n"
        "assert not first.flags.writeable\n"
        "index = (2, 3, 4, 5, 6)\n"
        "expected = sum(a[k][i]*c for k,i,c in "
        "zip(('t','x','y','vx','vy'), index, (1,2,3,4,5)))\n"
        "np.testing.assert_allclose(first[index], expected, rtol=1e-14, atol=1e-14)"
    )
    prediction, receipt, oracle = _run(tmp_path, statements, legacy=True)
    _assert_success(prediction, receipt)
    count = int(np.prod(PUBLIC_SHAPE))
    assert count == 1_146_717
    _assert_usage(receipt, {"source": count})
    assert sum(len(points) for _, points in oracle.calls) == count
    assert all(method == "source" and len(points) <= 8192 for method, points in oracle.calls)
    assert receipt["oracle_usage"]["source"]["calls"] == len(oracle.calls)


def test_spawn_keeps_parent_oracle_and_development_solution_unavailable(tmp_path):
    case = _case()
    target = SimpleNamespace(released_macros=lambda *a: "hidden completed answer")
    statements = (
        "session = open_public_case_session(case)\n"
        "np.testing.assert_allclose(session.rho0(np.asarray([[.2, .3]])), [1.008])\n"
        "try:\n"
        "    open_development_solution_session(case)\n"
        "except RuntimeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('development truth leaked into hidden solve')\n"
        "for name in ('released_macros', 'unpickleable_private_state', '_target'):\n"
        "    try:\n"
        "        getattr(session, name)\n"
        "    except AttributeError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('private oracle attribute leaked')"
    )
    with bind_development_solution_session_factory(
        lambda _: target, case, allowed_methods=("released_macros",),
    ):
        prediction, receipt, oracle = _run(tmp_path, statements, case=case)
    _assert_success(prediction, receipt)
    _assert_usage(receipt, {"rho0": 1})
    assert receipt["solver_received_origin_labels"] is False
    assert len(oracle.calls) == 1


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "exception"])
def test_parent_oracle_failure_is_unavailable_not_solver_hardness(tmp_path, failure):
    points = _points("source", 3)
    statements = f"open_public_case_session(case).source(np.asarray({points.tolist()!r}))"
    prediction, receipt, oracle = _run(
        tmp_path, statements, oracle=_RecordingOracle(failure),
    )
    assert prediction is None and not receipt["execution_passed"]
    assert receipt["reward"] is None and not receipt["reward_eligible"]
    assert receipt["validity_label_available"] is False
    assert len(oracle.calls) == 1
    assert receipt["oracle_usage"]["source"]["failures"] == 1


def test_solver_cannot_hide_parent_oracle_failure_by_catching_exception(tmp_path):
    points = _points("source", 3)
    statements = (
        "try:\n"
        f"    open_public_case_session(case).source(np.asarray({points.tolist()!r}))\n"
        "except Exception:\n"
        "    pass"
    )
    prediction, receipt, oracle = _run(
        tmp_path, statements, oracle=_RecordingOracle("exception"),
    )
    assert prediction is None and not receipt["execution_passed"]
    assert receipt["reward"] is None and not receipt["reward_eligible"]
    assert receipt["validity_label_available"] is False
    assert len(oracle.calls) == 1
    assert receipt["oracle_usage"]["source"]["failures"] == 1


@pytest.mark.parametrize("statements", [
    "open_public_case_session(case).released_macros(np.ones((3, 3)))",
    "open_public_case_session(case).source(np.ones((3, 4)))",
    "open_public_case_session(case).source(np.full((3, 5), np.nan))",
])
def test_invalid_oracle_requests_are_not_dispatched_or_rewarded(tmp_path, statements):
    prediction, receipt, oracle = _run(tmp_path, statements)
    assert prediction is None and not receipt["execution_passed"]
    assert receipt["reward"] is None and not receipt["reward_eligible"]
    assert receipt["validity_label_available"] is False
    assert oracle.calls == []


@pytest.mark.parametrize("when", ["before_dispatch", "after_oracle"])
def test_allocation_check_preserves_interruption_and_unavailable_receipt(tmp_path, when):
    case = _case()
    oracle = _RecordingOracle()
    asset = _asset(tmp_path, "open_public_case_session(case).rho0(np.asarray([[.2, .3]]))")
    interrupted = RuntimeError("synthetic scheduler allocation deadline")

    def check():
        if when == "before_dispatch" or oracle.calls:
            raise interrupted

    with pytest.raises(RuntimeError) as caught:
        run_demand_solver(
            asset, case, native_macro_query(case), public_session=oracle,
            budget_seconds=30., check=check,
        )
    assert caught.value is interrupted
    receipt = caught.value.demand_solver_receipt
    assert receipt["stage"] == "allocation_interrupted"
    assert not receipt["execution_passed"] and not receipt["validity_label_available"]
    assert receipt["reward"] is None and not receipt["reward_eligible"]
    assert len(oracle.calls) == (when == "after_oracle")
    assert sha256(Path(asset.path).read_bytes()).hexdigest() == asset.sha256
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("when", ["before", "inside"])
def test_frozen_solver_bytes_are_checked_before_and_after_demand_call(tmp_path, when):
    statements = "pass"
    if when == "inside":
        statements = "with open(__file__, 'a') as stream:\n    stream.write('\\n# mutation\\n')"
    asset = _asset(tmp_path, statements)
    if when == "before":
        Path(asset.path).write_text(_solver_code(statements) + "\n# mutation\n")
    case = _case()
    oracle = _RecordingOracle()
    prediction, receipt = run_demand_solver(
        asset, case, native_macro_query(case), public_session=oracle,
        budget_seconds=30.,
    )
    assert prediction is not None and receipt["execution_passed"]
    assert receipt["solver_prediction_attempted"] is True
    assert oracle.calls == []
    json.dumps(receipt, allow_nan=False)
