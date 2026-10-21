"""Small synthetic contracts only; real solver stability needs compute trials."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import pde_agent_verifier_solver.plugins.vpfp.frozen_solver_reward as frozen_module

from pde_agent_verifier_solver.plugins.vpfp.frozen_solver_reward import (
    PUBLIC_ARRAY_KEYS, PUBLIC_SHAPE, build_solver_public_case,
    load_frozen_solver, materialize_public_case, native_macro_query,
    public_case_receipt, run_frozen_solver, score_solver_prediction,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import VPFP_MACRO_FIELDS
from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, Case, CaseSource, CaseVisibility


def _public():
    nt, nx, ny, nvx, nvy = PUBLIC_SHAPE
    arrays = {
        "t": np.linspace(0, 1, nt),
        "x": np.arange(nx)*2*np.pi/nx, "y": np.arange(ny)*2*np.pi/ny,
        "vx": np.linspace(-5, 5, nvx), "vy": np.linspace(-5, 5, nvy),
        "velocity_weights": np.ones((nvx, nvy)), "rho0": np.ones((nx, ny)),
        "g0": np.ones((nx, ny, nvx, nvy)), "source": np.zeros((nt, nx, ny, nvx, nvy)),
        "momentum0": np.zeros((nx, ny, 2)), "kinetic_energy0": np.ones((nx, ny)),
    }
    question = {"collision_nu": 20., "temperature": 1., "background_density": 1.}
    return question, arrays


def _case():
    question, arrays = _public()
    return build_solver_public_case(question=question, arrays=arrays)


def _asset(tmp_path, body):
    path = tmp_path/"solver.py"
    path.write_text(body)
    return load_frozen_solver(path, sha256(path.read_bytes()).hexdigest())


_GOOD_SOURCE = '''
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView

def predict_macros(case, query, budget=None):
    assert case.case_id == 'frozen-strong-public-question'
    assert case.parent_case_id is None and case.tags == () and case.round_index is None
    assert set(case.public_data) == {'question', 'arrays'}
    assert set(case.public_data['question']) == {'collision_nu','temperature','background_density'}
    a = case.public_data['arrays']
    assert set(a) == {'t','x','y','vx','vy','velocity_weights','rho0','g0','source','momentum0','kinetic_energy0'}
    assert not a['source'].flags.writeable
    assert budget > 0
    shape = tuple(query.resolution[k] for k in ('t','x','y'))
    return AnswerView(kind=query.kind,fields={k:tuple(np.zeros(np.prod(shape))) for k in query.fields},
                      axes={k:tuple(a[k]) for k in ('t','x','y')},shape=shape)
'''


def test_materialized_case_drops_all_ancestry_and_hidden_extra_inputs():
    question, arrays = _public()
    question.update({"old_mms_error": 999., "case_id": "secret", "source_origin": "label"})
    arrays["hidden_full_solution"] = np.asarray([999.])
    frozen = build_solver_public_case(question=question, arrays=arrays)
    assert set(frozen.public_data["question"]) == {"collision_nu", "temperature", "background_density"}
    assert set(frozen.public_data["arrays"]) == set(PUBLIC_ARRAY_KEYS)
    assert frozen.parent_case_id is None and frozen.tags == ()
    assert frozen.source is CaseSource.PUBLIC
    assert frozen.visibility is CaseVisibility.HIDDEN
    arrays["rho0"][0, 0] = 2.
    assert frozen.public_data["arrays"]["rho0"][0, 0] == 1.
    assert not frozen.public_data["arrays"]["rho0"].flags.writeable


def test_public_identity_changes_with_operator_or_source_not_extra_labels():
    question, arrays = _public()
    first = public_case_receipt(build_solver_public_case(question=question, arrays=arrays))
    second = public_case_receipt(build_solver_public_case(question={**question, "old_score": 1e9}, arrays=arrays))
    assert first == second
    arrays["source"][0, 0, 0, 0, 0] = .125
    changed = public_case_receipt(build_solver_public_case(question=question, arrays=arrays))
    assert changed["public_input_sha256"] != first["public_input_sha256"]
    assert changed["array_sha256"]["rho0"] == first["array_sha256"]["rho0"]
    operator = public_case_receipt(build_solver_public_case(question={**question, "collision_nu": 21.}, arrays=arrays))
    assert operator["public_input_sha256"] != changed["public_input_sha256"]


def test_native_fp32_temperature_endpoint_is_not_misclassified_outside_regime():
    question, arrays = _public()
    question["temperature"] = np.float32(1.2)
    assert build_solver_public_case(question=question, arrays=arrays).public_data["question"]["temperature"] == float(np.float32(1.2))
    question["temperature"] = float(np.nextafter(np.float32(1.2), np.float32(2.)))
    with pytest.raises(ValueError, match="scope"):
        build_solver_public_case(question=question, arrays=arrays)


@pytest.mark.parametrize("change", ["velocity", "time", "shape", "nan", "nonuniform", "nu", "temperature"])
def test_fixed_public_problem_scope_rejects_wrong_inputs(change):
    question, arrays = _public()
    if change == "velocity": arrays["vx"] = np.linspace(-6, 6, 9)
    elif change == "time": arrays["t"] = np.linspace(0, 2, 13)
    elif change == "shape": arrays["x"] = np.linspace(0, 1, 17)
    elif change == "nan": arrays["source"][0, 0, 0, 0, 0] = np.nan
    elif change == "nonuniform": arrays["t"][1] += .001
    elif change == "nu": question["collision_nu"] = .1
    elif change == "temperature": question["temperature"] = -1.
    with pytest.raises((ValueError, TypeError)):
        build_solver_public_case(question=question, arrays=arrays)


class _PublicOracle:
    def __init__(self): self.calls = []
    def rho0(self, points):
        self.calls.append(("rho0", points.copy()))
        return np.ones(len(points))
    def g0(self, points):
        self.calls.append(("g0", points.copy()))
        return np.ones(len(points))*(1+points[:, 2]**2+points[:, 3]**2)
    def source(self, points):
        self.calls.append(("source", points.copy()))
        return points @ np.asarray([1., 2., 3., 4., 5.])
    def released_macros(self, *args):
        raise AssertionError("completed-solution access is forbidden")


def test_materialization_uses_only_three_public_methods_in_fixed_coordinate_order():
    question, arrays = _public()
    arrays.pop("g0")
    arrays.pop("source")
    arrays["rho0"][:] = 42.  # public oracle, not stale cache, supplies rho0.
    case = Case("private-root-label", CaseSource.GENERATED, CaseVisibility.DEVELOPMENT,
                {"question": question, "arrays": arrays, "metadata": {"old_error": 100}},
                parent_case_id="private-ancestor", tags=("private-generator-family",))
    oracle = _PublicOracle()
    result, receipt = materialize_public_case(case, oracle, chunk_size=65536)
    a = result.public_data["arrays"]
    assert np.all(a["rho0"] == 1)
    assert set(receipt["oracle_usage"]) == {"rho0", "g0", "source"}
    assert receipt["oracle_usage"]["source"]["points"] == int(np.prod(PUBLIC_SHAPE))
    assert all(len(points) <= 65536 for _, points in oracle.calls)
    index = (2, 3, 4, 5, 6)
    expected = sum(a[k][i]*c for k, i, c in zip(("t", "x", "y", "vx", "vy"), index, (1, 2, 3, 4, 5)))
    assert a["source"][index] == pytest.approx(expected)
    assert result.parent_case_id is None
    json.dumps(receipt, allow_nan=False)


def test_materialization_does_not_silently_downsample_or_accept_bad_oracle_shape():
    question, arrays = _public()
    case = Case("test", CaseSource.PUBLIC, CaseVisibility.HIDDEN, {"question": question, "arrays": arrays})
    oracle = _PublicOracle()
    oracle.rho0 = lambda points: np.ones((len(points), 1))
    with pytest.raises(ValueError, match="shape"):
        materialize_public_case(case, oracle)
    with pytest.raises(ValueError, match="chunk_size"):
        materialize_public_case(case, oracle, chunk_size=True)


def test_solver_asset_hash_and_symlink_guards(tmp_path):
    asset = _asset(tmp_path, _GOOD_SOURCE)
    assert asset.sha256 == sha256(Path(asset.path).read_bytes()).hexdigest()
    assert load_frozen_solver(asset.path, "0"*64).path == asset.path
    link = tmp_path/"link.py"
    link.symlink_to(asset.path)
    with pytest.raises(ValueError, match="symlink"):
        load_frozen_solver(link, asset.sha256)


def test_spawn_worker_receives_no_hidden_state_and_fresh_reward_is_pending(tmp_path):
    asset = _asset(tmp_path, _GOOD_SOURCE)
    case = _case()
    # Driver state deliberately includes an object that cannot be pickled.
    dirty = replace(case, public_data={**case.public_data, "truth": lambda: "secret"},
                    parent_case_id="old-label", tags=("old-error-100",))
    prediction, receipt = run_frozen_solver(asset, dirty, native_macro_query(case), budget_seconds=20.)
    assert prediction is not None and receipt["execution_passed"]
    assert receipt["reward"] is None and not receipt["reward_eligible"]
    assert receipt["reward_pending_verifier_comparison"]
    assert receipt["child_exitcode"] == 0
    assert receipt["solver_received_hidden_truth"] is False
    assert receipt["solver_prediction_attempted"] is True
    assert receipt["validity_label_available"] is False  # verifier comparison still pending
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("body,stage", [
    ("def predict_macros(*args,**kwargs):\n print('before failure')\n raise RuntimeError('numerical failure')", "solver_failed"),
    ("def predict_macros(*args,**kwargs): return {'rho':float('nan')}", "solver_failed"),
    ("import time\ndef predict_macros(*args,**kwargs): time.sleep(60)", "solver_timeout"),
])
def test_child_failure_and_timeout_never_become_high_rewards(tmp_path, body, stage):
    asset = _asset(tmp_path, body)
    case = _case()
    budget = .1 if stage == "solver_timeout" else 20.
    prediction, receipt = run_frozen_solver(asset, case, native_macro_query(case), budget_seconds=budget)
    assert prediction is None and receipt["stage"] == stage
    assert not receipt["execution_passed"] and not receipt["reward_eligible"]
    assert receipt["reward"] is None
    if "before failure" in body:
        assert "before failure" in receipt["stdout_tail"]
        assert receipt["error"]["type"] == "RuntimeError"
        assert receipt["solver_prediction_attempted"] is True
        assert receipt["validity_label_available"] is False  # unknown runtime exception
    if "return {'rho'" in body:
        assert receipt["validity_label_available"] is False  # malformed output, not numeric evidence
    if stage == "solver_timeout":
        assert receipt["validity_label_available"] == receipt["solver_prediction_attempted"]
    json.dumps(receipt, allow_nan=False)


def test_source_mutation_before_solver_execution_remains_failed(tmp_path):
    asset = _asset(tmp_path, _GOOD_SOURCE)
    Path(asset.path).write_text(_GOOD_SOURCE+"\n# changed\n")
    case = _case()
    prediction, receipt = run_frozen_solver(asset, case, native_macro_query(case))
    assert prediction is not None and receipt["execution_passed"]
    assert receipt["solver_prediction_attempted"] is True


def test_solver_source_is_checked_again_after_child_returns(tmp_path):
    # This synthetic child modifies only its own pytest temporary asset.
    body = _GOOD_SOURCE.replace("    a = case.public_data['arrays']", "    with open(__file__, 'a') as f: f.write('\\n# changed\\n')\n    a = case.public_data['arrays']")
    asset = _asset(tmp_path, body)
    case = _case()
    prediction, receipt = run_frozen_solver(asset, case, native_macro_query(case), budget_seconds=20.)
    assert prediction is not None and receipt["execution_passed"]
    assert receipt["solver_prediction_attempted"] is True


def test_query_change_is_refused_before_child_call(tmp_path):
    asset = _asset(tmp_path, _GOOD_SOURCE)
    case = _case()
    query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=65, x=33, y=33)
    prediction, receipt = run_frozen_solver(asset, case, query)
    assert prediction is None and "frozen public grid" in receipt["error"]["message"]
    assert receipt["solver_prediction_attempted"] is False and receipt["validity_label_available"] is False


@pytest.mark.parametrize("body,attempted,eligible", [
    (_GOOD_SOURCE, True, False),
    ("raise RuntimeError('import setup failed')", False, False),
    ("not_the_entrypoint = 1", False, False),
    ("def predict_macros(): pass", False, False),
    ("def predict_macros(*args,**kwargs): raise TypeError('API mismatch')", True, False),
    ("def predict_macros(*args,**kwargs): raise RuntimeError('unknown failure')", True, False),
    ("def predict_macros(*args,**kwargs): raise FloatingPointError('nonfinite iteration')", True, True),
    ("import numpy as np\ndef predict_macros(*args,**kwargs): raise np.linalg.LinAlgError('singular')", True, True),
    ("def predict_macros(*args,**kwargs): return {'rho':float('nan')}", True, False),
    (_GOOD_SOURCE.replace("tuple(np.zeros(np.prod(shape)))", "tuple(np.full(np.prod(shape), np.nan))"), True, True),
])
def test_worker_distinguishes_setup_api_numeric_failures_and_pending_scoring(body, attempted, eligible):
    # Direct synthetic worker contract; full spawn isolation is tested above.
    messages = []
    connection = SimpleNamespace(send=messages.append, close=lambda: None)
    event = Event()
    case = _case()
    data = body.encode()
    frozen_module._worker(connection, data, "synthetic-solver.py", sha256(data).hexdigest(),
                          case, native_macro_query(case), 10., event)
    assert len(messages) == 1 and event.is_set() is attempted
    assert messages[0]["validity_label_available"] is eligible
    if not attempted: assert messages[0]["failure_origin"] == "solver_setup"
    if eligible: assert messages[0]["failure_origin"] == "solver_numerical_failure"


@pytest.mark.parametrize("attempted", [False, True])
def test_parent_timeout_label_requires_shared_prediction_entry_event(tmp_path, monkeypatch, attempted):
    # Deterministic process double isolates the deadline classification. No
    # clocks/sleeps or claims of a real numerical-solver convergence result.
    event = Event()
    receive = SimpleNamespace(poll=lambda remaining: False, close=lambda: None)
    send = SimpleNamespace(close=lambda: None)
    class Process:
        pid = None
        alive = False
        def start(self):
            self.pid = 123
            self.alive = True
            if attempted: event.set()
        def is_alive(self): return self.alive
        def terminate(self): self.alive = False
        def kill(self): self.alive = False
        def join(self, timeout=None): pass
    context = SimpleNamespace(Pipe=lambda duplex: (receive, send), Event=lambda: event,
                              Process=lambda **kwargs: Process())
    monkeypatch.setattr(frozen_module.multiprocessing, "get_context", lambda name: context)
    asset = _asset(tmp_path, _GOOD_SOURCE)
    case = _case()
    prediction, receipt = run_frozen_solver(asset, case, native_macro_query(case), budget_seconds=20.)
    assert prediction is None and receipt["stage"] == "solver_timeout"
    assert receipt["solver_prediction_attempted"] is attempted
    assert receipt["validity_label_available"] is attempted
    assert receipt["reward"] is None and receipt["reward_eligible"] is False


def test_parent_spawn_setup_failure_has_no_validity_label(tmp_path, monkeypatch):
    def unavailable(name): raise OSError("synthetic process setup failure")
    monkeypatch.setattr(frozen_module.multiprocessing, "get_context", unavailable)
    asset = _asset(tmp_path, _GOOD_SOURCE)
    case = _case()
    prediction, receipt = run_frozen_solver(asset, case, native_macro_query(case))
    assert prediction is None and receipt["validity_label_available"] is False
    assert receipt["solver_prediction_attempted"] is False
    assert receipt["failure_origin"] == "parent_verification_or_setup"


def _view(values):
    query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=2, x=1, y=1)
    view = AnswerView(kind=query.kind, fields={k:tuple(values.get(k, [0., 0.])) for k in VPFP_MACRO_FIELDS},
                      axes={"t": (0., 1.), "x": (0.,), "y": (0.,)}, shape=(2, 1, 1))
    return view, query


def test_reward_uses_fresh_physical_units_not_tiny_truth_denominator():
    truth, query = _view({})
    prediction, _ = _view({"momentum_x": [1., 1.]})
    report = score_solver_prediction(prediction, truth, query, background_density=2., temperature=4.)
    assert report["reward_eligible"]
    assert report["per_field_physical_scaled_rmse"]["momentum_x"] == .25
    assert report["physical_scaled_rmse"] == .125
    assert report["reward"] == pytest.approx(np.log1p(.125)/(1+np.log1p(.125)))
    assert report["sdk_relative_l2_diagnostic_only"]["momentum_relative_l2"] > 1e12
    assert not report["old_error_labels_used"] and not report["reference_similarity_used"]
    assert not report["physical_regime_qualified"]
    json.dumps(report, allow_nan=False)


def test_reward_zero_and_monotone_bounded_for_increasing_actual_error():
    truth, query = _view({})
    rewards = []
    for error in (0., .1, 1., 100., 1e100):
        pred, _ = _view({k: [error, error] for k in VPFP_MACRO_FIELDS})
        report = score_solver_prediction(pred, truth, query, background_density=1., temperature=1.)
        assert report["execution_passed"]
        rewards.append(report["reward"])
    assert rewards[0] == 0.
    assert all(0 <= x < 1 for x in rewards)
    assert all(a < b for a, b in zip(rewards, rewards[1:]))


@pytest.mark.parametrize("bad", ["missing", "nan", "axis", "scale"])
def test_bad_scoring_inputs_are_invalid_not_rewards(bad):
    truth, query = _view({})
    pred, _ = _view({})
    if bad == "missing": pred = None
    elif bad == "nan": pred = replace(pred, fields={**pred.fields, "rho": (np.nan, 0.)})
    elif bad == "axis": pred = replace(pred, axes={**pred.axes, "t": (0., .9)})
    result = score_solver_prediction(pred, truth, query,
        background_density=0. if bad == "scale" else 1., temperature=1.)
    assert not result["execution_passed"] and result["reward"] is None
    json.dumps(result, allow_nan=False)


def test_reward_respects_temperature_and_density_units():
    truth, query = _view({})
    first, _ = _view({"rho": [1., 1.], "momentum_x": [2., 2.], "kinetic_energy": [3., 3.]})
    second, _ = _view({"rho": [2., 2.], "momentum_x": [8., 8.], "kinetic_energy": [24., 24.]})
    a = score_solver_prediction(first, truth, query, background_density=1., temperature=1.)
    b = score_solver_prediction(second, truth, query, background_density=2., temperature=4.)
    assert a["physical_scaled_rmse"] == b["physical_scaled_rmse"]
    assert a["reward"] == b["reward"]
