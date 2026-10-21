"""Basic-kernel dispatch and trusted-result policy, without real assets."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import pde_agent_verifier_solver.plugins.vpfp.perturbation_backend as module
from tests.test_vpfp_canonical_resident_runtime import resident_runtime, _resident_record


def _basic_stub(monkeypatch, calls, *, result_value=3.0):
    def source(model, points, latent, *, equation_condition, chunk_size):
        calls.append(("source", len(points)))
        return np.full(len(points), result_value)

    def initial(model, points, latent, *, equation_condition, field_name):
        calls.append((field_name, len(points)))
        assert np.all(points[:, 0] == 0.0)
        return np.full(len(points), result_value)

    monkeypatch.setitem(sys.modules,
        "pde_agent_verifier_solver.plugins.vpfp.basic_oracle",
        SimpleNamespace(evaluate_source_basic=source, evaluate_initial_basic=initial))


def test_basic_dispatch_uses_field_specific_kernels(resident_runtime, monkeypatch):
    runtime = resident_runtime
    runtime.oracle_kernel = "basic_vectorized_v1"
    record = _resident_record(runtime)
    calls = []
    _basic_stub(monkeypatch, calls)
    monkeypatch.setattr(runtime.expanded_model, "fields", lambda *a, **k: pytest.fail("full fields called"))
    monkeypatch.setattr(module, "_finite_oracle_array", lambda *a, **k: pytest.fail("online result scan"))
    assert runtime.evaluate_source(record.case.case_id, [[.2, .3, .4, .5, .6]]).tolist() == [3.]
    assert runtime.evaluate_g0(record.case.case_id, [[.3, .4, .5, .6]]).tolist() == [3.]
    assert runtime.evaluate_rho0(record.case.case_id, [[.3, .4]]).tolist() == [3.]
    assert calls == [("source", 1), ("g", 1), ("rho", 1)]


def test_trusted_result_path_does_not_scan_finiteness(resident_runtime, monkeypatch):
    runtime = resident_runtime
    runtime.oracle_kernel = "basic_vectorized_v1"
    record = _resident_record(runtime)
    _basic_stub(monkeypatch, [], result_value=float("nan"))
    result = runtime.evaluate_source(record.case.case_id, [[.2, .3, .4, .5, .6]])
    assert np.isnan(result).all()  # Offline assertion; not a repeated online scan.


@pytest.mark.parametrize("points", [[], [[0., 1.]], [[0., 0., 0., float("nan"), 0.]], [[2., 0., 0., 0., 0.]]])
def test_basic_preserves_public_coordinate_boundary(resident_runtime, monkeypatch, points):
    runtime = resident_runtime
    runtime.oracle_kernel = "basic_vectorized_v1"
    record = _resident_record(runtime)
    calls = []
    _basic_stub(monkeypatch, calls)
    with pytest.raises(ValueError):
        runtime.evaluate_source(record.case.case_id, points)
    assert not calls


@pytest.mark.parametrize("condition", [[-1., 1.], [1., 0.], [float("inf"), 1.], [1.], [[1., 1.], [1., 1.]]])
def test_basic_condition_checked_at_registration(resident_runtime, condition):
    runtime = resident_runtime
    record = _resident_record(runtime)
    runtime.release_cases((record.case.case_id,))
    runtime.oracle_kernel = "basic_vectorized_v1"
    with pytest.raises(ValueError, match="resident equation condition"):
        runtime._register_resident(record, np.asarray([1., 2.]), np.asarray(condition))
    assert not runtime._resident_cases


def test_reference_default_is_retained(resident_runtime, monkeypatch):
    runtime = resident_runtime
    record = _resident_record(runtime)
    monkeypatch.setitem(sys.modules,
        "pde_agent_verifier_solver.plugins.vpfp.basic_oracle", None)
    result = runtime.evaluate_source(record.case.case_id, [[.2, .3, .4, .5, .6]])
    np.testing.assert_array_equal(result, [2.])
    assert runtime.api.requested_source_sizes == [1]


def test_bad_kernel_rejected_before_loading_assets():
    with pytest.raises(ValueError, match="oracle_kernel"):
        module.VPFPLegacyStructuredRuntime(generator=None, base_model=None,
            expanded_decoder_checkpoint=None, output_root=None,
            materialized_grid=None, materialized_time=None, oracle_kernel="unknown")


def test_trusted_result_keeps_abi_shape_check():
    with pytest.raises(ValueError, match="shape"):
        module._trusted_oracle_array(np.zeros((2, 2)), (4,), "source")
