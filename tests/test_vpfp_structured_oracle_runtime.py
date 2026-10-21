"""Synthetic lifecycle/dispatch checks; not trained-model validation."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import pde_agent_verifier_solver.plugins.vpfp.perturbation_backend as module
from tests.test_vpfp_canonical_resident_runtime import resident_runtime, _resident_record


@pytest.fixture
def prepared_stub(monkeypatch):
    objects = []
    class Prepared:
        def __init__(self, model, latent, equation_condition):
            self.latent = np.asarray(latent).copy()
            self.condition = np.asarray(equation_condition).copy()
            self.calls = []
            objects.append(self)

        def source(self, points, *, chunk_size):
            assert chunk_size == 8192
            self.calls.append(("source", len(points)))
            return np.full(len(points), self.latent[0] + self.condition[1])

        def initial(self, points, *, field_name):
            assert np.all(points[:, 0] == 0.)
            self.calls.append((field_name, len(points)))
            return np.full(len(points), self.latent[0] + self.condition[1])

        def cache_info(self):
            return {"calls": len(self.calls)}

    monkeypatch.setitem(sys.modules, "pde_agent_verifier_solver.plugins.vpfp.structured_oracle",
                        SimpleNamespace(StructuredOracle=Prepared))
    return objects


@pytest.mark.parametrize("mode,count", [("structured_batch_v1", 3), ("structured_reuse_v1", 1)])
def test_lazy_preparation_and_shared_source_initial_state(resident_runtime, prepared_stub, monkeypatch, mode, count):
    runtime = resident_runtime
    runtime.oracle_kernel = mode
    record = _resident_record(runtime)
    identifier = record.case.case_id
    # Public metadata creation uses the old macro routine once; only query
    # hot paths must delegate device conversion to the prepared object.
    monkeypatch.setattr(module, "_latent_for_model", lambda *a, **k: pytest.fail("extra runtime latent transfer"))
    monkeypatch.setattr(module, "_finite_oracle_array", lambda *a, **k: pytest.fail("online result audit"))
    assert not prepared_stub  # Registering public data does not prepare a grid.
    assert runtime.evaluate_source(identifier, [[.5, 0., 0., 0., 0.]]).tolist() == [2.]
    assert runtime.evaluate_g0(identifier, [[0., 0., 0., 0.]]).tolist() == [2.]
    assert runtime.evaluate_rho0(identifier, [[0., 0.]]).tolist() == [2.]
    assert len(prepared_stub) == count
    info = runtime.prepared_oracle_cache_info()
    assert info["case_count"] == (1 if count == 1 else 0)
    assert info["hits"] == (2 if count == 1 else 0)
    assert info["misses"] == count
    assert "prepared_oracle_cache_info" not in runtime.public_session_methods


def test_lru_is_bounded_and_does_not_evict_physical_answers(resident_runtime, prepared_stub):
    runtime = resident_runtime
    runtime.oracle_kernel = "structured_reuse_v1"
    identifiers = []
    for index in range(17):
        identifier = _resident_record(runtime, case_id=f"case-{index}").case.case_id
        identifiers.append(identifier)
        runtime.evaluate_source(identifier, [[.5, 0., 0., 0., 0.]])
    info = runtime.prepared_oracle_cache_info()
    assert info["case_count"] == info["max_cases"] == 16
    assert info["evictions"] == 1
    assert identifiers[0] not in runtime._prepared_oracles
    assert len(runtime._resident_cases) == 17
    runtime.evaluate_source(identifiers[0], [[.5, 0., 0., 0., 0.]])
    assert runtime.prepared_oracle_cache_info()["evictions"] == 2
    assert len(prepared_stub) == 18
    runtime.release_cases(identifiers)
    assert not runtime._resident_cases
    assert runtime.prepared_oracle_cache_info()["case_count"] == 0


def test_release_and_reissue_same_identifier_never_reuses_old_state(resident_runtime, prepared_stub):
    runtime = resident_runtime
    runtime.oracle_kernel = "structured_reuse_v1"
    record = _resident_record(runtime)
    identifier = record.case.case_id
    points = [[.5, 0., 0., 0., 0.]]
    assert runtime.evaluate_source(identifier, points).tolist() == [2.]
    runtime.release_cases((identifier,))
    with pytest.raises(ValueError, match="no longer active"):
        runtime.evaluate_source(identifier, points)
    runtime._register_resident(record, np.asarray([7., 2.]), np.asarray([20., 3.]))
    assert runtime.evaluate_source(identifier, points).tolist() == [10.]
    assert len(prepared_stub) == 2


def test_distinct_cases_own_distinct_prepared_objects(resident_runtime, prepared_stub):
    runtime = resident_runtime
    runtime.oracle_kernel = "structured_reuse_v1"
    for identifier in ("parent", "perturbed"):
        record = _resident_record(runtime, case_id=identifier)
        runtime.evaluate_source(record.case.case_id, [[.5, 0., 0., 0., 0.]])
    assert len(prepared_stub) == 2
    assert runtime._prepared_oracles["parent"] is not runtime._prepared_oracles["perturbed"]


def test_existing_lock_serializes_first_preparation(resident_runtime, prepared_stub):
    runtime = resident_runtime
    runtime.oracle_kernel = "structured_reuse_v1"
    identifier = _resident_record(runtime).case.case_id
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda _: runtime.evaluate_source(identifier, [[.5, 0., 0., 0., 0.]]), range(12)))
    assert all(row.tolist() == [2.] for row in rows)
    assert len(prepared_stub) == 1
    assert runtime.prepared_oracle_cache_info()["hits"] == 11


@pytest.mark.parametrize("points", [[], [[0., 0.]], [[.5, 0., 0., np.nan, 0.]], [[2., 0., 0., 0., 0.]]])
def test_bad_requests_never_prepare_state(resident_runtime, prepared_stub, points):
    runtime = resident_runtime
    runtime.oracle_kernel = "structured_reuse_v1"
    identifier = _resident_record(runtime).case.case_id
    with pytest.raises(ValueError):
        runtime.evaluate_source(identifier, points)
    assert not prepared_stub


@pytest.mark.parametrize("mode", ["structured_batch_v1", "structured_reuse_v1"])
@pytest.mark.parametrize("condition", [[-1., 1.], [1., 0.], [1., np.nan]])
def test_immutable_conditions_checked_only_at_registration(resident_runtime, mode, condition):
    runtime = resident_runtime
    record = _resident_record(runtime)
    runtime.release_cases((record.case.case_id,))
    runtime.oracle_kernel = mode
    with pytest.raises(ValueError, match="resident equation condition"):
        runtime._register_resident(record, np.asarray([1., 2.]), np.asarray(condition))


def test_trusted_path_does_not_scan_returned_numerical_values(resident_runtime, prepared_stub):
    runtime = resident_runtime
    runtime.oracle_kernel = "structured_reuse_v1"
    identifier = _resident_record(runtime).case.case_id
    runtime.evaluate_source(identifier, [[.5, 0., 0., 0., 0.]])
    prepared_stub[0].latent[0] = np.nan  # Deliberate internal fault, offline test only.
    assert np.isnan(runtime.evaluate_source(identifier, [[.5, 0., 0., 0., 0.]])).all()
