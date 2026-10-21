"""Scoped Oracle copies preserve Cases/truth and return borrowed GPU memory."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import weakref

import numpy as np
import pytest

from tests.test_ns_oracle_numerical_hot_path import _field, _points, _session, torch
from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer
from pde_agent_verifier_solver.plugins.ns import evaluation_devices as devices
from pde_agent_verifier_solver.sdk.evaluation_devices import (
    EvaluationDevicePool, bind_evaluation_device_pool,
)


@pytest.fixture
def fake_cuda(monkeypatch):
    actions = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: actions.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: actions.append(("empty", None)))
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: actions.append(("blas-clear", None)))
    return actions


def test_default_and_unknown_fields_do_not_allocate(monkeypatch):
    answer = StructuredNSMMSAnswer(_field())
    def unexpected(*args):
        raise AssertionError("unexpected device allocation")
    monkeypatch.setattr(devices, "_NSDeviceAnswers", unexpected)
    assert devices.public_evaluation_answer("case", answer) == (answer, None)
    answer._thread_safe_field = False
    with EvaluationDevicePool(("cuda:0",)) as pool, bind_evaluation_device_pool(pool):
        assert devices.public_evaluation_answer("case", answer) == (answer, None)


def test_pool_routes_reuses_and_closes_public_sessions(monkeypatch, fake_cuda):
    answer = StructuredNSMMSAnswer(_field())
    original_coefficients = answer.raw_answer.coefficients.psi.clone()
    reference = answer.evaluate_forcing(_points()[:, (2, 0, 1)])
    copies, targets = [], []
    real_copy = devices._copy_answer
    def copy_on_cpu(original, target):
        result = real_copy(original, "cpu")
        copies.append(weakref.ref(result))
        targets.append(target)
        return result
    monkeypatch.setattr(devices, "_copy_answer", copy_on_cpu)
    pool = EvaluationDevicePool(("cuda:0", "cuda:1", "cuda:2"))
    def open_one(_):
        with bind_evaluation_device_pool(pool):
            return _session(answer, "shared-case")
    with ThreadPoolExecutor(max_workers=4) as workers:
        sessions = list(workers.map(open_one, range(4)))
    assert targets == [pool.device_for("shared-case")]
    for session in sessions:
        np.testing.assert_array_equal(session.query_forcing(_points()[:, (2, 0, 1)]), reference)
    pool.close()
    assert pool.cleanup_confirmed
    for session in sessions:
        assert session._forcing_evaluator is None
        with pytest.raises(RuntimeError, match="closed"):
            session.query_forcing(_points()[:, (2, 0, 1)])
    gc.collect()
    assert all(ref() is None for ref in copies)
    torch.testing.assert_close(answer.raw_answer.coefficients.psi, original_coefficients, rtol=0, atol=0)
    assert ("sync", targets[0]) in fake_cuda and ("empty", None) in fake_cuda


def test_cache_is_bounded_and_exception_cleanup_runs(monkeypatch, fake_cuda):
    answer = StructuredNSMMSAnswer(_field())
    monkeypatch.setattr(devices, "_copy_answer", lambda original, target: original)
    resource = devices._NSDeviceAnswers("cuda:7", cache_limit=2)
    for index in range(5):
        resource.answer(str(index), answer)
    assert len(resource._answers) == 2
    pool = EvaluationDevicePool(("cuda:7",))
    with pytest.raises(ValueError, match="evaluation failed"):
        with pool:
            pool.get_resource("resource", lambda: resource)
            raise ValueError("evaluation failed")
    assert pool.cleanup_confirmed and not resource._answers
    with pytest.raises(RuntimeError, match="closed"):
        resource.answer("new", answer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires scheduled CUDA allocation")
def test_cuda_public_copy_preserves_frozen_field_and_releases_memory():
    answer = StructuredNSMMSAnswer(_field())
    points = _points()[:, (2, 0, 1)]
    reference = answer.evaluate_forcing(points)
    reference_initial = answer.evaluate_velocity(np.column_stack((np.zeros(len(points)), points[:, 1:])))
    targets = tuple(f"cuda:{i}" for i in range(min(torch.cuda.device_count(), 3)))
    rng = torch.get_rng_state().clone()
    cuda_rng = [torch.cuda.get_rng_state(target).clone() for target in targets]
    # Normalize pre-existing library workspaces before the allocation baseline.
    # The after-scope assertion below relies solely on production cleanup.
    for target in targets:
        torch.cuda.synchronize(target)
    torch._C._cuda_clearCublasWorkspaces()
    for target in targets:
        with torch.cuda.device(target):
            torch.cuda.empty_cache()
    before = {target: torch.cuda.memory_allocated(target) for target in targets}
    pool = EvaluationDevicePool(targets)
    seen = set()
    with pool, bind_evaluation_device_pool(pool):
        for index in range(32):
            case_id = f"gpu-case-{index}"
            session = _session(answer, case_id)
            actual_device = str(session._forcing_evaluator.__self__.raw_answer.coefficients.psi.device)
            assert actual_device == pool.device_for(case_id)
            seen.add(actual_device)
            np.testing.assert_allclose(session.query_forcing(points), reference, rtol=2e-10, atol=2e-11)
            np.testing.assert_allclose(session.query_initial_velocity(points[:, 1:]), reference_initial,
                                       rtol=2e-10, atol=2e-11)
            session.close()
        assert seen == set(targets)
    assert answer.raw_answer.device.type == "cpu"
    gc.collect()
    for target in targets:
        assert torch.cuda.memory_allocated(target) == before[target]
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for target, expected in zip(targets, cuda_rng):
        torch.testing.assert_close(torch.cuda.get_rng_state(target), expected, rtol=0, atol=0)


def test_failed_field_copy_cannot_confirm_gpu_release(monkeypatch, fake_cuda):
    answer = StructuredNSMMSAnswer(_field())
    def fail_copy(original, target):
        partial_tensor = original.raw_answer.coefficients.psi.clone()
        raise ValueError("copy failed with retained tensor")
    monkeypatch.setattr(devices, "_copy_answer", fail_copy)
    pool = EvaluationDevicePool(("cuda:0",))
    resource = pool.get_resource("ns", lambda: devices._NSDeviceAnswers("cuda:0"))
    with pytest.raises(ValueError, match="copy failed"):
        resource.answer("case", answer)
    with pytest.raises(RuntimeError, match="cleanup"):
        pool.close()
    assert not pool.cleanup_confirmed
    assert not resource._answers
    with pytest.raises(RuntimeError, match="cleanup"):
        resource.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires scheduled CUDA allocation")
def test_cuda_frozen_bank_routes_public_only_and_preserves_reference(tmp_path):
    from tests.test_ns_continuous_case_bank import _bank
    from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
    from pde_agent_verifier_solver.plugins.ns.case_bank import load_ns_continuous_case_bank
    from pde_agent_verifier_solver.sdk import CaseAccessKind

    path, _, _, _ = _bank(tmp_path, _field())
    bank = load_ns_continuous_case_bank(path, expected_purpose="terminal", device="cpu")
    manifest_bytes = path.read_bytes()
    frozen = tuple((content_sha256(record.case), content_sha256(bank.plans[record.case.case_id]),
                    content_sha256(record.truth.reference), record.truth.reference)
                   for record in bank.records)
    points = _points()[:, (2, 0, 1)]
    reference = []
    for record in bank.records:
        session = bank.generator.open_case_session(
            bank.generator_state, record.case, CaseAccessKind.PUBLIC_PROBLEM)
        reference.append((session.query_forcing(points), session.query_initial_velocity(points[:, 1:])))
        session.close()
    targets = tuple(f"cuda:{i}" for i in range(min(torch.cuda.device_count(), 3)))
    rng = torch.get_rng_state().clone()
    cuda_rng = [torch.cuda.get_rng_state(target).clone() for target in targets]
    # Normalize pre-existing library workspaces before the allocation baseline.
    # The after-scope assertion below relies solely on production cleanup.
    for target in targets:
        torch.cuda.synchronize(target)
    torch._C._cuda_clearCublasWorkspaces()
    for target in targets:
        with torch.cuda.device(target):
            torch.cuda.empty_cache()
    before = {target: torch.cuda.memory_allocated(target) for target in targets}
    # Each singleton pool tests the actual bank opener on every available card;
    # the test above independently verifies the stable multi-card Case routing.
    for target in targets:
        with EvaluationDevicePool((target,)) as pool, bind_evaluation_device_pool(pool):
            for record, expected in zip(bank.records, reference):
                session = bank.generator.open_case_session(
                    bank.generator_state, record.case, CaseAccessKind.PUBLIC_PROBLEM)
                assert str(session._forcing_evaluator.__self__.raw_answer.device) == target
                np.testing.assert_allclose(session.query_forcing(points), expected[0], rtol=2e-10, atol=2e-11)
                np.testing.assert_allclose(session.query_initial_velocity(points[:, 1:]), expected[1],
                                           rtol=2e-10, atol=2e-11)
                session.close()
        assert pool.cleanup_confirmed
    for record, expected in zip(bank.records, frozen):
        plan = bank.plans[record.case.case_id]
        assert content_sha256(record.case) == expected[0]
        assert content_sha256(plan) == expected[1]
        assert content_sha256(record.truth.reference) == expected[2]
        assert record.truth.answer(plan.query) is expected[3]
        assert record.truth._answer.raw_answer.device.type == "cpu"
    assert path.read_bytes() == manifest_bytes
    gc.collect()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for target, expected in zip(targets, cuda_rng):
        torch.testing.assert_close(torch.cuda.get_rng_state(target), expected, rtol=0, atol=0)
        assert torch.cuda.memory_allocated(target) == before[target]
