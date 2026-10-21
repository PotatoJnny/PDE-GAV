"""Persistent CUDA graph reuse preserves frozen NS answers and scope ownership."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
import gc
from threading import Barrier
import weakref

import numpy as np
import pytest

from tests.test_ns_analytic_derivatives import _ABIS, _field, _reference, torch
from tests.test_ns_oracle_numerical_hot_path import _session
from pde_agent_verifier_solver.plugins.ns.analytic_cuda import (
    NSAnalyticCudaCache, _field_signature,
)
from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer
from pde_agent_verifier_solver.plugins.ns import evaluation_devices as devices
from pde_agent_verifier_solver.sdk.evaluation_devices import (
    EvaluationDevicePool, bind_evaluation_device_pool,
)


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires scheduled CUDA allocation")
_MULTI_CUDA = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two scheduled CUDA devices")
_RTOL, _ATOL = 2e-10, 1e-9


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _coordinates(field, count=13, *, uniform=True, seed=908):
    points = np.random.default_rng(seed).uniform(.08, .92, (count, 3))
    xmin, xmax, ymin, ymax = field.domain_bounds
    points[:, 0] = xmin + points[:, 0] * (xmax - xmin)
    points[:, 1] = ymin + points[:, 1] * (ymax - ymin)
    points[:, 2] = field.t0 + (.41 if uniform else points[:, 2]) * (field.t1 - field.t0)
    return points


def _expected(field, coordinates, kind):
    reference = _reference(field, coordinates, components=False)
    keys = ("forcing_u", "forcing_v") if kind == "forcing" else ("u", "v")
    return np.stack([reference[key].cpu().numpy() for key in keys], axis=-1)


def _assert_answer(actual, expected):
    assert isinstance(actual, np.ndarray) and actual.flags.owndata
    assert actual.dtype == np.float64 and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


def _capture_threshold(cache, kind):
    return cache.diagnostics["capture_after_occurrences"][kind]


def _variant(field, **overrides):
    parameters = dict(
        config=field.config, geometry_spec=field.geometry_spec,
        continuous_mms_abi=field.continuous_mms_abi,
        domain_bounds=field.domain_bounds, viscosity=field.viscosity,
        t0=field.t0, t1=field.t1, device=field.coefficients.psi.device,
    )
    parameters.update(overrides)
    coefficients = type(field.coefficients)(
        psi=field.coefficients.psi.detach().clone(),
        pressure=field.coefficients.pressure.detach().clone(),
    )
    return type(field)(coefficients, **parameters)


@pytest.mark.parametrize("capacity", [False, 0, -1, 9, 1.5])
def test_invalid_capacity_is_rejected_without_cuda_allocation(capacity):
    with pytest.raises(ValueError, match="max_entries"):
        NSAnalyticCudaCache("cuda:0", max_entries=capacity)


def test_cpu_device_is_rejected_without_cuda_allocation():
    with pytest.raises(ValueError, match="CUDA device"):
        NSAnalyticCudaCache("cpu")


def test_default_capture_thresholds_protect_low_frequency_queries():
    cache = NSAnalyticCudaCache("cuda:0")
    assert cache.diagnostics["capture_after_occurrences"] == {"forcing": 12, "velocity": 20}
    assert cache.diagnostics["capture_count"] == 0


@pytest.mark.parametrize("kind,points,reason", [
    ("pressure", [[.1, .2, .3]], "kind"),
    ("forcing", [], "coordinates"),
    ("velocity", [[.1, .2]], "coordinates"),
    ("forcing", [[[.1, .2, .3]]], "coordinates"),
    ("forcing", [[.1, .2, .3]], "reside"),
])
def test_invalid_request_is_rejected_before_cuda_work(kind, points, reason):
    cache = NSAnalyticCudaCache("cuda:0")
    with pytest.raises(ValueError, match=reason):
        cache.evaluate(_field(), points, kind=kind)
    assert cache.diagnostics["capture_count"] == 0
    assert cache.diagnostics["eager_calls"] == 0
    assert not cache.diagnostics["failed"]


def test_signature_distinguishes_scientific_configuration_but_reuses_coefficients():
    original = _field()
    signature = _field_signature(original)
    assert _field_signature(_field(seed=102)) == signature
    variants = [
        _variant(original, viscosity=.071),
        _variant(original, t0=.1),
        _variant(original, t1=.7),
        _variant(original, continuous_mms_abi=_ABIS[2]),
        _variant(original, config=replace(original.config, geometry_mask_transition=.07)),
        _field(shape="cw"), _field(shape="multi"), _field(shape="stretched"),
        _field(shape="legacy_geo"), _field(modes=(2, 3, 2)),
    ]
    assert all(_field_signature(other) != signature for other in variants)


def test_unused_cache_close_is_idempotent_without_allocating_a_stream(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    cache = NSAnalyticCudaCache("cuda:0")
    cache.close()
    cache.close()
    assert cache.diagnostics["closed"] and not cache.diagnostics["failed"]
    assert cache.diagnostics["entries"] == cache.diagnostics["pending_keys"] == 0


def test_ordinary_answer_and_cpu_pool_copy_do_not_enable_cuda_graphs(monkeypatch):
    from pde_agent_verifier_solver.plugins.ns import analytic_cuda

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU/development answer allocated a CUDA graph cache")
    monkeypatch.setattr(analytic_cuda, "NSAnalyticCudaCache", forbidden)
    original = StructuredNSMMSAnswer(_field())
    points = _coordinates(original.raw_answer)
    expected = _expected(original.raw_answer, points, "forcing")
    assert original._analytic_cuda_cache is None
    normal = _session(original, "ordinary-development")
    np.testing.assert_allclose(normal.query_forcing(points[:, (2, 0, 1)]), expected,
                               rtol=_RTOL, atol=_ATOL)
    normal.close()
    real_copy = devices._copy_answer
    monkeypatch.setattr(devices, "_copy_answer", lambda answer, device: real_copy(answer, "cpu"))
    pool = EvaluationDevicePool(("cuda:0",))
    with pool, bind_evaluation_device_pool(pool):
        session = _session(original, "cpu-test-copy")
        copy = session._forcing_evaluator.__self__
        assert copy is not original and copy._analytic_cuda_cache is None
        np.testing.assert_allclose(session.query_forcing(points[:, (2, 0, 1)]), expected,
                                   rtol=_RTOL, atol=_ATOL)
    assert pool.cleanup_confirmed and session._forcing_evaluator is None
    assert original._analytic_cuda_cache is None


def test_device_resource_drains_sessions_before_freeing_graph_buffers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    events = []
    class Session:
        def close(self):
            events.append("session-drained")
    class GraphCache:
        def close(self):
            assert events == ["session-drained"]
            events.append("graph-freed")
    resource = devices._NSDeviceAnswers("cuda:0")
    session = Session()
    resource.attach(session)
    resource._analytic_cuda_cache = GraphCache()
    resource.close()
    resource.close()
    assert events == ["session-drained", "graph-freed"]


@_CUDA
@pytest.mark.parametrize("kind", ["forcing", "velocity"])
@pytest.mark.parametrize("abi", _ABIS, ids=lambda abi: abi)
def test_replay_copies_both_coefficients_and_returns_independent_numpy(kind, abi):
    first = _field("cuda:0", abi=abi, seed=101)
    second = _field("cuda:0", abi=abi, seed=777)
    third = _variant(second)
    # Same psi but different pressure isolates pressure-buffer refresh, which
    # velocity-only comparisons cannot detect.
    third.coefficients.pressure.mul_(-2.5)
    points = _coordinates(first)
    later_points = _coordinates(first, seed=919)
    saved = [(field.coefficients.psi.clone(), field.coefficients.pressure.clone())
             for field in (first, second, third)]
    expected = [_expected(first, points, kind), _expected(second, points, kind),
                _expected(third, later_points, kind)]
    torch.cuda.synchronize(0)
    cache = NSAnalyticCudaCache("cuda:0")
    try:
        a = cache.evaluate(first, points, kind=kind)
        a_snapshot = a.copy()
        threshold = _capture_threshold(cache, kind)
        for _ in range(threshold - 2):
            _assert_answer(cache.evaluate(first, points, kind=kind), expected[0])
            assert cache.diagnostics["capture_count"] == 0
        # Capture using the first Case, then change psi and pressure on replay.
        _assert_answer(cache.evaluate(first, points, kind=kind), expected[0])
        assert cache.diagnostics["capture_count"] == 1
        b = cache.evaluate(second, points, kind=kind)
        b_snapshot = b.copy()
        c = cache.evaluate(third, later_points, kind=kind)
        for actual, reference in zip((a, b, c), expected):
            _assert_answer(actual, reference)
        np.testing.assert_array_equal(a, a_snapshot)
        np.testing.assert_array_equal(b, b_snapshot)
        assert not np.shares_memory(a, b) and not np.shares_memory(b, c)
        assert cache.diagnostics["capture_count"] == 1
        assert cache.diagnostics["eager_calls"] == threshold - 1
        assert cache.diagnostics["replays"] == 3
        for field, (psi, pressure) in zip((first, second, third), saved):
            torch.testing.assert_close(field.coefficients.psi, psi, rtol=0, atol=0)
            torch.testing.assert_close(field.coefficients.pressure, pressure, rtol=0, atol=0)
    finally:
        cache.close()


@_CUDA
@pytest.mark.parametrize("kind", ["forcing", "velocity"])
def test_eight_repetitions_stay_eager_and_only_the_default_threshold_captures(kind):
    field = _field("cuda:0")
    coordinates = _coordinates(field)
    expected = _expected(field, coordinates, kind)
    torch.cuda.synchronize(0)
    cache = NSAnalyticCudaCache("cuda:0")
    try:
        threshold = _capture_threshold(cache, kind)
        assert threshold > 8
        for _ in range(8):
            _assert_answer(cache.evaluate(field, coordinates, kind=kind), expected)
        assert cache.diagnostics["eager_calls"] == 8
        assert cache.diagnostics["capture_count"] == cache.diagnostics["entries"] == 0
        for _ in range(8, threshold - 1):
            _assert_answer(cache.evaluate(field, coordinates, kind=kind), expected)
            assert cache.diagnostics["capture_count"] == 0
        _assert_answer(cache.evaluate(field, coordinates, kind=kind), expected)
        assert cache.diagnostics["capture_count"] == cache.diagnostics["entries"] == 1
        assert cache.diagnostics["eager_calls"] == threshold - 1
        assert cache.diagnostics["replays"] == 1
        _assert_answer(cache.evaluate(field, coordinates, kind=kind), expected)
        assert cache.diagnostics["capture_count"] == 1
        assert cache.diagnostics["replays"] == 2
    finally:
        cache.close()


@_CUDA
def test_keys_separate_shape_time_pattern_kind_viscosity_geometry_and_interval():
    base = _field("cuda:0")
    specifications = [
        (base, _coordinates(base, 11), "forcing"),
        (base, _coordinates(base, 1), "forcing"),
        (base, _coordinates(base, 11, uniform=False), "forcing"),
        (base, _coordinates(base, 11), "velocity"),
        (_variant(base, viscosity=.113), _coordinates(base, 11), "forcing"),
        (_field("cuda:0", shape="multi"), _coordinates(base, 11), "forcing"),
        (_variant(base, t1=.8), _coordinates(base, 11), "forcing"),
    ]
    references = [_expected(field, points, kind) for field, points, kind in specifications]
    torch.cuda.synchronize(0)
    cache = NSAnalyticCudaCache("cuda:0")
    try:
        thresholds = [_capture_threshold(cache, kind) for _, _, kind in specifications]
        for iteration in range(max(thresholds)):
            for (field, points, kind), reference in zip(specifications, references):
                if iteration < _capture_threshold(cache, kind):
                    _assert_answer(cache.evaluate(field, points, kind=kind), reference)
        assert cache.diagnostics["capture_count"] == len(specifications)
        assert cache.diagnostics["entries"] == len(specifications)
        assert cache.diagnostics["eager_calls"] == sum(value - 1 for value in thresholds)
        assert cache.diagnostics["replays"] == len(specifications)
    finally:
        cache.close()


@_CUDA
def test_lru_capacity_pending_keys_and_large_tail_eager_fallback():
    field = _field("cuda:0", shape="empty", modes=(1, 1, 1))
    cache = NSAnalyticCudaCache("cuda:0", max_entries=2)
    try:
        threshold = _capture_threshold(cache, "velocity")
        for count in (5, 7):
            points = _coordinates(field, count)
            for _ in range(threshold):
                cache.evaluate(field, points, kind="velocity")
        cache.evaluate(field, _coordinates(field, 5), kind="velocity")
        for _ in range(threshold):
            cache.evaluate(field, _coordinates(field, 9), kind="velocity")
        assert cache.diagnostics["entries"] == 2 and cache.diagnostics["evictions"] == 1
        captures = cache.diagnostics["capture_count"]
        cache.evaluate(field, _coordinates(field, 5), kind="velocity")
        assert cache.diagnostics["capture_count"] == captures
        for _ in range(threshold - 1):
            cache.evaluate(field, _coordinates(field, 7), kind="velocity")
            assert cache.diagnostics["capture_count"] == captures
        cache.evaluate(field, _coordinates(field, 7), kind="velocity")
        assert cache.diagnostics["capture_count"] == captures + 1
        for count in range(20, 32):
            cache.evaluate(field, _coordinates(field, count), kind="velocity")
            assert cache.diagnostics["pending_keys"] <= 8
            assert cache.diagnostics["entries"] <= 2
        before = cache.diagnostics
        points = _coordinates(field, 4097)
        reference = _expected(field, points, "velocity")
        for _ in range(2):
            _assert_answer(cache.evaluate(field, points, kind="velocity"), reference)
        assert cache.diagnostics["eager_calls"] == before["eager_calls"] + 2
        assert cache.diagnostics["capture_count"] == before["capture_count"]
        assert cache.diagnostics["pending_keys"] == before["pending_keys"]
    finally:
        cache.close()


@_CUDA
def test_concurrent_cases_never_share_responses_or_static_coefficients():
    fields = [_field("cuda:0", seed=seed) for seed in (101, 102, 103, 104)]
    points = [_coordinates(field, seed=seed) for field, seed in zip(fields, (91, 92, 93, 94))]
    references = [_expected(field, coordinates, "forcing") for field, coordinates in zip(fields, points)]
    torch.cuda.synchronize(0)
    cache = NSAnalyticCudaCache("cuda:0")
    threshold = _capture_threshold(cache, "forcing")
    barrier = Barrier(4)
    def query(index):
        barrier.wait(timeout=10)
        results = [cache.evaluate(fields[index], points[index], kind="forcing") for _ in range(threshold)]
        for result in results:
            _assert_answer(result, references[index])
        return results
    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            futures = [workers.submit(query, index) for index in range(4)]
            results = [future.result(timeout=120) for future in futures]
        assert cache.diagnostics["capture_count"] == 1
        assert cache.diagnostics["eager_calls"] == threshold - 1
        assert cache.diagnostics["replays"] == 4 * threshold - (threshold - 1)
        assert not np.shares_memory(results[0][-1], results[-1][-1])
    finally:
        cache.close()


@_CUDA
def test_close_releases_graph_buffers_and_rejects_new_work():
    field = _field("cuda:0")
    points = _coordinates(field)
    cache = NSAnalyticCudaCache("cuda:0")
    for _ in range(_capture_threshold(cache, "forcing")):
        cache.evaluate(field, points, kind="forcing")
    references = [weakref.ref(entry.field) for entry in cache._entries.values()]
    assert references and all(reference() is not None for reference in references)
    cache.close()
    cache.close()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert cache.diagnostics["closed"] and cache.diagnostics["entries"] == 0
    assert cache.diagnostics["pending_keys"] == 0
    assert cache._stream is None
    with pytest.raises(RuntimeError, match="closed"):
        cache.evaluate(field, points, kind="forcing")


@_MULTI_CUDA
def test_explicit_cache_device_is_independent_of_query_thread_current_device():
    fields = [_field(f"cuda:{index}", seed=101 + index) for index in range(2)]
    points = [_coordinates(field) for field in fields]
    expected = [_expected(field, coords, "forcing") for field, coords in zip(fields, points)]
    for index in range(2):
        torch.cuda.synchronize(index)
    caches = [NSAnalyticCudaCache(f"cuda:{index}") for index in range(2)]
    barrier = Barrier(2)
    def query(index):
        with torch.cuda.device(1 - index):
            barrier.wait(timeout=10)
            for _ in range(_capture_threshold(caches[index], "forcing") + 1):
                _assert_answer(caches[index].evaluate(fields[index], points[index], kind="forcing"), expected[index])
            assert torch.cuda.current_device() == 1 - index
            assert caches[index].diagnostics["device"] == f"cuda:{index}"
            assert caches[index].diagnostics["capture_count"] == 1
            assert caches[index].diagnostics["replays"] == 2
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(query, index) for index in range(2)]
            for future in futures:
                future.result(timeout=120)
    finally:
        for cache in caches:
            cache.close()


@_CUDA
def test_hidden_pool_alone_uses_graphs_and_returns_gpu_allocations():
    original = StructuredNSMMSAnswer(_field())
    points = _coordinates(original.raw_answer)
    public_points = points[:, (2, 0, 1)]
    expected = _expected(original.raw_answer, points, "forcing")
    assert getattr(original, "_analytic_cuda_cache", None) is None
    normal = _session(original, "ordinary-development")
    _assert_answer(normal.query_forcing(public_points).copy(), expected)
    assert getattr(normal._forcing_evaluator.__self__, "_analytic_cuda_cache", None) is None
    normal.close()
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    with torch.cuda.device(0):
        torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(0)
    pool = EvaluationDevicePool(("cuda:0",))
    with pool, bind_evaluation_device_pool(pool):
        session = _session(original, "hidden-official")
        copied = session._forcing_evaluator.__self__
        cache = copied._analytic_cuda_cache
        assert isinstance(cache, NSAnalyticCudaCache)
        assert copied is not original and copied.raw_answer.coefficients.psi.device == torch.device("cuda:0")
        for _ in range(_capture_threshold(cache, "forcing") + 1):
            _assert_answer(session.query_forcing(public_points).copy(), expected)
        assert cache.diagnostics["capture_count"] == 1 and cache.diagnostics["replays"] == 2
        copied_reference = weakref.ref(copied)
        del copied
    assert pool.cleanup_confirmed and cache.diagnostics["closed"]
    assert session._forcing_evaluator is None
    assert cache.diagnostics["entries"] == 0
    assert original.raw_answer.coefficients.psi.device.type == "cpu"
    assert getattr(original, "_analytic_cuda_cache", None) is None
    gc.collect()
    assert copied_reference() is None
    assert torch.cuda.memory_allocated(0) == before
