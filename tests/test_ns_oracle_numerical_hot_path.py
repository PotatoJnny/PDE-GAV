"""Exact manufactured-field parity and safe query-local NS Oracle execution."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Barrier, Event

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy/ns/src"))
from agent_verifier.neural_generator import continuous_geometry as geometry
from agent_verifier.neural_generator import structured_ns_answer_first as mms
from agent_verifier.neural_generator import structured_ns_analytic as analytic
from pde_agent_verifier_solver.plugins.ns import continuous_backend as backend
from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA qualification requires a scheduled GPU")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield request.param
    torch.set_num_threads(previous)


def _field(device="cpu", *, abi=mms.STRUCTURED_NS_MMS_ABI_VERSION, reverse=False):
    config = mms.StructuredNSProjectionConfig(spatial_mode_x=2, spatial_mode_y=2, time_order=2)
    polygon = ((.3, .3), (.6, .3), (.45, .6))
    spec = geometry.ContinuousGeometrySpec(domain_bounds=(0., 1., 0., 1.),
        obstacle_polygons=(polygon[::-1] if reverse else polygon,))
    rng = torch.Generator().manual_seed(101)
    coefficients = mms.StructuredNSCaseCoefficients(
        psi=torch.randn((3, 9), generator=rng, dtype=torch.float64) * 1e-4,
        pressure=torch.randn((3, 9), generator=rng, dtype=torch.float64) * 1e-4)
    return mms.ContinuousStructuredNSMMS(coefficients, config=config, geometry_spec=spec,
        continuous_mms_abi=abi, domain_bounds=spec.domain_bounds,
        viscosity=.01, t0=0., t1=.12, device=device)


def _points():
    return np.array([[0., .5, 0.], [1., .5, .12], [.5, 0., .06], [.5, 1., .06],
        [.3, .3, .06], [.6, .3, .06], [.45, .6, .06], [.45, .4, .06],
        [.15, .25, .03], [.8, .8, .09], [.28, .3, .07], [.45, .62, .07]], dtype=np.float64)


def _uncached_masks(field, coords):
    mask = (geometry.legacy_c3_continuous_no_slip_masks
            if field.continuous_mms_abi == mms.LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION
            else geometry.continuous_no_slip_masks)
    return mask(coords, geometry=field.geometry_spec,
                transition=field.config.geometry_mask_transition)


@pytest.mark.parametrize("abi", [mms.LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION,
    mms.LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
    mms.LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION, mms.STRUCTURED_NS_MMS_ABI_VERSION])
@pytest.mark.parametrize("reverse", [False, True])
def test_hoisted_constants_preserve_every_mms_quantity(device, abi, reverse, monkeypatch):
    field = _field(device, abi=abi, reverse=reverse)
    points = _points()
    with monkeypatch.context() as patch:
        patch.setattr(field, "geometry_masks", lambda coords: _uncached_masks(field, coords))
        patch.setattr(field, "_spatial_frequencies", None)
        reference = {key: value.detach().clone() for key, value in
                     field.quantities(points, include_laplacian_components=True).items()}
    actual = field.quantities(points, include_laplacian_components=True)
    assert reference.keys() == actual.keys()
    for key in reference:
        assert torch.isfinite(actual[key]).all(), key
        torch.testing.assert_close(actual[key], reference[key], rtol=0., atol=0., msg=key)
    # Repeated differently ordered points must build fresh derivative graphs.
    repeated = field.quantities(points[::-1].copy(), include_laplacian_components=True)
    for key in reference:
        torch.testing.assert_close(repeated[key].flip(0), reference[key], rtol=0., atol=0., msg=key)


def test_field_hot_path_does_not_revalidate_or_rebuild_constants(device, monkeypatch):
    field = _field(device)
    reference = field.quantities(_points())
    def unexpected(*args, **kwargs):
        raise AssertionError("query rebuilt field-invariant geometry/frequencies")
    monkeypatch.setattr(geometry.ContinuousGeometrySpec, "validate", unexpected)
    monkeypatch.setattr(geometry, "_prepare_polygon", unexpected)
    monkeypatch.setattr(torch, "arange", unexpected)
    actual = field.quantities(_points())
    for key in reference:
        torch.testing.assert_close(actual[key], reference[key], rtol=0., atol=0.)
    # The standalone API remains a validation boundary.
    with pytest.raises(AssertionError, match="field-invariant"):
        geometry.continuous_no_slip_masks(torch.as_tensor(_points(), device=device),
            geometry=field.geometry_spec, transition=.05)


def _session(answer, case_id="case-1"):
    case = Case(case_id, CaseSource.GENERATED, CaseVisibility.DEVELOPMENT, {})
    return backend._NSPublicSession(case, answer.evaluate_forcing, answer.evaluate_velocity,
        0., answer.domain_bounds, answer.time_interval, 4096)


def test_session_keeps_public_boundary_validation_once(monkeypatch):
    answer = backend.StructuredNSMMSAnswer(_field())
    session = _session(answer)
    assert session.public_calls_thread_safe is True
    original = backend._physical_points
    calls = []
    def checked(*args, **kwargs):
        calls.append(kwargs["width"])
        return original(*args, **kwargs)
    monkeypatch.setattr(backend, "_physical_points", checked)
    session.query_forcing(_points()[:, (2, 0, 1)])
    session.query_initial_velocity(_points()[:, :2])
    assert calls == [3, 2]
    for invalid in (np.array([[0., np.nan, .5]]), np.array([[.2, .5, .5]]),
                    np.array([[0., 1.01, .5]]), np.ones((2, 2))):
        with pytest.raises(ValueError):
            session.query_forcing(invalid)
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.query_initial_velocity([[.5, .5]])


def test_two_cases_preserve_answers_and_cpu_overlap(device, monkeypatch):
    field = _field(device)
    answer = backend.StructuredNSMMSAnswer(field)
    first, second = _session(answer, "first"), _session(answer, "second")
    points = _points()[:, (2, 0, 1)]
    expected = first.query_forcing(points)
    original = analytic.analytic_forcing
    rendezvous = Barrier(2)
    def synchronized(*args, **kwargs):
        if device == "cpu":
            rendezvous.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(analytic, "analytic_forcing", synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(session.query_forcing, points) for session in (first, second)]
        for future in futures:
            np.testing.assert_array_equal(future.result(timeout=10), expected)
    assert field.coefficients.psi.grad is None
    assert field.coefficients.pressure.grad is None


def test_mutable_or_unknown_fields_do_not_opt_into_parallel_queries():
    field = _field()
    field.coefficients.psi.requires_grad_(True)
    assert _session(backend.StructuredNSMMSAnswer(field)).public_calls_thread_safe is False
    field = _field()
    field.quantities = lambda points: {}
    assert _session(backend.StructuredNSMMSAnswer(field)).public_calls_thread_safe is False
    class CustomField(type(_field())):
        pass
    field = _field()
    field.__class__ = CustomField
    assert _session(backend.StructuredNSMMSAnswer(field)).public_calls_thread_safe is False


def test_capture_guard_allows_readers_and_prioritizes_waiting_capture():
    guard = backend._NSCUDAWorkGuard()
    first_entered, release_first = Event(), Event()
    capture_waiting, capture_entered, release_capture = Event(), Event(), Event()
    last_entered = Event()

    def first_query():
        with guard.shared_query():
            first_entered.set()
            assert release_first.wait(5)
    def capture():
        capture_waiting.set()
        with guard:
            capture_entered.set()
            assert release_capture.wait(5)
    def last_query():
        with guard.shared_query():
            last_entered.set()
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(first_query)
        assert first_entered.wait(5)
        with guard.shared_query():  # Independent reader may overlap.
            assert not guard.acquire(blocking=False)
        writer = pool.submit(capture)
        assert capture_waiting.wait(5)
        # Wait on the actual writer state, avoiding a scheduling-time guess.
        with guard._condition:
            assert guard._condition.wait_for(lambda: guard._waiting_writers == 1, timeout=5)
        last = pool.submit(last_query)
        try:
            assert not capture_entered.wait(.05)
            assert not last_entered.wait(.05)
            release_first.set()
            assert capture_entered.wait(5)
            assert not last_entered.wait(.05)
        finally:
            release_first.set()
            release_capture.set()
        for future in (first, writer, last):
            future.result(timeout=5)
    assert last_entered.is_set()
    with guard:
        with guard:  # The original capture lock was reentrant.
            with guard.shared_query():
                pass
    assert guard.acquire(timeout=.1)
    guard.release()


def test_capture_guard_cancellation_credits_wait_and_releases_reader():
    from pde_agent_verifier_solver.sdk._candidate_timing import bind_queue_wait_observer
    guard = backend._NSCUDAWorkGuard()
    waiting, cancelled = Event(), Event()
    events = []
    def check():
        if cancelled.is_set():
            raise RuntimeError("revoked query")
    def observe(started):
        events.append(started)
        if started:
            waiting.set()
    def query():
        with bind_queue_wait_observer(observe, check_cancelled=check):
            with guard.shared_query():
                raise AssertionError("cancelled query entered CUDA work")
    with guard:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(query)
            assert waiting.wait(5)
            cancelled.set()
            with pytest.raises(RuntimeError, match="revoked query"):
                future.result(timeout=5)
    assert events == [True, False]
    with guard.shared_query():
        pass
    assert not guard._readers


def test_shared_queries_release_guard_after_scientific_exception():
    guard = backend._NSCUDAWorkGuard()
    with pytest.raises(ValueError, match="scientific error"):
        with guard.shared_query():
            raise ValueError("scientific error")
    assert guard.acquire(blocking=False)
    guard.release()


def test_known_cuda_field_waits_for_capture_before_launching(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA qualification requires a scheduled GPU")
    answer = backend.StructuredNSMMSAnswer(_field("cuda"))
    session = _session(answer)
    points = _points()[:, (2, 0, 1)]
    expected = session.query_forcing(points)
    started, entered = Event(), Event()
    original = analytic.analytic_forcing
    def observed(self, *args, **kwargs):
        entered.set()
        return original(self, *args, **kwargs)
    monkeypatch.setattr(analytic, "analytic_forcing", observed)
    def query():
        started.set()
        return session.query_forcing(points)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with backend._NS_CUDA_CAPTURE_LOCK:
            future = pool.submit(query)
            assert started.wait(5)
            assert not entered.wait(.1)
        np.testing.assert_array_equal(future.result(timeout=10), expected)
    assert entered.is_set()


def test_geometry_masks_keeps_other_query_dtype_compatibility():
    field = _field()
    points = torch.as_tensor(_points(), dtype=torch.float32).requires_grad_(True)
    actual = field.geometry_masks(points)
    reference = _uncached_masks(field, points)
    for a, b in zip(actual, reference):
        assert a.dtype == b.dtype == torch.float32
        torch.testing.assert_close(a, b, rtol=0., atol=0.)
