"""Small thread/RPC regressions for shared Artemis forward-AD exclusion.

No production model, GPU allocation, or deliberate unsafe concurrent torch AD.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Lock
from time import perf_counter, sleep
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidatePublicSessionClient, _CandidateOracleHost,
)
from pde_agent_verifier_solver.engine.evaluator import _PersistentPublicCaseSession
from pde_agent_verifier_solver.plugins.vp_1d1v.concurrency import (
    LocalArtemisSession, artemis_ad_compute,
)
from pde_agent_verifier_solver.plugins.vp_1d1v.continuous_truth import ArtemisContinuousMacroTruth
from pde_agent_verifier_solver.sdk import AnswerQuery, GenerationKind
from pde_agent_verifier_solver.sdk._candidate_timing import bind_queue_wait_observer
from tests.test_artemis_canonical import _binding, _request
from tests.test_artemis_continuous_truth import FIELDS, _Native, _Runtime
from tests.test_candidate_solve_budget import _measured_call


def _truth():
    runtime = _Runtime(False)
    query = AnswerQuery.at_points(fields=FIELDS, points=({"t": .5, "x": .125},))
    return ArtemisContinuousMacroTruth(_Native(), runtime, object()), query


@pytest.mark.parametrize("operation", [
    "source", "initial", "development_solution", "source_factory",
    "development_factory", "base_generation", "guided_generation", "guidance_fit",
    "release",
])
def test_continuous_truth_excludes_every_local_runtime_entry(operation, monkeypatch):
    truth, query = _truth()
    entered, release, waiting, invoked = Event(), Event(), Event(), Event()
    original = truth.runtime._factorized_tx_jet

    def blocked_jet(*args):
        entered.set()
        assert release.wait(3.0)
        return original(*args)

    monkeypatch.setattr(truth.runtime, "_factorized_tx_jet", blocked_jet)
    binding, source = _binding()
    backend = binding.plugin.generator.backend
    state = backend.checkpoint_payload
    record = backend.generate(state, None, _request())[0]

    def numerical_entry(*args, **kwargs):
        invoked.set()
        return ()

    # Factories themselves can materialize physical state. Session methods can
    # evaluate AD after opening, and must remain guarded independently of it.
    if operation in ("source", "initial", "development_solution"):
        target = SimpleNamespace(query_source=numerical_entry,
                                 query_initial_distribution=numerical_entry,
                                 query_completed_macros=numerical_entry)
        monkeypatch.setattr(source, "public_session_factory", lambda case: target)
        monkeypatch.setattr(source, "development_solution_session_factory", lambda case: target)
        if operation == "development_solution":
            session = backend.development_solution_session_factory(record.case)
            call = lambda: session.query_completed_macros(())
        else:
            session = backend.public_session_factory(record.case)
            method = "query_source" if operation == "source" else "query_initial_distribution"
            call = lambda: getattr(session, method)(())
    elif operation in ("source_factory", "development_factory"):
        method = ("public_session_factory" if operation == "source_factory"
                  else "development_solution_session_factory")
        monkeypatch.setattr(source, method, numerical_entry)
        call = lambda: getattr(backend, method)(record.case)
    elif operation == "base_generation":
        monkeypatch.setattr(source, "sample", numerical_entry)
        call = lambda: backend.generate(state, None, _request())
    elif operation in ("guided_generation", "guidance_fit"):
        backend.guidance_adapter = SimpleNamespace(generate=numerical_entry,
                                                  fit_guidance=numerical_entry)
        call = (lambda: backend.generate(state, object(),
                                         replace(_request(), kind=GenerationKind.GUIDED))) if (
            operation == "guided_generation") else (lambda: backend.fit_guidance(state, None, None, {}))
    else:
        monkeypatch.setattr(source, "release_cases", numerical_entry)
        call = lambda: backend.release_cases((record.case.case_id,))

    def contender():
        with bind_queue_wait_observer(lambda value: waiting.set() if value else None):
            return call()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(truth.answer, query)
        try:
            assert entered.wait(2.0)
            second = pool.submit(contender)
            assert waiting.wait(2.0)
            assert not invoked.is_set()
        finally:
            release.set()
        assert first.result(timeout=3.0).fields["M0"] == (.625,)
        second.result(timeout=3.0)
    assert invoked.is_set()


def test_truth_also_waits_for_source_and_native_cache_needs_no_ad_lock(monkeypatch):
    truth, query = _truth()
    entered, release, waiting = Event(), Event(), Event()

    def source(*args):
        entered.set()
        assert release.wait(3.0)
        return 7

    session = LocalArtemisSession(SimpleNamespace(query_source=source))
    native = AnswerQuery.at_points(fields=FIELDS, points=({"t": .5, "x": 0.},))

    def contender():
        with bind_queue_wait_observer(lambda value: waiting.set() if value else None):
            return truth.answer(query)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(session.query_source, ())
        try:
            assert entered.wait(2.0)
            assert truth.answer(native) == "native-cache"
            second = pool.submit(contender)
            assert waiting.wait(2.0)
            assert not truth.runtime.batch_sizes
        finally:
            release.set()
        assert first.result(timeout=3.0) == 7
        second.result(timeout=3.0)


def test_guard_is_reentrant_and_releases_on_exception_or_cancel():
    events = []
    with bind_queue_wait_observer(events.append), artemis_ad_compute():
        with pytest.raises(ValueError, match="numerical"):
            with artemis_ad_compute():
                raise ValueError("numerical")
    assert events == []

    waiting = Event()

    def cancelled():
        checks = 0
        def check():
            nonlocal checks
            checks += 1
            if checks > 1:
                raise RuntimeError("cancelled")
        with bind_queue_wait_observer(lambda value: waiting.set() if value else None,
                                      check_cancelled=check):
            with artemis_ad_compute():
                raise AssertionError("cancelled call must not compute")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with artemis_ad_compute():
            future = pool.submit(cancelled)
            assert waiting.wait(2.0)
            with pytest.raises(RuntimeError, match="cancelled"):
                future.result(timeout=2.0)
        # The cancelled acquisition did not strand the shared lock.
        future = pool.submit(lambda: LocalArtemisSession(SimpleNamespace(run=lambda: 4)).run())
        assert future.result(timeout=2.0) == 4


def test_independent_gpu_pool_is_not_serialized_by_parent_ad_lock():
    binding, _ = _binding()
    backend = binding.plugin.generator.backend
    remote_session, lane_lock = object(), object()
    backend._parallel_oracle_pool = SimpleNamespace(session=lambda case: remote_session,
                                                    operation_lock=lambda case_id: lane_lock)
    case = SimpleNamespace(case_id="remote")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with artemis_ad_compute():
            future = pool.submit(backend.public_session_factory, case)
            assert future.result(timeout=2.0) is remote_session
            assert binding.plugin.generator.public_session_operation_lock(None, case) is lane_lock


@pytest.mark.parametrize("fail", [False, True])
def test_backend_contention_refunds_wait_but_not_numerical_service(tmp_path, fail):
    binding, _ = _binding()
    backend = binding.plugin.generator.backend
    case = backend.generate(backend.checkpoint_payload, None, _request())[0].case
    invoked = Event()

    def marker():
        invoked.set()
        # A nested physical call is reentrant, not another refundable interval.
        with artemis_ad_compute():
            sleep(.04)  # Charged numerical service, not platform wait.
        if fail:
            raise ValueError("intentional numerical error")
        return 7.0

    session = _PersistentPublicCaseSession(
        opener=lambda *args: LocalArtemisSession(SimpleNamespace(marker=marker)),
        generator_state=object(), controller_case=case, candidate_case=case,
        methods=("marker",), operation_lock=Lock(),
    )
    host = _CandidateOracleHost(tmp_path / "arrays", case_ids=(case.case_id,),
        methods=("marker",), invoke=lambda cid, method, args, kwargs: session.invoke(method, args, kwargs),
        close=lambda cid: session.close())
    host.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with artemis_ad_compute():
                future = pool.submit(_measured_call,
                    CandidatePublicSessionClient(host.manifest_for_case(case.case_id)), case.case_id)
                deadline = perf_counter() + 2.0
                while host.state.queue_wait_seconds(case.case_id) < .12 and perf_counter() < deadline:
                    sleep(.002)
                assert host.state.queue_wait_seconds(case.case_id) >= .12
                assert not invoked.is_set()
            result = future.result(timeout=3.0)
        assert result["failed"] is fail
        assert result["value"] == (None if fail else 7.0)
        assert result["candidate_oracle_lock_wait_seconds"] >= .10
        assert result["candidate_oracle_timed_requests"] == 1.0
        assert result["active"] >= .035
        assert result["wall"] - result["active"] == pytest.approx(
            result["candidate_platform_wait_seconds"], abs=.002)
    finally:
        host.close()
