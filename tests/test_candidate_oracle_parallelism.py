"""Bounded parallel Oracle work preserves authorization, cancellation and budgets."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
import json
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from pde_agent_verifier_solver.engine import (
    CandidateExecutionError, CandidateExecutionTimeout,
    PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    CANDIDATE_ORACLE_REQUEST_SCHEMA, CandidatePublicSessionClient,
    _CandidateOracleHost, _TOKEN_HEADER,
)
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan
from tests.test_candidate_execution import _artifact, _case, _evaluation_settings, _query


def _service(tmp_path, *, budget=15):
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution", task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=2, oracle_max_workers=2,
        case_hard_wall_seconds=budget, oracle_drain_timeout_seconds=0.1,
    )
    plan = EvaluationPlan(phase=EvaluationPhase.OFFICIAL, query=query, settings=settings)
    return artifact, evaluation, service, plan


def test_queued_empty_hosts_do_not_eagerly_create_science_threads(tmp_path):
    host = _CandidateOracleHost(
        tmp_path / "arrays", case_ids=tuple(f"case-{i}" for i in range(30)),
        methods=("marker",), invoke=lambda *_args: 1, close=None, max_workers=28,
    )
    try:
        assert not host.state._workers
        host.start()
        assert not host.state._workers
    finally:
        host.close()


def test_more_concurrent_batches_than_case_workers_bound_actual_oracle_work(tmp_path):
    artifact, evaluation, service, plan = _service(tmp_path)
    rendezvous = Barrier(2)
    lock = Lock()
    active = peak = 0
    called = []

    def invoke(case_id, method, args, kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            called.append(case_id)
        try:
            # Six requests in three batches must overlap in pairs, even though
            # all batches have separate public capability hosts.
            rendezvous.wait(timeout=5)
            time.sleep(0.02)
            return float(args[0]) + 1
        finally:
            with lock:
                active -= 1

    def run(batch_index):
        cases = tuple(_case(f"batch-{batch_index}-{i}", value=2*batch_index+i,
                            use_oracle=True) for i in range(2))
        return service.execute_cases(
            artifact, cases, (plan, plan), evaluation,
            public_session_methods=("marker",), invoke_public_session=invoke,
            close_public_session=lambda _case_id: None,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(run, i) for i in range(3)]
        results = [future.result(timeout=25) for future in pending]
    assert peak == 2
    assert active == 0
    assert len(called) == len(set(called)) == 6
    for batch_index, outputs in enumerate(results):
        assert [row.prediction.fields["velocity_x"][0] for row in outputs] == [
            2*batch_index+1, 2*batch_index+2,
        ]
    assert len(service.audits) == 6
    assert all(audit.status == "completed" for audit in service.audits)
    assert all(audit.oracle_call_count == 1 for audit in service.audits)
    assert all(audit.oracle_service_wall_seconds >= 0.02 for audit in service.audits)
    assert all(audit.oracle_codec_wall_seconds >= 0 for audit in service.audits)


def test_parallel_http_rejects_duplicate_and_cross_case_then_revokes_queued_case(tmp_path):
    entered = {case_id: Event() for case_id in ("a", "b", "queued")}
    release = Event()
    closed = []

    def invoke(case_id, method, args, kwargs):
        entered[case_id].set()
        assert release.wait(8), "test Oracle was not released"
        return args[0]

    host = _CandidateOracleHost(
        tmp_path / "arrays", case_ids=tuple(entered), methods=("marker",),
        invoke=invoke, close=closed.append, max_workers=2,
    )
    host.start()
    clients = {case_id: CandidatePublicSessionClient(host.manifest_for_case(case_id))
               for case_id in entered}
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            running = {case_id: pool.submit(client.invoke, case_id, "marker", (1,), {})
                       for case_id, client in clients.items() if case_id != "queued"}
            try:
                assert entered["a"].wait(3) and entered["b"].wait(3)
                duplicate = CandidatePublicSessionClient(host.manifest_for_case("a"))
                with pytest.raises(CandidateExecutionError, match="public-oracle call failed"):
                    duplicate.invoke("a", "marker", (99,), {})
                # Test host-side scope validation, bypassing the client's own guard.
                client = clients["a"]
                payload = {
                    "schema": CANDIDATE_ORACLE_REQUEST_SCHEMA, "request_id": "cross-case",
                    "params": {"case_id": "b", "method": "marker",
                               "args": host.codec.encode((99,)), "kwargs": {}},
                }
                raw = json.dumps(payload).encode()
                request = Request(client.endpoint, data=raw, headers={
                    "Content-Type": "application/json", _TOKEN_HEADER: client.token,
                })
                with pytest.raises(HTTPError) as error:
                    build_opener(ProxyHandler({})).open(request, timeout=3)
                assert error.value.code == 403
                queued = pool.submit(clients["queued"].invoke, "queued", "marker", (2,), {})
                deadline = time.monotonic() + 3
                while host.state.queue_wait_seconds("queued") == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert host.state.queue_wait_seconds("queued") > 0
                assert not entered["queued"].is_set()
                host.finish_case("queued", drain_timeout_seconds=0.1)
                with pytest.raises(CandidateExecutionError):
                    queued.result(timeout=3)
                assert len(host.state._workers) == 2
            finally:
                release.set()
            assert [future.result(timeout=3) for future in running.values()] == [1, 1]
        host.finish_case("a")
        host.finish_case("b")
    finally:
        release.set()
        host.close(drain_timeout_seconds=1)
    assert not entered["queued"].is_set()
    assert sorted(closed) == ["a", "b", "queued"]
    assert not any(worker.is_alive() for worker in host.state._workers)


def test_parallel_pool_keeps_own_oracle_work_in_case_timeout(tmp_path):
    artifact, evaluation, service, plan = _service(tmp_path, budget=2.5)
    slow_entered, release = Event(), Event()
    cases = (_case("slow", value=1, use_oracle=True),
             _case("fast", value=2, use_oracle=True))

    def invoke(case_id, method, args, kwargs):
        if case_id == "slow":
            slow_entered.set()
            assert release.wait(8), "test Oracle was not released"
        return args[0]

    try:
        with pytest.raises(CandidateExecutionTimeout):
            service.execute_cases(
                artifact, cases, (plan, plan), evaluation, public_session_methods=("marker",),
                invoke_public_session=invoke, close_public_session=lambda _case_id: None,
            )
    finally:
        release.set()
    assert slow_entered.is_set()
    audits = {audit.case_id: audit for audit in service.audits}
    assert audits["fast"].status == "completed"
    slow = audits["slow"]
    assert slow.error_code == "candidate_case_timeout"
    assert slow.charged_wall_seconds >= 2.5
    assert slow.oracle_queue_wait_seconds < 0.5
    assert slow.oracle_service_wall_seconds >= 1
    assert slow.oracle_call_count == 1
    assert slow.process_group_cleaned
