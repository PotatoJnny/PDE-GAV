from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from threading import Event
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from pde_agent_verifier_solver.engine import (
    CandidateExecutionBatchTimeout,
    CandidateExecutionCancelled,
    CandidateExecutionError,
    CandidateExecutionTimeout,
    PythonCandidateCodec,
    PythonTaskWorkerSpec,
    SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    CANDIDATE_ORACLE_REQUEST_SCHEMA,
    CandidatePublicSessionClient,
    _CandidateOracleHost,
    _TOKEN_HEADER,
)
from pde_agent_verifier_solver.engine import candidate_execution as candidate_module
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
    EvaluationPhase,
    EvaluationPlan,
    TaskKind,
    write_candidate_bundle_manifest,
)


def _candidate_bundle(root: Path) -> Path:
    root.mkdir()
    (root / "submission.py").write_text(
        """\
import time

import numpy as np
import scipy

from pde_agent_verifier_solver.sdk import (
    AnswerView,
    QueryKind,
    open_public_case_session,
)


class Handle:
    def __init__(self, value):
        self.value = float(value)

    def evaluate(self, query):
        count = len(query.points)
        value = float(np.asarray([self.value], dtype=np.float64)[0])
        return AnswerView(
            kind=QueryKind.POINTS,
            fields={
                "velocity_x": tuple(value for _ in range(count)),
                "velocity_y": tuple(-value for _ in range(count)),
            },
            points=query.points,
            shape=(count,),
        )


def prepare_solution(case, solve_budget=None):
    del solve_budget
    assert scipy.__version__
    if case.public_data.get("fail"):
        raise RuntimeError("intentional private Candidate diagnostic")
    sleep_seconds = float(case.public_data.get("sleep_seconds", 0.0))
    if sleep_seconds:
        time.sleep(sleep_seconds)
    value = float(case.public_data["value"])
    if case.public_data.get("use_oracle"):
        value = open_public_case_session(case).marker(value)
    if case.public_data.get("fail_after_oracle"):
        raise RuntimeError("intentional failure after public oracle")
    return Handle(value)
""",
        encoding="utf-8",
    )
    write_candidate_bundle_manifest(root, files=("submission.py",))
    return root


def _artifact(tmp_path: Path):
    return PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        _candidate_bundle(tmp_path / "source"),
        storage_root=tmp_path / "sealed",
        artifact_id="candidate",
        name="Candidate",
        summary="test Candidate",
        usage="prepare then evaluate",
    )


def _query() -> AnswerQuery:
    return AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"),
        points=(
            {"t": 0.0, "x": 0.0, "y": 0.0},
            {"t": 1.0, "x": 0.5, "y": -0.5},
        ),
    )


def _case(identifier: str, **public_data: object) -> Case:
    return Case(
        identifier,
        CaseSource.GENERATED,
        CaseVisibility.DEVELOPMENT,
        public_data,
    )


def _evaluation_settings(query: AnswerQuery) -> dict[str, object]:
    return {
        "default_query": query,
        "primary_metric": "velocity_relative_l2",
        "include_pressure": False,
        "official_point_count": 2,
    }


def test_subprocess_candidate_execution_preserves_order_venv_and_public_oracle(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    cases = (
        _case("candidate-case-b", value=2.0),
        _case("candidate-case-a", value=3.0, use_oracle=True),
    )
    plans = tuple(
        EvaluationPlan(
            phase=EvaluationPhase.OFFICIAL,
            query=query,
            seed=index,
            settings=settings,
        )
        for index in range(len(cases))
    )
    calls: list[tuple[str, str, tuple[object, ...]]] = []
    closed: list[str] = []

    def invoke(case_id, method, args, kwargs):
        assert kwargs == {}
        calls.append((case_id, method, args))
        return float(args[0]) + 4.0

    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=2,
        case_hard_wall_seconds=30,
    )

    outputs = service.execute_cases(
        artifact,
        cases,
        plans,
        evaluation,
        public_session_methods=("marker",),
        invoke_public_session=invoke,
        close_public_session=closed.append,
    )

    assert outputs[0].prediction.fields["velocity_x"] == (2.0, 2.0)
    assert outputs[1].prediction.fields["velocity_x"] == (7.0, 7.0)
    assert calls == [("candidate-case-a", "marker", (3.0,))]
    assert sorted(closed) == ["candidate-case-a", "candidate-case-b"]
    assert len(service.audits) == 2
    assert {audit.status for audit in service.audits} == {"completed"}
    assert all(audit.controller_truth_shared is False for audit in service.audits)
    assert all(audit.process_group_cleaned for audit in service.audits)


def test_subprocess_candidate_execution_enforces_per_case_hard_wall(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    case = _case("candidate-case-timeout", value=1.0, sleep_seconds=30.0)
    plan = EvaluationPlan(
        phase=EvaluationPhase.TERMINAL,
        query=query,
        settings=settings,
    )
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=1,
        case_hard_wall_seconds=0.2,
    )

    with pytest.raises(CandidateExecutionTimeout):
        service.execute_cases(artifact, (case,), (plan,), evaluation)

    assert len(service.audits) == 1
    audit = service.audits[0]
    assert audit.status == "failed"
    assert audit.error_code == "candidate_case_timeout"
    assert audit.process_group_cleaned


def test_development_batch_hard_wall_stops_a_sleeping_worker(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    case = _case("candidate-case-development-wall", value=1.0, sleep_seconds=30.0)
    plan = EvaluationPlan(
        phase=EvaluationPhase.DEVELOPMENT,
        query=query,
        settings=settings,
    )
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=1,
        case_hard_wall_seconds=10,
        development_batch_hard_wall_seconds=0.2,
    )

    started = time.monotonic()
    with pytest.raises(CandidateExecutionBatchTimeout):
        service.execute_cases(artifact, (case,), (plan,), evaluation)
    assert time.monotonic() - started < 3.0

    assert len(service.audits) == 1
    audit = service.audits[0]
    assert audit.error_code == "candidate_development_batch_timeout"
    assert audit.process_group_cleaned


@pytest.mark.parametrize(
    "phase",
    (EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL),
)
def test_development_batch_hard_wall_does_not_limit_hidden_phases(
    tmp_path: Path,
    phase: EvaluationPhase,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    case = _case(
        f"candidate-case-{phase.value}-no-development-wall",
        value=4.0,
        sleep_seconds=0.3,
    )
    plan = EvaluationPlan(phase=phase, query=query, settings=settings)
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=1,
        case_hard_wall_seconds=10,
        development_batch_hard_wall_seconds=0.1,
    )

    outputs = service.execute_cases(artifact, (case,), (plan,), evaluation)

    assert outputs[0].prediction.fields["velocity_x"] == (4.0, 4.0)
    assert service.audits[0].status == "completed"


def test_failed_worker_revokes_case_token_and_closes_public_session_once(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    case = _case(
        "candidate-case-oracle-failure",
        value=1.0,
        use_oracle=True,
        fail_after_oracle=True,
    )
    plan = EvaluationPlan(
        phase=EvaluationPhase.OFFICIAL,
        query=query,
        settings=settings,
    )
    closed: list[str] = []
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=1,
        case_hard_wall_seconds=30,
    )

    with pytest.raises(CandidateExecutionError, match="diagnostics were retained"):
        service.execute_cases(
            artifact,
            (case,),
            (plan,),
            evaluation,
            public_session_methods=("marker",),
            invoke_public_session=(
                lambda _case_id, _method, args, _kwargs: float(args[0])
            ),
            close_public_session=closed.append,
        )

    assert closed == [case.case_id]


def test_candidate_failure_retains_private_diagnostics_and_cancels_peers(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    cases = (
        _case("candidate-case-fails", value=1.0, fail=True),
        _case("candidate-case-peer", value=2.0, sleep_seconds=30.0),
    )
    plans = tuple(
        EvaluationPlan(
            phase=EvaluationPhase.OFFICIAL,
            query=query,
            seed=index,
            settings=settings,
        )
        for index in range(2)
    )
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=2,
        case_hard_wall_seconds=10,
    )

    started = time.monotonic()
    with pytest.raises(CandidateExecutionError, match="diagnostics were retained") as captured:
        service.execute_cases(artifact, cases, plans, evaluation)
    from pde_agent_verifier_solver.engine.common_evaluation import _is_infrastructure_failure
    assert not _is_infrastructure_failure(captured.value)
    assert time.monotonic() - started < 8.0

    audits = service.audits
    assert len(audits) == 2
    assert all(audit.status == "failed" for audit in audits)
    assert all(audit.process_group_cleaned for audit in audits)
    failed = next(
        audit for audit in audits if audit.case_id == "candidate-case-fails"
    )
    response = json.loads(Path(failed.response_path).read_text(encoding="utf-8"))
    assert response["error_type"] == "RuntimeError"
    assert "intentional private Candidate diagnostic" in response["error_message"]
    assert "Traceback" in response["traceback"]
    assert Path(failed.stderr_path).is_file()


def test_unconfirmed_process_cleanup_poison_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    case = _case("candidate-case-unconfirmed-cleanup", value=1.0)
    plan = EvaluationPlan(
        phase=EvaluationPhase.OFFICIAL,
        query=query,
        settings=settings,
    )
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=1,
        case_hard_wall_seconds=30,
    )
    monkeypatch.setattr(candidate_module, "_cleanup_process_group", lambda _pid: False)

    with pytest.raises(
        CandidateExecutionError,
        match="process group cleanup could not be confirmed",
    ):
        service.execute_cases(artifact, (case,), (plan,), evaluation)

    with pytest.raises(
        CandidateExecutionError,
        match="unavailable after prior resource cleanup could not be confirmed",
    ):
        service.execute_cases(artifact, (case,), (plan,), evaluation)
    assert len(service.audits) == 1


def test_concurrent_batches_isolate_failure_cleanup(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    successful_case = _case(
        "candidate-case-concurrent-success",
        value=5.0,
        sleep_seconds=0.8,
    )
    failing_case = _case(
        "candidate-case-concurrent-failure",
        value=1.0,
        fail=True,
    )
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=2,
        case_hard_wall_seconds=10,
    )

    def run(case: Case):
        plan = EvaluationPlan(
            phase=EvaluationPhase.DEVELOPMENT,
            query=query,
            settings=settings,
        )
        return service.execute_cases(
            artifact,
            (case,),
            (plan,),
            evaluation,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        successful = executor.submit(run, successful_case)
        failing = executor.submit(run, failing_case)
        with pytest.raises(CandidateExecutionError):
            failing.result(timeout=8)
        output = successful.result(timeout=8)

    assert output[0].prediction.fields["velocity_x"] == (5.0, 5.0)
    audits = {audit.case_id: audit for audit in service.audits}
    assert audits[successful_case.case_id].status == "completed"
    assert audits[failing_case.case_id].status == "failed"
    assert all(audit.process_group_cleaned for audit in audits.values())


def test_caller_cancellation_kills_running_worker_without_cross_batch_kill(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    successful_case = _case(
        "candidate-case-cancellation-peer",
        value=5.0,
        sleep_seconds=1.0,
    )
    cancelled_case = _case(
        "candidate-case-cancelled-running",
        value=1.0,
        sleep_seconds=30.0,
    )
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=2,
        case_hard_wall_seconds=10,
    )
    cancellation = Event()

    def run(case: Case, event: Event):
        plan = EvaluationPlan(
            phase=EvaluationPhase.DEVELOPMENT,
            query=query,
            settings=settings,
        )
        return service.execute_cases(
            artifact,
            (case,),
            (plan,),
            evaluation,
            cancellation_event=event,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        successful = executor.submit(run, successful_case, Event())
        cancelled = executor.submit(run, cancelled_case, cancellation)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with service._process_lock:
                active = len(service._processes)
            if active == 2:
                break
            time.sleep(0.01)
        assert active == 2

        cancelled_at = time.monotonic()
        cancellation.set()
        with pytest.raises(CandidateExecutionCancelled):
            cancelled.result(timeout=3)
        assert time.monotonic() - cancelled_at < 2.0
        output = successful.result(timeout=8)

    assert output[0].prediction.fields["velocity_x"] == (5.0, 5.0)
    audits = {audit.case_id: audit for audit in service.audits}
    assert audits[cancelled_case.case_id].error_code == "candidate_execution_cancelled"
    assert audits[cancelled_case.case_id].process_group_cleaned
    assert audits[successful_case.case_id].status == "completed"


def test_caller_cancellation_returns_while_queued_and_preserves_owner(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    owner_case = _case("candidate-case-slot-owner", value=6.0, sleep_seconds=3.0)
    queued_case = _case("candidate-case-cancelled-queued", value=1.0)
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=1,
        case_hard_wall_seconds=10,
    )
    queued_cancellation = Event()

    def run(case: Case, event: Event):
        plan = EvaluationPlan(
            phase=EvaluationPhase.DEVELOPMENT,
            query=query,
            settings=settings,
        )
        return service.execute_cases(
            artifact,
            (case,),
            (plan,),
            evaluation,
            cancellation_event=event,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(run, owner_case, Event())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with service._process_lock:
                if service._processes:
                    break
            time.sleep(0.01)
        else:
            pytest.fail("slot owner Candidate process did not start")
        queued = executor.submit(run, queued_case, queued_cancellation)
        time.sleep(0.1)
        queued_cancellation.set()

        with pytest.raises(CandidateExecutionCancelled):
            queued.result(timeout=2)
        output = owner.result(timeout=8)

    assert output[0].prediction.fields["velocity_x"] == (6.0, 6.0)
    audits = {audit.case_id: audit for audit in service.audits}
    assert audits[queued_case.case_id].pid is None
    assert audits[queued_case.case_id].error_code == "candidate_execution_cancelled"
    assert audits[owner_case.case_id].status == "completed"


def test_hung_public_oracle_cleanup_is_bounded_and_service_fails_closed(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    case = _case(
        "candidate-case-hung-public-oracle",
        value=1.0,
        use_oracle=True,
    )
    plan = EvaluationPlan(
        phase=EvaluationPhase.DEVELOPMENT,
        query=query,
        settings=settings,
    )
    entered = Event()
    release = Event()
    cancellation = Event()
    closed: list[str] = []

    def invoke(_case_id, _method, args, _kwargs):
        entered.set()
        if not release.wait(10.0):
            raise RuntimeError("test oracle was not released")
        return float(args[0])

    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=1,
        case_hard_wall_seconds=10,
        oracle_drain_timeout_seconds=0.1,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                service.execute_cases,
                artifact,
                (case,),
                (plan,),
                evaluation,
                public_session_methods=("marker",),
                invoke_public_session=invoke,
                close_public_session=closed.append,
                cancellation_event=cancellation,
            )
            assert entered.wait(6.0)
            cancelled_at = time.monotonic()
            cancellation.set()
            with pytest.raises(CandidateExecutionCancelled):
                running.result(timeout=3.0)
            assert time.monotonic() - cancelled_at < 2.0

        with pytest.raises(
            CandidateExecutionError,
            match="unavailable after prior resource cleanup could not be confirmed",
        ):
            service.execute_cases(artifact, (case,), (plan,), evaluation)
        assert len(service.audits) == 1
    finally:
        release.set()

    deadline = time.monotonic() + 2.0
    while closed != [case.case_id] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert closed == [case.case_id]


def test_candidate_oracle_token_cannot_query_another_case(tmp_path: Path) -> None:
    calls: list[tuple[str, str, tuple[object, ...]]] = []
    closed: list[str] = []

    def invoke(case_id, method, args, kwargs):
        assert kwargs == {}
        calls.append((case_id, method, args))
        return float(args[0]) + 1.0

    host = _CandidateOracleHost(
        tmp_path / "oracle-arrays",
        case_ids=("candidate-case-a", "candidate-case-b"),
        methods=("marker",),
        invoke=invoke,
        close=closed.append,
    )
    host.start()
    try:
        manifest = host.manifest_for_case("candidate-case-a")
        client = CandidatePublicSessionClient(manifest)
        assert client.invoke("candidate-case-a", "marker", (2.0,), {}) == 3.0

        with pytest.raises(PermissionError, match="scoped to another Case"):
            client.invoke("candidate-case-b", "marker", (3.0,), {})

        request_id = "cross-case-request"
        body = json.dumps(
            {
                "schema": CANDIDATE_ORACLE_REQUEST_SCHEMA,
                "request_id": request_id,
                "params": {
                    "case_id": "candidate-case-b",
                    "method": "marker",
                    "args": host.codec.encode((3.0,)),
                    "kwargs": host.codec.encode({}),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            str(manifest["endpoint"]),
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                _TOKEN_HEADER: str(manifest["token"]),
            },
        )
        with pytest.raises(HTTPError) as caught:
            build_opener(ProxyHandler({})).open(request, timeout=5.0)
        assert caught.value.code == 403
        assert calls == [("candidate-case-a", "marker", (2.0,))]

        host.finish_case("candidate-case-a")
        host.finish_case("candidate-case-b")
    finally:
        host.close()

    assert closed == ["candidate-case-a", "candidate-case-b"]
