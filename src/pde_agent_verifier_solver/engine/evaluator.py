"""Agent-facing development evaluation and controller-only hidden verification."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
from threading import Event, Lock, Thread
import time
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    ArtifactEvaluation,
    Case,
    CaseAccessKind,
    CaseEvaluation,
    CaseRecord,
    CaseVisibility,
    DevelopmentCoordinateMode,
    EvaluationPhase,
    EvaluationPlan,
    EvaluationProgram,
    GeneratorProgram,
    GeneratorState,
    QueryKind,
    resolve_evaluation_program,
)
from pde_agent_verifier_solver.sdk._candidate_timing import controller_queue_wait
from pde_agent_verifier_solver.sdk.evaluation_devices import (
    EvaluationDevicePool,
    bind_evaluation_device_pool,
    current_evaluation_device_pool,
)
from pde_agent_verifier_solver.sdk.public_sessions import (
    bind_development_solution_session_factory,
    bind_public_session_factory,
    open_development_solution_session,
    open_public_case_session,
)
from pde_agent_verifier_solver.sdk.solve_budget import _record_oracle_lock_wait

from .campaign import EvaluatorPolicy
from .candidate_diagnostics import candidate_diagnostic
from .candidate_execution import (
    CandidateExecutionCancelled,
    CandidateExecutionInfrastructureError,
    CandidateExecutionService,
)
from .candidate_case import (
    candidate_facing_case,
    require_controller_case_id,
)


_REQUEST_CANCELLATION: ContextVar[Event | None] = ContextVar(
    "agent_evaluation_request_cancellation", default=None,
)


@contextmanager
def evaluation_request_cancellation(event: Event):
    """Bind one HTTP request without mutating the owning Agent's revocation."""
    token = _REQUEST_CANCELLATION.set(event)
    try:
        yield
    finally:
        _REQUEST_CANCELLATION.reset(token)


class _EvaluationCancellation:
    def __init__(self, owner: Event, request: Event | None):
        self.owner, self.request = owner, request

    def is_set(self) -> bool:
        return self.owner.is_set() or (
            self.request is not None and self.request.is_set()
        )


@dataclass(frozen=True)
class EvaluationRequest:
    case_ids: tuple[str, ...]
    query: AnswerQuery | None = None
    return_truth: bool = False


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    primary_error: float
    metrics: Mapping[str, float]
    prediction: AnswerView | None
    truth: AnswerView | None


@dataclass(frozen=True)
class EvaluationResponse:
    artifact_id: str
    primary_error: float
    metrics: Mapping[str, float]
    cases: tuple[EvaluationCaseResult, ...]


DevelopmentPlanSeedProvider = Callable[[int, int, str], int]
HiddenPlanSeedProvider = Callable[[int, str], int]


class _CaseSessionBindings:
    """Bind plugin-owned numerical sessions around exactly one candidate call."""

    def __init__(
        self,
        generator: GeneratorProgram | None = None,
        generator_state: GeneratorState | None = None,
        *,
        allow_development_solution: bool = False,
        operation_lock: object | None = None,
        evaluation_device_pool: EvaluationDevicePool | None = None,
    ) -> None:
        if (generator is None) != (generator_state is None):
            raise ValueError(
                "GeneratorProgram and GeneratorState must be provided together"
            )
        if not isinstance(allow_development_solution, bool):
            raise TypeError("allow_development_solution must be boolean")
        if evaluation_device_pool is not None and not isinstance(evaluation_device_pool, EvaluationDevicePool):
            raise TypeError("evaluation device pool must be EvaluationDevicePool or None")
        self._evaluation_device_pool = evaluation_device_pool
        self._generator = generator
        self._generator_state = generator_state
        self._allow_development_solution = allow_development_solution
        self._operation_lock = operation_lock or Lock()
        if not callable(getattr(self._operation_lock, "acquire", None)) or not callable(
            getattr(self._operation_lock, "release", None)
        ):
            raise TypeError("case session operation_lock must support acquire/release")
        self._open_session = (
            None
            if generator is None
            else getattr(generator, "open_case_session", None)
        )
        self._public_methods = self._session_methods(
            CaseAccessKind.PUBLIC_PROBLEM
        )
        self._development_methods = (
            self._session_methods(CaseAccessKind.DEVELOPMENT_SOLUTION)
            if allow_development_solution
            else ()
        )
        if (
            self._public_methods or self._development_methods
        ) and not callable(self._open_session):
            raise TypeError(
                "a Generator with session methods must implement open_case_session"
            )
        self._serialize_operations = bool(
            self._public_methods or self._development_methods
        )

    @contextmanager
    def bind(
        self,
        controller_case: Case,
        candidate_case: Case,
        phase: EvaluationPhase,
    ) -> Iterator[None]:
        if self._serialize_operations:
            lock_started = time.perf_counter()
            self._operation_lock.acquire()
            _record_oracle_lock_wait(time.perf_counter() - lock_started)
        try:
            public_factory = self._factory(
                controller_case,
                CaseAccessKind.PUBLIC_PROBLEM,
                self._public_methods,
            )
            development_methods = (
                self._development_methods
                if (
                    phase is EvaluationPhase.DEVELOPMENT
                    and self._allow_development_solution
                    and controller_case.visibility is CaseVisibility.DEVELOPMENT
                )
                else ()
            )
            development_factory = self._factory(
                controller_case,
                CaseAccessKind.DEVELOPMENT_SOLUTION,
                development_methods,
            )
            # Bind empty namespaces too. This shadows any outer evaluator
            # binding and makes phase/candidate revocation fail closed.
            with bind_public_session_factory(
                public_factory,
                candidate_case,
                allowed_methods=self._public_methods,
            ), bind_development_solution_session_factory(
                development_factory,
                candidate_case,
                allowed_methods=development_methods,
            ):
                yield
        finally:
            if self._serialize_operations:
                self._operation_lock.release()

    def _session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]:
        generator = self._generator
        if generator is None:
            return ()
        resolver = getattr(generator, "case_session_methods", None)
        if not callable(resolver):
            return ()
        values = resolver(access)
        if isinstance(values, (str, bytes)):
            raise TypeError("Generator case session methods must be a sequence")
        methods = tuple(values)
        if any(
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            for name in methods
        ):
            raise ValueError(
                "Generator case session methods must be public identifiers"
            )
        if len(methods) != len(set(methods)):
            raise ValueError("Generator case session methods must be unique")
        return methods

    def methods(self, access: CaseAccessKind) -> tuple[str, ...]:
        if not isinstance(access, CaseAccessKind):
            raise TypeError("case session access must be a CaseAccessKind")
        if access is CaseAccessKind.PUBLIC_PROBLEM:
            return self._public_methods
        if access is CaseAccessKind.DEVELOPMENT_SOLUTION:
            return self._development_methods
        return ()  # pragma: no cover - enum exhaustiveness

    def invoke_public(
        self,
        controller_case: Case,
        candidate_case: Case,
        phase: EvaluationPhase,
        method: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> object:
        """Execute one allowlisted public-session RPC in the Controller."""

        if method not in self._public_methods:
            raise AttributeError(
                f"public case session does not expose {method!r}"
            )
        if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
            raise TypeError("public session args must be a sequence")
        if not isinstance(kwargs, Mapping) or any(
            not isinstance(name, str) for name in kwargs
        ):
            raise TypeError("public session kwargs must be a string-keyed mapping")
        with self.bind(controller_case, candidate_case, phase):
            session = open_public_case_session(candidate_case)
            target = getattr(session, method)
            return target(*tuple(args), **dict(kwargs))

    def persistent_public_session(
        self,
        controller_case: Case,
        candidate_case: Case,
    ) -> "_PersistentPublicCaseSession":
        """Create one lazy Controller session for a remote Candidate/Case."""

        opener = self._open_session
        state = self._generator_state
        if not self._public_methods or not callable(opener) or state is None:
            raise RuntimeError("this equation does not expose a public case session")
        # An explicitly scoped plugin may own independent numerical replicas.
        # Default/shared generators retain the existing global operation lock.
        operation_lock = self._operation_lock
        lock_provider = getattr(self._generator, "public_session_operation_lock", None)
        if callable(lock_provider):
            with bind_evaluation_device_pool(self._evaluation_device_pool):
                provided = lock_provider(state, controller_case)
            if provided is not None:
                operation_lock = provided
        return _PersistentPublicCaseSession(
            opener=opener,
            generator_state=state,
            controller_case=controller_case,
            candidate_case=candidate_case,
            methods=self._public_methods,
            operation_lock=operation_lock,
            evaluation_device_pool=self._evaluation_device_pool,
        )

    def _factory(
        self,
        case: Case,
        access: CaseAccessKind,
        methods: Sequence[str],
    ) -> Callable[[Case], object] | None:
        if not methods:
            return None
        opener = self._open_session
        state = self._generator_state
        assert callable(opener) and state is not None

        def open_controller_session(_candidate_case: Case) -> object:
            with bind_evaluation_device_pool(self._evaluation_device_pool):
                return opener(state, case, access)

        return open_controller_session


class _PersistentPublicCaseSession:
    """One memoized plugin session scoped to one remote Candidate/Case worker."""

    def __init__(
        self,
        *,
        opener: Callable[[GeneratorState, Case, CaseAccessKind], object],
        generator_state: GeneratorState,
        controller_case: Case,
        candidate_case: Case,
        methods: Sequence[str],
        operation_lock: object,
        evaluation_device_pool: EvaluationDevicePool | None = None,
    ) -> None:
        self._evaluation_device_pool = evaluation_device_pool
        self._opener = opener
        self._generator_state = generator_state
        self._controller_case = controller_case
        self._candidate_case_id = candidate_case.case_id
        self._methods = frozenset(methods)
        self._operation_lock = operation_lock
        self._lifecycle_lock = Lock()
        self._accepting = True
        self._inflight = 0
        self._close_claimed = False
        self._target: object | None = None
        # A plugin may allow independent immutable sessions to overlap. Calls
        # within one session, and session creation, must still be serialized.
        self._invoke_lock = Lock()

    def invoke(
        self,
        method: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> object:
        if method not in self._methods:
            raise AttributeError(
                f"public case session does not expose {method!r}"
            )
        if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
            raise TypeError("public session args must be a sequence")
        if not isinstance(kwargs, Mapping) or any(
            not isinstance(name, str) for name in kwargs
        ):
            raise TypeError(
                "public session kwargs must be a string-keyed mapping"
            )
        with self._lifecycle_lock:
            if not self._accepting:
                raise RuntimeError("public case session is no longer active")
            self._inflight += 1

        result: object | None = None
        primary_error: BaseException | None = None
        operation_acquired = False
        invocation_acquired = False
        target_to_close: object | None = None
        try:
            with controller_queue_wait() as check_cancelled:
                lock_started = time.perf_counter()
                try:
                    while not self._invoke_lock.acquire(timeout=0.1):
                        check_cancelled()
                        with self._lifecycle_lock:
                            if not self._accepting:
                                raise RuntimeError("public case session is no longer active")
                    invocation_acquired = True
                    check_cancelled()
                finally:
                    # Classify the existing live wait; never add a second
                    # refundable interval to the Controller deadline ledger.
                    _record_oracle_lock_wait(time.perf_counter() - lock_started)
            with self._lifecycle_lock:
                if not self._accepting:
                    raise RuntimeError("public case session is no longer active")
                target = self._target
            if target is None or getattr(target, "public_calls_thread_safe", False) is not True:
                with controller_queue_wait() as check_cancelled:
                    lock_started = time.perf_counter()
                    try:
                        while not self._operation_lock.acquire(timeout=0.1):
                            check_cancelled()
                            with self._lifecycle_lock:
                                if not self._accepting:
                                    raise RuntimeError("public case session is no longer active")
                        operation_acquired = True
                        check_cancelled()
                    finally:
                        _record_oracle_lock_wait(time.perf_counter() - lock_started)
                with self._lifecycle_lock:
                    if not self._accepting:
                        raise RuntimeError("public case session is no longer active")
                if target is None:
                    # Oracle calls run in worker threads, where the Controller's
                    # ContextVar is absent. Carry the captured scope explicitly.
                    with bind_evaluation_device_pool(self._evaluation_device_pool):
                        target = self._opener(
                            self._generator_state,
                            self._controller_case,
                            CaseAccessKind.PUBLIC_PROBLEM,
                        )
                    if target is None:
                        raise TypeError("public session factory returned None")
                    with self._lifecycle_lock:
                        self._target = target
                # Only the trusted plugin can opt in. Opener/state changes stay
                # under the Generator lock; immutable per-session work need not.
                if getattr(target, "public_calls_thread_safe", False) is True:
                    self._operation_lock.release()
                    operation_acquired = False
            callable_method = getattr(target, method, None)
            if not callable(callable_method):
                raise TypeError(
                    f"declared public session method {method!r} is not callable"
                )
            result = callable_method(*tuple(args), **dict(kwargs))
        except BaseException as error:
            primary_error = error
        finally:
            if operation_acquired:
                self._operation_lock.release()
            if invocation_acquired:
                self._invoke_lock.release()
            with self._lifecycle_lock:
                self._inflight -= 1
                if not self._accepting and self._inflight == 0:
                    target_to_close = self._claim_target_for_close_locked()

        if target_to_close is not None:
            try:
                self._close_target(target_to_close)
            except BaseException as cleanup_error:
                if primary_error is None:
                    primary_error = cleanup_error
                else:
                    _attach_case_session_cleanup_error(
                        primary_error,
                        cleanup_error,
                    )
        if primary_error is not None:
            raise primary_error
        return result

    def close(self) -> None:
        with self._lifecycle_lock:
            if not self._accepting and self._close_claimed:
                return
            self._accepting = False
            target = (
                self._claim_target_for_close_locked()
                if self._inflight == 0
                else None
            )
        if target is not None:
            self._close_target(target)

    def _claim_target_for_close_locked(self) -> object | None:
        if self._close_claimed:
            return None
        self._close_claimed = True
        target = self._target
        self._target = None
        return target

    def _close_target(self, target: object) -> None:
        """Close now when safe, or defer behind a busy shared plugin lock."""

        if getattr(target, "public_calls_thread_safe", False) is True:
            close = getattr(target, "close", None)
            if callable(close):
                close()
            return
        try:
            acquired = self._operation_lock.acquire(False)
        except TypeError:
            acquired = False
        if acquired:
            try:
                close = getattr(target, "close", None)
                if callable(close):
                    close()
            finally:
                self._operation_lock.release()
            return

        Thread(
            target=self._close_target_after_operation_drains,
            args=(target,),
            name=f"public-session-close-{self._candidate_case_id[:12]}",
            daemon=True,
        ).start()

    def _close_target_after_operation_drains(self, target: object) -> None:
        self._operation_lock.acquire()
        try:
            close = getattr(target, "close", None)
            if callable(close):
                close()
        finally:
            self._operation_lock.release()


class AgentEvaluator:
    """Interactive development service executing one Task-owned EvaluationProgram."""

    def __init__(
        self,
        records: Sequence[CaseRecord],
        task_or_evaluation: object,
        policy: EvaluatorPolicy,
        settings: Mapping[str, object] | None = None,
        *,
        seed: int = 0,
        plan_seed_provider: DevelopmentPlanSeedProvider | None = None,
        generator: GeneratorProgram | None = None,
        generator_state: GeneratorState | None = None,
        session_lock: object | None = None,
        candidate_execution: CandidateExecutionService | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Evaluator seed must be a non-negative integer")
        self._records: dict[str, CaseRecord] = {}
        self._candidate_to_controller: dict[str, str] = {}
        self._controller_to_candidate: dict[str, str] = {}
        self._policy = policy
        # Case aliases are stable across Agent evaluators and process restarts.
        # Scientific plan seeds remain independently scoped below.
        self._candidate_id_seed = 0
        self._settings = {
            **dict(settings or {}),
            "default_query": policy.default_query,
            "development_coordinate_mode": (
                policy.development_coordinate_mode.value
            ),
            "development_point_count": policy.development_point_count,
        }
        self._evaluation = resolve_evaluation_program(
            task_or_evaluation,
            self._settings,
        )
        self._next_plan_seed = seed
        self._next_call_index = 1
        self._plan_seed_provider = plan_seed_provider
        self._case_sessions = _CaseSessionBindings(
            generator,
            generator_state,
            allow_development_solution=(
                policy.allow_development_solution_session
            ),
            operation_lock=session_lock,
        )
        if (
            candidate_execution is not None
            and policy.allow_development_solution_session
        ):
            raise ValueError(
                "split Candidate execution does not expose the Development "
                "solution-session namespace"
            )
        self._candidate_execution = candidate_execution
        self._evaluation_cancellation = Event()
        self._lock = Lock()
        self.register(records)

    @property
    def cases(self) -> tuple[Case, ...]:
        with self._lock:
            controller_case_ids = tuple(self._records)
        return self._candidate_cases_for_controller_ids(controller_case_ids)

    def _candidate_cases_for_controller_ids(
        self,
        controller_case_ids: Sequence[str],
    ) -> tuple[Case, ...]:
        """Project registered controller Cases for an engine-owned facade."""

        prepared = tuple(controller_case_ids)
        with self._lock:
            unknown = set(prepared) - set(self._records)
            if unknown:
                raise KeyError(
                    f"controller Cases are not registered: {sorted(unknown)}"
                )
            rows = tuple(
                (
                    self._records[case_id].case,
                    self._controller_to_candidate[case_id],
                )
                for case_id in prepared
            )
        # Return a fresh detached projection on every read. One Candidate may
        # mutate its local Case without changing controller state or a later
        # Candidate's view.
        projected = tuple(
            candidate_facing_case(case, id_seed=self._candidate_id_seed)
            for case, _ in rows
        )
        if any(
            candidate.case_id != expected_id
            for candidate, (_case, expected_id) in zip(projected, rows)
        ):
            raise RuntimeError("candidate Case registry became inconsistent")
        return projected

    def _controller_case_ids_for_candidate_ids(
        self,
        candidate_case_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Resolve opaque ids without exposing the mapping to Candidate code."""

        prepared = tuple(candidate_case_ids)
        with self._lock:
            unknown = set(prepared) - set(self._candidate_to_controller)
            if unknown:
                raise KeyError(
                    f"cases are not available to this Agent: {sorted(unknown)}"
                )
            return tuple(
                self._candidate_to_controller[case_id]
                for case_id in prepared
            )

    def invoke_case_session(
        self,
        case_id: str,
        access: CaseAccessKind,
        method: str,
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
    ) -> object:
        """Invoke one scoped plugin method without exposing its session."""

        if not isinstance(access, CaseAccessKind):
            raise TypeError("case session access must be a CaseAccessKind")
        if (
            not isinstance(method, str)
            or not method.isidentifier()
            or method.startswith("_")
        ):
            raise ValueError("case session method must be a public identifier")
        if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
            raise TypeError("case session args must be a sequence")
        prepared_args = tuple(args)
        if kwargs is not None and not isinstance(kwargs, Mapping):
            raise TypeError("case session kwargs must be a mapping")
        prepared_kwargs = dict(kwargs or {})
        if any(not isinstance(key, str) for key in prepared_kwargs):
            raise TypeError("case session keyword names must be strings")
        if (
            access is CaseAccessKind.DEVELOPMENT_SOLUTION
            and not self._policy.allow_development_solution_session
        ):
            raise PermissionError(
                "Development solution session is not authorized for this Run"
            )
        if method not in self.case_session_methods(access):
            raise AttributeError(
                f"case session does not expose {method!r} for {access.value}"
            )
        controller_id = self._controller_case_ids_for_candidate_ids((case_id,))[0]
        with self._lock:
            record = self._records[controller_id]
        candidate_case = candidate_facing_case(
            record.case,
            id_seed=self._candidate_id_seed,
        )
        with self._case_sessions.bind(
            record.case,
            candidate_case,
            EvaluationPhase.DEVELOPMENT,
        ):
            if access is CaseAccessKind.PUBLIC_PROBLEM:
                session = open_public_case_session(candidate_case)
            elif access is CaseAccessKind.DEVELOPMENT_SOLUTION:
                session = open_development_solution_session(candidate_case)
            else:  # pragma: no cover - enum exhaustiveness
                raise ValueError("unknown case session access")
            operation = getattr(session, method)
            return operation(*prepared_args, **prepared_kwargs)

    def case_session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]:
        """Return only the session methods authorized for this Evaluator."""

        return self._case_sessions.methods(access)

    @property
    def evaluation_program(self) -> EvaluationProgram:
        return self._evaluation

    @property
    def candidate_invocation_process(self) -> str:
        """Describe where untrusted Candidate invocation currently executes."""

        value = getattr(
            self._candidate_execution,
            "candidate_invocation_process",
            "controller",
        )
        if value == "bounded_same_user_subprocess":
            return value
        return "controller"

    def cancel_pending_evaluations(self) -> None:
        """Revoke current and queued Development evaluations for this Agent."""

        self._evaluation_cancellation.set()

    @property
    def development_coordinate_mode(self) -> DevelopmentCoordinateMode:
        return self._policy.development_coordinate_mode

    @property
    def development_point_count(self) -> int:
        return self._policy.development_point_count

    @property
    def allow_grid_truth(self) -> bool:
        return self._policy.allow_grid_truth

    @property
    def allow_point_truth(self) -> bool:
        return self._policy.allow_point_truth

    @property
    def fixed_query(self) -> AnswerQuery | None:
        """Expose the user-declared query only in the explicit fixed mode."""

        if (
            self._policy.development_coordinate_mode
            is DevelopmentCoordinateMode.USER_FIXED_QUERY
        ):
            return self._policy.default_query
        return None

    def register(self, records: Sequence[CaseRecord]) -> None:
        """Register controller-owned truth for newly sampled development Cases."""

        additions = _records_with_visibility(
            records,
            CaseVisibility.DEVELOPMENT,
            "the Agent Evaluator",
        )
        if len({record.case.case_id for record in additions}) != len(additions):
            raise ValueError("an Evaluator batch cannot contain duplicate Case ids")
        projections = tuple(
            candidate_facing_case(
                record.case,
                id_seed=self._candidate_id_seed,
            )
            for record in additions
        )
        if len({case.case_id for case in projections}) != len(projections):
            raise ValueError("an Evaluator batch produced duplicate candidate Case ids")
        with self._lock:
            duplicate_ids = {
                record.case.case_id
                for record in additions
                if record.case.case_id in self._records
            }
            if duplicate_ids:
                raise ValueError(
                    f"Evaluator already contains Cases: {sorted(duplicate_ids)}"
                )
            candidate_collisions = {
                projection.case_id
                for projection in projections
                if projection.case_id in self._candidate_to_controller
            }
            if candidate_collisions:
                raise ValueError(
                    "Evaluator candidate Case id collision: "
                    f"{sorted(candidate_collisions)}"
                )
            self._records.update(
                {record.case.case_id: record for record in additions}
            )
            self._candidate_to_controller.update(
                {
                    projection.case_id: record.case.case_id
                    for record, projection in zip(additions, projections)
                }
            )
            self._controller_to_candidate.update(
                {
                    record.case.case_id: projection.case_id
                    for record, projection in zip(additions, projections)
                }
            )

    def evaluate(
        self,
        artifact: Artifact,
        request: EvaluationRequest,
    ) -> EvaluationResponse:
        """Evaluate one Artifact using a fresh private development plan."""

        return self.evaluate_many((artifact,), request)[0]

    def evaluate_many(
        self,
        artifacts: Sequence[Artifact],
        request: EvaluationRequest,
    ) -> tuple[EvaluationResponse, ...]:
        """Evaluate Artifacts fairly on one shared, freshly sampled plan set."""

        self._raise_if_cancelled()
        if not isinstance(request, EvaluationRequest):
            raise TypeError("Evaluator needs an EvaluationRequest")
        prepared_artifacts = tuple(artifacts)
        if not prepared_artifacts:
            raise ValueError("Evaluator batch needs at least one Artifact")
        if not request.case_ids:
            raise ValueError("Evaluator request needs at least one case id")
        if len(set(request.case_ids)) != len(request.case_ids):
            raise ValueError("Evaluator request cannot contain duplicate case ids")
        development_batch_deadline = self._development_batch_deadline()
        controller_case_ids = self._controller_case_ids_for_candidate_ids(
            request.case_ids
        )
        with self._lock:
            records = tuple(
                self._records[case_id] for case_id in controller_case_ids
            )
            call_index = self._next_call_index
            first_seed = self._next_plan_seed
            # Allocate the namespace before executing Task code. Concurrent or
            # failed calls must never reuse another call's private plan seeds.
            self._next_plan_seed += len(records)
            self._next_call_index += 1
        plans = tuple(
            self._development_plan(
                record,
                seed=(
                    first_seed + index
                    if self._plan_seed_provider is None
                    else self._plan_seed_provider(
                        call_index,
                        index,
                        record.case.case_id,
                    )
                ),
                requested_query=request.query,
            )
            for index, record in enumerate(records)
        )
        if request.return_truth:
            for plan in plans:
                if plan.query.kind is QueryKind.GRID:
                    if not self._policy.allow_grid_truth:
                        raise PermissionError(
                            "grid truth views are disabled for this run"
                        )
                elif plan.query.kind is QueryKind.POINTS:
                    if not self._policy.allow_point_truth:
                        raise PermissionError(
                            "point truth views are disabled for this run"
                        )
                else:  # Defense in depth for a malformed custom EvaluationProgram.
                    raise TypeError("evaluation plan query kind is invalid")

        truths = _materialize_truths(records, plans, check_cancelled=self._raise_if_cancelled)
        self._raise_if_cancelled()
        responses = []
        public_ids = dict(zip(controller_case_ids, request.case_ids))
        for index, artifact in enumerate(prepared_artifacts):
            try:
                report = _evaluate_records_with_truths(
                    artifact, records, self._evaluation, plans, truths,
                    case_sessions=self._case_sessions,
                    candidate_id_seed=self._candidate_id_seed,
                    candidate_execution=self._candidate_execution,
                    cancellation_event=_EvaluationCancellation(
                        self._evaluation_cancellation, _REQUEST_CANCELLATION.get(),
                    ),
                    development_batch_deadline=development_batch_deadline,
                )
                responses.append(_evaluation_response(
                    report, public_ids, truths, return_truth=request.return_truth,
                ))
            except Exception as error:
                # Only scalar results for already authorized Development Cases
                # leave this boundary. Incomplete candidates have no aggregate.
                partial = tuple(getattr(error, "_partial_case_evaluations", ()))
                completed_cases = [
                    {"case_id": public_ids[row.case_id],
                     "primary_error": row.primary_error, "metrics": dict(row.metrics)}
                    for row in partial if row.case_id in public_ids
                ]
                completed_ids = {row["case_id"] for row in completed_cases}
                diagnostic = candidate_diagnostic(error)
                case_index = getattr(error, "_candidate_case_index", None)
                if diagnostic and isinstance(case_index, int) and 0 <= case_index < len(request.case_ids):
                    diagnostic["case_id"] = request.case_ids[case_index]
                    error.candidate_diagnostic = diagnostic
                error.evaluation_progress = {
                    "complete": False,
                    "completed_candidates": [
                        {"index": i, "artifact_id": response.artifact_id,
                         "primary_error": response.primary_error,
                         "metrics": dict(response.metrics),
                         "cases": [{"case_id": row.case_id,
                                    "primary_error": row.primary_error,
                                    "metrics": dict(row.metrics)} for row in response.cases]}
                        for i, response in enumerate(responses)
                    ],
                    "interrupted_candidate": {
                        "index": index, "artifact_id": artifact.artifact_id,
                        "status": "incomplete", "requested_case_count": len(request.case_ids),
                        "completed_cases": completed_cases,
                        **({"diagnostic": diagnostic} if diagnostic else {}),
                        "not_completed_case_ids": [case_id for case_id in request.case_ids
                                                   if case_id not in completed_ids],
                    },
                    "unattempted_candidate_indices": list(range(index + 1, len(prepared_artifacts))),
                }
                raise
        return tuple(responses)

    def _raise_if_cancelled(self) -> None:
        if _EvaluationCancellation(
            self._evaluation_cancellation, _REQUEST_CANCELLATION.get(),
        ).is_set():
            raise CandidateExecutionCancelled(
                "Candidate evaluation was cancelled by its Agent turn"
            )

    def _development_batch_deadline(self) -> float | None:
        if self._candidate_execution is None:
            return None
        raw_timeout = getattr(
            self._candidate_execution,
            "development_batch_hard_wall_seconds",
            None,
        )
        if raw_timeout is None:
            return None
        if isinstance(raw_timeout, bool) or not isinstance(
            raw_timeout,
            (int, float),
        ):
            raise TypeError(
                "CandidateExecutionService Development batch hard wall must "
                "be a number"
            )
        timeout = float(raw_timeout)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(
                "CandidateExecutionService Development batch hard wall must "
                "be positive and finite"
            )
        return time.monotonic() + timeout

    def _development_plan(
        self,
        record: CaseRecord,
        *,
        seed: int,
        requested_query: AnswerQuery | None,
    ) -> EvaluationPlan:
        mode = self._policy.development_coordinate_mode
        if mode is DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS:
            if requested_query is not None:
                raise PermissionError(
                    "the Agent cannot choose coordinates in private_random_points mode"
                )
            factory = getattr(
                self._evaluation,
                "make_private_development_plan",
                None,
            )
            if not callable(factory):
                raise TypeError(
                    "this Task does not support private random Development points; "
                    "select user_fixed_query explicitly"
                )
            plan = factory(record.case, self._settings, seed)
            if not isinstance(plan, EvaluationPlan):
                raise TypeError(
                    "Task private Development planner must return EvaluationPlan"
                )
            return self._validated_development_plan(
                plan,
                seed=seed,
                private_random=True,
            )

        if mode is DevelopmentCoordinateMode.USER_FIXED_QUERY:
            fixed = self._policy.default_query
            if requested_query is not None and requested_query != fixed:
                raise PermissionError(
                    "the Agent cannot override the user-fixed Development query"
                )
            requested_query = fixed

        expected_query = requested_query or self._policy.default_query
        plan = self._evaluation.make_plan(
            record.case,
            EvaluationPhase.DEVELOPMENT,
            self._settings,
            seed=seed,
            request=requested_query,
        )
        return self._validated_development_plan(
            plan,
            seed=seed,
            expected_query=expected_query,
        )

    def _validated_development_plan(
        self,
        plan: object,
        *,
        seed: int,
        expected_query: AnswerQuery | None = None,
        private_random: bool = False,
    ) -> EvaluationPlan:
        """Fail closed when a custom Task violates the coordinate contract."""

        if not isinstance(plan, EvaluationPlan):
            raise TypeError("Task Development planner must return EvaluationPlan")
        if plan.phase is not EvaluationPhase.DEVELOPMENT:
            raise ValueError("Task Development plan has the wrong phase")
        if plan.seed != seed:
            raise ValueError("Task Development plan changed its allocated seed")
        if private_random:
            if plan.query.kind is not QueryKind.POINTS:
                raise ValueError(
                    "private_random_points mode requires a POINTS evaluation plan"
                )
            if tuple(plan.query.fields) != tuple(self._policy.default_query.fields):
                raise ValueError(
                    "private_random_points mode changed the user-declared fields"
                )
            if len(plan.query.points) != self._policy.development_point_count:
                raise ValueError(
                    "private_random_points mode returned the wrong point count"
                )
        elif expected_query is None or plan.query != expected_query:
            raise ValueError(
                "Task Development plan changed the selected coordinate query"
            )
        return plan


class HiddenVerifier:
    """Controller-only service executing frozen official or terminal plans."""

    def __init__(
        self,
        task_or_evaluation: object,
        query: AnswerQuery,
        settings: Mapping[str, object] | None = None,
        *,
        phase: EvaluationPhase = EvaluationPhase.OFFICIAL,
        seed: int = 0,
        plan_seed_provider: HiddenPlanSeedProvider | None = None,
        generator: GeneratorProgram | None = None,
        generator_state: GeneratorState | None = None,
        session_lock: object | None = None,
        candidate_execution: CandidateExecutionService | None = None,
    ) -> None:
        if phase not in (EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL):
            raise ValueError("HiddenVerifier phase must be official or terminal")
        self._settings = {
            **dict(settings or {}),
            "default_query": query,
        }
        self._evaluation = resolve_evaluation_program(
            task_or_evaluation,
            self._settings,
        )
        self._phase = phase
        self._seed = seed
        self._candidate_id_seed = 0
        self._plan_seed_provider = plan_seed_provider
        self._case_sessions = _CaseSessionBindings(
            generator,
            generator_state,
            allow_development_solution=False,
            operation_lock=session_lock,
            evaluation_device_pool=current_evaluation_device_pool(),
        )
        self._candidate_execution = candidate_execution
        self._plans: dict[str, EvaluationPlan] = {}
        self._plan_lock = Lock()
        self._truths: dict[str, AnswerView] = {}
        self._truth_errors: dict[str, Exception] = {}
        self._truth_lock = Lock()

    @property
    def evaluation_program(self) -> EvaluationProgram:
        return self._evaluation

    def evaluate(
        self,
        artifact: Artifact,
        records: Sequence[CaseRecord],
    ) -> ArtifactEvaluation:
        hidden_records = _records_with_visibility(
            records,
            CaseVisibility.HIDDEN,
            "the Hidden Verifier",
        )
        if not hidden_records:
            raise ValueError("cannot evaluate an Artifact on an empty case set")
        try:
            plans = tuple(
                self._plan_for(record, index)
                for index, record in enumerate(hidden_records)
            )
            truths = tuple(
                self._truth_for(record, plan)
                for record, plan in zip(hidden_records, plans)
            )
        except Exception as error:
            failure = CandidateExecutionInfrastructureError(
                "Controller could not prepare the hidden evaluation plans or truth"
            )
            failure._hidden_evaluation_stage = "prepare_hidden_inputs"
            raise failure from error
        return _evaluate_records_with_truths(
            artifact,
            hidden_records,
            self._evaluation,
            plans,
            truths,
            case_sessions=self._case_sessions,
            candidate_id_seed=self._candidate_id_seed,
            candidate_execution=self._candidate_execution,
        )

    def evaluate_incremental(
        self, artifact, records, *, progress, before_batch, checkpoint, batch_size=28,
    ):
        """Continue one comparison using its unchanged plans and per-Case scores.

        Each bounded batch still uses the service's ordinary parallel execution.
        Aggregation runs once on the complete original cohort, never on means of
        smaller batches. The caller owns persistence and allocation decisions.
        """
        from .campaign_continuation import CampaignPaused

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("incremental hidden batch size must be positive")
        hidden_records = _records_with_visibility(records, CaseVisibility.HIDDEN, "the Hidden Verifier")
        if not hidden_records:
            raise ValueError("cannot evaluate an Artifact on an empty case set")
        self._plans = progress.setdefault("plans", {})
        self._truths = progress.setdefault("truths", {})
        completed = progress.setdefault("cases", {}).setdefault(artifact.artifact_id, {})
        identifiers = tuple(record.case.case_id for record in hidden_records)
        if len(identifiers) != len(set(identifiers)) or set(completed) - set(identifiers):
            raise ValueError("incremental hidden results do not match the frozen cohort")
        indices = {identifier: index for index, identifier in enumerate(identifiers)}
        for start in range(0, len(hidden_records), batch_size):
            chunk = tuple(record for record in hidden_records[start:start + batch_size]
                          if record.case.case_id not in completed)
            if not chunk:
                continue
            before_batch()
            try:
                plans = tuple(self._plan_for(record, indices[record.case.case_id]) for record in chunk)
                truths = tuple(self._truth_for(record, plan) for record, plan in zip(chunk, plans))
            except Exception as error:
                failure = CandidateExecutionInfrastructureError("Controller could not prepare the hidden evaluation plans or truth")
                failure._hidden_evaluation_stage = "prepare_hidden_inputs"
                raise failure from error
            # Persist exact finite plans/truth before launching Candidate work.
            checkpoint()
            try:
                results = _evaluate_records_with_truths(
                    artifact, chunk, self._evaluation, plans, truths,
                    case_sessions=self._case_sessions,
                    candidate_id_seed=self._candidate_id_seed,
                    candidate_execution=self._candidate_execution,
                    return_case_results=True,
                )
            except BaseException as error:
                partial = getattr(error, "_partial_case_evaluations", ())
                for result in partial:
                    if result.case_id not in {record.case.case_id for record in chunk}:
                        raise RuntimeError("partial evaluation returned an unknown Case") from error
                    completed[result.case_id] = result
                checkpoint()
                if getattr(error, "_allocation_pause_requested", False):
                    raise CampaignPaused(stage="hidden_evaluation") from error
                raise
            for result in results:
                completed[result.case_id] = result
            checkpoint()
        plans = tuple(self._plans[identifier] for identifier in identifiers)
        results = tuple(completed[identifier] for identifier in identifiers)
        return _aggregate_case_results(artifact, self._evaluation, results, plans)

    def _truth_for(
        self,
        record: CaseRecord,
        plan: EvaluationPlan,
    ) -> AnswerView:
        """Materialize one immutable hidden view for the whole comparison."""

        case_id = record.case.case_id
        # TruthProgram implementations may be stateful or backed by a live
        # numerical service. Serialize the first call and memoize failures so
        # every Candidate sees exactly the same hidden evaluation contract.
        with self._truth_lock:
            if case_id in self._truth_errors:
                raise self._truth_errors[case_id]
            cached = self._truths.get(case_id)
            if cached is not None:
                return cached
            try:
                truth = _freeze_answer_view(record.truth.answer(plan.query))
            except Exception as exc:
                self._truth_errors[case_id] = exc
                raise
            self._truths[case_id] = truth
            return truth

    def _plan_for(self, record: CaseRecord, index: int) -> EvaluationPlan:
        case_id = record.case.case_id
        with self._plan_lock:
            plan = self._plans.get(case_id)
            if plan is None:
                plan = self._evaluation.make_plan(
                    record.case,
                    self._phase,
                    self._settings,
                    seed=(
                        self._seed + index
                        if self._plan_seed_provider is None
                        else self._plan_seed_provider(index, case_id)
                    ),
                )
                self._plans[case_id] = plan
            return plan


def evaluate_records(
    artifact: Artifact,
    records: Sequence[CaseRecord],
    evaluation: EvaluationProgram,
    plans: Sequence[EvaluationPlan],
    *,
    case_sessions: _CaseSessionBindings | None = None,
    candidate_execution: CandidateExecutionService | None = None,
) -> tuple[ArtifactEvaluation, tuple[AnswerView, ...]]:
    prepared_records = tuple(records)
    prepared_plans = tuple(plans)
    if not prepared_records:
        raise ValueError("cannot evaluate an Artifact on an empty case set")
    if len(prepared_records) != len(prepared_plans):
        raise ValueError("evaluation records and plans must have equal length")

    truths = _materialize_truths(prepared_records, prepared_plans)
    report = _evaluate_records_with_truths(
        artifact,
        prepared_records,
        evaluation,
        prepared_plans,
        truths,
        case_sessions=case_sessions,
        candidate_execution=candidate_execution,
    )
    return report, truths


def _materialize_truths(
    records: Sequence[CaseRecord],
    plans: Sequence[EvaluationPlan],
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[AnswerView, ...]:
    answers = []
    for record, plan in zip(records, plans):
        if check_cancelled is not None:
            check_cancelled()
        answers.append(record.truth.answer(plan.query))
    return tuple(answers)


def _freeze_answer_view(view: AnswerView) -> AnswerView:
    """Copy a finite Truth view into the immutable evaluator boundary."""

    if not isinstance(view, AnswerView):
        raise TypeError("TruthProgram.answer() must return an AnswerView")
    return AnswerView(
        kind=view.kind,
        fields=MappingProxyType(
            {name: tuple(values) for name, values in view.fields.items()}
        ),
        axes=MappingProxyType(
            {name: tuple(values) for name, values in view.axes.items()}
        ),
        points=tuple(MappingProxyType(dict(point)) for point in view.points),
        shape=tuple(view.shape),
    )


def _evaluate_records_with_truths(
    artifact: Artifact,
    records: Sequence[CaseRecord],
    evaluation: EvaluationProgram,
    plans: Sequence[EvaluationPlan],
    truths: Sequence[AnswerView],
    *,
    case_sessions: _CaseSessionBindings | None = None,
    candidate_id_seed: int = 0,
    candidate_execution: CandidateExecutionService | None = None,
    cancellation_event: Event | None = None,
    development_batch_deadline: float | None = None,
    return_case_results: bool = False,
) -> ArtifactEvaluation | tuple[CaseEvaluation, ...]:
    bindings = (
        _CaseSessionBindings()
        if case_sessions is None
        else case_sessions
    )
    candidate_cases = tuple(
        candidate_facing_case(
            record.case,
            id_seed=candidate_id_seed,
        )
        for record in records
    )
    if candidate_execution is None:
        candidate_results = []
        for record, candidate_case, plan, truth in zip(
            records,
            candidate_cases,
            plans,
            truths,
        ):
            # In-process tasks cannot preempt an entered numerical function,
            # but cancellation must stop the next Case from starting.
            if cancellation_event is not None and cancellation_event.is_set():
                error = CandidateExecutionCancelled("Candidate evaluation was cancelled")
                _retain_scored_prefix(error, records, candidate_cases, candidate_results)
                raise error
            with bindings.bind(record.case, candidate_case, plan.phase):
                try:
                    result = evaluation.evaluate(
                        artifact,
                        candidate_case,
                        truth,
                        plan,
                    )
                except Exception as error:
                    _retain_scored_prefix(error, records, candidate_cases, candidate_results)
                    _mark_hidden_evaluation_stage(error, "evaluate_candidate_and_score")
                    raise
            candidate_results.append(result)
    else:
        execute = getattr(evaluation, "execute_candidate", None)
        score = getattr(evaluation, "score_candidate", None)
        if (
            getattr(evaluation, "supports_split_candidate_execution", False)
            is not True
            or not callable(execute)
            or not callable(score)
        ):
            raise TypeError(
                "configured CandidateExecutionService requires split Candidate "
                "execution from the Task EvaluationProgram"
            )
        invocation_rows = {
            candidate_case.case_id: (record, candidate_case, plan)
            for record, candidate_case, plan in zip(
                records,
                candidate_cases,
                plans,
            )
        }
        if len(invocation_rows) != len(candidate_cases):
            raise RuntimeError("candidate-facing hidden Case ids are not unique")

        public_methods = bindings.methods(CaseAccessKind.PUBLIC_PROBLEM)
        persistent_sessions = (
            {
                case_id: bindings.persistent_public_session(
                    record.case,
                    candidate_case,
                )
                for case_id, (record, candidate_case, _plan) in invocation_rows.items()
            }
            if public_methods
            else {}
        )

        def invoke_public(
            case_id: str,
            method: str,
            args: tuple[object, ...],
            kwargs: Mapping[str, object],
        ) -> object:
            try:
                session = persistent_sessions[case_id]
            except KeyError as exc:
                raise KeyError("Candidate requested an unknown hidden Case") from exc
            return session.invoke(method, args, kwargs)

        def close_public(case_id: str) -> None:
            try:
                session = persistent_sessions[case_id]
            except KeyError as exc:
                raise KeyError("Candidate requested an unknown hidden Case") from exc
            session.close()

        outputs: tuple[object, ...] | None = None
        primary_error: BaseException | None = None
        try:
            outputs = candidate_execution.execute_cases(
                artifact,
                candidate_cases,
                plans,
                evaluation,
                public_session_methods=public_methods,
                invoke_public_session=(invoke_public if public_methods else None),
                close_public_session=(close_public if public_methods else None),
                cancellation_event=cancellation_event,
                development_batch_deadline=development_batch_deadline,
            )
        except BaseException as error:
            primary_error = error
        finally:
            for session in persistent_sessions.values():
                try:
                    session.close()
                except BaseException as cleanup_error:
                    if primary_error is None:
                        primary_error = cleanup_error
                    else:
                        _attach_case_session_cleanup_error(
                            primary_error,
                            cleanup_error,
                        )
        if primary_error is not None:
            # The service can return already completed outputs when allocation
            # cancellation drains a parallel batch. Score and retain those rows
            # before propagating the pause; they must never execute twice.
            partial_results = []
            partial_errors = []
            for index, output in sorted(getattr(primary_error, "_candidate_partial_outputs", {}).items()):
                try:
                    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
                        raise ValueError("partial Candidate output index is invalid")
                    result = score(artifact, candidate_cases[index], truths[index], plans[index], output)
                    partial_results.append(_canonical_case_result(records[index], candidate_cases[index], result))
                except Exception as score_error:
                    partial_errors.append(score_error)
            primary_error._partial_case_evaluations = tuple(partial_results)
            if partial_errors:
                # A scoring failure remains a real scientific/system failure;
                # allocation cancellation must not silently turn it into a retry.
                score_error = partial_errors[0]
                score_error._partial_case_evaluations = tuple(partial_results)
                score_error._partial_case_scoring_errors = tuple(partial_errors)
                _mark_hidden_evaluation_stage(score_error, "score_candidate")
                raise score_error from primary_error
            _mark_hidden_evaluation_stage(primary_error, "execute_candidate")
            raise primary_error
        assert outputs is not None
        if len(outputs) != len(candidate_cases):
            raise RuntimeError(
                "CandidateExecutionService returned the wrong number of outputs"
            )
        candidate_results = []
        for candidate_case, truth, plan, output in zip(
            candidate_cases, truths, plans, outputs,
        ):
            try:
                result = score(artifact, candidate_case, truth, plan, output)
            except Exception as error:
                # Scorers also validate candidate output. Exception class alone
                # does not establish whether the candidate or scorer is wrong;
                # retain the exact stage and cause for trusted diagnostics.
                _retain_scored_prefix(error, records, candidate_cases, candidate_results)
                _mark_hidden_evaluation_stage(error, "score_candidate")
                raise
            candidate_results.append(result)

    results = tuple(_canonical_case_result(record, candidate_case, result)
                    for record, candidate_case, result in zip(records, candidate_cases, candidate_results))
    if return_case_results:
        return results
    try:
        return _aggregate_case_results(artifact, evaluation, results, plans)
    except Exception as error:
        error._partial_case_evaluations = results
        _mark_hidden_evaluation_stage(error, "aggregate_candidate")
        raise


def _retain_scored_prefix(error, records, candidate_cases, results):
    error._partial_case_evaluations = tuple(
        _canonical_case_result(record, candidate_case, result)
        for record, candidate_case, result in zip(records, candidate_cases, results)
    )


def _canonical_case_result(record, candidate_case, result):
    if result.case_id != candidate_case.case_id:
        raise ValueError("EvaluationProgram returned the wrong Case id")
    macro_match = result.macro_match
    if macro_match is not None:
        if macro_match.case_id != candidate_case.case_id:
            raise ValueError("EvaluationProgram returned the wrong macro Case id")
        macro_match = replace(macro_match, case_id=record.case.case_id)
    return replace(result, case_id=record.case.case_id, macro_match=macro_match)


def _aggregate_case_results(artifact, evaluation, results, plans):
    try:
        report = evaluation.aggregate(artifact, results, plans)
        if report.artifact is not artifact:
            raise ValueError("EvaluationProgram.aggregate returned the wrong Artifact")
    except Exception as error:
        if any(plan.phase is not EvaluationPhase.DEVELOPMENT for plan in plans):
            failure = CandidateExecutionInfrastructureError(
                "Controller could not aggregate the hidden evaluation results"
            )
            failure._hidden_evaluation_stage = "aggregate_case_scores"
            raise failure from error
        raise
    return report


def _mark_hidden_evaluation_stage(error: BaseException, stage: str) -> None:
    if not hasattr(error, "_hidden_evaluation_stage"):
        try:
            error._hidden_evaluation_stage = stage
        except Exception:  # pragma: no cover - immutable foreign exception
            pass


def _evaluation_response(
    evaluation: ArtifactEvaluation,
    candidate_ids_by_controller_id: Mapping[str, str],
    truths: Sequence[AnswerView],
    *,
    return_truth: bool,
) -> EvaluationResponse:
    # Per-Case scalar feedback is independent of permission to read answer
    # fields or their coordinates. Only public development Case IDs cross this
    # boundary; the controller-private report and answer views stay private.
    truth_by_id = dict(zip(candidate_ids_by_controller_id, truths)) if return_truth else {}
    return EvaluationResponse(
        artifact_id=evaluation.artifact.artifact_id,
        primary_error=evaluation.primary_error,
        metrics=evaluation.metrics,
        cases=tuple(
            EvaluationCaseResult(
                case_id=candidate_ids_by_controller_id[result.case_id],
                primary_error=result.primary_error,
                metrics=result.metrics,
                prediction=result.prediction if return_truth else None,
                truth=truth_by_id[result.case_id] if return_truth else None,
            )
            for result in evaluation.cases
        ),
    )


def _attach_case_session_cleanup_error(
    primary: BaseException,
    cleanup_error: BaseException,
) -> None:
    """Retain remote-session cleanup failures without hiding the first error."""

    attribute = "_pde_session_cleanup_errors"
    attached = tuple(getattr(primary, attribute, ())) + (cleanup_error,)
    try:
        setattr(primary, attribute, attached)
    except BaseException:  # pragma: no cover - unusual immutable exception
        pass
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(
            "case-session cleanup failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def _records_with_visibility(
    records: Sequence[CaseRecord],
    visibility: CaseVisibility,
    owner: str,
) -> tuple[CaseRecord, ...]:
    materialized = tuple(records)
    if any(not isinstance(record, CaseRecord) for record in materialized):
        raise TypeError(f"{owner} only accepts CaseRecord values")
    for record in materialized:
        require_controller_case_id(record.case.case_id)
    if any(record.case.visibility is not visibility for record in materialized):
        raise ValueError(f"{owner} only accepts {visibility.value} Cases")
    return materialized
