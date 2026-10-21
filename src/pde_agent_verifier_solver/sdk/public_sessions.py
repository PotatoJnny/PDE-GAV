"""Equation-neutral access to scoped numerical case sessions.

Some manufactured questions expose large initial data or source terms through
a bounded numerical session rather than copying those arrays into every
``Case``. Development may also expose a user-authorized completed-solution
session. The equation plugin owns both sessions. The shared evaluator binds
their factories only for the duration of one candidate call.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Callable, Iterator, Sequence

from .models import Case


PublicSessionFactory = Callable[[Case], object]


class _PublicSessionBinding:
    def __init__(
        self,
        factory: PublicSessionFactory | None,
        case: Case,
        allowed_methods: Sequence[str],
        *,
        unavailable_message: str = (
            "this equation does not expose a public case session"
        ),
    ) -> None:
        self._factory = factory
        self._case = case
        self._allowed_methods = frozenset(allowed_methods)
        self._unavailable_message = unavailable_message
        self._lock = RLock()
        self._active = True
        self._target: object | None = None
        self._proxy: _RevocablePublicSession | None = None

    def open(self, case: Case) -> object:
        with self._lock:
            self._require_active()
            # PDE Tasks such as GKW deliberately snapshot/freeze the Case
            # again before invoking Candidate code. Object identity therefore
            # is not a valid capability key; the evaluator-issued opaque id is.
            if case.case_id != self._case.case_id:
                raise ValueError(
                    "a public session can only open for the current Case id"
                )
            if self._factory is None:
                raise RuntimeError(self._unavailable_message)
            if self._proxy is None:
                target = self._factory(case)
                if target is None:
                    raise TypeError("public session factory returned None")
                self._target = target
                self._proxy = _RevocablePublicSession(self)
            return self._proxy

    def attribute(self, name: str) -> object:
        with self._lock:
            target = self._require_target()
            if name not in self._allowed_methods:
                raise AttributeError(
                    f"public case session does not expose {name!r}"
                )
            value = getattr(target, name, None)
            if not callable(value):
                raise TypeError(
                    f"declared public session method {name!r} is not callable"
                )
            return _RevocablePublicMethod(self, name)

    def invoke(self, name: str, args, kwargs) -> object:
        with self._lock:
            target = self._require_target()
            if name not in self._allowed_methods:
                raise AttributeError(
                    f"public case session does not expose {name!r}"
                )
            method = getattr(target, name, None)
            if not callable(method):
                raise TypeError(
                    f"declared public session method {name!r} is not callable"
                )
            return method(*args, **kwargs)

    def close(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            target = self._target
            try:
                close = getattr(target, "close", None) if target is not None else None
                if callable(close):
                    close()
            finally:
                self._target = None
                self._proxy = None

    def _require_target(self) -> object:
        self._require_active()
        if self._target is None:
            raise RuntimeError("public session target is unavailable")
        return self._target

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("public case session is no longer active")


class _RevocablePublicMethod:
    __slots__ = ("__binding", "__name")

    def __init__(self, binding: _PublicSessionBinding, name: str) -> None:
        self.__binding = binding
        self.__name = name

    def __call__(self, *args, **kwargs) -> object:
        binding = object.__getattribute__(
            self, "_RevocablePublicMethod__binding"
        )
        name = object.__getattribute__(self, "_RevocablePublicMethod__name")
        return binding.invoke(name, args, kwargs)

    def __getattribute__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(
                f"public session method does not expose {name!r}"
            )
        return object.__getattribute__(self, name)


class _RevocablePublicSession:
    __slots__ = ("__binding",)

    def __init__(self, binding: _PublicSessionBinding) -> None:
        self.__binding = binding

    def __getattribute__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(
                f"public case session does not expose {name!r}"
            )
        binding = object.__getattribute__(
            self, "_RevocablePublicSession__binding"
        )
        return binding.attribute(name)


_CURRENT_PUBLIC_SESSION_BINDING: ContextVar[_PublicSessionBinding | None] = (
    ContextVar("pde_candidate_public_session_binding", default=None)
)
_CURRENT_DEVELOPMENT_SOLUTION_SESSION_BINDING: ContextVar[
    _PublicSessionBinding | None
] = ContextVar("pde_candidate_development_solution_session_binding", default=None)


def open_public_case_session(case: Case) -> object:
    """Open the equation-owned public numerical session for ``case``.

    This function is part of the candidate-facing SDK. It is available only
    while a candidate is being evaluated and only when the selected equation
    plugin advertises a public session factory.
    """

    if not isinstance(case, Case):
        raise TypeError("a public case session requires a Case")
    binding = _CURRENT_PUBLIC_SESSION_BINDING.get()
    if binding is None:
        raise RuntimeError("this equation does not expose a public case session")
    return binding.open(case)


def open_development_solution_session(case: Case) -> object:
    """Open the user-authorized Development solution session for ``case``.

    The canonical Evaluator binds this namespace only for Development calls
    whose Run policy explicitly enables it. Official and Terminal evaluation
    never bind it, even when the Generator implements the capability.
    """

    if not isinstance(case, Case):
        raise TypeError("a development solution session requires a Case")
    binding = _CURRENT_DEVELOPMENT_SOLUTION_SESSION_BINDING.get()
    if binding is None:
        raise RuntimeError(
            "this evaluation does not expose a development solution session"
        )
    return binding.open(case)


@contextmanager
def bind_public_session_factory(
    factory: PublicSessionFactory | None,
    case: Case,
    *,
    allowed_methods: Sequence[str] = (),
) -> Iterator[None]:
    """Bind one allowlisted, memoized, revocable session to a candidate call."""

    if factory is not None and not callable(factory):
        raise TypeError("public session factory must be callable")
    if not isinstance(case, Case):
        raise TypeError("public session binding requires a Case")
    if isinstance(allowed_methods, (str, bytes)):
        raise TypeError("public session allowed_methods must be a sequence")
    methods = tuple(allowed_methods)
    if any(
        not isinstance(name, str)
        or not name.isidentifier()
        or name.startswith("_")
        for name in methods
    ):
        raise ValueError(
            "public session allowed methods must be public identifiers"
        )
    if len(methods) != len(set(methods)):
        raise ValueError("public session allowed methods must be unique")
    if factory is not None and not methods:
        raise ValueError("a public session factory needs an explicit method allowlist")
    if factory is None and methods:
        raise ValueError("public session methods require a factory")
    binding = _PublicSessionBinding(factory, case, methods)
    token = _CURRENT_PUBLIC_SESSION_BINDING.set(binding)
    with _close_and_reset_preserving_primary(
        binding,
        _CURRENT_PUBLIC_SESSION_BINDING,
        token,
    ):
        yield


@contextmanager
def bind_development_solution_session_factory(
    factory: PublicSessionFactory | None,
    case: Case,
    *,
    allowed_methods: Sequence[str] = (),
) -> Iterator[None]:
    """Bind one allowlisted Development solution session to a candidate call."""

    if factory is not None and not callable(factory):
        raise TypeError("development solution session factory must be callable")
    if not isinstance(case, Case):
        raise TypeError("development solution session binding requires a Case")
    if isinstance(allowed_methods, (str, bytes)):
        raise TypeError(
            "development solution session allowed_methods must be a sequence"
        )
    methods = tuple(allowed_methods)
    if any(
        not isinstance(name, str)
        or not name.isidentifier()
        or name.startswith("_")
        for name in methods
    ):
        raise ValueError(
            "development solution session allowed methods must be public identifiers"
        )
    if len(methods) != len(set(methods)):
        raise ValueError(
            "development solution session allowed methods must be unique"
        )
    if factory is not None and not methods:
        raise ValueError(
            "a development solution session factory needs an explicit method allowlist"
        )
    if factory is None and methods:
        raise ValueError(
            "development solution session methods require a factory"
        )
    binding = _PublicSessionBinding(
        factory,
        case,
        methods,
        unavailable_message=(
            "this evaluation does not expose a development solution session"
        ),
    )
    token = _CURRENT_DEVELOPMENT_SOLUTION_SESSION_BINDING.set(binding)
    with _close_and_reset_preserving_primary(
        binding,
        _CURRENT_DEVELOPMENT_SOLUTION_SESSION_BINDING,
        token,
    ):
        yield


@contextmanager
def _close_and_reset_preserving_primary(
    binding: _PublicSessionBinding,
    binding_var: ContextVar[_PublicSessionBinding | None],
    token,
) -> Iterator[None]:
    """Revoke one binding without replacing an in-flight Candidate failure."""

    primary: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            binding.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            binding_var.reset(token)
        except BaseException as exc:  # pragma: no cover - ContextVar invariant
            cleanup_errors.append(exc)

        if cleanup_errors:
            if primary is None:
                primary = cleanup_errors.pop(0)
                for cleanup_error in cleanup_errors:
                    _attach_cleanup_error(primary, cleanup_error)
                raise primary
            for cleanup_error in cleanup_errors:
                _attach_cleanup_error(primary, cleanup_error)


def _attach_cleanup_error(
    primary: BaseException,
    cleanup_error: BaseException,
) -> None:
    """Retain cleanup failures on the primary exception for audit/debugging."""

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


__all__ = [
    "query_public_cases",
    "PublicSessionFactory",
    "open_development_solution_session",
    "open_public_case_session",
]


class _BatchSessionBinding:
    def __init__(self, bindings, query_many=None):
        self.bindings = bindings
        self.query_many = query_many

    def open(self, case):
        try:
            binding = self.bindings[case.case_id]
        except KeyError:
            raise PermissionError("Case is not in this solver batch") from None
        return binding.open(case)


@contextmanager
def bind_public_session_batch(factories, cases, *, allowed_methods=(), query_many=None):
    """Worker-owned scope. Each Case retains its own revocable capability."""
    from contextlib import ExitStack
    cases = tuple(cases)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("batch Case ids must be distinct")
    with ExitStack() as stack:
        bindings = {}
        for case in cases:
            stack.enter_context(bind_public_session_factory(
                factories[case.case_id], case, allowed_methods=allowed_methods))
            bindings[case.case_id] = _CURRENT_PUBLIC_SESSION_BINDING.get()
        token = _CURRENT_PUBLIC_SESSION_BINDING.set(_BatchSessionBinding(bindings, query_many))
        try:
            yield
        finally:
            _CURRENT_PUBLIC_SESSION_BINDING.reset(token)


def query_public_cases(cases, method, points_by_case, *, max_workers=32, combine_transport=True):
    """Query different authorized Cases concurrently; preserve Case-keyed results.

    Open sessions on the bound thread before dispatch. Inputs stay per Case, so
    different geometry and point counts need no padding or interpolation.
    """
    from concurrent.futures import ThreadPoolExecutor
    cases = tuple(cases)
    if (isinstance(max_workers, bool) or not isinstance(max_workers, int)
            or not 1 <= max_workers <= 128):
        raise ValueError("max_workers must be in [1, 128]")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("batch Case ids must be distinct")
    if set(points_by_case) != {case.case_id for case in cases}:
        raise ValueError("supply exactly one point array per Case id")
    binding = _CURRENT_PUBLIC_SESSION_BINDING.get()
    if (combine_transport and isinstance(binding, _BatchSessionBinding)
            and binding.query_many is not None and cases):
        # Keep the same per-Case revocation and method boundaries as single calls.
        from contextlib import ExitStack
        with ExitStack() as stack:
            for case in sorted(cases, key=lambda c: c.case_id):
                binding.open(case)
                member = binding.bindings[case.case_id]
                stack.enter_context(member._lock)
                member.attribute(method)
            return binding.query_many(cases, method, points_by_case, max_workers)
    calls = [(case.case_id, getattr(open_public_case_session(case), method)) for case in cases]
    if not calls:
        return {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as executor:
        futures = {key: executor.submit(call, points_by_case[key]) for key, call in calls}
        return {key: future.result() for key, future in futures.items()}
