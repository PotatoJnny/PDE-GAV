"""Controller-private evaluation of equation-owned macroscopic objectives.

The physics stays in :class:`~pde_agent_verifier_solver.sdk.MacroMatcher`.
This module only fixes the interaction protocol around it: materialize a
cohort's targets once, give candidates isolated public snapshots, preserve a
common paired cohort, account for development credits, and return sanitized
scores without exposing ``MacroTarget`` or ``TruthProgram`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from inspect import getattr_static
from itertools import islice
from math import isfinite
from numbers import Real
from pathlib import Path
from random import SystemRandom
import struct
from threading import Lock
from time import perf_counter
from typing import Callable, ClassVar, Mapping, Sequence, TypeVar, cast

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    MacroMatcher,
    CaseSource,
    CaseVisibility,
    QueryKind,
    MacroTarget,
    materialize_macro_target,
)
from pde_agent_verifier_solver.sdk.public_sessions import (
    PublicSessionFactory,
    bind_public_session_factory,
)
from pde_agent_verifier_solver.sdk.continuous_solution import (
    PointQueryResult,
    PreparedSolutionResult,
    TimingAssurance,
    measure_point_query,
    measure_prepare_solution,
)


QueryFactory = Callable[[Case], AnswerQuery]
Clock = Callable[[], float]


from ._macro_cohort_store import (
    MacroCohortStoreError,
    PrivateMacroCohortStore,
    SafeNumpyValueCodec,
)
from .candidate_case import (
    CANDIDATE_CASE_ID_PREFIX,
    candidate_case_id,
    candidate_facing_case,
)

_SHA256_LENGTH = 64
_SnapshotT = TypeVar("_SnapshotT")
_SNAPSHOT_IN_PROGRESS = object()


def _snapshot_value(value: _SnapshotT) -> _SnapshotT:
    """Recursively detach one shared macro-protocol value.

    Public PDE cases commonly use immutable mappings and read-only NumPy
    arrays. deepcopy cannot copy MappingProxyType and does not state the
    writable-buffer semantics candidates need. This utility normalizes
    mappings to private dictionaries, copies arrays onto writable buffers, and
    reconstructs the shared Case/query/answer/target records recursively.
    """

    return cast(_SnapshotT, _snapshot_value_recursive(value, {}))


def _snapshot_value_recursive(
    value: object,
    memo: dict[int, object],
) -> object:
    if value is None or isinstance(
        value,
        (Enum, str, bytes, bool, int, float, complex, Real),
    ):
        return value

    if isinstance(value, bytearray):
        copied_bytes = bytearray(value)
        memo[id(value)] = copied_bytes
        return copied_bytes

    identity = id(value)
    cached = memo.get(identity)
    if cached is _SNAPSHOT_IN_PROGRESS:
        raise TypeError("cyclic immutable macro snapshot values are unsupported")
    if cached is not None:
        return cached

    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            copied = np.array(value, copy=True, order="K", subok=False)
            copied.setflags(write=True)
            memo[identity] = copied
            if copied.dtype.hasobject:
                for index in np.ndindex(value.shape):
                    copied[index] = _snapshot_value_recursive(value[index], memo)
            return copied
        if isinstance(value, np.generic):
            return value.copy()
        if isinstance(value, _MMapFloatSequence):
            copied = _MMapFloatSequence(
                np.array(value._array, copy=True, order="K", subok=False)
            )
            memo[identity] = copied
            return copied

    if isinstance(value, Case):
        memo[identity] = _SNAPSHOT_IN_PROGRESS
        copied = Case(
            case_id=value.case_id,
            source=value.source,
            visibility=value.visibility,
            public_data=_snapshot_value_recursive(value.public_data, memo),
            round_index=value.round_index,
            parent_case_id=value.parent_case_id,
            tags=tuple(
                _snapshot_value_recursive(item, memo) for item in value.tags
            ),
        )
        memo[identity] = copied
        return copied
    if isinstance(value, AnswerQuery):
        memo[identity] = _SNAPSHOT_IN_PROGRESS
        copied = AnswerQuery(
            kind=value.kind,
            fields=tuple(
                _snapshot_value_recursive(item, memo) for item in value.fields
            ),
            resolution=_snapshot_value_recursive(value.resolution, memo),
            points=tuple(
                _snapshot_value_recursive(point, memo) for point in value.points
            ),
        )
        memo[identity] = copied
        return copied
    if isinstance(value, AnswerView):
        memo[identity] = _SNAPSHOT_IN_PROGRESS
        copied = AnswerView(
            kind=value.kind,
            fields=_snapshot_value_recursive(value.fields, memo),
            axes=_snapshot_value_recursive(value.axes, memo),
            points=tuple(
                _snapshot_value_recursive(point, memo) for point in value.points
            ),
            shape=tuple(
                _snapshot_value_recursive(item, memo) for item in value.shape
            ),
        )
        memo[identity] = copied
        return copied
    if isinstance(value, MacroTarget):
        memo[identity] = _SNAPSHOT_IN_PROGRESS
        copied = MacroTarget(
            case_id=value.case_id,
            contract_id=value.contract_id,
            query=_snapshot_value_recursive(value.query, memo),
            values=_snapshot_value_recursive(value.values, memo),
            sample_weights=tuple(
                _snapshot_value_recursive(item, memo)
                for item in value.sample_weights
            ),
        )
        memo[identity] = copied
        return copied
    if isinstance(value, Mapping):
        copied_mapping: dict[object, object] = {}
        memo[identity] = copied_mapping
        for key, item in value.items():
            copied_mapping[_snapshot_value_recursive(key, memo)] = (
                _snapshot_value_recursive(item, memo)
            )
        return copied_mapping
    if isinstance(value, list):
        copied_list: list[object] = []
        memo[identity] = copied_list
        copied_list.extend(
            _snapshot_value_recursive(item, memo) for item in value
        )
        return copied_list
    if isinstance(value, tuple):
        memo[identity] = _SNAPSHOT_IN_PROGRESS
        copied_tuple = tuple(
            _snapshot_value_recursive(item, memo) for item in value
        )
        memo[identity] = copied_tuple
        return copied_tuple
    raise TypeError(
        f"unsupported macro snapshot value type: {type(value).__name__}"
    )


_STABLE_FAILURE_CODES = frozenset(
    {
        "candidate_execution_failed",
        "candidate_query_budget_exceeded",
        "candidate_solve_budget_exceeded",
        "candidate_two_phase_required",
        "candidate_output_invalid",
        "macro_evaluation_failed",
        "macro_gate_rejected",
    }
)


@dataclass(frozen=True)
class SanitizedMacroCaseScore:
    """Controller score; development responses apply a stricter projection."""

    case_id: str
    contract_id: str
    status: str
    primary_error: float | None
    metrics: Mapping[str, float]
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.case_id, "macro score case_id")
        _require_identifier(self.contract_id, "macro score contract_id")
        if self.status not in {"valid", "rejected", "invalid"}:
            raise ValueError(f"unknown macro score status: {self.status!r}")
        if self.status == "invalid":
            if self.primary_error is not None:
                raise ValueError("an invalid macro score cannot have a primary error")
            if self.metrics:
                raise ValueError("an invalid macro score cannot have metrics")
        elif (
            self.primary_error is None
            or isinstance(self.primary_error, bool)
            or not isinstance(self.primary_error, Real)
            or not isfinite(float(self.primary_error))
            or float(self.primary_error) < 0.0
        ):
            raise ValueError(
                "a valid or rejected score needs a finite nonnegative primary error"
            )
        if any(not isinstance(name, str) or not name.strip() for name in self.metrics):
            raise ValueError("sanitized macro metric names cannot be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            for value in self.metrics.values()
        ):
            raise ValueError("sanitized macro metrics must be finite numbers")
        if not isinstance(self.failures, tuple) or any(
            not isinstance(code, str) or not code.strip() for code in self.failures
        ):
            raise ValueError("sanitized macro failures must be nonempty string codes")
        if self.status == "valid" and self.failures:
            raise ValueError("a valid macro score cannot contain failures")


@dataclass(frozen=True)
class MacroCaseTimingAudit:
    """Controller-private timing attribution for one candidate/case call.

    A ``formal_timing_eligible`` row only establishes that solve and point
    projection were measured across the genuine two-phase boundary. Hard
    termination at either budget remains an isolated-runner responsibility.
    """

    case_id: str
    query_kind: QueryKind
    timing_assurance: TimingAssurance
    candidate_solve_wall_seconds: float | None = None
    candidate_query_wall_seconds: float | None = None
    candidate_query_point_count: int | None = None
    candidate_legacy_wall_seconds: float | None = None
    controller_truth_match_wall_seconds: float | None = None

    hard_process_deadline_enforcement: ClassVar[str] = (
        "isolated_runner_required"
    )

    def __post_init__(self) -> None:
        _require_identifier(self.case_id, "macro timing case_id")
        if not isinstance(self.query_kind, QueryKind):
            raise TypeError("macro timing query_kind must be QueryKind")
        if not isinstance(self.timing_assurance, TimingAssurance):
            raise TypeError("macro timing assurance must be TimingAssurance")
        for value, label in (
            (self.candidate_solve_wall_seconds, "candidate solve wall"),
            (self.candidate_query_wall_seconds, "candidate query wall"),
            (self.candidate_legacy_wall_seconds, "candidate legacy wall"),
            (
                self.controller_truth_match_wall_seconds,
                "controller truth/match wall",
            ),
        ):
            _validate_optional_wall_seconds(value, label)
        if self.candidate_query_point_count is not None and (
            isinstance(self.candidate_query_point_count, bool)
            or not isinstance(self.candidate_query_point_count, int)
            or self.candidate_query_point_count <= 0
        ):
            raise ValueError("candidate query point count must be positive")
        if (self.candidate_query_wall_seconds is None) != (
            self.candidate_query_point_count is None
        ):
            raise ValueError("candidate query wall and point count must align")
        if self.candidate_legacy_wall_seconds is not None and any(
            value is not None
            for value in (
                self.candidate_solve_wall_seconds,
                self.candidate_query_wall_seconds,
            )
        ):
            raise ValueError("legacy wall cannot be presented as solve/query timing")

    @property
    def formal_timing_eligible(self) -> bool:
        return (
            self.query_kind is QueryKind.POINTS
            and self.timing_assurance is TimingAssurance.TWO_PHASE_METERED
            and self.candidate_solve_wall_seconds is not None
            and self.candidate_query_wall_seconds is not None
            and self.candidate_query_point_count is not None
            and self.candidate_legacy_wall_seconds is None
        )

    @property
    def candidate_query_wall_seconds_per_point(self) -> float | None:
        if self.candidate_query_wall_seconds is None:
            return None
        assert self.candidate_query_point_count is not None
        return (
            self.candidate_query_wall_seconds
            / self.candidate_query_point_count
        )


@dataclass(frozen=True)
class MacroBatchEvaluation:
    """Controller-private audit for one exact, ordered cohort.

    This contains hidden scores, timing audits and `target_sha256`; do not
    serialize it into Agent-facing results. `cohort_sha256` is the public
    Case/query/order commitment.
    """

    candidate_id: str
    cohort_id: str
    cohort_sha256: str
    target_sha256: str
    contract_id: str
    scores: tuple[SanitizedMacroCaseScore, ...]
    root_ids: tuple[str, ...] | None = None
    case_timings: tuple[MacroCaseTimingAudit, ...] = ()
    execution_order: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "macro batch candidate_id")
        _require_identifier(self.cohort_id, "macro batch cohort_id")
        _require_sha256(self.cohort_sha256, "macro batch public cohort")
        _require_sha256(self.target_sha256, "macro batch private target")
        _require_identifier(self.contract_id, "macro batch contract_id")
        if not isinstance(self.scores, tuple) or not self.scores:
            raise ValueError("a macro batch needs at least one score")
        if any(score.contract_id != self.contract_id for score in self.scores):
            raise ValueError("macro batch score contracts must agree")
        case_ids = tuple(score.case_id for score in self.scores)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("macro batch score case ids must be unique")
        root_ids = (
            case_ids if self.root_ids is None else tuple(self.root_ids)
        )
        if len(root_ids) != len(case_ids) or any(
            not isinstance(root_id, str) or not root_id.strip()
            for root_id in root_ids
        ):
            raise ValueError("macro batch root ids must align with case ids")
        object.__setattr__(self, "root_ids", root_ids)
        if not isinstance(self.case_timings, tuple) or any(
            not isinstance(timing, MacroCaseTimingAudit)
            for timing in self.case_timings
        ):
            raise TypeError("macro batch case timings must be timing audits")
        if self.case_timings and tuple(
            timing.case_id for timing in self.case_timings
        ) != case_ids:
            raise ValueError("macro batch case timings must align with scores")
        if self.execution_order is not None:
            execution_order = tuple(self.execution_order)
            if (
                len(execution_order) != len(case_ids)
                or len(set(execution_order)) != len(execution_order)
                or set(execution_order) != set(case_ids)
            ):
                raise ValueError(
                    "macro batch execution order must permute its case ids"
                )
            object.__setattr__(self, "execution_order", execution_order)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(score.case_id for score in self.scores)

    @property
    def accepted(self) -> bool:
        return bool(self.scores) and all(
            score.status == "valid" for score in self.scores
        )


@dataclass(frozen=True)
class DevelopmentMacroCaseResult:
    """Agent-visible score with an optional finite truth view."""

    score: SanitizedMacroCaseScore
    finite_truth: AnswerView | None = None

    def __post_init__(self) -> None:
        if self.finite_truth is not None and not isinstance(
            self.finite_truth, AnswerView
        ):
            raise TypeError("finite_truth must be an AnswerView or None")


@dataclass(frozen=True)
class DevelopmentMacroResponse:
    candidate_id: str
    cohort_id: str
    cohort_sha256: str
    contract_id: str
    charged_case_credits: int
    remaining_case_credits: int
    cases: tuple[DevelopmentMacroCaseResult, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "development response candidate_id")
        _require_identifier(self.cohort_id, "development response cohort_id")
        _require_sha256(self.cohort_sha256, "development public cohort")
        _require_identifier(self.contract_id, "development response contract_id")
        for value, label in (
            (self.charged_case_credits, "charged_case_credits"),
            (self.remaining_case_credits, "remaining_case_credits"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("a development response needs at least one case")
        if any(
            row.score.contract_id != self.contract_id for row in self.cases
        ):
            raise ValueError("development response score contracts must agree")


class CaseCreditExceeded(RuntimeError):
    """The requested development evaluation exceeds the remaining credits."""


@dataclass(frozen=True)
class _PreparedCase:
    public_case: Case
    query: AnswerQuery
    target: MacroTarget

@dataclass(frozen=True)
class _PreparedCohortIdentity:
    case_ids: tuple[str, ...]
    public_row_sha256: tuple[str, ...]
    target_row_sha256: tuple[str, ...]
@dataclass(frozen=True)
class _NumericDigestSequence:
    """A numerical sequence hashed in bounded binary chunks."""

    values: Sequence[float]



class _MMapFloatSequence(Sequence[float]):
    """Truth weights backed by one read-only NPY mapping."""

    def __init__(self, array: object) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - store already needs NumPy
            raise MacroCohortStoreError(
                "numpy is required for mapped macro target weights"
            ) from exc
        values = np.asarray(array)
        if values.ndim != 1 or values.dtype.hasobject:
            raise MacroCohortStoreError(
                "mapped macro target weights must be a one-dimensional array"
            )
        self._array = values

    def __len__(self) -> int:
        return len(self._array)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, index):
        value = self._array[index]
        if isinstance(index, slice):
            return tuple(float(item) for item in value)
        return float(value)

    def __iter__(self):
        return (float(value) for value in self._array)


class _StoredPreparedCases(Sequence[_PreparedCase]):
    """Lazy row view over one sealed private cohort store."""

    def __init__(
        self,
        store: PrivateMacroCohortStore,
        references: Sequence[Mapping[str, object]],
    ) -> None:
        self.store = store
        self.references = tuple(dict(reference) for reference in references)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("macro cohort row index must be an integer or slice")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        payload = self.store.read_row(self.references[index])
        return _prepared_case_from_storage_payload(payload, self.store.codec)

    def discard(self) -> None:
        self.store.discard()


class _IndexedPreparedCases(Sequence[_PreparedCase]):
    """Lazy ordered index view without retaining decoded private rows."""

    def __init__(
        self,
        prepared: Sequence[_PreparedCase],
        indices: Sequence[int],
    ) -> None:
        self._prepared = prepared
        self._indices = tuple(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _IndexedPreparedCases(self._prepared, self._indices[index])
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("macro cohort row index must be an integer or slice")
        return self._prepared[self._indices[index]]


def _prepared_store(
    prepared: Sequence[_PreparedCase],
) -> PrivateMacroCohortStore | None:
    if isinstance(prepared, _StoredPreparedCases):
        return prepared.store
    if isinstance(prepared, _IndexedPreparedCases):
        return _prepared_store(prepared._prepared)


class PreparedMacroCohort:
    """Controller-only target cache for a fixed ordered cohort.

    ``CaseRecord`` and its raw ``TruthProgram`` are deliberately discarded
    after target materialization. Candidates never receive an object stored in
    this cache; every call gets a deep public snapshot instead.
    """

    def __init__(
        self,
        *,
        cohort_id: str,
        matcher: MacroMatcher,
        prepared: Sequence[_PreparedCase],
        _identity: _PreparedCohortIdentity | None = None,
        release_cases: Callable[[tuple[str, ...]], None] | None = None,
    ) -> None:
        _require_identifier(cohort_id, "macro cohort_id")
        if not prepared:
            raise ValueError("a macro cohort cannot be empty")
        _require_identifier(matcher.contract_id, "macro matcher contract_id")
        if _identity is None:
            rows = tuple(prepared)
            identity = _prepared_cohort_identity(rows, matcher.contract_id)
            prepared_rows: Sequence[_PreparedCase] = rows
        else:
            identity = _identity
            prepared_rows = prepared
            if len(identity.case_ids) != len(prepared_rows):
                raise ValueError("macro cohort identity length changed")
        case_ids = identity.case_ids
        if any(not case_id.strip() for case_id in case_ids):
            raise ValueError("macro cohort case ids cannot be empty")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("macro cohort case ids must be unique")
        if any(
            case_id.startswith(CANDIDATE_CASE_ID_PREFIX)
            for case_id in case_ids
        ):
            raise ValueError(
                "macro cohort raw case ids cannot use the candidate id prefix"
            )
        if len(identity.public_row_sha256) != len(case_ids) or len(
            identity.target_row_sha256
        ) != len(case_ids):
            raise ValueError("macro cohort digest rows do not align")
        for digest in (*identity.public_row_sha256, *identity.target_row_sha256):
            _require_sha256(digest, "macro cohort row")

        self.cohort_id = cohort_id
        self.contract_id = matcher.contract_id
        self._matcher = matcher
        self._prepared = prepared_rows
        self._identity = identity
        self._case_ids = case_ids
        if release_cases is not None and not callable(release_cases):
            raise TypeError("macro cohort release_cases must be callable")
        self._release_cases = release_cases
        self._store_discarded = False
        self._discarded = False
        self.public_cohort_sha256 = _public_cohort_digest(
            cohort_id, matcher.contract_id, identity.public_row_sha256
        )
        self.cohort_sha256 = self.public_cohort_sha256
        self._private_target_sha256 = _private_target_digest(
            self.public_cohort_sha256, identity.target_row_sha256
        )
        self._by_case_id = {
            case_id: index for index, case_id in enumerate(case_ids)
        }
        self._candidate_by_case_id = {
            candidate_case_id(case_id): index
            for case_id, index in self._by_case_id.items()
        }

    @classmethod
    def prepare(
        cls,
        *,
        cohort_id: str,
        records: Sequence[CaseRecord],
        matcher: MacroMatcher,
        query: AnswerQuery | QueryFactory,
        private_store_root: Path | None = None,
        release_cases: Callable[[tuple[str, ...]], None] | None = None,
    ) -> PreparedMacroCohort:
        """Materialize all targets before any candidate code can run."""

        if not records:
            raise ValueError("a macro cohort cannot be empty")
        if release_cases is not None and not callable(release_cases):
            raise TypeError("macro cohort release_cases must be callable")
        release_case_ids = tuple(
            dict.fromkeys(
                record.case.case_id
                for record in records
                if isinstance(record, CaseRecord)
                and isinstance(record.case.case_id, str)
                and record.case.case_id.strip()
            )
        )
        prepared: list[_PreparedCase] = []
        references: list[Mapping[str, object]] = []
        case_ids: list[str] = []
        public_digests: list[str] = []
        target_digests: list[str] = []
        seen: set[str] = set()
        store = (
            None
            if private_store_root is None
            else PrivateMacroCohortStore(
                Path(private_store_root), cohort_id=cohort_id
            )
        )
        try:
            for index, record in enumerate(records):
                if not isinstance(record, CaseRecord):
                    raise TypeError("macro cohort records must be CaseRecord objects")
                case_id = record.case.case_id
                _require_identifier(case_id, "macro cohort case_id")
                if case_id in seen:
                    raise ValueError("macro cohort case ids must be unique")
                seen.add(case_id)
                controller_case = _snapshot_value(record.case)
                raw_query = (
                    query(_snapshot_value(controller_case))
                    if callable(query)
                    else query
                )
                if not isinstance(raw_query, AnswerQuery):
                    raise TypeError("macro query factory must return AnswerQuery")
                controller_query = _snapshot_value(raw_query)
                target = materialize_macro_target(
                    matcher,
                    CaseRecord(_snapshot_value(controller_case), record.truth),
                    _snapshot_value(controller_query),
                )
                row = _PreparedCase(
                    public_case=_snapshot_value(controller_case),
                    query=_snapshot_value(controller_query),
                    target=_snapshot_value(target),
                )
                _validate_prepared_case(row, matcher.contract_id)
                public_digest, target_digest = _prepared_case_identity(row)
                case_ids.append(case_id)
                public_digests.append(public_digest)
                target_digests.append(target_digest)
                if store is None:
                    prepared.append(row)
                else:
                    references.append(
                        store.write_row(
                            index,
                            _prepared_case_storage_payload(row, store.codec),
                        )
                    )
            identity = _PreparedCohortIdentity(
                tuple(case_ids),
                tuple(public_digests),
                tuple(target_digests),
            )
            if store is None:
                rows: Sequence[_PreparedCase] = tuple(prepared)
            else:
                public_sha256 = _public_cohort_digest(
                    cohort_id, matcher.contract_id, identity.public_row_sha256
                )
                target_sha256 = _private_target_digest(
                    public_sha256, identity.target_row_sha256
                )
                store.seal(
                    {
                        "schema": "macro-private-cohort-store-v1",
                        "cohort_id": cohort_id,
                        "contract_id": matcher.contract_id,
                        "case_ids": list(identity.case_ids),
                        "public_cohort_sha256": public_sha256,
                        "private_target_sha256": target_sha256,
                        "rows": [
                            {
                                **dict(reference),
                                "case_id": case_id,
                                "public_row_sha256": public_digest,
                                "target_row_sha256": target_digest,
                            }
                            for (
                                reference,
                                case_id,
                                public_digest,
                                target_digest,
                            ) in zip(
                                references,
                                identity.case_ids,
                                identity.public_row_sha256,
                                identity.target_row_sha256,
                            )
                        ],
                    }
                )
                rows = _StoredPreparedCases(store, references)
            return cls(
                cohort_id=cohort_id,
                matcher=matcher,
                prepared=rows,
                _identity=identity,
                release_cases=release_cases,
            )
        except BaseException:
            try:
                if store is not None:
                    store.discard()
            finally:
                if release_cases is not None:
                    release_cases(release_case_ids)
            raise

    @property
    def case_ids(self) -> tuple[str, ...]:
        return self._case_ids

    @property
    def disk_backed(self) -> bool:
        return _prepared_store(self._prepared) is not None

    @property
    def private_manifest_path(self) -> Path | None:
        store = _prepared_store(self._prepared)
        return None if store is None else store.manifest_path

    def public_cases(self) -> tuple[Case, ...]:
        """Return isolated Agent-visible cases, never controller cache objects."""

        self._require_available()
        return tuple(
            _snapshot_value(candidate_facing_case(row.public_case))
            for row in self._prepared
        )

    def _controller_cases(self) -> tuple[Case, ...]:
        """Return issued envelopes only to trusted controller collaborators."""

        self._require_available()
        return tuple(
            _snapshot_value(row.public_case) for row in self._prepared
        )

    def public_queries(self) -> tuple[AnswerQuery, ...]:
        """Return isolated public queries in the same order as ``public_cases``."""

        self._require_available()
        return tuple(_snapshot_value(row.query) for row in self._prepared)

    def _select(self, case_ids: Sequence[str] | None) -> Sequence[_PreparedCase]:
        self._require_available()
        if case_ids is None:
            return self._prepared
        indices = self._resolve_case_indices(case_ids)
        return _IndexedPreparedCases(self._prepared, indices)

    def _resolve_case_indices(
        self,
        case_ids: Sequence[str],
    ) -> tuple[int, ...]:
        if isinstance(case_ids, (str, bytes)):
            raise ValueError("case_ids must be a sequence of identifiers")
        requested = tuple(case_ids)
        if any(
            not isinstance(case_id, str) or not case_id.strip()
            for case_id in requested
        ):
            raise ValueError("case_ids must contain nonempty strings")
        if not requested:
            raise ValueError("an evaluation request needs at least one case id")
        if len(requested) != len(set(requested)):
            raise ValueError("an evaluation request cannot repeat case ids")
        unknown = (
            set(requested)
            - set(self._by_case_id)
            - set(self._candidate_by_case_id)
        )
        if unknown:
            raise KeyError(f"macro cases are unavailable: {sorted(unknown)}")
        indices = tuple(
            self._by_case_id[case_id]
            if case_id in self._by_case_id
            else self._candidate_by_case_id[case_id]
            for case_id in requested
        )
        if len(indices) != len(set(indices)):
            raise ValueError("an evaluation request cannot repeat macro cases")
        return indices

    def _subset(
        self,
        *,
        cohort_id: str,
        case_ids: Sequence[str],
    ) -> PreparedMacroCohort:
        self._require_available()
        requested = tuple(case_ids)
        indices = self._resolve_case_indices(requested)
        prepared = _IndexedPreparedCases(self._prepared, indices)
        raw_case_ids = tuple(self._case_ids[index] for index in indices)
        identity = _PreparedCohortIdentity(
            raw_case_ids,
            tuple(self._identity.public_row_sha256[index] for index in indices),
            tuple(self._identity.target_row_sha256[index] for index in indices),
        )
        return PreparedMacroCohort(
            cohort_id=cohort_id,
            matcher=self._matcher,
            prepared=prepared,
            _identity=identity,
        )

    def discard(self) -> None:
        """Release temporary prepared rows while retaining commitment metadata."""

        if self._discarded:
            return
        store = _prepared_store(self._prepared)
        if store is not None and not self._store_discarded:
            store.discard()
            self._store_discarded = True
        release_cases = self._release_cases
        if release_cases is not None:
            release_cases(self._case_ids)
            self._release_cases = None
        self._prepared = ()
        self._by_case_id = {}
        self._candidate_by_case_id = {}
        self._discarded = True

    def _require_available(self) -> None:
        if self._discarded:
            raise RuntimeError("prepared macro cohort was discarded")


class ControllerMacroEvaluator:
    """Fail-closed candidate scoring on a prepared private cohort.

    This in-process layer attributes candidate solve, candidate point-query,
    and controller truth/matching wall time independently. The compute budget
    is forwarded as the per-case solve budget. When a point-query latency
    limit is configured, the formal two-phase POINTS ABI is mandatory and the
    query phase is independently eligibility-bounded. Hard process termination
    is not claimed here and must be enforced by the isolated candidate runner.
    """

    def __init__(
        self,
        cohort: PreparedMacroCohort,
        *,
        compute_budget: float | None = None,
        query_seconds_per_point_limit: float | None = None,
        public_session_factory: PublicSessionFactory | None = None,
        public_session_methods: Sequence[str] = (),
        candidate_clock: Clock = perf_counter,
        controller_clock: Clock = perf_counter,
        candidate_order_rng: object | None = None,
    ) -> None:
        self._cohort = cohort
        self._compute_budget = _validated_compute_budget(compute_budget)
        self._query_seconds_per_point_limit = (
            _validated_query_seconds_per_point_limit(
                query_seconds_per_point_limit
            )
        )
        if public_session_factory is not None and not callable(
            public_session_factory
        ):
            raise TypeError("public session factory must be callable")
        methods = tuple(public_session_methods)
        if any(
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            for name in methods
        ) or len(methods) != len(set(methods)):
            raise ValueError(
                "public session methods must be unique public identifiers"
            )
        if (public_session_factory is None) != (not methods):
            raise ValueError(
                "public session factory and explicit method allowlist must align"
            )
        if not callable(candidate_clock):
            raise TypeError("candidate clock must be callable")
        if not callable(controller_clock):
            raise TypeError("controller clock must be callable")
        self._public_session_factory = public_session_factory
        self._public_session_methods = methods
        self._candidate_clock = candidate_clock
        self._controller_clock = controller_clock
        order_rng = (
            SystemRandom() if candidate_order_rng is None else candidate_order_rng
        )
        order_shuffle = getattr(order_rng, "shuffle", None)
        if not callable(order_shuffle):
            raise TypeError("candidate_order_rng must provide shuffle")
        self._candidate_order_shuffle = order_shuffle
        self._candidate_order_lock = Lock()

    @property
    def cohort_id(self) -> str:
        return self._cohort.cohort_id

    @property
    def cohort_sha256(self) -> str:
        return self._cohort.cohort_sha256

    def evaluate(
        self,
        candidate_id: str,
        candidate: object,
    ) -> MacroBatchEvaluation:
        """Run a fresh private case order; return frozen cohort order."""

        self._cohort._require_available()
        return self._evaluate(candidate_id, candidate, self._cohort._prepared)

    def evaluate_many(
        self,
        candidates: Sequence[tuple[str, object]],
    ) -> tuple[MacroBatchEvaluation, ...]:
        """Evaluate programs case-major on one frozen paired cohort.

        A fresh private case order is shared; candidate order is preserved. Each gets
        its own public snapshots and public-session binding for every case;
        only the loop order changes so equation backends may retain bounded
        per-case state while several programs consume it.
        """

        self._cohort._require_available()
        return self._evaluate_many(candidates, self._cohort._prepared)

    def evaluate_pair(
        self,
        first_id: str,
        first: object,
        second_id: str,
        second: object,
    ) -> tuple[MacroBatchEvaluation, MacroBatchEvaluation]:
        """Private-order, case-major convenience for a paired comparison."""

        first_batch, second_batch = self.evaluate_many(
            ((first_id, first), (second_id, second))
        )
        return first_batch, second_batch

    def _evaluate(
        self,
        candidate_id: str,
        candidate: object,
        prepared: Sequence[_PreparedCase],
    ) -> MacroBatchEvaluation:
        return self._evaluate_many(
            ((candidate_id, candidate),),
            prepared,
        )[0]

    def _evaluate_many(
        self,
        candidates: Sequence[tuple[str, object]],
        prepared: Sequence[_PreparedCase],
    ) -> tuple[MacroBatchEvaluation, ...]:
        entries = tuple(candidates)
        if not entries:
            raise ValueError("candidates must contain at least one program")
        for candidate_id, _candidate in entries:
            _require_identifier(candidate_id, "candidate_id")

        prepared_rows = tuple(prepared)
        score_rows: list[dict[int, SanitizedMacroCaseScore]] = [
            {} for _entry in entries
        ]
        timing_rows: list[dict[int, MacroCaseTimingAudit]] = [
            {} for _entry in entries
        ]
        evaluation_order = list(range(len(prepared_rows)))
        with self._candidate_order_lock:
            self._candidate_order_shuffle(evaluation_order)
        if sorted(evaluation_order) != list(range(len(prepared_rows))):
            raise RuntimeError(
                "candidate_order_rng returned an invalid permutation"
            )
        for row_index in evaluation_order:
            row = prepared_rows[row_index]
            for scores, timings, (_candidate_id, candidate) in zip(
                score_rows,
                timing_rows,
                entries,
            ):
                score, timing = self._evaluate_case(candidate, row)
                scores[row_index] = score
                timings[row_index] = timing
        frozen_root_ids = tuple(
            row.public_case.parent_case_id or row.public_case.case_id
            for row in prepared_rows
        )
        original_order = range(len(prepared_rows))
        return tuple(
            MacroBatchEvaluation(
                candidate_id=candidate_id,
                cohort_id=self._cohort.cohort_id,
                cohort_sha256=self._cohort.cohort_sha256,
                target_sha256=self._cohort._private_target_sha256,
                contract_id=self._cohort.contract_id,
                scores=tuple(scores[index] for index in original_order),
                root_ids=frozen_root_ids,
                case_timings=tuple(timings[index] for index in original_order),
                execution_order=tuple(
                    prepared_rows[index].public_case.case_id
                    for index in evaluation_order
                ),
            )
            for (candidate_id, _candidate), scores, timings in zip(
                entries,
                score_rows,
                timing_rows,
            )
        )

    def _evaluate_case(
        self,
        candidate: object,
        prepared: _PreparedCase,
    ) -> tuple[SanitizedMacroCaseScore, MacroCaseTimingAudit]:
        case_id = prepared.public_case.case_id
        timing_assurance = TimingAssurance.LEGACY_ONE_SHOT_UNASSURED
        prepared_result: PreparedSolutionResult | None = None
        query_result: PointQueryResult | None = None
        legacy_wall_seconds: float | None = None
        try:
            controller_case = _snapshot_value(prepared.public_case)
            public_case = _snapshot_value(candidate_facing_case(controller_case))

            def open_controller_session(_candidate_case: Case) -> object:
                assert self._public_session_factory is not None
                return self._public_session_factory(controller_case)

            candidate_session_factory = (
                None
                if self._public_session_factory is None
                else open_controller_session
            )
            uses_two_phase_points = _uses_two_phase_points(
                candidate,
                prepared.query,
                allow_deferred=(
                    self._query_seconds_per_point_limit is not None
                ),
            )
            if (
                self._query_seconds_per_point_limit is not None
                and not uses_two_phase_points
            ):
                return (
                    _invalid_score(
                        case_id,
                        self._cohort.contract_id,
                        "candidate_two_phase_required",
                    ),
                    _macro_case_timing(
                        case_id=case_id,
                        query_kind=prepared.query.kind,
                        timing_assurance=timing_assurance,
                        prepared_result=None,
                        query_result=None,
                        legacy_wall_seconds=None,
                    ),
                )
            if uses_two_phase_points:
                timing_assurance = TimingAssurance.TWO_PHASE_METERED
                with bind_public_session_factory(
                    candidate_session_factory,
                    public_case,
                    allowed_methods=self._public_session_methods,
                ):
                    prepared_result = measure_prepare_solution(
                        candidate,
                        public_case,
                        self._compute_budget,
                        clock=self._candidate_clock,
                    )
                if (
                    self._compute_budget is not None
                    and prepared_result.candidate_solve_wall_seconds
                    > self._compute_budget
                ):
                    return (
                        _invalid_score(
                            case_id,
                            self._cohort.contract_id,
                            "candidate_solve_budget_exceeded",
                        ),
                        _macro_case_timing(
                            case_id=case_id,
                            query_kind=prepared.query.kind,
                            timing_assurance=prepared_result.timing_assurance,
                            prepared_result=prepared_result,
                            query_result=None,
                            legacy_wall_seconds=None,
                        ),
                    )
                timing_assurance = prepared_result.timing_assurance
                query_result = measure_point_query(
                    prepared_result.handle,
                    _snapshot_value(prepared.query),
                    clock=self._candidate_clock,
                )
                if (
                    self._query_seconds_per_point_limit is not None
                    and query_result.candidate_query_wall_seconds_per_point
                    > self._query_seconds_per_point_limit
                ):
                    return (
                        _invalid_score(
                            case_id,
                            self._cohort.contract_id,
                            "candidate_query_budget_exceeded",
                        ),
                        _macro_case_timing(
                            case_id=case_id,
                            query_kind=prepared.query.kind,
                            timing_assurance=prepared_result.timing_assurance,
                            prepared_result=prepared_result,
                            query_result=query_result,
                            legacy_wall_seconds=None,
                        ),
                    )
                prediction = query_result.answer
            else:
                with bind_public_session_factory(
                    candidate_session_factory,
                    public_case,
                    allowed_methods=self._public_session_methods,
                ):
                    legacy_started = _clock_value(
                        self._candidate_clock,
                        "candidate clock",
                    )
                    try:
                        prediction = _predict_legacy(
                            candidate,
                            public_case,
                            _snapshot_value(prepared.query),
                            self._compute_budget,
                        )
                    finally:
                        legacy_wall_seconds = _wall_elapsed(
                            legacy_started,
                            _clock_value(
                                self._candidate_clock,
                                "candidate clock",
                            ),
                        )
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException:
            return (
                _invalid_score(
                    case_id,
                    self._cohort.contract_id,
                    "candidate_execution_failed",
                ),
                _macro_case_timing(
                    case_id=case_id,
                    query_kind=prepared.query.kind,
                    timing_assurance=timing_assurance,
                    prepared_result=prepared_result,
                    query_result=query_result,
                    legacy_wall_seconds=legacy_wall_seconds,
                ),
            )
        if not isinstance(prediction, AnswerView):
            return (
                _invalid_score(
                    case_id,
                    self._cohort.contract_id,
                    "candidate_output_invalid",
                ),
                _macro_case_timing(
                    case_id=case_id,
                    query_kind=prepared.query.kind,
                    timing_assurance=timing_assurance,
                    prepared_result=prepared_result,
                    query_result=query_result,
                    legacy_wall_seconds=legacy_wall_seconds,
                ),
            )

        controller_started = _clock_value(
            self._controller_clock,
            "controller clock",
        )
        try:
            result = self._cohort._matcher.match(
                _snapshot_value(prepared.public_case),
                _snapshot_value(prediction),
                _snapshot_value(prepared.target),
            )
            if result.case_id != case_id:
                raise ValueError("macro matcher changed the case identity")
            if result.contract_id != self._cohort.contract_id:
                raise ValueError("macro matcher changed the contract identity")
            status = "rejected" if result.passed is False else "valid"
            score = SanitizedMacroCaseScore(
                case_id=result.case_id,
                contract_id=result.contract_id,
                status=status,
                primary_error=float(result.primary_error),
                metrics={
                    name: float(value) for name, value in result.metrics.items()
                },
                failures=tuple(result.failures),
            )
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException:
            score = _invalid_score(
                case_id,
                self._cohort.contract_id,
                "macro_evaluation_failed",
            )
        controller_wall_seconds = _wall_elapsed(
            controller_started,
            _clock_value(self._controller_clock, "controller clock"),
        )
        return (
            score,
            _macro_case_timing(
                case_id=case_id,
                query_kind=prepared.query.kind,
                timing_assurance=timing_assurance,
                prepared_result=prepared_result,
                query_result=query_result,
                legacy_wall_seconds=legacy_wall_seconds,
                controller_wall_seconds=controller_wall_seconds,
            ),
        )


class MacroBatchStreamAccumulator:
    """Combine bounded in-memory evaluations into one common-cohort batch.

    Official selection and terminal evaluation need one logical cohort shared
    by every candidate, but they do not need every manufactured answer to be
    resident at once. This accumulator retains only sanitized scores, timing
    audits, root ids, and per-row commitments. In particular it never keeps a
    prepared cohort, public case, query, target, or truth program alive after
    consume returns.

    Chunks must use the final logical cohort_id and must be memory-backed. The
    latter is a fail-closed guard against accidentally turning transient
    official questions into disk artifacts.
    """

    def __init__(
        self,
        *,
        cohort_id: str,
        contract_id: str,
        candidate_ids: Sequence[str],
    ) -> None:
        _require_identifier(cohort_id, "streamed macro cohort_id")
        _require_identifier(contract_id, "streamed macro contract_id")
        ids = tuple(candidate_ids)
        if not ids:
            raise ValueError("a streamed macro evaluation needs candidates")
        for candidate_id in ids:
            _require_identifier(candidate_id, "streamed candidate_id")
        if len(ids) != len(set(ids)):
            raise ValueError("streamed candidate ids must be unique")
        self._cohort_id = cohort_id
        self._contract_id = contract_id
        self._candidate_ids = ids
        self._case_ids: list[str] = []
        self._seen_case_ids: set[str] = set()
        self._public_row_sha256: list[str] = []
        self._target_row_sha256: list[str] = []
        self._root_ids: list[str] = []
        self._execution_order: list[str] = []
        self._execution_order_available = True
        self._score_rows: list[list[SanitizedMacroCaseScore]] = [
            [] for _candidate_id in ids
        ]
        self._timing_rows: list[list[MacroCaseTimingAudit]] = [
            [] for _candidate_id in ids
        ]
        self._finalized = False

    @property
    def case_count(self) -> int:
        return len(self._case_ids)

    def consume(
        self,
        cohort: PreparedMacroCohort,
        batches: Sequence[MacroBatchEvaluation],
    ) -> None:
        """Copy one evaluated chunk's bounded audit data into the stream."""

        if self._finalized:
            raise RuntimeError("a finalized macro stream cannot consume chunks")
        if not isinstance(cohort, PreparedMacroCohort):
            raise TypeError("streamed chunk must be a PreparedMacroCohort")
        cohort._require_available()
        if cohort.disk_backed:
            raise ValueError(
                "official streamed macro chunks must not use private disk storage"
            )
        if cohort.cohort_id != self._cohort_id:
            raise ValueError("streamed chunk changed the common cohort identity")
        if cohort.contract_id != self._contract_id:
            raise ValueError("streamed chunk changed the macro contract")

        rows = tuple(batches)
        if len(rows) != len(self._candidate_ids):
            raise ValueError("streamed chunk candidate count changed")
        if tuple(row.candidate_id for row in rows) != self._candidate_ids:
            raise ValueError("streamed chunk candidate order changed")
        chunk_case_ids = cohort.case_ids
        chunk_case_id_set = set(chunk_case_ids)
        if chunk_case_id_set & self._seen_case_ids:
            raise ValueError("streamed common cohort repeated case ids")
        for row in rows:
            if row.cohort_id != self._cohort_id:
                raise ValueError("streamed batch changed the common cohort id")
            if row.contract_id != self._contract_id:
                raise ValueError("streamed batch changed the macro contract")
            if row.case_ids != chunk_case_ids:
                raise ValueError("streamed batch case order changed")
            if row.cohort_sha256 != cohort.cohort_sha256 or (
                row.target_sha256 != cohort._private_target_sha256
            ):
                raise ValueError("streamed batch commitment changed")
            if len(row.case_timings) != len(row.scores):
                raise ValueError("streamed batch timing audits must be complete")
        execution_orders = tuple(row.execution_order for row in rows)
        if len(set(execution_orders)) != 1:
            raise ValueError(
                "streamed candidates used different execution orders"
            )
        chunk_execution_order = execution_orders[0]
        if chunk_execution_order is None:
            self._execution_order_available = False
        elif self._execution_order_available:
            self._execution_order.extend(chunk_execution_order)

        identity = cohort._identity
        self._case_ids.extend(identity.case_ids)
        self._seen_case_ids.update(chunk_case_id_set)
        self._public_row_sha256.extend(identity.public_row_sha256)
        self._target_row_sha256.extend(identity.target_row_sha256)
        self._root_ids.extend(rows[0].root_ids or ())
        for target_scores, target_timings, row in zip(
            self._score_rows,
            self._timing_rows,
            rows,
        ):
            if row.root_ids != rows[0].root_ids:
                raise ValueError("streamed candidate root order changed")
            target_scores.extend(row.scores)
            target_timings.extend(row.case_timings)

    def finalize(self) -> tuple[MacroBatchEvaluation, ...]:
        """Seal the logical cohort commitment and return ordinary ABI batches."""

        if self._finalized:
            raise RuntimeError("a macro evaluation stream is single-use")
        if not self._case_ids:
            raise ValueError("a macro evaluation stream cannot be empty")
        self._finalized = True
        public_sha256 = _public_cohort_digest(
            self._cohort_id,
            self._contract_id,
            self._public_row_sha256,
        )
        target_sha256 = _private_target_digest(
            public_sha256,
            self._target_row_sha256,
        )
        root_ids = tuple(self._root_ids)
        return tuple(
            MacroBatchEvaluation(
                candidate_id=candidate_id,
                cohort_id=self._cohort_id,
                cohort_sha256=public_sha256,
                target_sha256=target_sha256,
                contract_id=self._contract_id,
                scores=tuple(scores),
                root_ids=root_ids,
                case_timings=tuple(timings),
                execution_order=(
                    tuple(self._execution_order)
                    if self._execution_order_available
                    else None
                ),
            )
            for candidate_id, scores, timings in zip(
                self._candidate_ids,
                self._score_rows,
                self._timing_rows,
            )
        )


class DevelopmentMacroEvaluator:
    """Budgeted Agent-facing view over the same controller evaluator."""

    def __init__(
        self,
        cohort: PreparedMacroCohort,
        *,
        case_credit_limit: int,
        allow_finite_truth: bool = False,
        compute_budget: float | None = None,
        query_seconds_per_point_limit: float | None = None,
        safe_metric_names: Sequence[str] = (),
        public_session_factory: PublicSessionFactory | None = None,
        public_session_methods: Sequence[str] = (),
    ) -> None:
        if (
            isinstance(case_credit_limit, bool)
            or not isinstance(case_credit_limit, int)
            or case_credit_limit <= 0
        ):
            raise ValueError("case_credit_limit must be a positive integer")
        if not isinstance(allow_finite_truth, bool):
            raise TypeError("allow_finite_truth must be bool")
        if isinstance(safe_metric_names, (str, bytes)):
            raise ValueError("safe_metric_names must be a sequence of names")
        metric_names = tuple(safe_metric_names)
        if any(
            not isinstance(name, str) or not name.strip()
            for name in metric_names
        ):
            raise ValueError("safe_metric_names must contain nonempty strings")
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("safe_metric_names cannot contain duplicates")
        self._controller = ControllerMacroEvaluator(
            cohort,
            compute_budget=compute_budget,
            query_seconds_per_point_limit=query_seconds_per_point_limit,
            public_session_factory=public_session_factory,
            public_session_methods=public_session_methods,
        )
        self._cohort = cohort
        self._case_credit_limit = case_credit_limit
        self._charged_case_credits = 0
        self._allow_finite_truth = allow_finite_truth
        self._safe_metric_names = frozenset(metric_names)
        self._lock = Lock()

    @property
    def charged_case_credits(self) -> int:
        with self._lock:
            return self._charged_case_credits

    @property
    def remaining_case_credits(self) -> int:
        with self._lock:
            return self._case_credit_limit - self._charged_case_credits

    @property
    def public_cases(self) -> tuple[Case, ...]:
        return self._cohort.public_cases()

    @property
    def public_queries(self) -> tuple[AnswerQuery, ...]:
        return self._cohort.public_queries()

    @property
    def cohort_id(self) -> str:
        return self._cohort.cohort_id

    @property
    def cohort_sha256(self) -> str:
        return self._cohort.cohort_sha256

    @property
    def contract_id(self) -> str:
        return self._cohort.contract_id

    def evaluate(
        self,
        candidate_id: str,
        candidate: object,
        *,
        case_ids: Sequence[str] | None = None,
        return_finite_truth: bool = False,
    ) -> DevelopmentMacroResponse:
        _require_identifier(candidate_id, "candidate_id")
        if not isinstance(return_finite_truth, bool):
            raise TypeError("return_finite_truth must be bool")
        prepared = self._cohort._select(case_ids)
        if return_finite_truth and not self._allow_finite_truth:
            raise PermissionError("finite macro truth views are disabled")

        charge = len(prepared)
        with self._lock:
            remaining = self._case_credit_limit - self._charged_case_credits
            if charge > remaining:
                raise CaseCreditExceeded(
                    f"evaluation needs {charge} case credits but only "
                    f"{remaining} remain"
                )
            self._charged_case_credits += charge
            charged_total = self._charged_case_credits
            remaining_after = self._case_credit_limit - charged_total

        batch = self._controller._evaluate(candidate_id, candidate, prepared)
        if return_finite_truth:
            results = tuple(
                DevelopmentMacroCaseResult(
                    score=_agent_visible_score(score, self._safe_metric_names),
                    finite_truth=_snapshot_value(row.target.values),
                )
                for row, score in zip(prepared, batch.scores)
            )
        else:
            results = tuple(
                DevelopmentMacroCaseResult(
                    score=_agent_visible_score(score, self._safe_metric_names),
                    finite_truth=None,
                )
                for score in batch.scores
            )
        return DevelopmentMacroResponse(
            candidate_id=batch.candidate_id,
            cohort_id=batch.cohort_id,
            cohort_sha256=batch.cohort_sha256,
            contract_id=batch.contract_id,
            charged_case_credits=charged_total,
            remaining_case_credits=remaining_after,
            cases=results,
        )


def _prepared_cohort_identity(
    prepared: Sequence[_PreparedCase],
    contract_id: str,
) -> _PreparedCohortIdentity:
    case_ids: list[str] = []
    public_digests: list[str] = []
    target_digests: list[str] = []
    for row in prepared:
        _validate_prepared_case(row, contract_id)
        public_digest, target_digest = _prepared_case_identity(row)
        case_ids.append(row.public_case.case_id)
        public_digests.append(public_digest)
        target_digests.append(target_digest)
    return _PreparedCohortIdentity(
        tuple(case_ids),
        tuple(public_digests),
        tuple(target_digests),
    )


def _validate_prepared_case(row: _PreparedCase, contract_id: str) -> None:
    if not isinstance(row, _PreparedCase):
        raise TypeError("prepared macro rows must be _PreparedCase objects")
    case_id = row.public_case.case_id
    _require_identifier(case_id, "macro cohort case_id")
    if row.target.case_id != case_id:
        raise ValueError("macro cohort target case identity changed")
    if row.target.contract_id != contract_id:
        raise ValueError("macro cohort target contract changed")


def _prepared_case_identity(row: _PreparedCase) -> tuple[str, str]:
    public = _digest_payload(
        {
            "schema": "macro-public-case-v2",
            "case": _case_payload(row.public_case),
            "query": _query_payload(row.query),
        }
    )
    target = _digest_payload(
        {
            "schema": "macro-private-target-case-v2",
            "case_id": row.target.case_id,
            "contract_id": row.target.contract_id,
            "query": _query_payload(row.target.query),
            "values": _view_payload(row.target.values),
            "sample_weights": _NumericDigestSequence(row.target.sample_weights),
        }
    )
    return public, target


def _public_cohort_digest(
    cohort_id: str,
    contract_id: str,
    ordered_case_sha256: Sequence[str],
) -> str:
    return _digest_payload(
        {
            "schema": "macro-public-cohort-v2",
            "cohort_id": cohort_id,
            "contract_id": contract_id,
            "ordered_case_sha256": list(ordered_case_sha256),
        }
    )


def _private_target_digest(
    public_cohort_sha256: str,
    ordered_target_sha256: Sequence[str],
) -> str:
    return _digest_payload(
        {
            "schema": "macro-private-target-v2",
            "public_cohort_sha256": public_cohort_sha256,
            "ordered_target_sha256": list(ordered_target_sha256),
        }
    )


def _prepared_case_storage_payload(
    row: _PreparedCase,
    codec: SafeNumpyValueCodec,
) -> Mapping[str, object]:
    return {
        "schema": "macro-prepared-case-store-v1",
        "case": {
            "case_id": row.public_case.case_id,
            "source": row.public_case.source.value,
            "visibility": row.public_case.visibility.value,
            "public_data": codec.encode(row.public_case.public_data),
            "round_index": row.public_case.round_index,
            "parent_case_id": row.public_case.parent_case_id,
            "tags": list(row.public_case.tags),
        },
        "query": _query_storage_payload(row.query, codec),
        "target": {
            "case_id": row.target.case_id,
            "contract_id": row.target.contract_id,
            "query": _query_storage_payload(row.target.query, codec),
            "values": _view_storage_payload(row.target.values, codec),
            "sample_weights": _numeric_storage_payload(
                row.target.sample_weights, codec
            ),
        },
    }


def _query_storage_payload(
    query: AnswerQuery,
    codec: SafeNumpyValueCodec,
) -> Mapping[str, object]:
    return {
        "kind": query.kind.value,
        "fields": list(query.fields),
        "resolution": codec.encode(query.resolution),
        "points": codec.encode(query.points),
    }


def _view_storage_payload(
    view: AnswerView,
    codec: SafeNumpyValueCodec,
) -> Mapping[str, object]:
    return {
        "kind": view.kind.value,
        "fields": {
            name: _numeric_storage_payload(values, codec)
            for name, values in sorted(view.fields.items())
        },
        "axes": {
            name: _numeric_storage_payload(values, codec)
            for name, values in sorted(view.axes.items())
        },
        "points": codec.encode(view.points),
        "shape": list(view.shape),
    }


def _numeric_storage_payload(
    values: Sequence[float],
    codec: SafeNumpyValueCodec,
) -> Mapping[str, object]:
    np = codec.np
    if isinstance(values, np.ndarray):
        array = np.asarray(values)
    else:
        array = np.fromiter(
            (float(value) for value in values),
            dtype=np.float64,
            count=len(values),
        )
    if array.ndim != 1:
        raise ValueError("macro cohort numerical values must be one-dimensional")
    return codec.encode(array)


def _prepared_case_from_storage_payload(
    payload: Mapping[str, object],
    codec: SafeNumpyValueCodec,
) -> _PreparedCase:
    if set(payload) != {"schema", "case", "query", "target"} or payload.get(
        "schema"
    ) != "macro-prepared-case-store-v1":
        raise MacroCohortStoreError("prepared macro cohort row schema changed")
    case_node = _required_mapping(payload, "case")
    query_node = _required_mapping(payload, "query")
    target_node = _required_mapping(payload, "target")
    case_expected = {
        "case_id",
        "source",
        "visibility",
        "public_data",
        "round_index",
        "parent_case_id",
        "tags",
    }
    if set(case_node) != case_expected:
        raise MacroCohortStoreError("stored macro Case fields changed")
    tags = case_node.get("tags")
    if not isinstance(tags, list) or any(
        not isinstance(tag, str) for tag in tags
    ):
        raise MacroCohortStoreError("stored macro Case tags are invalid")
    round_index = case_node.get("round_index")
    if round_index is not None and (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
    ):
        raise MacroCohortStoreError("stored macro Case round is invalid")
    parent_case_id = case_node.get("parent_case_id")
    if parent_case_id is not None and (
        not isinstance(parent_case_id, str) or not parent_case_id.strip()
    ):
        raise MacroCohortStoreError("stored macro Case parent is invalid")
    try:
        case = Case(
            case_id=_required_storage_text(case_node, "case_id"),
            source=CaseSource(_required_storage_text(case_node, "source")),
            visibility=CaseVisibility(
                _required_storage_text(case_node, "visibility")
            ),
            public_data=_decoded_mapping(codec, case_node.get("public_data")),
            round_index=round_index,
            parent_case_id=parent_case_id,
            tags=tuple(tags),
        )
        query = _query_from_storage_payload(query_node, codec)
        target = _target_from_storage_payload(target_node, codec)
    except (TypeError, ValueError) as exc:
        raise MacroCohortStoreError(
            "stored prepared macro case is invalid"
        ) from exc
    row = _PreparedCase(case, query, target)
    _validate_prepared_case(row, target.contract_id)
    return row


def _query_from_storage_payload(
    payload: Mapping[str, object],
    codec: SafeNumpyValueCodec,
) -> AnswerQuery:
    if set(payload) != {"kind", "fields", "resolution", "points"}:
        raise MacroCohortStoreError("stored macro query fields changed")
    fields = payload.get("fields")
    if not isinstance(fields, list) or any(
        not isinstance(field, str) or not field.strip() for field in fields
    ):
        raise MacroCohortStoreError("stored macro query fields are invalid")
    points = codec.decode(payload.get("points"))
    if not isinstance(points, tuple) or any(
        not isinstance(point, Mapping) for point in points
    ):
        raise MacroCohortStoreError("stored macro query points are invalid")
    return AnswerQuery(
        kind=QueryKind(_required_storage_text(payload, "kind")),
        fields=tuple(fields),
        resolution={
            str(name): int(value)
            for name, value in _decoded_mapping(
                codec, payload.get("resolution")
            ).items()
        },
        points=tuple(dict(point) for point in points),
    )


def _target_from_storage_payload(
    payload: Mapping[str, object],
    codec: SafeNumpyValueCodec,
) -> MacroTarget:
    if set(payload) != {
        "case_id",
        "contract_id",
        "query",
        "values",
        "sample_weights",
    }:
        raise MacroCohortStoreError("stored macro target fields changed")
    weights = codec.decode(payload.get("sample_weights"))
    return MacroTarget(
        case_id=_required_storage_text(payload, "case_id"),
        contract_id=_required_storage_text(payload, "contract_id"),
        query=_query_from_storage_payload(
            _required_mapping(payload, "query"), codec
        ),
        values=_view_from_storage_payload(
            _required_mapping(payload, "values"), codec
        ),
        sample_weights=_MMapFloatSequence(weights),
    )


def _view_from_storage_payload(
    payload: Mapping[str, object],
    codec: SafeNumpyValueCodec,
) -> AnswerView:
    if set(payload) != {"kind", "fields", "axes", "points", "shape"}:
        raise MacroCohortStoreError("stored macro AnswerView fields changed")
    field_nodes = _required_mapping(payload, "fields")
    axis_nodes = _required_mapping(payload, "axes")
    points = codec.decode(payload.get("points"))
    shape = payload.get("shape")
    if (
        not isinstance(points, tuple)
        or any(not isinstance(point, Mapping) for point in points)
        or not isinstance(shape, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in shape
        )
    ):
        raise MacroCohortStoreError("stored macro AnswerView metadata is invalid")
    fields = {
        str(name): _decoded_numeric_array(codec, node)
        for name, node in field_nodes.items()
    }
    axes = {
        str(name): tuple(
            float(value) for value in _decoded_numeric_array(codec, node)
        )
        for name, node in axis_nodes.items()
    }
    return AnswerView(
        kind=QueryKind(_required_storage_text(payload, "kind")),
        fields=fields,
        axes=axes,
        points=tuple(dict(point) for point in points),
        shape=tuple(shape),
    )


def _decoded_numeric_array(
    codec: SafeNumpyValueCodec,
    node: object,
):
    values = codec.decode(node)
    if not isinstance(values, codec.np.ndarray) or values.ndim != 1:
        raise MacroCohortStoreError(
            "stored macro numerical values are not a one-dimensional array"
        )
    return values


def _decoded_mapping(
    codec: SafeNumpyValueCodec,
    node: object,
) -> Mapping[str, object]:
    value = codec.decode(node)
    if not isinstance(value, Mapping):
        raise MacroCohortStoreError("stored macro value is not a mapping")
    return value


def _required_mapping(
    payload: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise MacroCohortStoreError(f"stored macro {name} must be an object")
    return value


def _required_storage_text(
    payload: Mapping[str, object],
    name: str,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise MacroCohortStoreError(
            f"stored macro {name} must be nonempty text"
        )
    return value


def _case_payload(case: Case) -> Mapping[str, object]:
    return {
        "case_id": case.case_id,
        "source": case.source.value,
        "visibility": case.visibility.value,
        "public_data": case.public_data,
        "round_index": case.round_index,
        "parent_case_id": case.parent_case_id,
        "tags": list(case.tags),
    }

def _query_payload(query: AnswerQuery) -> Mapping[str, object]:
    return {
        "kind": query.kind.value,
        "fields": list(query.fields),
        "resolution": query.resolution,
        "points": query.points,
    }


def _view_payload(view: AnswerView) -> Mapping[str, object]:
    return {
        "kind": view.kind.value,
        "fields": {
            name: _NumericDigestSequence(values)
            for name, values in sorted(view.fields.items())
        },
        "axes": {
            name: _NumericDigestSequence(values)
            for name, values in sorted(view.axes.items())
        },
        "points": view.points,
        "shape": list(view.shape),
    }


def _digest_payload(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    _update_canonical_digest(digest, payload)
    return digest.hexdigest()


def _digest_token(
    digest: object,
    tag: str,
    payload: bytes = b"",
) -> None:
    encoded_tag = tag.encode("utf-8")
    digest.update(struct.pack(">I", len(encoded_tag)))
    digest.update(encoded_tag)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _update_canonical_digest(digest: object, value: object) -> None:
    if isinstance(value, _NumericDigestSequence):
        _update_numeric_digest(digest, value.values)
        return
    if isinstance(value, Enum):
        _update_canonical_digest(digest, value.value)
        return
    if value is None:
        _digest_token(digest, "none")
        return
    if isinstance(value, bool):
        _digest_token(digest, "bool", b"1" if value else b"0")
        return
    if isinstance(value, str):
        _digest_token(digest, "str", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _digest_token(digest, "bytes", value)
        return
    if isinstance(value, bytearray):
        _digest_token(digest, "bytearray", bytes(value))
        return
    if isinstance(value, int):
        _digest_token(digest, "int", str(value).encode("ascii"))
        return
    if isinstance(value, Real):
        converted = float(value)
        if not isfinite(converted):
            raise ValueError("cohort digest values must be finite")
        _digest_token(digest, "float64", struct.pack(">d", converted))
        return
    if isinstance(value, complex):
        real, imag = float(value.real), float(value.imag)
        if not isfinite(real) or not isfinite(imag):
            raise ValueError("cohort digest values must be finite")
        _digest_token(digest, "complex128", struct.pack(">dd", real, imag))
        return
    if isinstance(value, Mapping):
        if any(
            not isinstance(key, str) or not key.strip()
            for key in value
        ):
            raise TypeError("cohort digest mapping keys must be nonempty strings")
        keys = sorted(value)
        _digest_token(digest, "mapping", struct.pack(">Q", len(keys)))
        for key in keys:
            _update_canonical_digest(digest, key)
            _update_canonical_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        kind = "tuple" if isinstance(value, tuple) else "list"
        _digest_token(digest, kind, struct.pack(">Q", len(value)))
        for item in value:
            _update_canonical_digest(digest, item)
        return

    try:
        import numpy as np
    except ImportError as exc:
        raise TypeError(
            f"unsupported cohort digest value type: {type(value).__name__}"
        ) from exc
    if isinstance(value, np.generic):
        _update_canonical_digest(digest, value.item())
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("cohort digest arrays cannot use object dtype")
        metadata = (
            value.dtype.str.encode("ascii")
            + b"\0"
            + b",".join(str(size).encode("ascii") for size in value.shape)
        )
        _digest_token(digest, "ndarray", metadata)
        iterator = np.nditer(
            value,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            order="C",
            buffersize=65_536,
        )
        for chunk in iterator:
            if chunk.dtype.kind in {"f", "c"} and not bool(
                np.all(np.isfinite(chunk))
            ):
                raise ValueError("cohort digest arrays must be finite")
            digest.update(np.asarray(chunk).tobytes(order="C"))
        return
    raise TypeError(
        f"unsupported cohort digest value type: {type(value).__name__}"
    )


def _update_numeric_digest(
    digest: object,
    values: Sequence[float],
) -> None:
    count = len(values)
    _digest_token(digest, "numeric-f64", struct.pack(">Q", count))
    try:
        import numpy as np
    except ImportError:
        buffer = bytearray()
        processed = 0
        for value in values:
            converted = float(value)
            if not isfinite(converted):
                raise ValueError("cohort numerical digest values must be finite")
            buffer.extend(struct.pack("<d", converted))
            processed += 1
            if len(buffer) >= 512 * 1024:
                digest.update(buffer)
                buffer.clear()
        digest.update(buffer)
    else:
        iterator = iter(values)
        processed = 0
        while True:
            chunk = np.fromiter(
                islice(iterator, 65_536),
                dtype="<f8",
            )
            if not chunk.size:
                break
            if not bool(np.all(np.isfinite(chunk))):
                raise ValueError("cohort numerical digest values must be finite")
            digest.update(chunk.tobytes(order="C"))
            processed += int(chunk.size)
    if processed != count:
        raise ValueError("cohort numerical digest length changed")




def _require_identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} sha256 must be 64 lowercase hex characters")


_MISSING_CANDIDATE_ATTRIBUTE = object()


def _uses_two_phase_points(
    candidate: object,
    query: AnswerQuery,
    *,
    allow_deferred: bool = False,
) -> bool:
    """Select the POINTS ABI without executing candidate attribute hooks."""

    if query.kind is not QueryKind.POINTS:
        return False
    declared = getattr_static(
        candidate,
        "prepare_solution",
        _MISSING_CANDIDATE_ATTRIBUTE,
    )
    if declared is not _MISSING_CANDIDATE_ATTRIBUTE:
        return True
    controller_loaded = getattr_static(
        candidate,
        "_controller_loaded_two_phase_abi",
        False,
    )
    if controller_loaded is True:
        return True
    deferred = getattr_static(candidate, "_deferred_solution_abi", False)
    return allow_deferred and deferred is True


def _predict_legacy(
    candidate: object,
    case: Case,
    query: AnswerQuery,
    budget: float | None,
) -> object:
    """Explicit one-shot compatibility path; never formal timing evidence."""

    predict = getattr(candidate, "predict_macros", None)
    if not callable(predict):
        raise TypeError("legacy candidate must define callable predict_macros")
    return predict(case, query, budget=budget)


def _macro_case_timing(
    *,
    case_id: str,
    query_kind: QueryKind,
    timing_assurance: TimingAssurance,
    prepared_result: PreparedSolutionResult | None,
    query_result: PointQueryResult | None,
    legacy_wall_seconds: float | None,
    controller_wall_seconds: float | None = None,
) -> MacroCaseTimingAudit:
    return MacroCaseTimingAudit(
        case_id=case_id,
        query_kind=query_kind,
        timing_assurance=timing_assurance,
        candidate_solve_wall_seconds=(
            None
            if prepared_result is None
            else prepared_result.candidate_solve_wall_seconds
        ),
        candidate_query_wall_seconds=(
            None
            if query_result is None
            else query_result.candidate_query_wall_seconds
        ),
        candidate_query_point_count=(
            None if query_result is None else query_result.point_count
        ),
        candidate_legacy_wall_seconds=legacy_wall_seconds,
        controller_truth_match_wall_seconds=controller_wall_seconds,
    )


def _clock_value(clock: Clock, label: str) -> float:
    value = clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must return a finite nonnegative number")
    return float(value)


def _wall_elapsed(started: float, finished: float) -> float:
    if finished < started:
        raise RuntimeError("wall clock moved backwards")
    return finished - started


def _validate_optional_wall_seconds(
    value: float | None,
    label: str,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")


def _validated_compute_budget(value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("compute_budget must be a finite positive number or None")
    return float(value)


def _validated_query_seconds_per_point_limit(
    value: float | None,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(
            "query_seconds_per_point_limit must be a finite positive number or None"
        )
    return float(value)


def _invalid_score(
    case_id: str,
    contract_id: str,
    failure_code: str,
) -> SanitizedMacroCaseScore:
    if failure_code not in _STABLE_FAILURE_CODES:
        raise ValueError("invalid internal macro failure code")
    return SanitizedMacroCaseScore(
        case_id=case_id,
        contract_id=contract_id,
        status="invalid",
        primary_error=None,
        metrics={},
        failures=(failure_code,),
    )


def _agent_visible_score(
    score: SanitizedMacroCaseScore,
    safe_metric_names: frozenset[str],
) -> SanitizedMacroCaseScore:
    metrics = {
        name: score.metrics[name]
        for name in sorted(safe_metric_names)
        if name in score.metrics
    }
    if score.status == "invalid":
        metrics = {}

    if score.status == "valid":
        failures: tuple[str, ...] = ()
    elif score.status == "rejected":
        failures = ("macro_gate_rejected",)
    else:
        failures = tuple(
            code
            for code in score.failures
            if code in _STABLE_FAILURE_CODES and code != "macro_gate_rejected"
        )
        if not failures:
            failures = ("macro_evaluation_failed",)
    return SanitizedMacroCaseScore(
        case_id=candidate_case_id(score.case_id),
        contract_id=score.contract_id,
        status=score.status,
        primary_error=score.primary_error,
        metrics=metrics,
        failures=failures,
    )
