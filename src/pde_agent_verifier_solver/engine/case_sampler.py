"""One round-local sampling service shared by adaptive and fixed workflows.

One instance belongs to one Agent in one round.  Original, base, guided, and
perturbed Cases share an optional cumulative allowance. None removes that
allowance; issued Cases and source counts are still tracked. A failed or
invalid batch never consumes the allowance.
"""

from __future__ import annotations

from hashlib import sha256
from threading import Lock
from typing import Callable, Mapping, Sequence

from pde_agent_verifier_solver.sdk.models import (
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorCapabilities,
    GeneratorMode,
    GeneratorState,
    PreparedDataset,
)
from pde_agent_verifier_solver.sdk.programs import GeneratorProgram, ProgressCallback
from pde_agent_verifier_solver.sdk.sampling import (
    PairedSampleBatch,
    PairedSampleRequest,
    PairedSamplingAttemptAudit,
    PerturbCasesRequest,
    RecoverableSamplingRejection,
    SampleBatch,
    SampleCasesRequest,
    SampleSource,
    SamplingStrategy,
)


class CaseReservationConflict(ValueError):
    """A pre-commit registry rejected ids already owned by another batch."""

    def __init__(
        self,
        message: str,
        *,
        retained_generated_case_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.retained_generated_case_ids = frozenset(
            retained_generated_case_ids
        )


class RoundCaseSampler:
    """Serve development Cases to one Agent during one Campaign round.

    ``SamplingStrategy`` changes who calls this service, not how Cases are
    generated.  ``FIXED_RECIPE`` lets a Controller make the calls before the
    Agent starts; ``AGENT_ADAPTIVE`` lets the Agent make them while working.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        round_index: int,
        max_cases: int | None,
        strategy: SamplingStrategy,
        dataset: PreparedDataset,
        generator: GeneratorProgram | None = None,
        generator_state: GeneratorState | None = None,
        seed: int = 0,
        original_offset: int | None = None,
        generation_settings: Mapping[str, object] | None = None,
        visibility: CaseVisibility = CaseVisibility.DEVELOPMENT,
        purpose: GenerationPurpose = GenerationPurpose.DEVELOPMENT,
        seed_namespace: str = "",
        generation_progress: ProgressCallback | None = None,
        generator_lock: object | None = None,
        record_reserver: Callable[[Sequence[CaseRecord]], None] | None = None,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("round Case sampler needs a nonempty agent_id")
        _nonnegative_int(round_index, "round index")
        if max_cases is not None:
            _nonnegative_int(max_cases, "maximum Case count")
        _nonnegative_int(seed, "sampling seed")
        if original_offset is not None:
            _nonnegative_int(original_offset, "original Case offset")
        if not isinstance(strategy, SamplingStrategy):
            raise TypeError("sampling strategy must be a SamplingStrategy")
        if not isinstance(dataset, PreparedDataset):
            raise TypeError("round Case sampler needs a PreparedDataset")
        if not isinstance(visibility, CaseVisibility):
            raise TypeError("sampling visibility must be a CaseVisibility")
        if not isinstance(purpose, GenerationPurpose):
            raise TypeError("sampling purpose must be a GenerationPurpose")
        if not isinstance(seed_namespace, str):
            raise TypeError("sampling seed namespace must be a string")
        if generation_progress is not None and not callable(generation_progress):
            raise TypeError("sampling progress callback must be callable")
        if generator_lock is not None and (
            not callable(getattr(generator_lock, "acquire", None))
            or not callable(getattr(generator_lock, "release", None))
        ):
            raise TypeError("generator_lock must provide acquire and release")
        if record_reserver is not None and not callable(record_reserver):
            raise TypeError("record_reserver must be callable")
        if (generator is None) != (generator_state is None):
            raise ValueError("GeneratorProgram and GeneratorState must be provided together")

        original_records = tuple(dataset.development_cases)
        _validate_original_pool(original_records)
        if original_offset is not None:
            if original_offset > len(original_records):
                raise ValueError("original Case offset exceeds the development pool")
            original_records = original_records[original_offset:]
        elif original_records:
            # Rotate instead of shuffle: each round can start from a fresh part
            # of D0 while sampling remains deterministic and without replacement.
            shift = seed % len(original_records)
            original_records = original_records[shift:] + original_records[:shift]

        self.agent_id = agent_id
        self.round_index = round_index
        self.max_cases = max_cases
        self.strategy = strategy
        self._original_records = original_records
        self._generator = generator
        self._generator_state = generator_state
        self._seed = seed
        self._generation_settings = dict(generation_settings or {})
        self._visibility = visibility
        self._purpose = purpose
        self._seed_namespace = seed_namespace
        self._generation_progress = generation_progress
        self._generator_lock = generator_lock if generator_lock is not None else Lock()
        self._record_reserver = record_reserver

        self._original_offset = 0
        self._successful_batches = 0
        self._next_sampling_operation = 0
        self._issued_records: dict[str, CaseRecord] = {}
        # Live Generator-owned resources. Rejected attempts are released and
        # removed immediately; committed Cases remain until this cohort closes.
        self._generated_case_ids: set[str] = set()
        self._perturbable_case_ids: set[str] = set()
        self._clean_source_by_case_id: dict[str, SampleSource] = {}
        self._logical_root_index_by_case_id: dict[str, int] = {}
        self._clean_root_offsets = {
            SampleSource.BASE: 0,
            SampleSource.GUIDED: 0,
        }
        self._sampling_attempts: list[PairedSamplingAttemptAudit] = []
        self._closed = False
        self._lock = Lock()

    @property
    def issued_count(self) -> int:
        with self._lock:
            return len(self._issued_records)

    @property
    def remaining(self) -> int | None:
        with self._lock:
            return None if self.max_cases is None else self.max_cases - len(self._issued_records)

    @property
    def issued_cases(self) -> tuple[Case, ...]:
        with self._lock:
            return tuple(record.case for record in self._issued_records.values())

    @property
    def generated_case_ids(self) -> tuple[str, ...]:
        """Generator-owned Case ids still live in this cohort."""

        with self._lock:
            return tuple(sorted(self._generated_case_ids))

    @property
    def sampling_attempts(self) -> tuple[PairedSamplingAttemptAudit, ...]:
        """Immutable audit rows for paired and child-only retry attempts."""

        with self._lock:
            return tuple(self._sampling_attempts)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Release this cohort through the exact GeneratorState that created it."""

        with self._lock:
            if self._closed:
                return
            identifiers = tuple(sorted(self._generated_case_ids))
            generator = self._generator
            state = self._generator_state
            if generator is not None and state is not None:
                self._generator_lock.acquire()
                try:
                    generator.release(state, identifiers)
                finally:
                    self._generator_lock.release()
            elif identifiers:
                raise RuntimeError(
                    "generated Cases cannot be released without a Generator"
                )
            self._generated_case_ids.clear()
            self._perturbable_case_ids.clear()
            self._clean_source_by_case_id.clear()
            self._logical_root_index_by_case_id.clear()
            self._closed = True

    def record_for(self, case_id: str) -> CaseRecord:
        """Return the controller-private record for one issued public Case."""

        with self._lock:
            try:
                return self._issued_records[case_id]
            except KeyError as error:
                raise KeyError(f"Case {case_id!r} was not issued by this sampler") from error

    def sample_cases(self, request: SampleCasesRequest) -> SampleBatch:
        """Sample clean original, base, or guided Cases as one atomic batch."""

        if not isinstance(request, SampleCasesRequest):
            raise TypeError("sample_cases needs a SampleCasesRequest")
        with self._lock:
            self._require_open()
            operation_index = self._begin_sampling_operation()
            self._require_capacity(request.count)
            if request.source is SampleSource.ORIGINAL:
                if self._visibility is not CaseVisibility.DEVELOPMENT:
                    raise PermissionError(
                        "hidden Generator sampling cannot draw original development Cases"
                    )
                records = self._take_original(request.count)
                self._validate_new_records(
                    records,
                    expected_count=request.count,
                    expected_kind=request.source,
                )
            else:
                kind = (
                    GenerationKind.BASE
                    if request.source is SampleSource.BASE
                    else GenerationKind.GUIDED
                )
                logical_start = self._clean_root_offsets[request.source]
                records = self._sample_validated(
                    kind=kind,
                    count=request.count,
                    expected_kind=request.source,
                    seed=self._seed + operation_index,
                    settings=self._settings_for_root(logical_start),
                )
            batch = self._commit(records)
            if request.source is SampleSource.ORIGINAL:
                self._original_offset += request.count
            else:
                self._perturbable_case_ids.update(
                    record.case.case_id for record in records
                )
                self._clean_source_by_case_id.update(
                    (record.case.case_id, request.source) for record in records
                )
                self._logical_root_index_by_case_id.update(
                    (record.case.case_id, logical_start + index)
                    for index, record in enumerate(records)
                )
                self._clean_root_offsets[request.source] += request.count
            return batch

    def sample_paired_cases(
        self,
        request: PairedSampleRequest,
    ) -> PairedSampleBatch:
        """Atomically create fresh roots and children with root-local retries."""

        if not isinstance(request, PairedSampleRequest):
            raise TypeError("sample_paired_cases needs a PairedSampleRequest")
        with self._lock:
            self._require_open()
            operation_index = self._begin_sampling_operation()
            self._require_capacity(2 * request.count)
            logical_start = self._clean_root_offsets[request.source]
            clean_kind = (
                GenerationKind.BASE
                if request.source is SampleSource.BASE
                else GenerationKind.GUIDED
            )
            accepted_pairs: list[tuple[CaseRecord, CaseRecord]] = []
            owned: dict[str, CaseRecord] = {}
            operation_audits: list[PairedSamplingAttemptAudit] = []

            def own(records: tuple[CaseRecord, ...]) -> None:
                identifiers = tuple(record.case.case_id for record in records)
                if len(identifiers) != len(set(identifiers)) or set(
                    identifiers
                ).intersection(owned):
                    self._release_records(records)
                    for case_id in identifiers:
                        owned.pop(case_id, None)
                    raise ValueError("paired sampling reused an active Case id")
                owned.update(zip(identifiers, records))

            def release(records: tuple[CaseRecord, ...]) -> None:
                self._release_records(records)
                for record in records:
                    owned.pop(record.case.case_id, None)

            try:
                for root_index in range(request.count):
                    settings = self._settings_for_root(logical_start + root_index)
                    for attempt_index in range(request.max_attempts_per_root):
                        clean_seed = self._paired_attempt_seed(
                            request.source,
                            clean_kind,
                            operation_index,
                            root_index,
                            attempt_index,
                            request.count,
                        )
                        perturbed_seed = self._paired_attempt_seed(
                            request.source,
                            GenerationKind.PERTURBED,
                            operation_index,
                            root_index,
                            attempt_index,
                            request.count,
                        )
                        root_records: tuple[CaseRecord, ...] = ()
                        root_case_id: str | None = None
                        try:
                            root_records = self._sample_validated(
                                kind=clean_kind,
                                count=1,
                                expected_kind=request.source,
                                seed=clean_seed,
                                settings=settings,
                            )
                            own(root_records)
                            root = root_records[0]
                            root_case_id = root.case.case_id
                            child_records = self._sample_validated(
                                kind=GenerationKind.PERTURBED,
                                count=1,
                                anchors=(root.case,),
                                expected_parents=(root_case_id,),
                                seed=perturbed_seed,
                                settings=settings,
                            )
                            own(child_records)
                        except RecoverableSamplingRejection as error:
                            if root_records:
                                release(root_records)
                            audit = self._attempt_audit(
                                source=request.source,
                                operation_index=operation_index,
                                root_index=root_index,
                                attempt_index=attempt_index,
                                maximum_attempts=request.max_attempts_per_root,
                                clean_seed=clean_seed,
                                perturbed_seed=perturbed_seed,
                                root_case_id=root_case_id,
                                child_case_id=None,
                                accepted=False,
                                rejection_audit=error.audit,
                            )
                            self._sampling_attempts.append(audit)
                            operation_audits.append(audit)
                            continue

                        child = child_records[0]
                        accepted_pairs.append((root, child))
                        audit = self._attempt_audit(
                            source=request.source,
                            operation_index=operation_index,
                            root_index=root_index,
                            attempt_index=attempt_index,
                            maximum_attempts=request.max_attempts_per_root,
                            clean_seed=clean_seed,
                            perturbed_seed=perturbed_seed,
                            root_case_id=root_case_id,
                            child_case_id=child.case.case_id,
                            accepted=True,
                        )
                        self._sampling_attempts.append(audit)
                        operation_audits.append(audit)
                        break
                    else:
                        raise RecoverableSamplingRejection(
                            "paired Case sampling exhausted its fresh-draw attempt budget",
                            audit=self._exhaustion_audit(
                                request,
                                operation_index,
                                root_index,
                                operation_audits,
                            ),
                        )
            except BaseException:
                release(tuple(owned.values()))
                raise

            records = tuple(record for pair in accepted_pairs for record in pair)
            remaining = self._commit_records(records)
            roots = tuple(pair[0].case for pair in accepted_pairs)
            children = tuple(pair[1].case for pair in accepted_pairs)
            self._perturbable_case_ids.update(case.case_id for case in roots)
            self._clean_source_by_case_id.update(
                (case.case_id, request.source) for case in roots
            )
            self._logical_root_index_by_case_id.update(
                (case.case_id, logical_start + index)
                for index, case in enumerate(roots)
            )
            self._clean_root_offsets[request.source] += request.count
            return PairedSampleBatch(
                roots=roots,
                children=children,
                remaining=remaining,
                attempts=tuple(operation_audits),
            )

    def perturb_cases(self, request: PerturbCasesRequest) -> SampleBatch:
        """Route clean parent Cases to the PDE plugin's perturbation method.

        The shared engine only establishes lineage and counts Cases.  It does
        not know or choose how the physical perturbation is constructed.
        """

        if not isinstance(request, PerturbCasesRequest):
            raise TypeError("perturb_cases needs a PerturbCasesRequest")
        with self._lock:
            self._require_open()
            operation_index = self._begin_sampling_operation()
            self._require_capacity(len(request.parent_case_ids))
            parents = self._resolve_clean_parents(request.parent_case_ids)
            accepted: list[CaseRecord] = []
            owned: dict[str, CaseRecord] = {}
            operation_audits: list[PairedSamplingAttemptAudit] = []
            try:
                for root_index, parent in enumerate(parents):
                    source = self._clean_source_by_case_id[parent.case_id]
                    for attempt_index in range(16):
                        perturbed_seed = self._paired_attempt_seed(
                            source,
                            GenerationKind.PERTURBED,
                            operation_index,
                            root_index,
                            attempt_index,
                            len(parents),
                        )
                        try:
                            records = self._sample_validated(
                                kind=GenerationKind.PERTURBED,
                                count=1,
                                anchors=(parent,),
                                expected_parents=(parent.case_id,),
                                seed=perturbed_seed,
                                settings=self._settings_for_root(
                                    self._logical_root_index_by_case_id[parent.case_id]
                                ),
                            )
                        except RecoverableSamplingRejection as error:
                            audit = self._attempt_audit(
                                source=source,
                                operation_index=operation_index,
                                root_index=root_index,
                                attempt_index=attempt_index,
                                maximum_attempts=16,
                                clean_seed=None,
                                perturbed_seed=perturbed_seed,
                                root_case_id=parent.case_id,
                                child_case_id=None,
                                accepted=False,
                                rejection_audit=error.audit,
                            )
                            self._sampling_attempts.append(audit)
                            operation_audits.append(audit)
                            continue
                        child_id = records[0].case.case_id
                        if child_id in owned:
                            self._release_records(records)
                            owned.pop(child_id, None)
                            raise ValueError(
                                "perturbation sampling reused an active Case id"
                            )
                        owned[child_id] = records[0]
                        accepted.append(records[0])
                        audit = self._attempt_audit(
                            source=source,
                            operation_index=operation_index,
                            root_index=root_index,
                            attempt_index=attempt_index,
                            maximum_attempts=16,
                            clean_seed=None,
                            perturbed_seed=perturbed_seed,
                            root_case_id=parent.case_id,
                            child_case_id=records[0].case.case_id,
                            accepted=True,
                        )
                        self._sampling_attempts.append(audit)
                        operation_audits.append(audit)
                        break
                    else:
                        raise RecoverableSamplingRejection(
                            "perturbed Case sampling exhausted its attempt budget",
                            audit={
                                "schema": "paired_sampling_exhaustion_v1",
                                "agent_id": self.agent_id,
                                "round_index": self.round_index,
                                "purpose": self._purpose.value,
                                "operation_index": operation_index,
                                "root_index": root_index,
                                "maximum_attempts": 16,
                                "attempts": self._audit_payload(operation_audits),
                            },
                        )
            except BaseException:
                self._release_records(tuple(owned.values()))
                raise
            return self._commit(tuple(accepted))

    def _take_original(self, count: int) -> tuple[CaseRecord, ...]:
        end = self._original_offset + count
        if end > len(self._original_records):
            available = len(self._original_records) - self._original_offset
            raise ValueError(
                f"requested {count} original Cases, but only {available} remain"
            )
        return self._original_records[self._original_offset:end]

    def _generate(
        self,
        *,
        kind: GenerationKind,
        count: int,
        anchors: tuple[Case, ...] = (),
        seed: int | None = None,
        settings: Mapping[str, object] | None = None,
    ) -> tuple[CaseRecord, ...]:
        generator = self._generator
        state = self._generator_state
        if generator is None or state is None:
            raise ValueError(f"{kind.value} sampling needs a loaded Generator")
        capabilities = generator.capabilities
        if kind is GenerationKind.GUIDED:
            if not capabilities.guidance:
                raise ValueError("this Generator does not support guided sampling")
            if state.guidance is None:
                raise ValueError("guided sampling needs guidance frozen for this round")
        if kind is GenerationKind.PERTURBED and not capabilities.perturbations:
            raise ValueError("this Generator does not support perturbation")

        # GeneratorMode is a temporary adapter for older PDE plugins. Adaptive
        # versus fixed sampling says who chooses Cases; it does not change what
        # numerical service the plugin is capable of running.
        compatibility_mode = select_compatible_generator_mode(capabilities)
        request = GenerationRequest(
            round_index=self.round_index,
            count=count,
            mode=compatibility_mode,
            kind=kind,
            visibility=self._visibility,
            seed=(self._seed + self._successful_batches if seed is None else seed),
            purpose=self._purpose,
            anchors=anchors,
            settings=self._generation_settings if settings is None else settings,
        )
        self._generator_lock.acquire()
        try:
            records = tuple(generator.sample(state, request, self._generation_progress))
        finally:
            self._generator_lock.release()
        self._generated_case_ids.update(
            record.case.case_id
            for record in records
            if isinstance(record, CaseRecord)
        )
        return records

    def _sample_validated(
        self,
        *,
        kind: GenerationKind,
        count: int,
        anchors: tuple[Case, ...] = (),
        expected_kind: SampleSource | None = None,
        expected_parents: tuple[str, ...] = (),
        seed: int | None = None,
        settings: Mapping[str, object] | None = None,
    ) -> tuple[CaseRecord, ...]:
        records = self._generate(
            kind=kind,
            count=count,
            anchors=anchors,
            seed=seed,
            settings=settings,
        )
        try:
            self._validate_new_records(
                records,
                expected_count=count,
                expected_kind=expected_kind,
                expected_parents=expected_parents,
            )
        except BaseException:
            self._release_records(records)
            raise
        return records

    def _resolve_clean_parents(
        self,
        parent_case_ids: tuple[str, ...],
    ) -> tuple[Case, ...]:
        parents: list[Case] = []
        for case_id in parent_case_ids:
            record = self._issued_records.get(case_id)
            if record is None:
                raise ValueError(
                    f"perturbation parent {case_id!r} was not issued by this sampler"
                )
            if case_id not in self._perturbable_case_ids:
                raise ValueError("perturbation parents must be clean base or guided Cases")
            case = record.case
            parents.append(case)
        return tuple(parents)

    def _validate_new_records(
        self,
        records: tuple[CaseRecord, ...],
        *,
        expected_count: int,
        expected_kind: SampleSource | None = None,
        expected_parents: tuple[str, ...] = (),
    ) -> None:
        if len(records) != expected_count:
            raise ValueError(
                f"sampling requested {expected_count} Cases but received {len(records)}"
            )
        if any(not isinstance(record, CaseRecord) for record in records):
            raise TypeError("Generator sampling must return CaseRecord values")
        cases = tuple(record.case for record in records)
        ids = tuple(case.case_id for case in cases)
        if len(ids) != len(set(ids)):
            raise ValueError("sampled Case ids must be unique within a batch")
        duplicates = set(ids).intersection(self._issued_records)
        if duplicates:
            duplicate = sorted(duplicates)[0]
            raise ValueError(f"sampled Case id {duplicate!r} was already issued")
        if any(case.visibility is not self._visibility for case in cases):
            raise ValueError(
                f"Generator sampling must return {self._visibility.value} Cases"
            )

        if expected_kind is SampleSource.ORIGINAL:
            return
        if expected_kind in (SampleSource.BASE, SampleSource.GUIDED):
            if any(
                case.source is CaseSource.PERTURBED
                or case.parent_case_id is not None
                for case in cases
            ):
                raise ValueError("base and guided sampling must return clean Cases")
            return

        parent_ids = tuple(case.parent_case_id for case in cases)
        if any(case.source is not CaseSource.PERTURBED for case in cases):
            raise ValueError("perturbation must return perturbed Cases")
        if len(set(parent_ids)) != len(parent_ids) or set(parent_ids) != set(expected_parents):
            raise ValueError("perturbation must return exactly one child for each parent Case")

    def _require_capacity(self, count: int) -> None:
        if self.max_cases is None:
            return
        remaining = self.max_cases - len(self._issued_records)
        if count > remaining:
            raise ValueError(
                f"sampling requested {count} Cases but only {remaining} remain "
                f"for Agent {self.agent_id!r} in round {self.round_index}"
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("round Case sampler cohort is closed")

    def _begin_sampling_operation(self) -> int:
        operation_index = self._next_sampling_operation
        self._next_sampling_operation += 1
        return operation_index

    def _release_records(self, records: tuple[CaseRecord, ...]) -> None:
        identifiers = tuple(
            dict.fromkeys(
                record.case.case_id
                for record in records
                if isinstance(record, CaseRecord)
                and isinstance(record.case.case_id, str)
                and record.case.case_id.strip()
            )
        )
        if not identifiers:
            return
        generator = self._generator
        state = self._generator_state
        if generator is None or state is None:
            raise RuntimeError("generated Cases cannot be released without a Generator")
        self._generator_lock.acquire()
        try:
            generator.release(state, identifiers)
        finally:
            self._generator_lock.release()
        self._generated_case_ids.difference_update(identifiers)

    def _settings_for_root(self, root_index: int) -> Mapping[str, object]:
        settings = self._generation_settings
        logical_offset = settings.get("logical_root_offset")
        if isinstance(logical_offset, int) and not isinstance(logical_offset, bool):
            return {
                **settings,
                "logical_root_offset": logical_offset + root_index,
            }
        return settings

    def _paired_attempt_seed(
        self,
        source: SampleSource,
        kind: GenerationKind,
        operation_index: int,
        root_index: int,
        attempt_index: int,
        root_count: int,
    ) -> int:
        label = "\0".join(
            (
                "canonical-paired-root-local-retry-v1",
                str(self._seed),
                self._seed_namespace,
                self.agent_id,
                str(self.round_index),
                self._purpose.value,
                source.value,
                kind.value,
                str(operation_index),
                str(root_count),
                str(root_index),
                str(attempt_index),
            )
        )
        return int.from_bytes(
            sha256(label.encode("utf-8")).digest()[:8],
            "big",
        ) % (2**31)

    def _attempt_audit(
        self,
        *,
        source: SampleSource,
        operation_index: int,
        root_index: int,
        attempt_index: int,
        maximum_attempts: int,
        clean_seed: int | None,
        perturbed_seed: int,
        root_case_id: str | None,
        child_case_id: str | None,
        accepted: bool,
        rejection_audit: Mapping[str, object] | None = None,
    ) -> PairedSamplingAttemptAudit:
        return PairedSamplingAttemptAudit(
            agent_id=self.agent_id,
            round_index=self.round_index,
            purpose=self._purpose.value,
            source=source,
            operation_index=operation_index,
            root_index=root_index,
            attempt_index=attempt_index,
            max_attempts_per_root=maximum_attempts,
            clean_seed=clean_seed,
            perturbed_seed=perturbed_seed,
            root_case_id=root_case_id,
            child_case_id=child_case_id,
            accepted=accepted,
            rejection_audit={} if rejection_audit is None else rejection_audit,
        )

    @staticmethod
    def _audit_payload(
        attempts: list[PairedSamplingAttemptAudit],
    ) -> list[dict[str, object]]:
        return [
            {
                "source": row.source.value,
                "operation_index": row.operation_index,
                "root_index": row.root_index,
                "attempt_index": row.attempt_index,
                "clean_seed": row.clean_seed,
                "perturbed_seed": row.perturbed_seed,
                "root_case_id": row.root_case_id,
                "child_case_id": row.child_case_id,
                "accepted": row.accepted,
                "rejection_audit": dict(row.rejection_audit),
            }
            for row in attempts
        ]

    def _exhaustion_audit(
        self,
        request: PairedSampleRequest,
        operation_index: int,
        root_index: int,
        attempts: list[PairedSamplingAttemptAudit],
    ) -> dict[str, object]:
        return {
            "schema": "paired_sampling_exhaustion_v1",
            "agent_id": self.agent_id,
            "round_index": self.round_index,
            "purpose": self._purpose.value,
            "source": request.source.value,
            "operation_index": operation_index,
            "root_index": root_index,
            "maximum_attempts": request.max_attempts_per_root,
            "attempts": self._audit_payload(attempts),
        }

    def _commit_records(self, records: tuple[CaseRecord, ...]) -> int | None:
        if self._record_reserver is not None:
            try:
                self._record_reserver(records)
            except BaseException as exc:
                retained = (
                    exc.retained_generated_case_ids
                    if isinstance(exc, CaseReservationConflict)
                    else frozenset()
                )
                generated = tuple(
                    record
                    for record in records
                    if record.case.case_id in self._generated_case_ids
                    and record.case.case_id not in retained
                )
                # A duplicate id already names a resource owned by the earlier
                # committed batch. Do not release that shared identifier here;
                # remove it from this sampler so Campaign cleanup releases it
                # exactly once through the original owner.
                self._generated_case_ids.difference_update(retained)
                if generated:
                    self._release_records(generated)
                raise
        for record in records:
            self._issued_records[record.case.case_id] = record
        self._successful_batches += 1
        return None if self.max_cases is None else self.max_cases - len(self._issued_records)

    def _commit(self, records: tuple[CaseRecord, ...]) -> SampleBatch:
        remaining = self._commit_records(records)
        return SampleBatch(
            cases=tuple(record.case for record in records),
            remaining=remaining,
        )


def _validate_original_pool(records: tuple[CaseRecord, ...]) -> None:
    if any(not isinstance(record, CaseRecord) for record in records):
        raise TypeError("PreparedDataset development_cases must contain CaseRecord values")
    ids = tuple(record.case.case_id for record in records)
    if len(ids) != len(set(ids)):
        raise ValueError("PreparedDataset development Case ids must be unique")
    if any(
        record.case.visibility is not CaseVisibility.DEVELOPMENT
        for record in records
    ):
        raise ValueError("PreparedDataset development Cases must have development visibility")


def _nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def select_compatible_generator_mode(
    capabilities: GeneratorCapabilities,
) -> GeneratorMode:
    """Choose the best legacy mode that a Generator actually supports.

    New Campaign sampling is always on demand from the Controller's point of
    view. Some numerical backends still expose the older mode field, so prefer
    their online implementation when present and otherwise use offline.
    """

    if not isinstance(capabilities, GeneratorCapabilities):
        raise TypeError("Generator capabilities must be GeneratorCapabilities")
    if capabilities.online:
        return GeneratorMode.ONLINE
    if capabilities.offline:
        return GeneratorMode.OFFLINE
    raise ValueError("the selected Generator supports neither online nor offline sampling")
