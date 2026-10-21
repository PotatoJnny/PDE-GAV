"""Thin bridge from a physical GKW runtime to unified generation requests.

The runtime owns the hidden state, smooth perturbation, and nonlinear operator.
This module owns only dispatch, private root handles, validation, and lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Protocol, Sequence

from pde_agent_verifier_solver.sdk import (
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
)

from .contract import gkw_macro_query_panel, gkw_public_query_domain
from .public_contract import GKW_PUBLIC_ARRAY_SPECS, GKW_PUBLIC_SCALAR_NAMES
from .public_problem import (
    GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    GKW_PUBLIC_SESSION_METHODS,
)
from .release_io import validate_gkw_oracle_protocol


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_KEYS = frozenset(
    {"equation", "arrays", "scalars", "metadata", "oracle_protocol"}
)
_LEGACY_PUBLIC_KEYS = _PUBLIC_KEYS | {"macro_query"}
GKW_QUALIFIED_PERTURBATION_SEED_MODULUS = 2_147_483_647
_PURPOSE_COHORTS = {
    GenerationPurpose.DEVELOPMENT: "development",
    GenerationPurpose.DEVELOPMENT_SCREEN: "development_screen",
    GenerationPurpose.SELECTION: "selection",
    GenerationPurpose.PROMOTION: "promotion",
    GenerationPurpose.TERMINAL: "terminal",
    GenerationPurpose.GUIDANCE_FEEDBACK: "guidance_feedback",
}
_PURPOSE_LANES = {
    purpose: index for index, purpose in enumerate(_PURPOSE_COHORTS, start=1)
}


@dataclass(frozen=True)
class GKWManufacturingAudit:
    """O(1) provenance proof returned by the trusted physical runtime.

    Identifiers are opaque lineage tokens, not hashes of the hidden state.
    Expensive physical qualification belongs to the offline runtime stage; the
    online controller only checks that every public component and macro truth
    names the same manufactured answer.
    """

    seed: int
    answer_id: str
    source_answer_id: str
    initial_answer_id: str
    boundary_answer_id: str
    truth_answer_id: str
    answer_first: bool
    passed: bool
    parent_answer_id: str | None = None
    perturbation_id: str | None = None
    perturbation_qualification_id: str | None = None
    perturbation_qualification_seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("GKW manufacturing seed must be a nonnegative integer")
        for name in (
            "answer_id",
            "source_answer_id",
            "initial_answer_id",
            "boundary_answer_id",
            "truth_answer_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"GKW manufacturing {name} must be nonempty")
        for name in (
            "parent_answer_id",
            "perturbation_id",
            "perturbation_qualification_id",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"GKW manufacturing {name} must be nonempty")
        qualification_seed = self.perturbation_qualification_seed
        if qualification_seed is not None and (
            isinstance(qualification_seed, bool)
            or not isinstance(qualification_seed, int)
            or not 0 <= qualification_seed
            < GKW_QUALIFIED_PERTURBATION_SEED_MODULUS
        ):
            raise ValueError(
                "GKW perturbation qualification seed is invalid"
            )
        if not isinstance(self.answer_first, bool) or not isinstance(
            self.passed, bool
        ):
            raise TypeError("GKW manufacturing gates must be boolean")


@dataclass(frozen=True)
class GKWManufacturedCase:
    """One runtime result; its hidden handle never enters the public Case."""

    case_id: str
    public_data: Mapping[str, object]
    truth: object
    hidden_handle: object
    audit: GKWManufacturingAudit

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("GKW manufactured case id cannot be empty")
        if not isinstance(self.public_data, Mapping):
            raise TypeError("GKW manufactured public data must be a mapping")
        if not callable(getattr(self.truth, "answer", None)):
            raise TypeError("GKW manufactured truth must implement answer(query)")
        if self.hidden_handle is None:
            raise ValueError("GKW manufactured case needs a private hidden handle")
        if not isinstance(self.audit, GKWManufacturingAudit):
            raise TypeError("GKW manufactured case needs a manufacturing audit")


class GKWManufacturingRuntime(Protocol):
    """Narrow boundary implemented by the existing physical GKW runtime."""

    @property
    def implementation_sha256(self) -> str: ...

    @property
    def supports_perturbations(self) -> bool: ...

    def manufacture_clean(
        self,
        base_state: object,
        request: GenerationRequest,
        seeds: tuple[int, ...],
        progress=None,
    ) -> tuple[GKWManufacturedCase, ...]: ...

    def manufacture_perturbed(
        self,
        base_state: object,
        parent: GKWManufacturedCase,
        request: GenerationRequest,
        seed: int,
        progress=None,
    ) -> GKWManufacturedCase: ...

    def commit_manufactured(
        self,
        items: tuple[GKWManufacturedCase, ...],
    ) -> None: ...

    def abort_manufactured(
        self,
        items: tuple[GKWManufacturedCase, ...],
    ) -> None: ...

    def release_manufactured(self, handles: Sequence[object]) -> None: ...

    def public_session_factory(self, case: Case) -> object: ...

    def development_solution_session_factory(self, case: Case) -> object: ...


@dataclass(frozen=True)
class _CachedRoot:
    record: CaseRecord
    manufactured: GKWManufacturedCase
    provenance: tuple[str, str, str]


@dataclass(frozen=True)
class _IssuedCase:
    """Trusted controller-owned binding for one active public case."""

    case: Case
    manufactured: GKWManufacturedCase


class GKWManufacturedGeneratorBackend:
    """Implement BASE and PERTURBED using one injected physical runtime."""

    public_session_methods = GKW_PUBLIC_SESSION_METHODS
    development_solution_session_methods = (
        GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS
    )

    def __init__(self, runtime: GKWManufacturingRuntime) -> None:
        implementation = getattr(runtime, "implementation_sha256", None)
        if not isinstance(implementation, str) or not _SHA256.fullmatch(implementation):
            raise ValueError("GKW physical runtime needs an implementation/config digest")
        if not callable(getattr(runtime, "public_session_factory", None)):
            raise TypeError(
                "GKW physical runtime needs a public numerical-oracle session"
            )
        if not callable(
            getattr(runtime, "development_solution_session_factory", None)
        ):
            raise TypeError(
                "GKW physical runtime needs a DEVELOPMENT completed-solution session"
            )
        if not callable(getattr(runtime, "commit_manufactured", None)):
            raise TypeError("GKW physical runtime needs an atomic commit hook")
        if not callable(getattr(runtime, "abort_manufactured", None)):
            raise TypeError("GKW physical runtime needs an atomic abort hook")
        if not callable(getattr(runtime, "release_manufactured", None)):
            raise TypeError("GKW physical runtime needs a resident release hook")
        self.implementation_sha256 = implementation
        self.runtime = runtime
        self._roots: dict[str, _CachedRoot] = {}
        self._issued_cases: dict[str, _IssuedCase] = {}
        self._resident_references: dict[int, tuple[object, int]] = {}

    @property
    def supports_perturbations(self) -> bool:
        return getattr(self.runtime, "supports_perturbations", False) is True

    def public_session_factory(self, case: Case) -> object:
        if not isinstance(case, Case):
            raise TypeError("GKW public session requires a Case")
        issued = self._issued_cases.get(case.case_id)
        if issued is None:
            raise ValueError("GKW public session requires an active issued case id")
        _validate_presented_snapshot(issued.case, case)
        session = self.runtime.public_session_factory(issued.case)
        try:
            _validate_runtime_session(issued.case, case, session)
        except BaseException as exc:
            try:
                close = getattr(session, "close", None)
                if callable(close):
                    close()
            except BaseException as cleanup_exc:
                exc.add_note(
                    "GKW invalid public-session cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
        return session

    def development_solution_session_factory(self, case: Case) -> object:
        """Open truth access only for an active trusted DEVELOPMENT case."""

        if not isinstance(case, Case):
            raise TypeError("GKW completed-solution session requires a Case")
        if case.visibility is not CaseVisibility.DEVELOPMENT:
            raise PermissionError(
                "GKW completed-solution access is DEVELOPMENT-only"
            )
        issued = self._issued_cases.get(case.case_id)
        if issued is None:
            raise ValueError(
                "GKW completed-solution session requires an active issued case id"
            )
        if issued.case.visibility is not CaseVisibility.DEVELOPMENT:
            raise PermissionError(
                "GKW completed-solution access is DEVELOPMENT-only"
            )
        _validate_presented_snapshot(issued.case, case)
        session = self.runtime.development_solution_session_factory(
            issued.case
        )
        try:
            _validate_development_solution_session(case, session)
        except BaseException as exc:
            try:
                close = getattr(session, "close", None)
                if callable(close):
                    close()
            except BaseException as cleanup_exc:
                exc.add_note(
                    "GKW invalid completed-solution session cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
        return session

    def release_cases(self, case_ids: Sequence[str]) -> None:
        """Release active case residents exactly once; repeated calls are safe."""

        identifiers = tuple(case_ids)
        if any(
            not isinstance(case_id, str) or not case_id.strip()
            for case_id in identifiers
        ):
            raise ValueError("released GKW case ids must be nonempty strings")
        requested = tuple(dict.fromkeys(identifiers))
        issued = tuple(
            self._issued_cases[case_id]
            for case_id in requested
            if case_id in self._issued_cases
        )
        if not issued:
            return

        decrements: dict[int, int] = {}
        for row in issued:
            key = id(row.manufactured.hidden_handle)
            decrements[key] = decrements.get(key, 0) + 1
        released_handles: list[object] = []
        for key, decrement in decrements.items():
            handle, count = self._resident_references[key]
            if decrement > count:
                raise RuntimeError("GKW resident reference count underflow")
            if decrement == count:
                released_handles.append(handle)

        # Mutate local ownership only after runtime release succeeds, so a
        # failure leaves the operation retryable.
        if released_handles:
            self.runtime.release_manufactured(tuple(released_handles))
        for row in issued:
            case_id = row.case.case_id
            self._issued_cases.pop(case_id, None)
            self._roots.pop(case_id, None)
        for key, decrement in decrements.items():
            handle, count = self._resident_references[key]
            remaining = count - decrement
            if remaining:
                self._resident_references[key] = (handle, remaining)
            else:
                self._resident_references.pop(key, None)

    def train_base(self, dataset, settings, progress=None):
        raise RuntimeError("the unified GKW backend requires an already-frozen model")

    def fit_guidance(
        self,
        base_state,
        previous_guidance,
        feedback,
        settings,
        progress=None,
    ):
        raise RuntimeError("the unified GKW backend does not update model guidance")

    def generate(
        self,
        base_state: object,
        guidance_state: object | None,
        request: GenerationRequest,
        progress=None,
    ) -> tuple[CaseRecord, ...]:
        if guidance_state is not None:
            raise ValueError("the GKW manufactured backend is unguided")
        self._validate_request(request)
        if request.kind is GenerationKind.BASE:
            return self._generate_clean(base_state, request, progress)
        if request.kind is GenerationKind.PERTURBED:
            return self._generate_perturbed(base_state, request, progress)
        raise ValueError("the GKW backend supports BASE and PERTURBED only")

    def _generate_clean(self, base_state, request, progress) -> tuple[CaseRecord, ...]:
        provenance = _provenance(request)
        if provenance[1] != self.implementation_sha256:
            raise ValueError("GKW request artifact does not match the physical runtime")
        seeds = tuple(_derived_seed(request, index) for index in range(request.count))
        manufactured = tuple(
            self.runtime.manufacture_clean(base_state, request, seeds, progress)
        )
        try:
            if len(manufactured) != request.count:
                raise ValueError("GKW runtime returned the wrong clean case count")
            records = tuple(
                self._record(item, request, seed, parent=None)
                for item, seed in zip(manufactured, seeds)
            )
            identifiers = tuple(record.case.case_id for record in records)
            self._validate_new_ids(identifiers, "clean")
            pending_roots = {
                record.case.case_id: _CachedRoot(record, item, provenance)
                for record, item in zip(records, manufactured)
            }
            pending_cases = {
                record.case.case_id: _IssuedCase(record.case, item)
                for record, item in zip(records, manufactured)
            }
            self.runtime.commit_manufactured(manufactured)
        except BaseException:
            self.runtime.abort_manufactured(manufactured)
            raise
        self._roots.update(pending_roots)
        self._register_issued(pending_cases)
        return records

    def _generate_perturbed(
        self, base_state, request, progress
    ) -> tuple[CaseRecord, ...]:
        if not self.supports_perturbations:
            raise ValueError("GKW physical runtime does not support perturbations")
        provenance = _provenance(request)
        if provenance[1] != self.implementation_sha256:
            raise ValueError("GKW request artifact does not match the physical runtime")
        cached = []
        for anchor in request.anchors:
            root = self._roots.get(anchor.case_id)
            if root is None or root.record.case is not anchor:
                raise ValueError("GKW perturbation anchor is not a cached clean root")
            if root.provenance != provenance:
                raise ValueError("GKW perturbation anchor provenance changed")
            cached.append(root)
        seeds = tuple(_derived_seed(request, index) for index in range(request.count))
        manufactured_rows = []
        try:
            for root, seed in zip(cached, seeds):
                manufactured_rows.append(
                    self.runtime.manufacture_perturbed(
                        base_state, root.manufactured, request, seed, progress
                    )
                )
            manufactured = tuple(manufactured_rows)
            records = tuple(
                self._record(item, request, seed, parent=root)
                for item, seed, root in zip(manufactured, seeds, cached)
            )
            identifiers = tuple(record.case.case_id for record in records)
            self._validate_new_ids(identifiers, "perturbed")
            pending_cases = {
                record.case.case_id: _IssuedCase(record.case, item)
                for record, item in zip(records, manufactured)
            }
            self.runtime.commit_manufactured(manufactured)
        except BaseException:
            if manufactured_rows:
                self.runtime.abort_manufactured(tuple(manufactured_rows))
            raise
        self._register_issued(pending_cases)
        for anchor in request.anchors:
            self._roots.pop(anchor.case_id, None)
        return records

    def _register_issued(self, rows: Mapping[str, _IssuedCase]) -> None:
        self._issued_cases.update(rows)
        for row in rows.values():
            handle = row.manufactured.hidden_handle
            key = id(handle)
            current = self._resident_references.get(key)
            if current is None:
                self._resident_references[key] = (handle, 1)
            else:
                registered, count = current
                if registered is not handle:
                    raise RuntimeError("GKW resident object identity was reused")
                self._resident_references[key] = (handle, count + 1)

    def _validate_new_ids(self, identifiers: tuple[str, ...], label: str) -> None:
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"GKW runtime returned duplicate {label} case ids")
        duplicate = set(identifiers) & set(self._issued_cases)
        if duplicate:
            raise ValueError(f"GKW runtime reused case ids: {sorted(duplicate)}")

    def _record(
        self,
        item: GKWManufacturedCase,
        request: GenerationRequest,
        seed: int,
        *,
        parent: _CachedRoot | None,
    ) -> CaseRecord:
        if not isinstance(item, GKWManufacturedCase):
            raise TypeError("GKW runtime must return GKWManufacturedCase values")
        audit = item.audit
        if audit.seed != seed:
            raise ValueError("GKW runtime changed the deterministic case seed")
        if not audit.passed:
            raise ValueError("GKW runtime returned an inadmissible manufactured case")
        if not audit.answer_first:
            raise ValueError("GKW public problem was not derived answer-first")
        if not (
            audit.answer_id
            == audit.source_answer_id
            == audit.initial_answer_id
            == audit.boundary_answer_id
            == audit.truth_answer_id
        ):
            raise ValueError(
                "GKW source, initial/boundary data, and macro truth "
                "do not share one answer lineage"
            )
        if parent is None:
            if any(
                value is not None
                for value in (
                    audit.parent_answer_id,
                    audit.perturbation_id,
                    audit.perturbation_qualification_id,
                    audit.perturbation_qualification_seed,
                )
            ):
                raise ValueError("clean GKW cases cannot carry perturbation provenance")
            source = CaseSource.GENERATED
            parent_case_id = None
            tags = ("gkw", "generated", "clean", "answer-first")
        else:
            parent_audit = parent.manufactured.audit
            if (
                audit.parent_answer_id != parent_audit.answer_id
                or audit.perturbation_id is None
                or audit.perturbation_qualification_id is None
                or audit.perturbation_qualification_seed
                != seed % GKW_QUALIFIED_PERTURBATION_SEED_MODULUS
                or audit.answer_id == parent_audit.answer_id
            ):
                raise ValueError("perturbed GKW case has invalid answer lineage")
            source = CaseSource.PERTURBED
            parent_case_id = parent.record.case.case_id
            tags = (
                "gkw",
                "generated",
                "perturbed",
                "answer-first",
                "smooth-perturbation",
            )
        case = Case(
            case_id=item.case_id,
            source=source,
            visibility=request.visibility,
            public_data=item.public_data,
            round_index=request.round_index,
            parent_case_id=parent_case_id,
            tags=tags,
        )
        self._validate_public_case(case)
        if parent is not None:
            _validate_pair_geometry(parent.record.case, case)
        return CaseRecord(case, item.truth)

    @staticmethod
    def _validate_request(request: GenerationRequest) -> None:
        if not isinstance(request, GenerationRequest):
            raise TypeError("GKW backend needs a GenerationRequest")
        if (
            isinstance(request.round_index, bool)
            or not isinstance(request.round_index, int)
            or request.round_index < 0
        ):
            raise ValueError("GKW generation round must be an integer >= 0")
        if (
            isinstance(request.count, bool)
            or not isinstance(request.count, int)
            or request.count < 1
        ):
            raise ValueError("GKW generation count must be an integer >= 1")
        if request.kind is GenerationKind.BASE:
            if request.anchors:
                raise ValueError("clean GKW generation cannot have anchors")
        elif request.kind is GenerationKind.PERTURBED:
            if len(request.anchors) != request.count:
                raise ValueError("GKW perturbation needs one anchor per child")
            if len({anchor.case_id for anchor in request.anchors}) != request.count:
                raise ValueError("GKW perturbation anchors must be unique")
        else:
            raise ValueError("GKW generation kind must be BASE or PERTURBED")
        if (
            isinstance(request.seed, bool)
            or not isinstance(request.seed, int)
            or request.seed < 0
        ):
            raise ValueError("GKW generation request needs a nonnegative integer seed")
        _generation_cohort(request)
        _provenance(request)

    @staticmethod
    def _validate_public_case(case: Case) -> None:
        """Check only immutable packet headers; physics was qualified offline."""

        import numpy as np

        public = case.public_data
        if set(public) not in (_PUBLIC_KEYS, _LEGACY_PUBLIC_KEYS):
            raise ValueError("GKW runtime public-data keys changed")
        if public.get("equation") != "GKW electrostatic gyrokinetics":
            raise ValueError("GKW runtime equation label changed")
        arrays = public.get("arrays")
        scalars = public.get("scalars")
        metadata = public.get("metadata")
        if not all(isinstance(value, Mapping) for value in (arrays, scalars, metadata)):
            raise TypeError("GKW runtime public equation components are invalid")
        if set(arrays) != set(GKW_PUBLIC_ARRAY_SPECS):
            raise ValueError("GKW runtime public array keys changed")
        for name, (shape, dtype_name) in GKW_PUBLIC_ARRAY_SPECS.items():
            value = np.asarray(arrays[name])
            if value.shape != shape or value.dtype != np.dtype(dtype_name):
                raise TypeError(f"GKW public array {name} header changed")
            if value.flags.writeable:
                raise ValueError("GKW runtime public arrays must be read-only")
        if set(scalars) != GKW_PUBLIC_SCALAR_NAMES:
            raise ValueError("GKW runtime public scalar keys changed")
        for name in GKW_PUBLIC_SCALAR_NAMES:
            value = np.asarray(scalars[name])
            if value.shape != () or value.dtype != np.dtype("float64"):
                raise TypeError(f"GKW public scalar {name} header changed")
            if not np.isfinite(value.item()):
                raise ValueError(f"GKW public scalar {name} must be finite")
        if metadata.get("case_id") != case.case_id:
            raise ValueError("GKW public metadata case id changed")
        if metadata.get("source_origin") != "generated_solution_residual":
            raise ValueError("GKW generated cases need a reconstructed residual source")
        oracle = public.get("oracle_protocol")
        if not isinstance(oracle, Mapping):
            raise TypeError("GKW runtime needs a public numerical-oracle contract")
        validate_gkw_oracle_protocol(
            oracle,
            case_id=case.case_id,
            time_nodes=metadata["time_nodes"],
        )
        if "macro_query" in public:
            gkw_macro_query_panel(case)
        else:
            gkw_public_query_domain(case)


def _derived_seed(request: GenerationRequest, index: int) -> int:
    """Derive a local sampling seed without hashing case data or global RNG."""

    mask = (1 << 64) - 1
    kind_lane = 0 if request.kind is GenerationKind.BASE else 1
    value = (
        int(request.seed)
        ^ ((request.round_index + 1) * 0xD2B74407B1CE6E93)
        ^ ((kind_lane + 1) * 0xCA5A826395121157)
        ^ (_PURPOSE_LANES[request.purpose] * 0xA24BAED4963EE407)
        ^ ((index + 1) * 0x9E3779B97F4A7C15)
    ) & mask
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return value & ((1 << 63) - 1)


def _provenance(request: GenerationRequest) -> tuple[str, str, str]:
    generator_id = request.settings.get("fixed_generator_id")
    generator_sha256 = request.settings.get("fixed_generator_sha256")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (generator_id, generator_sha256)
    ):
        raise ValueError("GKW request lacks fixed Generator provenance")
    if not _SHA256.fullmatch(generator_sha256):
        raise ValueError("GKW Generator artifact sha256 is invalid")
    return generator_id, generator_sha256, _generation_cohort(request)


def _generation_cohort(request: GenerationRequest) -> str:
    try:
        return _PURPOSE_COHORTS[request.purpose]
    except (KeyError, TypeError) as exc:
        raise ValueError("GKW request purpose is unsupported") from exc


def _plain_contract(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise TypeError("GKW public metadata keys must be strings")
        return {name: _plain_contract(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_contract(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_contract(item())
    raise TypeError("GKW public metadata is not scalar JSON data")


def _validate_presented_snapshot(trusted: Case, presented: Case) -> None:
    """Bind an evaluator deep snapshot without scanning large public arrays."""

    import numpy as np

    envelope = (
        "case_id",
        "source",
        "visibility",
        "round_index",
        "parent_case_id",
        "tags",
    )
    if any(getattr(trusted, name) != getattr(presented, name) for name in envelope):
        raise ValueError("GKW public session Case envelope differs from issued case")
    left = trusted.public_data
    right = presented.public_data
    if (
        set(left) not in (_PUBLIC_KEYS, _LEGACY_PUBLIC_KEYS)
        or set(right) != set(left)
        or right.get("equation") != left.get("equation")
    ):
        raise ValueError("GKW public session contract differs from issued case")
    left_arrays = left.get("arrays")
    right_arrays = right.get("arrays")
    if not isinstance(right_arrays, Mapping) or set(right_arrays) != set(left_arrays):
        raise ValueError("GKW public session array headers differ from issued case")
    for name in left_arrays:
        trusted_value = np.asarray(left_arrays[name])
        presented_value = np.asarray(right_arrays[name])
        if (
            presented_value.shape != trusted_value.shape
            or presented_value.dtype != trusted_value.dtype
        ):
            raise ValueError("GKW public session array headers differ from issued case")
    left_scalars = left.get("scalars")
    right_scalars = right.get("scalars")
    if not isinstance(right_scalars, Mapping) or set(right_scalars) != set(left_scalars):
        raise ValueError("GKW public session scalars differ from issued case")
    if any(
        np.asarray(right_scalars[name]).reshape(()).item()
        != np.asarray(left_scalars[name]).reshape(()).item()
        for name in left_scalars
    ):
        raise ValueError("GKW public session scalars differ from issued case")
    for name in ("metadata", "oracle_protocol"):
        if _plain_contract(right.get(name)) != _plain_contract(left.get(name)):
            raise ValueError(
                f"GKW public session {name.replace('_', ' ')} differs from issued case"
            )
    if "macro_query" in left:
        left_panel = left.get("macro_query")
        right_panel = right.get("macro_query")
        if (
            not isinstance(left_panel, Mapping)
            or not isinstance(right_panel, Mapping)
            or set(right_panel) != set(left_panel)
            or right_panel.get("output_contract") != left_panel.get("output_contract")
        ):
            raise ValueError("GKW public session macro panel differs from issued case")
        for name in ("times", "field_indices"):
            trusted_value = np.asarray(left_panel[name])
            presented_value = np.asarray(right_panel[name])
            if (
                presented_value.shape != trusted_value.shape
                or presented_value.dtype != trusted_value.dtype
            ):
                raise ValueError("GKW public session macro panel differs from issued case")


def _validate_runtime_session(
    trusted: Case, presented: Case, session: object
) -> None:
    """Require the runtime session to expose the trusted active packet."""

    if str(getattr(session, "case_id", "")) != presented.case_id:
        raise ValueError("GKW public session case id differs from issued case")
    for name in ("arrays", "scalars", "metadata", "oracle_protocol"):
        if getattr(session, name, None) is not trusted.public_data[name]:
            raise ValueError(
                f"GKW runtime session {name.replace('_', ' ')} is not trusted"
            )
    for name in ("query_source", "query_initial_h", "query_initial_phi"):
        if not callable(getattr(session, name, None)):
            raise TypeError(f"GKW public session must expose {name}")


def _validate_development_solution_session(
    case: Case, session: object
) -> None:
    """Reject mixed public/private sessions and any direct hidden payload."""

    if str(getattr(session, "case_id", "")) != case.case_id:
        raise ValueError(
            "GKW completed-solution session case id differs from issued case"
        )
    if not callable(getattr(session, "query_completed_macros", None)):
        raise TypeError(
            "GKW completed-solution session must expose query_completed_macros"
        )
    if any(
        callable(getattr(session, name, None))
        for name in ("query_source", "query_initial_h", "query_initial_phi")
    ):
        raise TypeError(
            "GKW completed-solution and public equation sessions must be distinct"
        )
    if any(
        hasattr(session, name)
        for name in ("truth", "hidden_handle", "generator", "latent")
    ):
        raise TypeError(
            "GKW completed-solution session exposed a private implementation object"
        )


def _validate_pair_geometry(parent: Case, child: Case) -> None:
    parent_has_panel = "macro_query" in parent.public_data
    child_has_panel = "macro_query" in child.public_data
    if (
        any(
            parent.public_data[group] is not child.public_data[group]
            for group in ("arrays", "scalars")
        )
        or parent_has_panel != child_has_panel
        or (
            parent_has_panel
            and parent.public_data["macro_query"] is not child.public_data["macro_query"]
        )
        or _case_identity_blind_contract(parent.public_data["metadata"])
        != _case_identity_blind_contract(child.public_data["metadata"])
        or _case_identity_blind_contract(parent.public_data["oracle_protocol"])
        != _case_identity_blind_contract(child.public_data["oracle_protocol"])
    ):
        raise ValueError(
            "GKW perturbation did not reuse its trusted public geometry/panel"
        )


def _case_identity_blind_contract(value: object) -> Mapping[str, object]:
    """Compare a parent/child public contract while allowing only its Case id."""

    plain = _plain_contract(value)
    if not isinstance(plain, dict) or "case_id" not in plain:
        raise TypeError("GKW public geometry contract needs a case id")
    return {name: item for name, item in plain.items() if name != "case_id"}
