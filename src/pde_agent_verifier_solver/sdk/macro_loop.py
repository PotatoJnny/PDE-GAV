"""Thin contracts for one equation-neutral macroscopic Agent loop.

The physics boundary is deliberately explicit. A PDE owns its manufactured
case source and its MacroMatcher: density, momentum, fluxes, quadrature and
the single-case metric are not shared-controller policy. Everything
downstream of MacroMatchResult can be shared.
The bound Generator also owns every equation- and smoothness-specific
perturbation. Shared code only requests a perturbed child from a clean anchor
and validates the returned lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Callable, Mapping, Protocol

from .models import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseAccessKind,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    FormField,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    GeneratorFeedback,
    GeneratorMode,
    GeneratorState,
    PanelCard,
    PreparedDataset,
    ProgramInfo,
)
from .programs import GeneratorProgram
from .macro_matching import MacroMatcher
from .public_problem import PublicProblemBundle
from .sampling import RecoverableSamplingRejection


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CohortKind(str, Enum):
    """Controller-owned purposes with disjoint seed blocks."""

    DEVELOPMENT = "development"
    DEVELOPMENT_SCREEN = "development_screen"
    GUIDANCE_FEEDBACK = "guidance_feedback"
    SELECTION = "selection"
    PROMOTION = "promotion"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class SampleRequest:
    """Ask a frozen Generator for base cases or anchored perturbed children."""

    round_index: int
    cohort: CohortKind
    seed: int
    count: int
    settings: Mapping[str, object] = field(default_factory=dict)
    kind: GenerationKind = GenerationKind.BASE
    anchors: tuple[Case, ...] = ()

    def __post_init__(self) -> None:
        _require_integer(self.round_index, "sample round_index", minimum=0)
        if not isinstance(self.cohort, CohortKind):
            raise TypeError("sample cohort must be a CohortKind")
        _require_integer(self.seed, "sample seed", minimum=0)
        _require_integer(self.count, "sample count", minimum=1)
        if not isinstance(self.kind, GenerationKind):
            raise TypeError("sample kind must be a GenerationKind")
        if self.kind not in (GenerationKind.BASE, GenerationKind.PERTURBED):
            raise ValueError(
                "macro sampling kind must be BASE or PERTURBED"
            )
        anchors = tuple(self.anchors)
        if any(not isinstance(anchor, Case) for anchor in anchors):
            raise TypeError("sample anchors must contain Case values")
        anchor_ids = tuple(anchor.case_id for anchor in anchors)
        if any(
            not isinstance(case_id, str) or not case_id.strip() for case_id in anchor_ids
        ):
            raise ValueError("sample anchor ids cannot be empty")
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("sample anchor ids must be unique")
        if self.kind is GenerationKind.PERTURBED:
            if len(anchors) != self.count:
                raise ValueError(
                    "perturbed sampling needs exactly one anchor per case"
                )
            expected_visibility = (
                CaseVisibility.DEVELOPMENT
                if self.cohort is CohortKind.DEVELOPMENT
                else CaseVisibility.HIDDEN
            )
            for anchor in anchors:
                if not isinstance(anchor.source, CaseSource):
                    raise ValueError("sample anchor source must be a CaseSource")
                if (
                    anchor.source is CaseSource.PERTURBED
                    or anchor.parent_case_id is not None
                ):
                    raise ValueError(
                        "perturbed sampling anchors must be clean root cases"
                    )
                if anchor.visibility is not expected_visibility:
                    raise ValueError(
                        "sample anchor visibility must match the requested cohort"
                    )
                if anchor.round_index is not None:
                    _require_integer(
                        anchor.round_index,
                        "sample anchor round_index",
                        minimum=0,
                    )
                    if anchor.round_index != self.round_index:
                        raise ValueError(
                            "sample anchor round must match the request"
                        )
        elif anchors:
            raise ValueError("only perturbed sampling accepts anchors")
        object.__setattr__(self, "anchors", anchors)


class ManufacturedCaseSource(Protocol):
    """A bound, already-trained answer-first Generator.

    Implementations load and freeze model weights outside the controller. The
    controller can sample cases but has no train or update operation.
    """

    @property
    def generator_id(self) -> str: ...

    @property
    def artifact_sha256(self) -> str: ...

    @property
    def supports_perturbations(self) -> bool: ...

    def sample(self, request: SampleRequest) -> tuple[CaseRecord, ...]: ...

    def release_cases(self, case_ids: tuple[str, ...]) -> None: ...


class ManufacturedSourceGeneratorProgram:
    """Compatibility adapter for an old already-loaded case source.

    It is deliberately non-trainable. New PDE plugins should provide a normal
    GeneratorProgram and bind its loaded checkpoint with LoadedGenerator.
    """

    def __init__(self, source: ManufacturedCaseSource) -> None:
        self.source = source
        self.info = ProgramInfo(
            source.generator_id,
            source.generator_id,
            "Pre-trained manufactured Generator loaded through the compatibility adapter.",
        )
        self.capabilities = GeneratorCapabilities(
            offline=True,
            online=True,
            guidance=False,
            perturbations=(getattr(source, "supports_perturbations", False) is True),
            trainable=False,
        )

    def setup_fields(self) -> tuple[FormField, ...]:
        return ()

    def train(
        self,
        dataset: PreparedDataset,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorCheckpoint:
        raise RuntimeError(
            "this Generator was supplied pre-trained; bind its checkpoint instead"
        )

    def load(
        self,
        checkpoint: GeneratorCheckpoint,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorState:
        if checkpoint.generator_id != self.info.program_id:
            raise ValueError("manufactured source checkpoint id does not match")
        return GeneratorState(checkpoint=checkpoint, base=self.source)

    def update_guidance(
        self,
        state: GeneratorState,
        feedback: GeneratorFeedback,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorState:
        raise RuntimeError(
            "the legacy manufactured source has no unified guidance update"
        )

    def sample(
        self,
        state: GeneratorState,
        request: GenerationRequest,
        progress=None,
    ) -> tuple[CaseRecord, ...]:
        if request.kind is GenerationKind.GUIDED:
            raise ValueError(
                "legacy manufactured sources cannot sample guided Cases"
            )
        try:
            cohort = CohortKind(request.purpose.value)
        except ValueError as exc:
            raise ValueError(
                f"unsupported macro sampling purpose {request.purpose.value!r}"
            ) from exc
        return tuple(
            self.source.sample(
                SampleRequest(
                    round_index=request.round_index,
                    cohort=cohort,
                    seed=request.seed,
                    count=request.count,
                    settings=request.settings,
                    kind=request.kind,
                    anchors=request.anchors,
                )
            )
        )

    def open_case_session(
        self,
        state: GeneratorState,
        case: Case,
        access: CaseAccessKind,
    ) -> object | None:
        name = {
            CaseAccessKind.PUBLIC_PROBLEM: "public_session_factory",
            CaseAccessKind.DEVELOPMENT_SOLUTION: (
                "development_solution_session_factory"
            ),
        }[access]
        factory = getattr(self.source, name, None)
        return factory(case) if callable(factory) else None

    def case_session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]:
        name = {
            CaseAccessKind.PUBLIC_PROBLEM: "public_session_methods",
            CaseAccessKind.DEVELOPMENT_SOLUTION: (
                "development_solution_session_methods"
            ),
        }[access]
        return tuple(getattr(self.source, name, ()))

    def release(
        self,
        state: GeneratorState,
        case_ids: tuple[str, ...],
    ) -> None:
        release = getattr(self.source, "release_cases", None)
        if callable(release):
            release(case_ids)

    def preview(self, record: CaseRecord) -> PanelCard:
        return PanelCard(
            title=f"Generated case {record.case.case_id}",
            summary="Case sampled from a pre-trained Generator.",
            values={"source": record.case.source.value},
        )


@dataclass(frozen=True)
class LoadedGenerator:
    """One Generator checkpoint loaded for a Controller run."""

    program: GeneratorProgram
    state: GeneratorState
    artifact_sha256: str
    guidance_backend: object | None = None
    settings: Mapping[str, object] = field(default_factory=dict)
    mode: GeneratorMode = GeneratorMode.OFFLINE

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_sha256, "Generator artifact")
        if self.state.checkpoint.generator_id != self.program.info.program_id:
            raise ValueError("loaded Generator program and checkpoint ids differ")
        if not isinstance(self.mode, GeneratorMode):
            raise TypeError("loaded Generator mode must be a GeneratorMode")
        object.__setattr__(self, "settings", dict(self.settings))

    @classmethod
    def from_checkpoint(
        cls,
        *,
        program: GeneratorProgram,
        checkpoint: GeneratorCheckpoint,
        artifact_sha256: str,
        settings: Mapping[str, object] | None = None,
        guidance_backend: object | None = None,
    ) -> "LoadedGenerator":
        resolved_settings = dict(settings or {})
        state = program.load(checkpoint, resolved_settings)
        return cls(
            program=program,
            state=state,
            artifact_sha256=artifact_sha256,
            guidance_backend=guidance_backend,
            settings=resolved_settings,
        )

    @classmethod
    def from_program(
        cls,
        *,
        program: GeneratorProgram,
        base_state: object,
        artifact_sha256: str,
        settings: Mapping[str, object] | None = None,
        guidance_backend: object | None = None,
    ) -> "LoadedGenerator":
        """Compatibility helper for experiments that already hold base state."""

        checkpoint = GeneratorCheckpoint(
            generator_id=program.info.program_id,
            payload=base_state,
            report={"source": "pretrained"},
        )
        return cls.from_checkpoint(
            program=program,
            checkpoint=checkpoint,
            artifact_sha256=artifact_sha256,
            settings=settings,
            guidance_backend=guidance_backend,
        )

    @classmethod
    def from_manufactured_source(
        cls,
        source: ManufacturedCaseSource,
        *,
        guidance_backend: object | None = None,
    ) -> "LoadedGenerator":
        program = ManufacturedSourceGeneratorProgram(source)
        checkpoint = GeneratorCheckpoint(
            generator_id=source.generator_id,
            payload=source,
            report={"source": "legacy_manufactured_case_source"},
        )
        state = program.load(checkpoint, {})
        return cls(
            program=program,
            state=state,
            artifact_sha256=source.artifact_sha256,
            guidance_backend=guidance_backend,
        )

    @property
    def generator_id(self) -> str:
        return self.state.checkpoint.generator_id

    @property
    def supports_perturbations(self) -> bool:
        return self.program.capabilities.perturbations is True

    @property
    def supports_guidance(self) -> bool:
        return (
            self.program.capabilities.guidance is True
            or self.guidance_backend is not None
        )

    def update_guidance(
        self,
        feedback: GeneratorFeedback,
        settings: Mapping[str, object] | None = None,
        progress=None,
    ) -> "LoadedGenerator":
        resolved_settings = {**self.settings, **dict(settings or {})}
        updated_state = self.program.update_guidance(
            self.state,
            feedback,
            resolved_settings,
            progress,
        )
        return LoadedGenerator(
            program=self.program,
            state=updated_state,
            artifact_sha256=self.artifact_sha256,
            guidance_backend=self.guidance_backend,
            settings=resolved_settings,
            mode=self.mode,
        )

    def open_case_session(
        self,
        case: Case,
        access: CaseAccessKind,
    ) -> object | None:
        return self.program.open_case_session(self.state, case, access)

    def case_session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]:
        return tuple(self.program.case_session_methods(access))

    def sample(self, request: SampleRequest) -> tuple[CaseRecord, ...]:
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort is CohortKind.DEVELOPMENT
            else CaseVisibility.HIDDEN
        )
        settings = {
            **self.settings,
            **request.settings,
            "seed": request.seed,
            "cohort": request.cohort.value,
            "fixed_generator_id": self.generator_id,
            "fixed_generator_sha256": self.artifact_sha256,
        }
        return tuple(
            self.program.sample(
                self.state,
                GenerationRequest(
                    round_index=request.round_index,
                    count=request.count,
                    mode=self.mode,
                    kind=request.kind,
                    visibility=visibility,
                    seed=request.seed,
                    purpose=GenerationPurpose(request.cohort.value),
                    anchors=request.anchors,
                    settings=settings,
                ),
            )
        )

    def release_cases(self, case_ids: tuple[str, ...]) -> None:
        self.program.release(self.state, tuple(case_ids))


class MacroObjective(Protocol):
    """The complete equation-specific macroscopic matching boundary."""

    @property
    def matcher(self) -> MacroMatcher: ...
    @property
    def agent_visible_metrics(self) -> tuple[str, ...]: ...
    @property
    def published_metric_names(self) -> tuple[str, ...]: ...


    def query(self, case: Case, profile: str) -> AnswerQuery: ...


class MacroROMProgram(Protocol):
    """Legacy one-shot ABI retained for compatibility adapters."""

    def predict_macros(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView: ...


@dataclass(frozen=True)
class CandidateRef:
    """Immutable identity passed between rounds and processes."""

    candidate_id: str
    bundle_sha256: str
    round_index: int
    agent_id: str
    parent_sha256: str | None = None
    bundle_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.agent_id.strip():
            raise ValueError("candidate and Agent ids cannot be empty")
        _require_integer(self.round_index, "candidate round_index", minimum=0)
        if self.bundle_uri is not None and not self.bundle_uri.strip():
            raise ValueError("candidate bundle_uri cannot be empty")
        _require_sha256(self.bundle_sha256, "candidate bundle")
        if self.parent_sha256 is not None:
            _require_sha256(self.parent_sha256, "candidate parent")


@dataclass(frozen=True)
class CandidateSubmission:
    """Runtime program plus its persistent, immutable identity.

    A controller may attach ``program_factory`` to obtain one fresh runtime per
    formal cohort. Without it, stateless and legacy in-process integrations
    reuse ``program`` for compatibility.
    """

    ref: CandidateRef
    program: MacroROMProgram
    program_factory: Callable[[], MacroROMProgram] | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        if self.program_factory is not None and not callable(
            self.program_factory
        ):
            raise TypeError("candidate program_factory must be callable or None")

    def fresh_program(self) -> MacroROMProgram:
        """Return one controller-owned runtime for a formal cohort."""

        if self.program_factory is None:
            return self.program
        program = self.program_factory()
        if program is None:
            raise TypeError("candidate program_factory returned None")
        return program


@dataclass(frozen=True, init=False)
class MacroLoopPlugin:
    """The scientific capabilities supplied by one equation plugin."""

    plugin_id: str
    generator: LoadedGenerator | ManufacturedCaseSource
    objective: MacroObjective
    public_session_factory: Callable[[Case], object] | None = None
    public_session_methods: tuple[str, ...] = ()
    development_solution_session_factory: Callable[[Case], object] | None = None
    development_solution_session_methods: tuple[str, ...] = ()
    public_problem_bundle: PublicProblemBundle | None = None
    public_result_kind: str = "macro_target_rom"
    public_result_task: str = "macro_quantity_matching"

    def __init__(
        self,
        plugin_id: str,
        generator: LoadedGenerator | ManufacturedCaseSource | None = None,
        objective: MacroObjective | None = None,
        public_session_factory: Callable[[Case], object] | None = None,
        public_session_methods: tuple[str, ...] = (),
        development_solution_session_factory: (
            Callable[[Case], object] | None
        ) = None,
        development_solution_session_methods: tuple[str, ...] = (),
        public_problem_bundle: PublicProblemBundle | None = None,
        public_result_kind: str = "macro_target_rom",
        public_result_task: str = "macro_quantity_matching",
        *,
        cases: ManufacturedCaseSource | None = None,
    ) -> None:
        if generator is not None and cases is not None:
            raise ValueError("provide generator, not both generator and cases")
        resolved_generator = generator if generator is not None else cases
        if resolved_generator is None:
            raise ValueError("MacroLoopPlugin needs a loaded Generator")
        if objective is None:
            raise ValueError("MacroLoopPlugin needs a MacroObjective")
        values = {
            "plugin_id": plugin_id,
            "generator": resolved_generator,
            "objective": objective,
            "public_session_factory": public_session_factory,
            "public_session_methods": public_session_methods,
            "development_solution_session_factory": (
                development_solution_session_factory
            ),
            "development_solution_session_methods": (
                development_solution_session_methods
            ),
            "public_problem_bundle": public_problem_bundle,
            "public_result_kind": public_result_kind,
            "public_result_task": public_result_task,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("macro-loop plugin_id cannot be empty")
        if not isinstance(self.generator, LoadedGenerator):
            object.__setattr__(
                self,
                "generator",
                LoadedGenerator.from_manufactured_source(self.generator),
            )
        for label, value in (
            ("public_result_kind", self.public_result_kind),
            ("public_result_task", self.public_result_task),
        ):
            if (
                not isinstance(value, str)
                or not value.isidentifier()
                or value.startswith("_")
            ):
                raise ValueError(f"{label} must be a public identifier")
        if not self.generator.generator_id.strip():
            raise ValueError("Generator id cannot be empty")
        _require_sha256(self.generator.artifact_sha256, "Generator artifact")
        if self.public_session_factory is not None and not callable(
            self.public_session_factory
        ):
            raise TypeError("public session factory must be callable")
        methods = tuple(self.public_session_methods)
        if any(
            not isinstance(method, str)
            or not method.isidentifier()
            or method.startswith("_")
            for method in methods
        ):
            raise ValueError(
                "public session method allowlist needs public identifiers"
            )
        if len(methods) != len(set(methods)):
            raise ValueError("public session method allowlist must be unique")
        if methods and self.public_session_factory is None:
            raise ValueError(
                "public session methods require a public session factory"
            )
        if self.public_session_factory is not None and not methods:
            raise ValueError(
                "public session factory needs a method allowlist"
            )
        bundle = self.public_problem_bundle
        if bundle is not None and not isinstance(bundle, PublicProblemBundle):
            raise TypeError(
                "public_problem_bundle must be a PublicProblemBundle"
            )
        if bundle is not None:
            declared = {method.name for method in bundle.oracle_methods}
            if set(methods) != declared:
                raise ValueError(
                    "public session method allowlist must match the public "
                    "problem oracle contract"
                )

        development_factory = self.development_solution_session_factory
        if development_factory is not None and not callable(development_factory):
            raise TypeError(
                "development solution session factory must be callable"
            )
        development_methods = tuple(
            self.development_solution_session_methods
        )
        if any(
            not isinstance(method, str)
            or not method.isidentifier()
            or method.startswith("_")
            for method in development_methods
        ):
            raise ValueError(
                "development solution method allowlist needs public identifiers"
            )
        if len(development_methods) != len(set(development_methods)):
            raise ValueError(
                "development solution method allowlist must be unique"
            )
        if development_methods and development_factory is None:
            raise ValueError(
                "development solution methods require their own session factory"
            )
        if development_factory is not None and not development_methods:
            raise ValueError(
                "development solution session factory needs a method allowlist"
            )
        if set(methods) & set(development_methods):
            raise ValueError(
                "candidate and development solution session methods must be disjoint"
            )
        if (
            self.public_session_factory is not None
            and development_factory is not None
            and _same_callable(
                self.public_session_factory, development_factory
            )
        ):
            raise ValueError(
                "candidate and development solution sessions need distinct factories"
            )
        if bundle is not None:
            declared = {
                method.name
                for method in bundle.development_solution_oracle_methods
            }
            if set(development_methods) != declared:
                raise ValueError(
                    "development solution method allowlist must match the "
                    "public problem contract"
                )
        object.__setattr__(self, "public_session_methods", methods)
        object.__setattr__(
            self,
            "development_solution_session_methods",
            development_methods,
        )

    @property
    def cases(self) -> LoadedGenerator:
        """Deprecated name retained for older experiment scripts."""

        if not isinstance(self.generator, LoadedGenerator):
            raise TypeError("macro plugin Generator binding was not normalized")
        return self.generator


@dataclass(frozen=True)
class FixedGeneratorCaseSource:
    """Bind an existing GeneratorProgram to frozen, pre-trained state.

    This compatibility binding intentionally exposes only sample. It never
    calls train_base or fit_guidance, making the fixed-model invariant
    executable rather than documentary.
    """

    generator_id: str
    artifact_sha256: str
    generator: GeneratorProgram
    base_state: object
    settings: Mapping[str, object] = field(default_factory=dict)
    mode: GeneratorMode = GeneratorMode.OFFLINE

    def __post_init__(self) -> None:
        if not self.generator_id.strip():
            raise ValueError("Generator id cannot be empty")
        if not isinstance(self.mode, GeneratorMode):
            raise TypeError("fixed Generator mode must be a GeneratorMode")
        _require_sha256(self.artifact_sha256, "Generator artifact")

    @property
    def supports_perturbations(self) -> bool:
        return self.generator.capabilities.perturbations is True

    def sample(self, request: SampleRequest) -> tuple[CaseRecord, ...]:
        if (
            request.kind is GenerationKind.PERTURBED
            and not self.supports_perturbations
        ):
            raise ValueError("fixed Generator does not support perturbations")
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort is CohortKind.DEVELOPMENT
            else CaseVisibility.HIDDEN
        )
        settings = {
            **self.settings,
            **request.settings,
            "seed": request.seed,
            "cohort": request.cohort.value,
            "fixed_generator_id": self.generator_id,
            "fixed_generator_sha256": self.artifact_sha256,
        }
        generation_request = GenerationRequest(
            round_index=request.round_index,
            count=request.count,
            mode=self.mode,
            kind=request.kind,
            visibility=visibility,
            seed=request.seed,
            purpose=GenerationPurpose(request.cohort.value),
            anchors=request.anchors,
            settings=settings,
        )
        load = getattr(self.generator, "load", None)
        sample = getattr(self.generator, "sample", None)
        if callable(load) and callable(sample):
            checkpoint = GeneratorCheckpoint(
                generator_id=self.generator.info.program_id,
                payload=self.base_state,
                report={"source": "fixed_generator_compatibility"},
            )
            state = load(checkpoint, self.settings)
            records = tuple(sample(state, generation_request))
        else:
            # Old research backends remain readable while they migrate.
            records = tuple(
                self.generator.generate(
                    self.base_state,
                    None,
                    generation_request,
                )
            )
        try:
            if len(records) != request.count:
                raise ValueError(
                    "fixed Generator returned "
                    f"{len(records)} cases for a request of {request.count}"
                )
            if any(not isinstance(record, CaseRecord) for record in records):
                raise TypeError("fixed Generator responses must contain CaseRecord values")
            case_ids = tuple(record.case.case_id for record in records)
            if len(set(case_ids)) != len(case_ids):
                raise ValueError("fixed Generator returned duplicate case ids")
            if any(record.case.visibility is not visibility for record in records):
                raise ValueError("fixed Generator returned the wrong case visibility")
        except BaseException:
            self._release_records(records)
            raise
        return records

    def release_cases(self, case_ids: tuple[str, ...]) -> None:
        """Release optional equation-private state for a completed cohort."""

        identifiers = tuple(case_ids)
        if any(
            not isinstance(case_id, str) or not case_id.strip()
            for case_id in identifiers
        ):
            raise ValueError("released case ids must be nonempty strings")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("released case ids cannot contain duplicates")
        release = getattr(self.generator, "release", None)
        if callable(release):
            checkpoint = GeneratorCheckpoint(
                generator_id=self.generator.info.program_id,
                payload=self.base_state,
                report={"source": "fixed_generator_compatibility"},
            )
            load = getattr(self.generator, "load", None)
            state = (
                load(checkpoint, self.settings)
                if callable(load)
                else GeneratorState(checkpoint, self.base_state)
            )
            release(state, identifiers)
            return
        legacy_release = getattr(self.generator, "release_cases", None)
        if callable(legacy_release):
            legacy_release(identifiers)

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
        if identifiers:
            self.release_cases(identifiers)


def _same_callable(left, right) -> bool:
    left_owner = getattr(left, "__self__", None)
    right_owner = getattr(right, "__self__", None)
    left_function = getattr(left, "__func__", left)
    right_function = getattr(right, "__func__", right)
    return left_owner is right_owner and left_function is right_function


def _require_integer(value: int, label: str, *, minimum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} sha256 must be 64 lowercase hex characters")


class MacroQueryFactory(Protocol):
    def __call__(self, case: Case, profile: str) -> AnswerQuery: ...


class ControllerSeededMacroQueryFactory(Protocol):
    """Equation query factory driven by controller-private entropy."""

    def __call__(
        self, case: Case, profile: str, controller_seed: int
    ) -> AnswerQuery: ...


@dataclass(frozen=True)
class BoundMacroObjective:
    """Bind an equation-owned matcher to its equation-owned query builder."""

    matcher: MacroMatcher
    query_factory: MacroQueryFactory
    agent_visible_metrics: tuple[str, ...] = ()
    published_metric_names: tuple[str, ...] = ()
    controller_query_factory: ControllerSeededMacroQueryFactory | None = None

    def __post_init__(self) -> None:
        agent_names = _validated_metric_names(
            self.agent_visible_metrics,
            label="Agent-visible",
        )
        published_names = _validated_metric_names(
            self.published_metric_names,
            label="published",
        )
        object.__setattr__(self, "agent_visible_metrics", agent_names)
        object.__setattr__(self, "published_metric_names", published_names)

        if self.controller_query_factory is not None and not callable(
            self.controller_query_factory
        ):
            raise TypeError("controller query factory must be callable")

    def query(self, case: Case, profile: str) -> AnswerQuery:
        return self.query_factory(case, profile)

    def query_from_controller_seed(
        self,
        case: Case,
        profile: str,
        controller_seed: int,
    ) -> AnswerQuery:
        """Build from sealed controller entropy with legacy fallback."""

        _require_integer(
            controller_seed, "controller query seed", minimum=0
        )
        if self.controller_query_factory is None:
            return self.query(case, profile)
        return self.controller_query_factory(case, profile, controller_seed)


def _validated_metric_names(
    raw_names: tuple[str, ...],
    *,
    label: str,
) -> tuple[str, ...]:
    if isinstance(raw_names, (str, bytes)):
        raise ValueError(f"{label} metrics must be a sequence of names")
    names = tuple(raw_names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} metric names must be nonempty strings")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} metric names must be unique")
    return names
