"""Project-level reconstruction contracts for experimental observations.

Reconstruction is an optional data-preparation step. It is deliberately
outside CampaignRecipe and the round Controller: one Project freezes one
reconstruction artifact and every Run in that Project reuses it.

The public types separate the data visible to a reconstruction method from the
private reconstruction-validation reference. A ReconstructionProgram receives
only ReconstructionInput; the Problem plugin retains the validation reference
and evaluates the returned completed fields afterwards. That reference is not
a downstream Development, Official, or Terminal truth source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from .models import DatasetInput, FormField, PreparedDataset, ProgramInfo


ReconstructionProgress = Callable[[str, float], None]


@dataclass(frozen=True)
class ReconstructionInput:
    """The complete, safe view supplied to a reconstruction method.

    Scientific arrays remain plugin-owned opaque values. For example, an NS
    adapter may supply trajectory snapshots while a VPFP adapter supplies
    phase-space fields. Most importantly, this object never contains the
    sealed reconstruction-validation reference used to score the reconstruction.
    """

    # Publication-safe alias only. Private split manifests and paths remain in
    # Problem-owned data_settings and must not be copied here.
    public_split_id: str
    fit_data: object
    target_observations: object
    geometry: object
    noise_model: object | None = None
    pretrained_basis: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.public_split_id, "reconstruction public_split_id")
        _json_safe_mapping(self.metadata, "reconstruction input metadata")


@dataclass(frozen=True)
class PreparedReconstructionSource:
    """Problem-owned split of method-visible input and validation reference."""

    program_input: ReconstructionInput
    reconstruction_validation_reference: object

    def __post_init__(self) -> None:
        if not isinstance(self.program_input, ReconstructionInput):
            raise TypeError("program_input must be a ReconstructionInput")


@dataclass(frozen=True)
class ReconstructionOutput:
    """Completed fields returned by a method, before private evaluation.

    completed_data is a derived training artifact, never experimental truth.
    The Problem plugin converts it to a PreparedDataset while preserving the
    original truth providers separately.
    """

    method_id: str
    completed_data: object
    fields: tuple[str, ...]
    case_counts: Mapping[str, int]
    selected_rank: int | None = None
    regularization: float | None = None
    # Normalized, publication-safe method settings used for this artifact.
    # Raw recipe settings can contain private paths and stay outside the report.
    public_settings: Mapping[str, object] = field(default_factory=dict)
    # These diagnostics are included in the user-visible report. Programs must
    # only place publication-safe numerical summaries here, never source paths,
    # credentials, or held-out observations.
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.method_id, "reconstruction method_id")
        if not isinstance(self.fields, (tuple, list)):
            raise TypeError("reconstruction output fields must be a list or tuple")
        fields = tuple(self.fields)
        if not fields or any(
            not isinstance(value, str) or not value.strip()
            for value in fields
        ):
            raise ValueError("reconstruction output needs nonempty field names")
        object.__setattr__(self, "fields", fields)
        _count_mapping(self.case_counts, "reconstruction case_counts")
        if self.selected_rank is not None:
            if (
                isinstance(self.selected_rank, bool)
                or not isinstance(self.selected_rank, int)
                or self.selected_rank < 1
            ):
                raise ValueError("selected_rank must be a positive integer")
        if self.regularization is not None:
            if isinstance(self.regularization, bool) or not isinstance(
                self.regularization,
                (int, float),
            ):
                raise TypeError("regularization must be a real number")
            value = float(self.regularization)
            if not isfinite(value) or value < 0.0:
                raise ValueError("regularization must be finite and nonnegative")
        _json_safe_mapping(self.public_settings, "reconstruction public_settings")
        _json_safe_mapping(self.diagnostics, "reconstruction diagnostics")
        _warning_tuple(self.warnings)


@dataclass(frozen=True)
class ReconstructionEvaluation:
    """Private-reference scores produced by the trusted Problem adapter.

    Keys in ``per_target_metrics`` are public aliases, not private source IDs.
    The trusted plugin is responsible for sanitizing aliases and warnings.
    """

    aggregate_metrics: Mapping[str, float]
    per_target_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _metric_mapping(self.aggregate_metrics, "aggregate reconstruction metrics")
        if not isinstance(self.per_target_metrics, Mapping):
            raise TypeError("per_target_metrics must be a mapping")
        for target_id, metrics in self.per_target_metrics.items():
            _nonempty(target_id, "reconstruction target id")
            _metric_mapping(metrics, f"metrics for reconstruction target {target_id!r}")
        _count_mapping(self.counts, "reconstruction evaluation counts")
        _warning_tuple(self.warnings)


@dataclass(frozen=True)
class ReconstructionReport:
    """Small JSON-safe, user-visible record of one frozen reconstruction."""

    artifact_id: str
    program_id: str
    method_id: str
    public_split_id: str
    seed: int
    fields: tuple[str, ...]
    case_counts: Mapping[str, int]
    selected_rank: int | None
    regularization: float | None
    aggregate_metrics: Mapping[str, float]
    per_target_metrics: Mapping[str, Mapping[str, float]]
    warnings: tuple[str, ...]
    resolved_settings: Mapping[str, object]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.artifact_id, "reconstruction artifact_id"),
            (self.program_id, "reconstruction program_id"),
            (self.method_id, "reconstruction method_id"),
            (self.public_split_id, "reconstruction public_split_id"),
        ):
            _nonempty(value, name)
        _nonnegative_seed(self.seed)
        if not isinstance(self.fields, (tuple, list)):
            raise TypeError("reconstruction report fields must be a list or tuple")
        fields = tuple(self.fields)
        if not fields or any(
            not isinstance(value, str) or not value.strip()
            for value in fields
        ):
            raise ValueError("reconstruction report needs nonempty field names")
        object.__setattr__(self, "fields", fields)
        _count_mapping(self.case_counts, "reconstruction report case_counts")
        if self.selected_rank is not None and (
            isinstance(self.selected_rank, bool)
            or not isinstance(self.selected_rank, int)
            or self.selected_rank < 1
        ):
            raise ValueError("selected_rank must be a positive integer")
        if self.regularization is not None:
            if isinstance(self.regularization, bool) or not isinstance(
                self.regularization,
                (int, float),
            ):
                raise TypeError("regularization must be a real number")
            value = float(self.regularization)
            if not isfinite(value) or value < 0.0:
                raise ValueError("regularization must be finite and nonnegative")
        _metric_mapping(self.aggregate_metrics, "reconstruction report metrics")
        if not isinstance(self.per_target_metrics, Mapping):
            raise TypeError("per_target_metrics must be a mapping")
        for target_id, metrics in self.per_target_metrics.items():
            _nonempty(target_id, "reconstruction report target id")
            _metric_mapping(metrics, f"report metrics for target {target_id!r}")
        _warning_tuple(self.warnings)
        _json_safe_mapping(self.resolved_settings, "reconstruction settings")
        _json_safe_mapping(self.diagnostics, "reconstruction report diagnostics")
        # Frozen dataclasses do not recursively freeze mappings. This report is
        # reused by every Project Run, so detach and recursively freeze every
        # JSON-like container rather than exposing mutable nested state.
        for attribute in (
            "case_counts",
            "aggregate_metrics",
            "per_target_metrics",
            "resolved_settings",
            "diagnostics",
        ):
            object.__setattr__(self, attribute, _freeze_json(getattr(self, attribute)))

    def as_dict(self) -> dict[str, object]:
        """Return the stable customer/Panel representation."""

        return {
            "artifact_id": self.artifact_id,
            "program_id": self.program_id,
            "method_id": self.method_id,
            "public_split_id": self.public_split_id,
            "seed": self.seed,
            "fields": list(self.fields),
            "case_counts": _thaw_json(self.case_counts),
            "selected_rank": self.selected_rank,
            "regularization": self.regularization,
            "aggregate_metrics": _thaw_json(self.aggregate_metrics),
            "per_target_metrics": _thaw_json(self.per_target_metrics),
            "warnings": list(self.warnings),
            "resolved_settings": _thaw_json(self.resolved_settings),
            "diagnostics": _thaw_json(self.diagnostics),
            "truth_semantics": (
                "completed fields are reconstructed Generator-training targets, "
                "not experimental truth"
            ),
        }


class ReconstructionProgram(Protocol):
    """Replaceable method that completes method-visible observations."""

    @property
    def info(self) -> ProgramInfo: ...

    def setup_fields(self) -> tuple[FormField, ...]: ...

    def reconstruct(
        self,
        source: ReconstructionInput,
        settings: Mapping[str, object],
        seed: int,
        progress: ReconstructionProgress | None = None,
    ) -> ReconstructionOutput: ...


class ExperimentalProblemProgram(Protocol):
    """Optional Problem capability used only when Reconstruction is selected."""

    def prepare_reconstruction_source(
        self,
        dataset: DatasetInput,
        settings: Mapping[str, object],
        seed: int,
    ) -> PreparedReconstructionSource: ...

    def evaluate_reconstruction(
        self,
        output: ReconstructionOutput,
        reconstruction_validation_reference: object,
        settings: Mapping[str, object],
    ) -> ReconstructionEvaluation: ...

    def prepare_reconstructed_dataset(
        self,
        dataset: DatasetInput,
        output: ReconstructionOutput,
    ) -> PreparedDataset: ...


@dataclass(frozen=True)
class ReconstructionRecipe:
    """One explicit Project-level reconstruction selection."""

    program: ReconstructionProgram
    data_settings: Mapping[str, object] = field(default_factory=dict)
    method_settings: Mapping[str, object] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        info = getattr(self.program, "info", None)
        if not isinstance(info, ProgramInfo):
            raise TypeError("reconstruction program must expose ProgramInfo as info")
        if not callable(getattr(self.program, "reconstruct", None)):
            raise TypeError("reconstruction program must implement reconstruct()")
        _nonnegative_seed(self.seed)
        _json_safe_mapping(self.data_settings, "reconstruction data_settings")
        _json_safe_mapping(self.method_settings, "reconstruction method_settings")


def build_reconstruction_report(
    *,
    artifact_id: str,
    program: ReconstructionProgram,
    source: ReconstructionInput,
    output: ReconstructionOutput,
    evaluation: ReconstructionEvaluation,
    recipe: ReconstructionRecipe,
) -> ReconstructionReport:
    """Combine method diagnostics with independently computed private scores."""

    return ReconstructionReport(
        artifact_id=artifact_id,
        program_id=program.info.program_id,
        method_id=output.method_id,
        public_split_id=source.public_split_id,
        seed=recipe.seed,
        fields=output.fields,
        case_counts=dict(output.case_counts),
        selected_rank=output.selected_rank,
        regularization=output.regularization,
        aggregate_metrics=dict(evaluation.aggregate_metrics),
        per_target_metrics={
            target_id: dict(metrics)
            for target_id, metrics in evaluation.per_target_metrics.items()
        },
        warnings=tuple((*output.warnings, *evaluation.warnings)),
        # Raw recipe settings can contain local data/checkpoint paths. They are
        # intentionally not copied into the report shown by the Panel.
        resolved_settings={
            **dict(output.public_settings),
            "selected_rank": output.selected_rank,
            "regularization": output.regularization,
        },
        diagnostics={
            **dict(output.diagnostics),
            "evaluation_counts": dict(evaluation.counts),
        },
    )


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _nonnegative_seed(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("reconstruction seed must be an integer")
    if value < 0:
        raise ValueError("reconstruction seed must be nonnegative")


def _warning_tuple(values: object) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise TypeError("reconstruction warnings must be a tuple of nonempty strings")


def _count_mapping(values: object, name: str) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key, value in values.items():
        _nonempty(key, f"{name} key")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} values must be nonnegative integers")


def _metric_mapping(values: object, name: str) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key, value in values.items():
        _nonempty(key, f"{name} key")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} values must be real numbers")
        if not isfinite(float(value)):
            raise ValueError(f"{name} values must be finite")


def _json_safe_mapping(values: object, name: str) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _json_safe(values, name)


def _json_safe(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} cannot contain non-finite floats")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} mapping keys must be strings")
            _json_safe(item, name)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _json_safe(item, name)
        return
    raise TypeError(f"{name} must contain only JSON-safe primitive values")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
