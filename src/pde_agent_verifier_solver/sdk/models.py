"""Small data objects shared by PDE, Task, Generator, Engine, and Panel code.

The public ``Case`` intentionally contains only information an Agent may see.
Continuous answers live behind ``TruthProgram`` in a controller-only
``CaseRecord`` and can only be materialized through an ``AnswerQuery``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from .macro_matching import MacroMatchResult
    from .programs import TruthProgram


class DatasetKind(str, Enum):
    NUMERICAL = "numerical"
    EXPERIMENTAL = "experimental"


class CaseSource(str, Enum):
    INITIAL = "initial"
    GENERATED = "generated"
    PERTURBED = "perturbed"
    PUBLIC = "public"


class CaseVisibility(str, Enum):
    DEVELOPMENT = "development"
    HIDDEN = "hidden"


class QueryKind(str, Enum):
    GRID = "grid"
    POINTS = "points"


class GeneratorMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


class GenerationKind(str, Enum):
    BASE = "base"
    GUIDED = "guided"
    PERTURBED = "perturbed"


class GenerationPurpose(str, Enum):
    """Why the Controller is asking for a fresh Generator batch.

    Purpose is deliberately separate from visibility and distribution kind.
    For example, both a selection batch and a terminal batch are hidden, and
    either one may be sampled from the base or guided distribution.
    """

    DEVELOPMENT = "development"
    DEVELOPMENT_SCREEN = "development_screen"
    GUIDANCE_FEEDBACK = "guidance_feedback"
    SELECTION = "selection"
    PROMOTION = "promotion"
    TERMINAL = "terminal"


class CaseAccessKind(str, Enum):
    """Numerical access that may be opened for one issued Case."""

    PUBLIC_PROBLEM = "public_problem"
    DEVELOPMENT_SOLUTION = "development_solution"


class TaskKind(str, Enum):
    SOLVER = "solver"
    OBSERVABLE_MODEL = "observable_model"
    REDUCED_DYNAMICS = "reduced_dynamics"
    STATISTICS = "statistics"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ProgramInfo:
    program_id: str
    title: str
    description: str


@dataclass(frozen=True)
class FormField:
    """A deliberately small field description understood by the shared panel."""

    name: str
    label: str
    field_type: str
    default: object | None = None
    help_text: str = ""
    choices: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PanelCard:
    title: str
    summary: str
    values: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetInput:
    dataset_id: str
    kind: DatasetKind
    location: str | None = None
    problem: Mapping[str, object] = field(default_factory=dict)
    geometry: Mapping[str, object] = field(default_factory=dict)
    settings: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Case:
    """The complete Agent-visible form of one PDE question."""

    case_id: str
    source: CaseSource
    visibility: CaseVisibility
    public_data: Mapping[str, object]
    round_index: int | None = None
    parent_case_id: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerQuery:
    """Request either a sampled grid or truth at explicit points."""

    kind: QueryKind
    fields: tuple[str, ...]
    resolution: Mapping[str, int] = field(default_factory=dict)
    points: tuple[Mapping[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, QueryKind):
            raise TypeError("answer query kind must be a QueryKind")
        if not self.fields:
            raise ValueError("an answer query needs at least one field")
        if self.kind is QueryKind.GRID:
            if not self.resolution:
                raise ValueError("a grid query needs a resolution")
            if any(value <= 0 for value in self.resolution.values()):
                raise ValueError("grid resolutions must be positive")
        if self.kind is QueryKind.POINTS and not self.points:
            raise ValueError("a point query needs at least one point")

    @classmethod
    def grid(cls, *, fields: Sequence[str], **resolution: int) -> AnswerQuery:
        return cls(QueryKind.GRID, tuple(fields), dict(resolution))

    @classmethod
    def at_points(
        cls,
        *,
        fields: Sequence[str],
        points: Sequence[Mapping[str, float]],
    ) -> AnswerQuery:
        return cls(QueryKind.POINTS, tuple(fields), points=tuple(points))


@dataclass(frozen=True)
class AnswerView:
    """A finite numerical view; never the underlying continuous answer object.

    Coordinate metadata is part of the response ABI.  Grid producers should
    declare every axis and a shape consistent with the flattened field order;
    point producers should echo the requested points.  A PDE-owned evaluator
    may require those coordinates to match its query exactly.
    """

    kind: QueryKind
    fields: Mapping[str, tuple[float, ...]]
    axes: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    points: tuple[Mapping[str, float], ...] = ()
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class CaseRecord:
    """Controller-owned Case plus its non-public truth provider."""

    case: Case
    truth: TruthProgram
    # Experimental reconstruction may deliberately give the Generator a
    # completed/denoised training target while retaining the measured
    # reference as truth. Numerical datasets normally leave this unset.
    generator_target: TruthProgram | None = None


@dataclass(frozen=True)
class PreparedDataset:
    dataset: DatasetInput
    development_cases: tuple[CaseRecord, ...]
    hidden_cases: tuple[CaseRecord, ...]
    training_summary: Mapping[str, object] = field(default_factory=dict)
    # Generator fitting/validation are separate scientific roles from Agent
    # development and Official/Terminal evaluation. Existing numerical
    # plugins may leave these empty and keep using development_cases.
    generator_training_cases: tuple[CaseRecord, ...] = ()
    generator_validation_cases: tuple[CaseRecord, ...] = ()


@dataclass(frozen=True)
class Artifact:
    """A candidate scientific program produced by an Agent."""

    artifact_id: str
    kind: TaskKind
    name: str
    program: object
    summary: str
    usage: str
    round_index: int
    agent_id: str
    parent_artifact_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    primary_error: float
    metrics: Mapping[str, float]
    prediction: AnswerView | None = None
    macro_match: MacroMatchResult | None = None


@dataclass(frozen=True)
class ArtifactEvaluation:
    artifact: Artifact
    primary_error: float
    metrics: Mapping[str, float]
    cases: tuple[CaseEvaluation, ...]


@dataclass(frozen=True)
class GeneratorCapabilities:
    offline: bool = True
    online: bool = False
    guidance: bool = False
    perturbations: bool = False
    trainable: bool = True


@dataclass(frozen=True)
class GeneratorCheckpoint:
    """Output of Generator training, or a reference to an existing model.

    ``payload`` is intentionally owned by the PDE plugin. A small local
    Generator may keep an in-memory object here; a cluster implementation may
    store a path or a lightweight checkpoint descriptor. Training and loading
    remain two explicit operations even though both use this one value.
    """

    generator_id: str
    payload: object
    location: str | None = None
    report: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.generator_id, str) or not self.generator_id.strip():
            raise ValueError("Generator checkpoint needs a nonempty generator_id")
        if self.location is not None and (
            not isinstance(self.location, str) or not self.location.strip()
        ):
            raise ValueError("Generator checkpoint location cannot be empty")


@dataclass(frozen=True)
class GeneratorState:
    """One loaded Generator used by a Run.

    The Controller carries this value but never interprets ``base`` or
    ``guidance``. ``update_guidance`` returns a replacement state, leaving the
    base Generator unchanged.
    """

    checkpoint: GeneratorCheckpoint
    base: object
    guidance: object | None = None


@dataclass(frozen=True)
class GenerationRequest:
    round_index: int
    count: int
    mode: GeneratorMode
    kind: GenerationKind
    visibility: CaseVisibility
    seed: int = 0
    purpose: GenerationPurpose = GenerationPurpose.DEVELOPMENT
    anchors: tuple[Case, ...] = ()
    settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("generation round_index must be nonnegative")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 1
        ):
            raise ValueError("generation count must be positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("generation seed must be nonnegative")
        anchors = tuple(self.anchors)
        if any(not isinstance(anchor, Case) for anchor in anchors):
            raise TypeError("generation anchors must contain Case values")
        if self.kind is GenerationKind.PERTURBED:
            if any(
                anchor.source is CaseSource.PERTURBED
                or anchor.parent_case_id is not None
                for anchor in anchors
            ):
                raise ValueError(
                    "perturbed generation anchors must be clean roots"
                )
        elif anchors:
            raise ValueError("only perturbed generation accepts anchors")
        object.__setattr__(self, "anchors", anchors)


@dataclass(frozen=True)
class CaseFeedback:
    case: Case
    primary_error: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class RoundFeedback:
    round_index: int
    selected_artifact: Artifact
    cases: tuple[CaseFeedback, ...]


@dataclass(frozen=True)
class GeneratorFeedback:
    """Controller-private Solver errors used for one guidance update."""

    source_round_index: int
    target_round_index: int
    selected_program_id: str
    cases: tuple[CaseFeedback, ...]
    selected_program: object | None = None
    settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_round_index < 0:
            raise ValueError("Generator feedback source round cannot be negative")
        if self.target_round_index != self.source_round_index + 1:
            raise ValueError("Generator feedback target must be the next round")
        if (
            not isinstance(self.selected_program_id, str)
            or not self.selected_program_id.strip()
        ):
            raise ValueError("Generator feedback needs a selected program id")
        if not self.cases:
            raise ValueError("Generator feedback needs at least one scored Case")
