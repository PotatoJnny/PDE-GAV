"""Minimal extension contracts exercised by the first product slice."""

from __future__ import annotations

from typing import Callable, Mapping, Protocol, Sequence

from .evaluation import EvaluationProgram
from .models import (
    AnswerQuery,
    AnswerView,
    Artifact,
    ArtifactEvaluation,
    Case,
    CaseAccessKind,
    CaseEvaluation,
    CaseRecord,
    DatasetInput,
    FormField,
    GenerationRequest,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    GeneratorFeedback,
    GeneratorState,
    PanelCard,
    PreparedDataset,
    ProgramInfo,
    TaskKind,
)

ProgressCallback = Callable[[str, float], None]


class TruthProgram(Protocol):
    """Controller-only continuous or high-accuracy answer source."""

    def answer(self, query: AnswerQuery) -> AnswerView:
        """Materialize only the requested finite answer view."""


class ProblemProgram(Protocol):
    @property
    def info(self) -> ProgramInfo: ...

    def setup_fields(self) -> tuple[FormField, ...]: ...

    def prepare_dataset(self, dataset: DatasetInput) -> PreparedDataset: ...

    def case_card(self, case: Case) -> PanelCard: ...


class TaskProgram(Protocol):
    @property
    def info(self) -> ProgramInfo: ...

    @property
    def kind(self) -> TaskKind: ...

    def setup_fields(self) -> tuple[FormField, ...]: ...

    def agent_brief(self, settings: Mapping[str, object]) -> str: ...

    def create_evaluation(
        self,
        settings: Mapping[str, object],
    ) -> EvaluationProgram:
        """Build the Task-owned scientific evaluation policy for one Run."""

    # Temporary compatibility methods while existing PDE Tasks migrate.
    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> CaseEvaluation: ...

    def select(
        self,
        candidates: Sequence[ArtifactEvaluation],
        incumbent: ArtifactEvaluation | None,
    ) -> ArtifactEvaluation: ...

    def result_card(self, evaluation: ArtifactEvaluation) -> PanelCard: ...


class GeneratorProgram(Protocol):
    """The one Generator contract implemented by every PDE plugin.

    Training and loading are explicit, separate operations. The Run engine may
    train a new checkpoint or receive an existing one, but the Controller only
    samples from a loaded GeneratorState.
    """

    @property
    def info(self) -> ProgramInfo: ...

    @property
    def capabilities(self) -> GeneratorCapabilities: ...

    def setup_fields(self) -> tuple[FormField, ...]: ...

    def train(
        self,
        dataset: PreparedDataset,
        settings: Mapping[str, object],
        progress: ProgressCallback | None = None,
    ) -> GeneratorCheckpoint: ...

    def load(
        self,
        checkpoint: GeneratorCheckpoint,
        settings: Mapping[str, object],
        progress: ProgressCallback | None = None,
    ) -> GeneratorState: ...

    def update_guidance(
        self,
        state: GeneratorState,
        feedback: GeneratorFeedback,
        settings: Mapping[str, object],
        progress: ProgressCallback | None = None,
    ) -> GeneratorState: ...

    def sample(
        self,
        state: GeneratorState,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
    ) -> tuple[CaseRecord, ...]: ...

    def open_case_session(
        self,
        state: GeneratorState,
        case: Case,
        access: CaseAccessKind,
    ) -> object | None: ...

    def case_session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]: ...

    def release(
        self,
        state: GeneratorState,
        case_ids: tuple[str, ...],
    ) -> None: ...

    def preview(self, record: CaseRecord) -> PanelCard: ...
