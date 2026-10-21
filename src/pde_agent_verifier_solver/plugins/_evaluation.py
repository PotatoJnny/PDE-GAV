"""Small native EvaluationProgram adapter shared by product PDE plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    ArtifactEvaluation,
    Case,
    CandidateEvaluationOutput,
    CaseEvaluation,
    EvaluationPhase,
    EvaluationPlan,
    aggregate_case_evaluations,
)


CaseEvaluator = Callable[
    [Artifact, Case, AnswerView, AnswerQuery],
    CaseEvaluation,
]
CandidateExecutor = Callable[
    [Artifact, Case, EvaluationPlan],
    CandidateEvaluationOutput,
]
CandidateScorer = Callable[
    [Artifact, Case, AnswerView, EvaluationPlan, CandidateEvaluationOutput],
    CaseEvaluation,
]
PrivateDevelopmentQueryFactory = Callable[
    [Case, Mapping[str, object], int],
    AnswerQuery,
]


@dataclass(frozen=True)
class PluginTaskEvaluationProgram:
    """Adapt a plugin Task's existing per-Case evaluator to the native contract."""

    program_id: str
    default_query: AnswerQuery
    _evaluate_case: CaseEvaluator | None = field(default=None, repr=False)
    _private_development_query: PrivateDevelopmentQueryFactory | None = field(
        default=None,
        repr=False,
    )
    _execute_candidate: CandidateExecutor | None = field(default=None, repr=False)
    _score_candidate: CandidateScorer | None = field(default=None, repr=False)
    _execute_candidate_batch: Callable | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.program_id, str) or not self.program_id.strip():
            raise ValueError("native evaluation program needs a nonempty program_id")
        if not isinstance(self.default_query, AnswerQuery):
            raise TypeError("native evaluation program needs an AnswerQuery default_query")
        if self._evaluate_case is not None and not callable(self._evaluate_case):
            raise TypeError("native per-Case evaluator must be callable")
        if self._private_development_query is not None and not callable(
            self._private_development_query
        ):
            raise TypeError("private Development query factory must be callable")
        if self._execute_candidate_batch is not None and (
                not callable(self._execute_candidate_batch) or self._execute_candidate is None):
            raise TypeError("batch executor requires a callable and the single-Case split executor")
        has_executor = self._execute_candidate is not None
        has_scorer = self._score_candidate is not None
        if has_executor != has_scorer:
            raise ValueError(
                "split Candidate evaluation needs both execute_candidate and "
                "score_candidate"
            )
        if has_executor:
            if not callable(self._execute_candidate):
                raise TypeError("Candidate executor must be callable")
            if not callable(self._score_candidate):
                raise TypeError("Candidate scorer must be callable")
        elif self._evaluate_case is None:
            raise ValueError(
                "native evaluation program needs a per-Case evaluator or split "
                "Candidate evaluation callbacks"
            )
        object.__setattr__(self, "default_query", _immutable_query(self.default_query))

    @classmethod
    def from_settings(
        cls,
        *,
        program_id: str,
        settings: Mapping[str, object],
        evaluate_case: CaseEvaluator | None = None,
        private_development_query: PrivateDevelopmentQueryFactory | None = None,
        execute_candidate: CandidateExecutor | None = None,
        score_candidate: CandidateScorer | None = None,
        execute_candidate_batch: Callable | None = None,
    ) -> PluginTaskEvaluationProgram:
        query = settings.get("default_query")
        if not isinstance(query, AnswerQuery):
            raise TypeError("native evaluation settings need an AnswerQuery default_query")
        return cls(
            program_id=program_id,
            default_query=query,
            _evaluate_case=evaluate_case,
            _private_development_query=private_development_query,
            _execute_candidate=execute_candidate,
            _score_candidate=score_candidate,
            _execute_candidate_batch=execute_candidate_batch,
        )

    @property
    def supports_split_candidate_execution(self) -> bool:
        """Whether Candidate execution can run without private truth."""

        return self._execute_candidate is not None

    def execute_candidate(
        self,
        artifact: Artifact,
        case: Case,
        plan: EvaluationPlan,
    ) -> CandidateEvaluationOutput:
        executor = self._execute_candidate
        if executor is None:
            raise ValueError(
                f"EvaluationProgram {self.program_id!r} does not support split "
                "Candidate execution"
            )
        output = executor(artifact, case, plan)
        if not isinstance(output, CandidateEvaluationOutput):
            raise TypeError(
                "Candidate executor must return a CandidateEvaluationOutput"
            )
        return output

    @property
    def supports_batch_candidate_execution(self):
        return self._execute_candidate_batch is not None

    def execute_candidate_batch(self, artifact, cases, plans):
        if self._execute_candidate_batch is None:
            raise ValueError("Task does not support batch preparation")
        outputs = tuple(self._execute_candidate_batch(artifact, cases, plans))
        if len(outputs) != len(cases) or any(not isinstance(o, CandidateEvaluationOutput) for o in outputs):
            raise TypeError("batch executor must return one output per Case")
        return outputs

    def score_candidate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        plan: EvaluationPlan,
        output: CandidateEvaluationOutput,
    ) -> CaseEvaluation:
        scorer = self._score_candidate
        if scorer is None:
            raise ValueError(
                f"EvaluationProgram {self.program_id!r} does not support split "
                "Candidate scoring"
            )
        result = scorer(artifact, case, truth, plan, output)
        if not isinstance(result, CaseEvaluation):
            raise TypeError("Candidate scorer must return a CaseEvaluation")
        return result

    def make_private_development_plan(
        self,
        case: Case,
        settings: Mapping[str, object],
        seed: int,
    ) -> EvaluationPlan:
        """Let the Task select fresh coordinates without an Agent query."""

        factory = self._private_development_query
        if factory is None:
            raise ValueError(
                f"EvaluationProgram {self.program_id!r} does not support "
                "private random Development points"
            )
        query = factory(case, settings, seed)
        if not isinstance(query, AnswerQuery):
            raise TypeError(
                "private Development query factory must return an AnswerQuery"
            )
        return EvaluationPlan(
            phase=EvaluationPhase.DEVELOPMENT,
            query=_immutable_query(query),
            seed=seed,
            settings=settings,
        )

    def make_plan(
        self,
        case: Case,
        phase: EvaluationPhase,
        settings: Mapping[str, object],
        seed: int,
        request: AnswerQuery | None = None,
    ) -> EvaluationPlan:
        del case
        return EvaluationPlan(
            phase=phase,
            query=(
                self.default_query
                if request is None
                else _immutable_query(request)
            ),
            seed=seed,
            settings=settings,
        )

    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        plan: EvaluationPlan,
    ) -> CaseEvaluation:
        if self.supports_split_candidate_execution:
            output = self.execute_candidate(artifact, case, plan)
            return self.score_candidate(artifact, case, truth, plan, output)
        evaluator = self._evaluate_case
        if evaluator is None:  # pragma: no cover - guarded by __post_init__
            raise RuntimeError("EvaluationProgram has no per-Case evaluator")
        return evaluator(artifact, case, truth, plan.query)

    def aggregate(
        self,
        artifact: Artifact,
        case_results: Sequence[CaseEvaluation],
        plans: Sequence[EvaluationPlan],
    ) -> ArtifactEvaluation:
        return aggregate_case_evaluations(artifact, case_results, plans)


def _immutable_query(query: AnswerQuery) -> AnswerQuery:
    """Snapshot the finite query so callers cannot mutate a frozen plan by alias."""

    return AnswerQuery(
        kind=query.kind,
        fields=tuple(query.fields),
        resolution=MappingProxyType(dict(query.resolution)),
        points=tuple(MappingProxyType(dict(point)) for point in query.points),
    )


__all__ = ["PluginTaskEvaluationProgram"]
