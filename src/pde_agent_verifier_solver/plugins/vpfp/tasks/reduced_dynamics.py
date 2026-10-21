"""VPFP task with an explicit reduced state and reduced evolution."""

from __future__ import annotations

from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    ArtifactEvaluation,
    Case,
    CaseEvaluation,
    FormField,
    MacroMatcher,
    PanelCard,
    ProgramInfo,
    QueryKind,
    TaskKind,
    evaluate_macro_prediction,
)

from ..._evaluation import PluginTaskEvaluationProgram
from ..macro_matching import VPFPMacroMatcher
from ._shared import choose_smallest_error, private_vpfp_macro_query


def _private_development_query(
    case: Case,
    settings: Mapping[str, object],
    seed: int,
) -> AnswerQuery:
    return private_vpfp_macro_query(case, settings, seed)


class VPFPReducedDynamicsTask:
    """Require initialize, advance and observe rather than only outputs."""

    info = ProgramInfo(
        "vpfp-reduced-dynamics",
        "Find explicit VPFP reduced dynamics",
        "Return a reduced state, evolve it, then reconstruct requested moments.",
    )
    kind = TaskKind.REDUCED_DYNAMICS

    def __init__(self, macro_matcher: MacroMatcher | None = None) -> None:
        self.macro_matcher = (
            VPFPMacroMatcher() if macro_matcher is None else macro_matcher
        )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "maximum_state_dimension",
                "Maximum reduced-state dimension",
                "integer",
                128,
            ),
            FormField(
                "primary_metric",
                "Primary metric",
                "text",
                "vpfp_macro_match_v1_relative_l2",
            ),
        )

    def agent_brief(self, settings: Mapping[str, object]) -> str:
        maximum = int(settings.get("maximum_state_dimension", 128))
        return (
            "Return a model with initialize(case), "
            "advance(state, query_times, budget=None), and "
            "observe(reduced_trajectory, query) -> AnswerView. "
            f"The exposed reduced state dimension must not exceed {maximum}. "
            "Coordinate metadata is mandatory: for a GRID query, return "
            "shape in (t, x, y) order and the exact axes obtained from "
            "case.public_data['arrays'] after applying query.resolution "
            "(native coordinates when the count is omitted or unchanged, "
            "the midpoint for count 1, otherwise inclusive linear spacing "
            "between the public endpoints); flatten fields in that order. "
            "For a POINTS query, copy query.points exactly."
        )

    def create_evaluation(
        self,
        settings: Mapping[str, object],
    ) -> PluginTaskEvaluationProgram:
        return PluginTaskEvaluationProgram.from_settings(
            program_id="vpfp-reduced-dynamics-evaluation-v1",
            settings=settings,
            evaluate_case=self.evaluate,
            private_development_query=_private_development_query,
        )

    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> CaseEvaluation:
        model = artifact.program
        required = ("initialize", "advance", "observe")
        if any(not hasattr(model, name) for name in required):
            raise TypeError(
                "VPFP Reduced Dynamics Task expects initialize, advance "
                "and observe"
            )
        if query.kind is QueryKind.GRID:
            query_times = truth.axes.get("t", ())
        else:
            query_times = tuple(
                sorted({float(point["t"]) for point in query.points})
            )
        state = model.initialize(case)
        trajectory = model.advance(
            state,
            query_times,
            budget=None,
        )
        prediction = model.observe(trajectory, query)
        return evaluate_macro_prediction(
            matcher=self.macro_matcher,
            case=case,
            prediction=prediction,
            truth=truth,
            query=query,
        )

    def select(
        self,
        candidates: Sequence[ArtifactEvaluation],
        incumbent: ArtifactEvaluation | None,
    ) -> ArtifactEvaluation:
        return choose_smallest_error(candidates, incumbent)

    def result_card(self, evaluation: ArtifactEvaluation) -> PanelCard:
        return PanelCard(
            title=evaluation.artifact.name,
            summary="Selected VPFP model with explicit reduced dynamics.",
            values={
                "hidden_vpfp_macro_match_v1_relative_l2": (
                    evaluation.metrics.get(
                        "vpfp_macro_match_v1_relative_l2",
                        evaluation.primary_error,
                    )
                ),
                "hidden_macro_relative_l2": evaluation.primary_error,
                "state_dimension": evaluation.artifact.metadata.get(
                    "state_dimension"
                ),
                "usage": evaluation.artifact.usage,
            },
        )
