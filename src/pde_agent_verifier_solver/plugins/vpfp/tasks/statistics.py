"""VPFP task for reduced-order scalar statistics."""

from __future__ import annotations

from statistics import fmean
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    ArtifactEvaluation,
    Case,
    CaseEvaluation,
    FormField,
    PanelCard,
    ProgramInfo,
    TaskKind,
)

from ..._evaluation import PluginTaskEvaluationProgram
from ._shared import (
    choose_smallest_error,
    relative_l2,
    scalar_truth_statistics,
)


class VPFPStatisticsTask:
    """Predict named statistics instead of a complete trajectory."""

    info = ProgramInfo(
        "vpfp-statistics",
        "Find VPFP reduced-order statistics",
        "Predict task-selected long-time or ensemble statistics.",
    )
    kind = TaskKind.STATISTICS

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "requested_statistics",
                "Statistics",
                "text",
                "rho.mean,kinetic_energy.rms",
            ),
        )

    def agent_brief(self, settings: Mapping[str, object]) -> str:
        requested = str(
            settings.get(
                "requested_statistics",
                "rho.mean,kinetic_energy.rms",
            )
        )
        return (
            "Write statistics(case, requested, settings) -> mapping. "
            f"The requested scalar outputs are: {requested}."
        )

    def create_evaluation(
        self,
        settings: Mapping[str, object],
    ) -> PluginTaskEvaluationProgram:
        return PluginTaskEvaluationProgram.from_settings(
            program_id="vpfp-statistics-evaluation-v1",
            settings=settings,
            evaluate_case=self.evaluate,
        )

    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> CaseEvaluation:
        model = artifact.program
        if not hasattr(model, "statistics"):
            raise TypeError(
                "VPFP Statistics Task expects "
                "statistics(case, requested, settings)"
            )
        requested = tuple(
            str(value)
            for value in artifact.metadata.get(
                "requested_statistics",
                ("rho.mean", "kinetic_energy.rms"),
            )
        )
        prediction = model.statistics(case, requested, {})
        expected = scalar_truth_statistics(truth, requested)
        errors = {
            name: relative_l2(
                (float(prediction[name]),),
                (float(expected[name]),),
            )
            for name in requested
        }
        primary = fmean(errors.values())
        return CaseEvaluation(
            case_id=case.case_id,
            primary_error=primary,
            metrics={
                "statistics_relative_error": primary,
                **{
                    f"{name}_relative_error": value
                    for name, value in errors.items()
                },
            },
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
            summary="Selected VPFP statistics program.",
            values={
                "hidden_statistics_relative_error": (
                    evaluation.primary_error
                ),
                "usage": evaluation.artifact.usage,
            },
        )
