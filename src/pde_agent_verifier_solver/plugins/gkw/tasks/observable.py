"""GKW Task for the stable four-field macroscopic output ABI."""

from __future__ import annotations

from random import Random
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
    TaskKind,
)

from ..._evaluation import PluginTaskEvaluationProgram
from ..._private_queries import default_query_fields, development_point_count
from ..contract import (
    GKW_MACRO_FIELDS,
    gkw_macro_query_panel,
    gkw_public_query_domain,
)
from ..isolation import snapshot_gkw_case, snapshot_gkw_query
from ..macro_matching import GKWMacroMatcher


def _private_development_query(
    case: Case,
    settings: Mapping[str, object],
    seed: int,
) -> AnswerQuery:
    """Sample a legal query from a released panel or generated public domain."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("GKW private Development query seed must be nonnegative")
    fields = default_query_fields(settings)
    if set(fields) != set(GKW_MACRO_FIELDS):
        raise ValueError("GKW private Development query requires all macro fields")
    point_count = development_point_count(settings)
    random = Random(seed)
    if "macro_query" not in case.public_data:
        (time_lower, time_upper), shape = gkw_public_query_domain(case)
        points = tuple(
            {
                "t": time_lower + (time_upper - time_lower) * random.random(),
                "s_index": random.randrange(shape[0]),
                "kx_index": random.randrange(shape[1]),
                "ky_index": random.randrange(shape[2]),
            }
            for _index in range(point_count)
        )
        return AnswerQuery.at_points(fields=fields, points=points)

    times, field_indices = gkw_macro_query_panel(case)
    panel_size = len(times) * len(field_indices)
    if point_count > panel_size:
        raise ValueError(
            "GKW private Development point count exceeds the released public panel"
        )
    points = []
    for flat_index in random.sample(range(panel_size), point_count):
        time_index, field_index = divmod(flat_index, len(field_indices))
        s_index, kx_index, ky_index = field_indices[field_index]
        points.append(
            {
                "t": float(times[time_index]),
                "s_index": int(s_index),
                "kx_index": int(kx_index),
                "ky_index": int(ky_index),
            }
        )
    return AnswerQuery.at_points(fields=fields, points=points)


class GKWObservableTask:
    info = ProgramInfo(
        "gkw-observable-model",
        "Find a GKW macroscopic observable model",
        "Predict particle density, parallel momentum, kinetic energy and phi; "
        "the internal method may be full-state, learned or reduced.",
    )
    kind = TaskKind.OBSERVABLE_MODEL

    def __init__(self, macro_matcher: MacroMatcher | None = None) -> None:
        self.macro_matcher = (
            GKWMacroMatcher() if macro_matcher is None else macro_matcher
        )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "outputs",
                "Observable fields",
                "text",
                "particle_density,parallel_momentum,kinetic_energy,phi",
            ),
            FormField(
                "primary_metric",
                "Primary metric",
                "text",
                "gkw_macro_match_v1_relative_l2",
            ),
        )

    def agent_brief(self, settings: Mapping[str, object]) -> str:
        del settings
        return (
            "Write predict_observables(case, query, budget=None) -> AnswerView "
            "for particle_density, parallel_momentum, kinetic_energy and phi. "
            "Values are complex and must be encoded with a final length-2 "
            "[real, imag] axis. For GRID, use the exact public times and "
            "(s,kx,ky) rows in case.public_data['macro_query']; return shape "
            "(t, field_row, complex_component), axes t/field_row/"
            "complex_component, and C-order flattened fields. For POINTS, "
            "released Cases require coordinates from their public macro panel; "
            "generated Cases may use the public oracle time interval and full "
            "(s,kx,ky) grid. Each point supplies t, s_index, kx_index and "
            "ky_index; echo query.points and return shape (point, complex_component)."
        )

    def create_evaluation(
        self,
        settings: Mapping[str, object],
    ) -> PluginTaskEvaluationProgram:
        return PluginTaskEvaluationProgram.from_settings(
            program_id="gkw-observable-model-evaluation-v1",
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
        if not hasattr(model, "predict_observables"):
            raise TypeError(
                "GKW Observable Task expects "
                "predict_observables(case, query, budget=None)"
            )
        controller_case = snapshot_gkw_case(case)
        controller_query = snapshot_gkw_query(query)
        target = self.macro_matcher.prepare_target(
            controller_case, truth, controller_query
        )
        if (
            target.case_id != controller_case.case_id
            or target.contract_id != self.macro_matcher.contract_id
            or target.query != controller_query
        ):
            raise ValueError("GKW macro matcher prepared an incoherent target")

        prediction = model.predict_observables(
            snapshot_gkw_case(case), snapshot_gkw_query(query), budget=None
        )
        result = self.macro_matcher.match(
            controller_case, prediction, target
        )
        if (
            result.case_id != controller_case.case_id
            or result.contract_id != self.macro_matcher.contract_id
        ):
            raise ValueError("GKW macro matcher returned an incoherent result")
        metrics = dict(result.metrics)
        metrics["macro_match_error"] = result.primary_error
        if result.passed is not None:
            metrics["macro_match_passed"] = 1.0 if result.passed else 0.0
        return CaseEvaluation(
            case_id=controller_case.case_id,
            primary_error=result.primary_error,
            metrics=metrics,
            prediction=prediction,
            macro_match=result,
        )

    def select(
        self,
        candidates: Sequence[ArtifactEvaluation],
        incumbent: ArtifactEvaluation | None,
    ) -> ArtifactEvaluation:
        choices = tuple(candidates) + (() if incumbent is None else (incumbent,))
        if not choices:
            raise ValueError("GKW selection needs at least one evaluated artifact")
        return min(choices, key=lambda value: value.primary_error)

    def result_card(self, evaluation: ArtifactEvaluation) -> PanelCard:
        return PanelCard(
            title=evaluation.artifact.name,
            summary="Selected GKW macroscopic observable model.",
            values={
                "hidden_gkw_macro_match_v1_relative_l2": (
                    evaluation.metrics.get(
                        "gkw_macro_match_v1_relative_l2",
                        evaluation.primary_error,
                    )
                ),
                "usage": evaluation.artifact.usage,
            },
        )
