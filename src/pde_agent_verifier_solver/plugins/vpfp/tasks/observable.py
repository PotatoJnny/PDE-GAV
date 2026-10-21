"""VPFP task that predicts macroscopic observable fields."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from inspect import getattr_static
from math import isfinite
from typing import ClassVar, Mapping, Sequence

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
    FormField,
    MacroMatcher,
    PanelCard,
    ProgramInfo,
    QueryKind,
    TaskKind,
    evaluate_macro_prediction,
    measure_point_query,
    measure_prepare_solution,
)
from pde_agent_verifier_solver.sdk.solve_budget import (
    bind_candidate_solve_budget, candidate_solve_remaining_seconds,
)

from ..._evaluation import PluginTaskEvaluationProgram
from ...plasma_transport import VPFP_TRANSPORT_FIELDS
from ...plasma_transport_settings import configured_matcher, transport_brief
from ..macro_matching import VPFPMacroMatcher
from ._shared import choose_smallest_error, private_vpfp_macro_query


def _private_development_query(
    case: Case,
    settings: Mapping[str, object],
    seed: int,
) -> AnswerQuery:
    return private_vpfp_macro_query(case, settings, seed)


class VPFPObservableEvaluationProgram(PluginTaskEvaluationProgram):
    """Optional private Official/Terminal points, without changing old queries."""

    # The VPFP parent/perturbation comparison uses identical hidden coordinates.
    # This opt-in leaves other Tasks' frozen evaluation plans unchanged.
    pair_hidden_plan_seeds: ClassVar[bool] = True

    def make_plan(self, case, phase, settings, seed, request=None):
        count_key = {
            EvaluationPhase.OFFICIAL: "official_point_count",
            EvaluationPhase.TERMINAL: "terminal_point_count",
        }.get(phase)
        if request is None and count_key is not None and count_key in settings:
            query = private_vpfp_macro_query(case, {
                **settings,
                "default_query": self.default_query,
                "development_point_count": settings[count_key],
            }, seed)
            return super().make_plan(case, phase, settings, seed, query)
        return super().make_plan(case, phase, settings, seed, request)


class VPFPObservableTask:
    """Predict moments without constraining the model's internal state."""

    info = ProgramInfo(
        "vpfp-observable-model",
        "Find a VPFP macroscopic observable model",
        "Predict density, momentum and kinetic energy from the full kinetic "
        "question. The implementation may be kinetic, learned or reduced.",
    )
    kind = TaskKind.OBSERVABLE_MODEL

    def __init__(self, macro_matcher: MacroMatcher | None = None) -> None:
        self.macro_matcher = (
            VPFPMacroMatcher() if macro_matcher is None else macro_matcher
        )
        if getattr(self.macro_matcher, "calibration", None) is not None:
            self.info = ProgramInfo(
                "vpfp-observable-transport-v2", "Find a VPFP macroscopic transport model",
                "Predict density, self-consistent electric field, particle flux, "
                "central pressure and central heat flux. The implementation may "
                "be kinetic, learned or reduced.",
            )

    def setup_fields(self) -> tuple[FormField, ...]:
        transport = getattr(self.macro_matcher, "calibration", None) is not None
        return (
            FormField(
                "outputs",
                "Observable fields",
                "text",
                (",".join(VPFP_TRANSPORT_FIELDS) if transport else "rho,momentum_x,momentum_y,kinetic_energy"),
            ),
            FormField(
                "primary_metric",
                "Primary metric",
                "text",
                "plasma_macro_transport_v2" if transport else "vpfp_macro_match_v1_relative_l2",
            ),
        )

    def agent_brief(self, settings: Mapping[str, object]) -> str:
        matcher = configured_matcher(settings, self.macro_matcher, "vpfp")
        brief = (
            "Treat VPFP as a continuous public problem. Use "
            "open_public_case_session(case) to request source at (t,x,y,vx,vy), "
            "rho0 at (x,y), and g0 at (x,y,vx,vy), only where your method needs "
            "values. Choose your own mesh, basis, velocity truncation and "
            "adaptivity; do not require precomputed source or g0 arrays. "
            "Implement prepare_solution(case, solve_budget=None) -> handle, "
            "then handle.evaluate(query) -> AnswerView for POINTS queries. "
            "Preparation receives no scoring query; the point query is passed "
            "only after preparation returns. Return only rho, momentum_x, "
            "momentum_y and kinetic_energy; neither the full kinetic solution "
            "nor a reduced internal state is required. Copy query.points "
            "exactly, preserving order and duplicates, with shape (N,). "
            "The optional candidate_compute_budget is passed as solve_budget; "
            "when absent this Task passes solve_budget=None. Oracle waiting "
            "in Controller queues and shared locks is excluded from the solve "
            "budget, but Oracle numerical execution, transfer and solver work "
            "remain charged. For cooperative deadlines use differences of "
            "pde_agent_verifier_solver.sdk.solve_budget.candidate_solve_clock, "
            "not time.perf_counter or time.monotonic. A formal subprocess "
            "execution service independently enforces its per-Case charged "
            "wall limit. Agent total wall time and Development batch deadlines "
            "still include all waiting. POINTS candidates declaring prepare_solution use this "
            "path preferentially; preparation errors do not fall back. "
            "For legacy compatibility, predict_observables(case, query, "
            "budget=None) remains supported when no prepare_solution is "
            "declared, and is the supported GRID entry point. For a "
            "GRID query, return shape in (t, x, y) order and the exact axes "
            "obtained from case.public_data['arrays'] after applying "
            "query.resolution (native coordinates when the count is omitted "
            "or unchanged, the midpoint for count 1, otherwise inclusive "
            "linear spacing between the public endpoints); flatten fields in "
            "that order. For a POINTS query, copy query.points exactly."
        )
        if settings.get("candidate_budget_mode") == "ns_remaining_case_v1":
            brief = brief.replace(
                "The optional candidate_compute_budget is passed as solve_budget; "
                "when absent this Task passes solve_budget=None. ",
                "Preparation receives solve_budget equal to the remaining shared "
                "Case time from the execution-owned budget meter. ",
            )
            brief += (
                " This run uses ns_remaining_case_v1: there is no separate "
                "preparation limit. solve_budget is the remaining shared Case "
                "budget at preparation entry, after startup/import work. The "
                "900-second charged Case limit also covers evaluate and result "
                "serialization; preparation does not reset or extend it."
            )
        calibration = getattr(matcher, "calibration", None)
        if calibration is not None:
            brief = brief.replace(
                "Return only rho, momentum_x, momentum_y and kinetic_energy;",
                "Return all ten scalar fields " + ", ".join(VPFP_TRANSPORT_FIELDS) + ";",
            )
            brief += transport_brief(calibration)
        return brief

    def create_evaluation(
        self,
        settings: Mapping[str, object],
    ) -> PluginTaskEvaluationProgram:
        matcher = configured_matcher(settings, self.macro_matcher, "vpfp")
        task = self if matcher is self.macro_matcher else type(self)(matcher)
        if getattr(matcher, "calibration", None) is not None:
            query = settings.get("default_query")
            if not isinstance(query, AnswerQuery) or set(query.fields) != set(VPFP_TRANSPORT_FIELDS):
                raise ValueError("transport VPFP default_query must request all ten macro fields")
        return VPFPObservableEvaluationProgram.from_settings(
            program_id=("vpfp-observable-model-evaluation-v2" if getattr(matcher, "calibration", None)
                        is not None else "vpfp-observable-model-evaluation-v1"),
            settings=settings,
            execute_candidate=task.execute_candidate,
            score_candidate=task.score_candidate,
            private_development_query=_private_development_query,
        )

    def execute_candidate(
        self,
        artifact: Artifact,
        case: Case,
        plan: EvaluationPlan,
    ) -> CandidateEvaluationOutput:
        """Produce finite macros without receiving Controller-private truth."""
        query = plan.query
        model = artifact.program
        timing_metrics = {}
        require_prepared = plan.settings.get("require_prepared_solution", False)
        if type(require_prepared) is not bool:
            raise TypeError("VPFP require_prepared_solution must be boolean")
        solve_budget = plan.settings.get("candidate_compute_budget")
        budget_mode = plan.settings.get("candidate_budget_mode")
        if budget_mode not in (None, "ns_remaining_case_v1"):
            raise ValueError("unknown VPFP candidate_budget_mode")
        shared_case_budget = budget_mode == "ns_remaining_case_v1"
        if shared_case_budget and solve_budget is not None:
            raise ValueError("ns_remaining_case_v1 forbids a separate candidate_compute_budget")
        if solve_budget is not None and (
            isinstance(solve_budget, bool) or not isinstance(solve_budget, (int, float))
            or not isfinite(solve_budget) or solve_budget <= 0
        ):
            raise ValueError("VPFP candidate_compute_budget must be finite and positive")
        use_prepared_solution = False
        if query.kind is QueryKind.POINTS:
            missing = object()
            try:
                getattr(model, "prepare_solution")
            except AttributeError:
                # A declared but broken descriptor is not an absent ABI. Do
                # not silently switch solvers when resolving preparation fails.
                if getattr_static(model, "prepare_solution", missing) is not missing:
                    raise
            else:
                use_prepared_solution = True
        if use_prepared_solution:
            if not callable(getattr(model, "prepare_solution", None)):
                raise TypeError("program must define callable prepare_solution")
            # Existing SDK helpers validate the handle and finite point answer.
            # No query is supplied to preparation. The SDK deducts trusted
            # Controller queue/lock waits exactly once, not numerical service.
            # The SDK's timing wrapper retains evaluate but not close. Capture
            # the original handle only inside preparation so it is released
            # after the finite answer has been materialized, including errors.
            owned_handles = []

            class Preparation:
                def prepare_solution(self, public_case, solve_budget=None):
                    handle = model.prepare_solution(public_case, solve_budget)
                    owned_handles.append(handle)
                    return handle

            try:
                # Reuse the worker-owned meter; local calls get the same SDK
                # accounting without resetting any existing Case deadline.
                with bind_candidate_solve_budget():
                    if shared_case_budget:
                        solve_budget = candidate_solve_remaining_seconds()
                        if solve_budget is not None and solve_budget <= 0:
                            raise TimeoutError("VPFP shared Case budget exhausted before preparation")
                    # Match the upstream Artemis compatibility path for frozen
                    # Python bundles. Only entry-module stopwatch aliases are
                    # rebound during preparation; never change bundle bytes or
                    # the process-wide time module. The SDK timer stays raw.
                    clock_scope = getattr(model, "_solve_budget_clock_scope", None)
                    with clock_scope() if callable(clock_scope) else nullcontext(()) as aliases:
                        prepared = measure_prepare_solution(
                            Preparation(), case, solve_budget=solve_budget)
                    if (not shared_case_budget and solve_budget is not None
                            and prepared.candidate_solve_wall_seconds > solve_budget):
                        raise TimeoutError("VPFP Candidate exceeded preparation budget")
                    queried = measure_point_query(prepared.handle, query)
                    prediction = queried.answer
                    timing_metrics = {
                        "candidate_solve_wall_seconds": prepared.candidate_solve_wall_seconds,
                        "candidate_solve_queue_wait_seconds": prepared.candidate_solve_queue_wait_seconds,
                        "candidate_query_wall_seconds": queried.candidate_query_wall_seconds,
                        "candidate_query_queue_wait_seconds": queried.candidate_query_queue_wait_seconds,
                        "candidate_query_point_count": float(queried.point_count),
                        "candidate_query_wall_seconds_per_point": queried.candidate_query_wall_seconds_per_point,
                        "candidate_budget_clock_compat_aliases": float(len(aliases)),
                        "candidate_formal_two_phase": 1.0,
                    }
            finally:
                for handle in owned_handles:
                    close = getattr(handle, "close", None)
                    if callable(close):
                        close()
        else:
            if require_prepared:
                raise TypeError("formal VPFP Candidate requires prepare_solution and POINTS")
            predict = getattr(model, "predict_observables", None)
            if not callable(predict):
                raise TypeError(
                    "VPFP Observable Task expects prepare_solution for POINTS "
                    "or callable predict_observables(case, query, budget=None) "
                    "for legacy POINTS/GRID compatibility"
                )
            with bind_candidate_solve_budget() if shared_case_budget else nullcontext():
                legacy_budget = candidate_solve_remaining_seconds() if shared_case_budget else None
                if legacy_budget is not None and legacy_budget <= 0:
                    raise TimeoutError("VPFP shared Case budget exhausted before preparation")
                prediction = predict(case, query, budget=legacy_budget)
        return CandidateEvaluationOutput(prediction, timing_metrics)

    def score_candidate(self, artifact, case, truth, plan, output):
        """Apply the unchanged macro matcher only in the Controller."""
        del artifact
        if not isinstance(output, CandidateEvaluationOutput):
            raise TypeError("VPFP scoring requires a CandidateEvaluationOutput")
        result = evaluate_macro_prediction(
            matcher=self.macro_matcher,
            case=case,
            prediction=output.prediction,
            truth=truth,
            query=plan.query,
        )
        return replace(result, metrics={**result.metrics, **dict(output.metrics)})

    def evaluate(self, artifact, case, truth, query):
        """Legacy in-process entrypoint with the same native split scorer."""
        plan = EvaluationPlan(phase=EvaluationPhase.DEVELOPMENT, query=query)
        output = self.execute_candidate(artifact, case, plan)
        return self.score_candidate(artifact, case, truth, plan, output)

    def select(
        self,
        candidates: Sequence[ArtifactEvaluation],
        incumbent: ArtifactEvaluation | None,
    ) -> ArtifactEvaluation:
        return choose_smallest_error(candidates, incumbent)

    def result_card(self, evaluation: ArtifactEvaluation) -> PanelCard:
        if "plasma_macro_transport_v2" in evaluation.metrics:
            return PanelCard(
                title=evaluation.artifact.name,
                summary="Selected VPFP macroscopic transport model.",
                values={"metric_profile": "plasma_macro_transport_v2",
                        "transport_error": evaluation.primary_error,
                        "metrics": dict(evaluation.metrics),
                        "usage": evaluation.artifact.usage},
            )
        return PanelCard(
            title=evaluation.artifact.name,
            summary="Selected VPFP macroscopic observable model.",
            values={
                "hidden_vpfp_macro_match_v1_relative_l2": (
                    evaluation.metrics.get(
                        "vpfp_macro_match_v1_relative_l2",
                        evaluation.primary_error,
                    )
                ),
                "hidden_macro_relative_l2": evaluation.primary_error,
                "usage": evaluation.artifact.usage,
            },
        )
