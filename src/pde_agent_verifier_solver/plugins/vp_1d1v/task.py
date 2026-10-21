"""Native macro-only Task with the admitted Artemis causal information view."""

from __future__ import annotations

from contextlib import ExitStack
from contextvars import copy_context
from dataclasses import dataclass, replace
from math import isfinite
import random
from time import perf_counter

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, CandidateEvaluationOutput, EvaluationPhase, EvaluationPlan, FormField, PanelCard, ProgramInfo,
    TaskKind, aggregate_case_evaluations, evaluate_macro_prediction,
    open_public_case_session, open_development_solution_session,
)
from pde_agent_verifier_solver.sdk.public_sessions import (
    bind_public_session_factory, bind_development_solution_session_factory,
)
from pde_agent_verifier_solver.sdk.solve_budget import (
    bind_candidate_solve_budget, candidate_solve_clock, candidate_solve_timing,
    candidate_solve_remaining_seconds,
)
from .macro_matching import VP1DMacroMatcher
from .problem import VP1D_MACRO_FIELDS, causal_horizon_case
from .public_problem import VP1D_PUBLIC_SESSION_METHODS, VP1D_DEVELOPMENT_SOLUTION_SESSION_METHODS
from ..plasma_transport_settings import configured_matcher, transport_brief


class LegacyMacroCandidateAdapter:
    """Reuse an existing macro Candidate without requiring a kinetic output."""

    def __init__(self, program):
        if not callable(getattr(program, "predict_macros", None)):
            raise TypeError("legacy Artemis Candidate requires predict_macros")
        self.program = program

    def predict_observables(self, case, query, budget=None):
        return self.program.predict_macros(case, query, budget=budget)


class _CausalSession:
    def __init__(self, delegate, start, horizon):
        self.delegate, self.start, self.horizon = delegate, start, horizon

    def query_initial_distribution(self, points):
        return self.delegate.query_initial_distribution(points)

    def query_source(self, points):
        self._check(points)
        return self.delegate.query_source(points)

    def query_completed_macros(self, points):
        self._check(points)
        return self.delegate.query_completed_macros(points)

    def _check(self, points):
        import numpy as np
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
            raise ValueError("Artemis oracle points must be finite rows")
        if np.any(values[:, 0] < self.start) or np.any(values[:, 0] > self.horizon):
            raise ValueError("Artemis oracle query leaves the causal interval")

    def close(self):
        # The outer Controller binding owns and revokes the resident session.
        pass


def _native_query(case, count, seed):
    arrays = case.public_data["arrays"]
    time, space = tuple(arrays["t"]), tuple(arrays["x"])
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(space):
        raise ValueError("Artemis causal point count must lie between 1 and native nx")
    if len(time) < 2:
        raise ValueError("Artemis scoring requires a non-initial native time")
    rng = random.Random(seed)
    horizon = float(rng.choice(time[1:]))
    return AnswerQuery.at_points(
        fields=VP1D_MACRO_FIELDS,
        points=tuple({"t": horizon, "x": float(x)} for x in rng.sample(space, count)),
    )


def _private_query(case, count, seed, mode="native_x_v1"):
    """New-run continuous scoring; legacy native plans remain byte-for-byte unchanged.

    The number of scoring points must not enlarge the public convenience grid
    or the source Oracle. Parent/child plans retain the same seed and horizon.
    """
    if mode == "native_x_v1":
        return _native_query(case, count, seed)
    if mode != "continuous_x_v1":
        raise ValueError("unknown Artemis scoring_coordinate_mode")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Artemis continuous point count must be a positive integer")
    time = tuple(case.public_data["arrays"]["t"])
    if len(time) < 2:
        raise ValueError("Artemis scoring requires a non-initial native time")
    left, right = map(float, case.public_data["equation"]["domain"]["x_interval"])
    if not isfinite(left) or not isfinite(right) or right <= left:
        raise ValueError("Artemis periodic spatial interval must be finite and ordered")
    rng = random.Random(seed)
    horizon = float(rng.choice(time[1:]))
    return AnswerQuery.at_points(
        fields=VP1D_MACRO_FIELDS,
        points=tuple({"t": horizon, "x": left + (right - left) * rng.random()}
                     for _ in range(count)),
    )


@dataclass(frozen=True)
class ArtemisEvaluation:
    # Preserve Artemis parent/perturbation query pairing through the shared
    # Controller's optional capability, without changing other Backend banks.
    pair_hidden_plan_seeds = True

    default_query: AnswerQuery
    matcher: VP1DMacroMatcher
    causal_horizon: bool = True
    require_prepared_solution: bool = False
    solve_budget: float | None = None
    program_id: str = "vp-1d1v-observable-evaluation-v1"
    scoring_coordinate_mode: str = "native_x_v1"
    candidate_budget_mode: str | None = None

    def __post_init__(self):
        if self.scoring_coordinate_mode not in {"native_x_v1", "continuous_x_v1"}:
            raise ValueError("unknown Artemis scoring_coordinate_mode")
        if not isinstance(self.default_query, AnswerQuery) or tuple(self.default_query.fields) != VP1D_MACRO_FIELDS:
            raise ValueError("Artemis evaluation default_query must request M0,M1,M2,M3,E")
        for name in ("causal_horizon", "require_prepared_solution"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Artemis {name} must be boolean")
        if self.candidate_budget_mode not in (None, "ns_remaining_case_v1"):
            raise ValueError("unknown Artemis candidate_budget_mode")
        if self.candidate_budget_mode == "ns_remaining_case_v1" and self.solve_budget is not None:
            raise ValueError("ns_remaining_case_v1 forbids a separate candidate_compute_budget")
        if self.solve_budget is not None and (
            isinstance(self.solve_budget, bool) or not isinstance(self.solve_budget, (int, float))
            or not isfinite(self.solve_budget) or self.solve_budget <= 0
        ):
            raise ValueError("Artemis candidate_compute_budget must be finite and positive")
        immutable = EvaluationPlan(EvaluationPhase.DEVELOPMENT, self.default_query).query
        object.__setattr__(self, "default_query", immutable)

    def make_private_development_plan(self, case, settings, seed):
        query = _private_query(case, settings.get("development_point_count", 16), seed,
                               self.scoring_coordinate_mode)
        return EvaluationPlan(EvaluationPhase.DEVELOPMENT, query, seed, settings)

    def make_plan(self, case, phase, settings, seed, request=None):
        if phase is EvaluationPhase.DEVELOPMENT:
            query = request or self.default_query
        else:
            # Official coordinates belong to the Task and are frozen once for
            # every Candidate on the common cohort, independent of nominations.
            count = settings.get("official_point_count", len(self.default_query.points) or 16)
            if phase is EvaluationPhase.TERMINAL:
                count = settings.get("terminal_point_count", count)
            query = _private_query(case, count, seed, self.scoring_coordinate_mode)
        if self.causal_horizon:
            causal_horizon_case(case, query, scoring_coordinate_mode=self.scoring_coordinate_mode)
        return EvaluationPlan(phase, query, seed, settings)

    @property
    def supports_split_candidate_execution(self):
        return True

    def execute_candidate(self, artifact, case, plan):
        """Prepare and query macros using only the causal public information."""
        projected = causal_horizon_case(case, plan.query, scoring_coordinate_mode=self.scoring_coordinate_mode) if self.causal_horizon else case
        with ExitStack() as stack:
            stack.enter_context(bind_candidate_solve_budget())
            if projected is not case:
                start, horizon = projected.public_data["equation"]["domain"]["time_interval"]
                # Capture each outer binding before shadowing it. The inner
                # revocable proxy also rejects future oracle calls, including
                # retained bound methods after the Candidate returns.
                outer_context = copy_context()
                for opener, binder, methods in (
                    (open_public_case_session, bind_public_session_factory, VP1D_PUBLIC_SESSION_METHODS),
                    (open_development_solution_session, bind_development_solution_session_factory, VP1D_DEVELOPMENT_SOLUTION_SESSION_METHODS),
                ):
                    def factory(_case, open_outer=opener):
                        # Evaluate in the captured outer scope lazily. An
                        # unavailable or failed physical session propagates its
                        # exact error; it is never converted to an array fallback.
                        delegate = outer_context.run(open_outer, case)
                        return _CausalSession(delegate, start, horizon)
                    stack.enter_context(binder(factory, projected, allowed_methods=methods))
            model = artifact.program
            began = perf_counter()
            active_began = candidate_solve_clock()
            timing_before = candidate_solve_timing()
            shared_case_budget = self.candidate_budget_mode == "ns_remaining_case_v1"
            solve_budget = self.solve_budget
            # Narrow compatibility for immutable, previously nominated Python
            # bundles. Never patch time.perf_counter process-wide or rewrite
            # the submitted scientific program to repair its budget clock.
            aliases = ()
            prepare_scope = stack.enter_context(ExitStack())
            clock_scope = getattr(model, "_solve_budget_clock_scope", None)
            if callable(clock_scope):
                aliases = prepare_scope.enter_context(clock_scope())
            prepare = getattr(model, "prepare_solution", None)
            if callable(prepare):
                if shared_case_budget:
                    # Read after lazy Candidate import/ABI resolution. Startup,
                    # preparation, query and serialization share one deadline;
                    # this hint never starts a new preparation timer.
                    solve_budget = candidate_solve_remaining_seconds()
                    if solve_budget is not None and solve_budget <= 0:
                        raise TimeoutError("Artemis shared Case budget exhausted before preparation")
                handle = prepare(projected, solve_budget=solve_budget)
                solve_wall = perf_counter() - began
                solve_active = candidate_solve_clock() - active_began
                timing_after = candidate_solve_timing()
                prepare_scope.close()
                try:
                    if not callable(getattr(handle, "evaluate", None)):
                        raise TypeError("Artemis prepare_solution must return a handle with evaluate(query)")
                    query_start = perf_counter()
                    prediction = handle.evaluate(plan.query)
                    query_wall = perf_counter() - query_start
                finally:
                    close = getattr(handle, "close", None)
                    if callable(close):
                        close()
                prepared = True
            else:
                if self.require_prepared_solution:
                    raise TypeError("this Artemis Task requires prepare_solution before scoring points are revealed")
                predict = getattr(model, "predict_observables", None)
                if not callable(predict):
                    raise TypeError("Artemis Task needs predict_observables or prepare_solution; only M0--M3,E are required")
                if shared_case_budget:
                    solve_budget = candidate_solve_remaining_seconds()
                    if solve_budget is not None and solve_budget <= 0:
                        raise TimeoutError("Artemis shared Case budget exhausted before preparation")
                prediction = predict(projected, plan.query, budget=solve_budget)
                solve_wall = perf_counter() - began
                solve_active = candidate_solve_clock() - active_began
                timing_after = candidate_solve_timing()
                prepare_scope.close()
                query_wall, prepared = 0.0, False
            if not shared_case_budget and solve_budget is not None and solve_active > solve_budget:
                raise TimeoutError("Artemis Candidate exceeded its solve budget")
            return CandidateEvaluationOutput(prediction, {
                "candidate_solve_wall_seconds": solve_wall,
                "candidate_solve_active_seconds": solve_active,
                "candidate_budget_clock_compat_aliases": float(len(aliases)),
                **{key: timing_after[key] - value for key, value in timing_before.items()},
                "candidate_query_wall_seconds": query_wall,
                "query_blind_preparation": float(prepared),
            })

    def score_candidate(self, artifact, case, truth, plan, output):
        """Apply the unchanged macro contract to Controller-private truth."""
        del artifact
        if not isinstance(output, CandidateEvaluationOutput):
            raise TypeError("Artemis scoring requires a CandidateEvaluationOutput")
        projected = causal_horizon_case(case, plan.query, scoring_coordinate_mode=self.scoring_coordinate_mode) if self.causal_horizon else case
        result = evaluate_macro_prediction(
            matcher=self.matcher, case=projected, prediction=output.prediction,
            truth=truth, query=plan.query,
        )
        return replace(result, metrics={**result.metrics, **dict(output.metrics)})

    def evaluate(self, artifact, case, truth, plan):
        """Legacy in-process entrypoint sharing the split scientific scorer."""
        output = self.execute_candidate(artifact, case, plan)
        return self.score_candidate(artifact, case, truth, plan, output)

    def aggregate(self, artifact, case_results, plans):
        return aggregate_case_evaluations(artifact, case_results, plans)


class ArtemisObservableTask:
    info = ProgramInfo("vp-1d1v-observable-model", "Find an Artemis macroscopic model",
                       "Predict M0, M1, M2, M3 and E; no phase-space distribution output is required.")
    kind = TaskKind.OBSERVABLE_MODEL

    def __init__(self, macro_matcher=None):
        self.macro_matcher = macro_matcher or VP1DMacroMatcher()
        if getattr(self.macro_matcher, "calibration", None) is not None:
            self.info = ProgramInfo("vp-1d1v-observable-transport-v2", "Find an Artemis macroscopic transport model",
                "Predict density, electric field, particle flux, central pressure and central heat flux "
                "through the raw M0,M1,M2,M3,E output interface.")

    def setup_fields(self):
        return (
            FormField("outputs", "Observable fields", "text", "M0,M1,M2,M3,E"),
            FormField("causal_horizon", "Causal source prefix", "boolean", True),
            FormField("require_prepared_solution", "Prepare before seeing scoring points", "boolean", False),
            FormField("scoring_coordinate_mode", "Spatial scoring coordinates", "text", "native_x_v1"),
        )

    def agent_brief(self, settings):
        matcher = configured_matcher(settings, self.macro_matcher, "vp_1d1v")
        brief = (
            "For canonical file-submitted Candidates, export prepare_solution(case, solve_budget=None) "
            "returning a handle with evaluate(query) -> AnswerView for M0,M1,M2,M3,E. "
            "The Python Candidate publisher requires this two-phase ABI: prepare without scoring "
            "coordinates, then evaluate the supplied POINTS. No f(t,x,v) output or kinetic "
            "representation is required. predict_observables(case, query, budget=None) remains a "
            "legacy in-process compatibility ABI, not a file-submission entrypoint. Only the causal "
            "f0/source prefix is available during Official/Terminal Candidate evaluation. Completed-"
            "solution access requires explicit Run Development authorization and is never available "
            "in Official/Terminal; the submitted Candidate must work without it. Copy POINTS "
            "coordinates exactly, with shape (N,). Use the declared public "
            "oracle; exact-coordinate cases have non-authoritative convenience arrays."
            " For cooperative preparation deadlines, import candidate_solve_clock from "
            "pde_agent_verifier_solver.sdk.solve_budget instead of using a wall stopwatch. "
            "The solve_budget is charged preparation time: trusted shared-service queue/lock "
            "waiting is excluded, but your requested oracle computation is charged. An "
            "independent elapsed-time safety limit still applies."
        )
        if settings.get("candidate_budget_mode") == "ns_remaining_case_v1":
            brief = brief.replace(
                "The solve_budget is charged preparation time: trusted shared-service queue/lock "
                "waiting is excluded, but your requested oracle computation is charged. An "
                "independent elapsed-time safety limit still applies.",
                "This run uses ns_remaining_case_v1: there is no separate preparation limit. "
                "Preparation receives solve_budget equal to the remaining shared Case time "
                "from the execution-owned budget meter, after startup/import work. The "
                "900-second charged Case limit also covers evaluate and result serialization; "
                "preparation does not reset or extend it. Trusted shared-service queue/lock "
                "waiting is excluded, but oracle numerical execution, transfer and solver work "
                "remain charged. The subprocess execution service enforces that shared limit. "
                "Agent total wall time and Development batch deadlines still include all waiting.",
            )
        calibration = getattr(matcher, "calibration", None)
        if calibration is not None:
            brief += (" Return the existing raw M0,M1,M2,M3,E interface. The verifier "
                      "converts n=M0, j=M1, u=M1/M0, P=M2-M1^2/M0 and "
                      "q_h=0.5*(M3-3*u*M2+2*M0*u^3), separately for your "
                      "prediction and the reference.") + transport_brief(calibration)
        return brief

    def create_evaluation(self, settings):
        matcher = configured_matcher(settings, self.macro_matcher, "vp_1d1v")
        query = settings.get("default_query")
        if not isinstance(query, AnswerQuery) or tuple(query.fields) != VP1D_MACRO_FIELDS:
            raise ValueError("Artemis evaluation default_query must request M0,M1,M2,M3,E")
        return ArtemisEvaluation(query, matcher,
            program_id=("vp-1d1v-observable-evaluation-v2" if getattr(matcher, "calibration", None)
                        is not None else "vp-1d1v-observable-evaluation-v1"),
            causal_horizon=settings.get("causal_horizon", True),
            require_prepared_solution=settings.get("require_prepared_solution", False),
            solve_budget=settings.get("candidate_compute_budget"),
            scoring_coordinate_mode=settings.get("scoring_coordinate_mode", "native_x_v1"),
            candidate_budget_mode=settings.get("candidate_budget_mode"))

    def select(self, candidates, incumbent):
        return min(tuple(candidates) + ((incumbent,) if incumbent else ()), key=lambda value: value.primary_error)

    def evaluate(self, artifact, case, truth, query):
        """Compatibility entrypoint using the same native macro evaluator."""
        evaluation = self.create_evaluation({"default_query": query})
        return evaluation.evaluate(artifact, case, truth,
            EvaluationPlan(EvaluationPhase.DEVELOPMENT, query))

    def result_card(self, evaluation):
        if "plasma_macro_transport_v2" in evaluation.metrics:
            return PanelCard(evaluation.artifact.name, "Selected Artemis macroscopic transport model.",
                {"metric_profile": "plasma_macro_transport_v2", "transport_error": evaluation.primary_error,
                 "metrics": dict(evaluation.metrics), "usage": evaluation.artifact.usage})
        return PanelCard(evaluation.artifact.name, "Selected Artemis macroscopic model.",
                         {"macro_relative_l2": evaluation.primary_error, "usage": evaluation.artifact.usage})
