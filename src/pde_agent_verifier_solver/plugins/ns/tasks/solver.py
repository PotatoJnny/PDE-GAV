"""Full velocity-pressure solver task for Navier--Stokes."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from inspect import getattr_static
from math import isfinite, sqrt
from random import Random
from statistics import fmean
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    ArtifactEvaluation,
    CandidateEvaluationOutput,
    Case,
    CaseEvaluation,
    EvaluationPhase,
    EvaluationPlan,
    FormField,
    PanelCard,
    ProgramInfo,
    QueryKind,
    TaskKind,
    measure_point_query,
    measure_prepare_solution,
)

from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_remaining_seconds

from ..._evaluation import PluginTaskEvaluationProgram
from ..._private_queries import (
    default_query_fields,
    development_point_count,
    public_axis_bounds,
)


_MISSING_PROGRAM_ATTRIBUTE = object()
_NS_DEFAULT_POINT_COUNTS = {
    EvaluationPhase.DEVELOPMENT: 32,
    EvaluationPhase.OFFICIAL: 96,
    EvaluationPhase.TERMINAL: 128,
}


def _private_development_query(
    case: Case,
    settings: Mapping[str, object],
    seed: int,
) -> AnswerQuery:
    """Sample a fresh Task-owned Development plan for either NS Case schema."""

    return _private_ns_point_query(
        case,
        settings,
        seed,
        point_count=development_point_count(settings),
    )


def _private_ns_point_query(
    case: Case,
    settings: Mapping[str, object],
    seed: int,
    *,
    point_count: int,
) -> AnswerQuery:
    """Build finite points from public D0 axes or a continuous NS domain."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("NS private query seed must be nonnegative")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 1
    ):
        raise ValueError("NS private query point count must be positive")
    coordinates = case.public_data.get("coordinates")
    geometry = case.public_data.get("geometry")
    if (
        isinstance(coordinates, Mapping)
        and isinstance(geometry, Mapping)
        and geometry.get("topology") == "point_cloud"
    ):
        points = _point_cloud_points(coordinates, seed, point_count)
    else:
        arrays = case.public_data.get("arrays")
        if isinstance(arrays, Mapping):
            points = _regular_grid_points(case, arrays, seed, point_count)
        else:
            points = _continuous_domain_points(case, seed, point_count)
    return AnswerQuery.at_points(
        fields=default_query_fields(settings),
        points=points,
    )


def _point_cloud_points(
    coordinates: Mapping[str, object],
    seed: int,
    point_count: int,
) -> tuple[Mapping[str, float], ...]:
    """Privately draw native PIV space-time samples from public coordinates."""

    import numpy as np

    x = np.asarray(coordinates.get("x"), dtype=np.float64)
    y = np.asarray(coordinates.get("y"), dtype=np.float64)
    time = np.asarray(coordinates.get("time"), dtype=np.float64)
    if (
        x.ndim != 1
        or y.ndim != 1
        or len(x) != len(y)
        or len(x) < 1
        or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
    ):
        raise ValueError(
            "NS point-cloud query needs matching finite public x/y vectors"
        )
    if (
        time.ndim != 1
        or len(time) < 1
        or not np.all(np.isfinite(time))
        or (len(time) > 1 and not np.all(np.diff(time) > 0.0))
    ):
        raise ValueError(
            "NS point-cloud query needs a finite increasing public time vector"
        )
    random = Random(seed)
    points = []
    for _index in range(point_count):
        spatial_index = random.randrange(len(x))
        time_index = random.randrange(len(time))
        points.append(
            {
                "t": float(time[time_index]),
                "x": float(x[spatial_index]),
                "y": float(y[spatial_index]),
            }
        )
    return tuple(points)


def _regular_grid_points(
    case: Case,
    arrays: Mapping[str, object],
    seed: int,
    point_count: int,
) -> tuple[Mapping[str, float], ...]:
    """Use only public D0 coordinates and its fluid mask."""

    question = case.public_data.get("question")
    if not isinstance(question, Mapping):
        raise ValueError("NS regular-grid query needs a public question")
    time = question.get("time")
    if not isinstance(time, Mapping) or "t0" not in time or "t1" not in time:
        raise ValueError("NS regular-grid query needs public question.time t0/t1")
    try:
        t0, t1 = float(time["t0"]), float(time["t1"])
    except (TypeError, ValueError) as exc:
        raise ValueError("NS public question.time t0/t1 are invalid") from exc
    if (
        isinstance(time["t0"], bool)
        or isinstance(time["t1"], bool)
        or not (isfinite(t0) and isfinite(t1))
        or t1 <= t0
    ):
        raise ValueError("NS public question.time t0/t1 must be finite and ordered")

    public_axis_bounds(case, ("x", "y"))

    import numpy as np

    x = np.asarray(arrays["x"], dtype=np.float64)
    y = np.asarray(arrays["y"], dtype=np.float64)
    fluid_mask = np.asarray(arrays.get("fluid_mask"))
    if fluid_mask.dtype.kind != "b" or fluid_mask.shape != (len(x), len(y)):
        raise ValueError(
            "NS private Development query needs a boolean public fluid_mask "
            "matching x/y"
        )
    fluid_locations = np.argwhere(fluid_mask)
    if len(fluid_locations) == 0:
        raise ValueError("NS regular-grid public fluid domain is empty")

    random = Random(seed)
    points = []
    for _index in range(point_count):
        x_index, y_index = fluid_locations[random.randrange(len(fluid_locations))]
        points.append(
            {
                "t": t0 + (t1 - t0) * random.random(),
                "x": float(x[x_index]),
                "y": float(y[y_index]),
            }
        )
    return tuple(points)


def _continuous_domain_points(
    case: Case,
    seed: int,
    point_count: int,
) -> tuple[Mapping[str, float], ...]:
    """Draw continuous physical coordinates while excluding public obstacles."""

    domain = case.public_data.get("domain")
    if not isinstance(domain, Mapping):
        raise ValueError(
            "NS private query needs regular-grid arrays or a public continuous domain"
        )
    bounds = domain.get("bounds")
    interval = domain.get("time_interval")
    if not isinstance(bounds, Mapping):
        raise ValueError("NS continuous query needs public spatial bounds")
    try:
        t0, t1 = (float(value) for value in interval)
        x0, x1 = (float(value) for value in bounds["x"])
        y0, y1 = (float(value) for value in bounds["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("NS continuous public domain is invalid") from exc
    values = (t0, t1, x0, x1, y0, y1)
    if not all(isfinite(value) for value in values):
        raise ValueError("NS continuous public domain must be finite")
    if t1 <= t0 or x1 <= x0 or y1 <= y0:
        raise ValueError("NS continuous public domain is empty")
    obstacles = _obstacle_polygons(domain.get("geometry"))
    random = Random(seed)
    points: list[Mapping[str, float]] = []
    maximum_attempts = max(1024, 64 * point_count)
    for _attempt in range(maximum_attempts):
        if len(points) == point_count:
            break
        point = {
            "t": t0 + (t1 - t0) * random.random(),
            "x": x0 + (x1 - x0) * random.random(),
            "y": y0 + (y1 - y0) * random.random(),
        }
        if any(
            _inside_polygon(point["x"], point["y"], polygon)
            for polygon in obstacles
        ):
            continue
        points.append(point)
    if len(points) != point_count:
        raise ValueError("NS fluid-domain query rejection sampler exhausted")
    return tuple(points)


def _obstacle_polygons(
    geometry: object,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not isinstance(geometry, Mapping):
        raise ValueError("NS continuous query needs public geometry")
    values = geometry.get("obstacle_polygons", geometry.get("obstacles", ()))
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, SequenceABC):
        raise ValueError("NS public obstacle polygons are invalid")
    polygons = []
    for raw_polygon in values:
        if isinstance(raw_polygon, Mapping):
            raw_polygon = raw_polygon.get("vertices")
        if isinstance(raw_polygon, (str, bytes)) or not isinstance(
            raw_polygon, SequenceABC
        ):
            raise ValueError("NS public obstacle polygon is invalid")
        polygon = []
        for raw_point in raw_polygon:
            try:
                x, y = (float(value) for value in raw_point)
            except (TypeError, ValueError) as exc:
                raise ValueError("NS public obstacle vertex is invalid") from exc
            if not (isfinite(x) and isfinite(y)):
                raise ValueError("NS public obstacle vertex is non-finite")
            polygon.append((x, y))
        if len(polygon) < 3:
            raise ValueError("NS public obstacle polygon needs three vertices")
        polygons.append(tuple(polygon))
    return tuple(polygons)


def _inside_polygon(
    x: float,
    y: float,
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """Return true for polygon interior or boundary."""

    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        cross = (x - previous_x) * (current_y - previous_y) - (
            y - previous_y
        ) * (current_x - previous_x)
        if abs(cross) <= 1.0e-12 and (
            min(previous_x, current_x) - 1.0e-12
            <= x
            <= max(previous_x, current_x) + 1.0e-12
            and min(previous_y, current_y) - 1.0e-12
            <= y
            <= max(previous_y, current_y) + 1.0e-12
        ):
            return True
        if (current_y > y) != (previous_y > y):
            intersection = current_x + (y - current_y) * (
                previous_x - current_x
            ) / (previous_y - current_y)
            if x < intersection:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


class NSEvaluationProgram(PluginTaskEvaluationProgram):
    """Use one Task rule while adapting hidden points to each NS Case schema."""

    def make_plan(
        self,
        case: Case,
        phase: EvaluationPhase,
        settings: Mapping[str, object],
        seed: int,
        request: AnswerQuery | None = None,
    ) -> EvaluationPlan:
        # Explicit Development coordinates remain governed by the user's
        # user_fixed_query/agent_selected_query choice in AgentEvaluator.
        if request is not None or phase is EvaluationPhase.DEVELOPMENT:
            return super().make_plan(case, phase, settings, seed, request)
        query = _private_ns_point_query(
            case,
            settings,
            seed,
            point_count=_phase_point_count(settings, phase),
        )
        return EvaluationPlan(
            phase=phase,
            query=query,
            seed=seed,
            settings=settings,
        )


def _phase_point_count(
    settings: Mapping[str, object],
    phase: EvaluationPhase,
) -> int:
    if phase not in _NS_DEFAULT_POINT_COUNTS:
        raise ValueError(f"unsupported NS evaluation phase {phase!r}")
    setting_name = {
        EvaluationPhase.DEVELOPMENT: "development_point_count",
        EvaluationPhase.OFFICIAL: "official_point_count",
        EvaluationPhase.TERMINAL: "terminal_point_count",
    }[phase]
    value = settings.get(setting_name, _NS_DEFAULT_POINT_COUNTS[phase])
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"NS {setting_name} must be a positive integer")
    return value


class NSSolverTask:
    """Ask an Agent for a callable full-field NS solver."""

    info = ProgramInfo(
        "ns-full-field-solver",
        "Find a Navier--Stokes solver",
        "Return velocity and pressure on requested time-space grids or points.",
    )
    kind = TaskKind.SOLVER

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "primary_metric",
                "Primary metric",
                "text",
                "velocity_relative_l2",
                "Mean relative L2 error of the two velocity components.",
            ),
            FormField(
                "include_pressure",
                "Score pressure",
                "boolean",
                False,
            ),
            FormField(
                "official_point_count",
                "Official points per Case",
                "integer",
                96,
                "Fresh Task-owned hidden points shared by all Candidates.",
            ),
            FormField(
                "terminal_point_count",
                "Terminal points per Case",
                "integer",
                128,
                "Fresh Task-owned terminal points shared by all selected rounds.",
            ),
        )

    def agent_brief(self, settings: Mapping[str, object]) -> str:
        return (
            "Return a prepared continuous handle for velocity_x and velocity_y "
            "at physical (t,x,y) points. Pressure may also be returned. "
            "In isolated execution, solve_budget gives "
            "the remaining per-Case charged time. Measure elapsed solve time "
            "with pde_agent_verifier_solver.sdk.solve_budget.candidate_solve_clock, "
            "which excludes Controller-metered queue and lock waits."
        )

    def create_evaluation(
        self,
        settings: Mapping[str, object],
    ) -> PluginTaskEvaluationProgram:
        return NSEvaluationProgram.from_settings(
            program_id="ns-full-field-solver-evaluation-v1",
            settings=settings,
            private_development_query=_private_development_query,
            execute_candidate=self.execute_candidate,
            execute_candidate_batch=self.execute_candidate_batch,
            score_candidate=self.score_candidate,
        )

    def execute_candidate(
        self,
        artifact: Artifact,
        case: Case,
        plan: EvaluationPlan,
    ) -> CandidateEvaluationOutput:
        """Run Candidate code without receiving the Controller-owned truth."""

        query = plan.query
        solver = artifact.program
        timing_metrics: dict[str, float] = {}
        if _uses_two_phase_solution(solver):
            if query.kind is not QueryKind.POINTS:
                raise ValueError(
                    "formal NS prepare_solution evaluation requires a POINTS query"
                )
            solve_budget = candidate_solve_remaining_seconds()
            if solve_budget is not None and solve_budget <= 0.0:
                raise TimeoutError("NS Candidate exhausted its per-Case budget before preparation")
            prepared = measure_prepare_solution(solver, case, solve_budget=solve_budget)
            queried = measure_point_query(prepared.handle, query)
            prediction = queried.answer
            timing_metrics = {
                "candidate_solve_wall_seconds": (
                    prepared.candidate_solve_wall_seconds
                ),
                "candidate_query_wall_seconds": (
                    queried.candidate_query_wall_seconds
                ),
                "candidate_solve_queue_wait_seconds": (
                    prepared.candidate_solve_queue_wait_seconds
                ),
                "candidate_query_queue_wait_seconds": (
                    queried.candidate_query_queue_wait_seconds
                ),
                "candidate_query_point_count": float(queried.point_count),
                "candidate_query_wall_seconds_per_point": (
                    queried.candidate_query_wall_seconds_per_point
                ),
                "candidate_formal_two_phase": 1.0,
            }
        else:
            solve = getattr(solver, "solve", None)
            if not callable(solve):
                raise TypeError(
                    "NS Solver Task expects prepare_solution(case, solve_budget) "
                    "or legacy solve(case, query, budget)"
                )
            prediction = solve(case, query, budget=None)
        return CandidateEvaluationOutput(prediction, timing_metrics)

    def execute_candidate_batch(self, artifact, cases, plans):
        from pde_agent_verifier_solver.sdk.continuous_solution import measure_prepare_solutions
        if any(plan.query.kind is not QueryKind.POINTS for plan in plans):
            raise ValueError("batch NS requires POINTS queries")
        budget = candidate_solve_remaining_seconds()
        if budget is not None and budget <= 0:
            raise TimeoutError("NS batch exhausted its budget before preparation")
        handles, prepare_wall = measure_prepare_solutions(artifact.program, cases, solve_budget=budget)
        # No scoring coordinates are passed to either preparation interface.
        outputs = []
        for case, plan in zip(cases, plans):
            queried = measure_point_query(handles[case.case_id], plan.query)
            outputs.append(CandidateEvaluationOutput(queried.answer, {
                "candidate_batch_size": float(len(cases)),
                "candidate_batch_prepare_wall_seconds": prepare_wall,
                "candidate_query_wall_seconds": queried.candidate_query_wall_seconds,
                "candidate_query_point_count": float(queried.point_count),
                "candidate_formal_two_phase": 1.0,
            }))
        return tuple(outputs)

    def score_candidate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        plan: EvaluationPlan,
        output: CandidateEvaluationOutput,
    ) -> CaseEvaluation:
        """Score a finite Candidate prediction inside the Controller."""

        del artifact, plan
        prediction = output.prediction
        component_errors = {
            field: _relative_l2(
                prediction.fields[field],
                truth.fields[field],
            )
            for field in ("velocity_x", "velocity_y")
        }
        primary = fmean(component_errors.values())
        metrics = {
            "velocity_relative_l2": primary,
            **{
                f"{name}_relative_l2": value
                for name, value in component_errors.items()
            },
            **dict(output.metrics),
        }
        if (
            "pressure" in truth.fields
            and "pressure" in prediction.fields
        ):
            metrics["pressure_relative_l2"] = _relative_l2(
                prediction.fields["pressure"],
                truth.fields["pressure"],
            )
        return CaseEvaluation(
            case.case_id,
            primary,
            metrics,
            prediction,
        )

    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> CaseEvaluation:
        plan = EvaluationPlan(
            phase=EvaluationPhase.DEVELOPMENT,
            query=query,
        )
        output = self.execute_candidate(artifact, case, plan)
        return self.score_candidate(artifact, case, truth, plan, output)

    def select(
        self,
        candidates: Sequence[ArtifactEvaluation],
        incumbent: ArtifactEvaluation | None,
    ) -> ArtifactEvaluation:
        choices = tuple(candidates) + ((incumbent,) if incumbent else ())
        return min(choices, key=lambda value: value.primary_error)

    def result_card(self, evaluation: ArtifactEvaluation) -> PanelCard:
        return PanelCard(
            title=evaluation.artifact.name,
            summary="Selected full-field Navier--Stokes solver.",
            values={
                "hidden_velocity_relative_l2": evaluation.primary_error,
                "usage": evaluation.artifact.usage,
            },
        )


def _uses_two_phase_solution(program: object) -> bool:
    """Select the formal ABI without loading a deferred Candidate early."""

    declared = getattr_static(
        program,
        "prepare_solution",
        _MISSING_PROGRAM_ATTRIBUTE,
    )
    if declared is not _MISSING_PROGRAM_ATTRIBUTE:
        return True
    return (
        getattr_static(program, "_controller_loaded_two_phase_abi", False)
        is True
        or getattr_static(program, "_deferred_solution_abi", False) is True
    )


def _relative_l2(actual, expected) -> float:
    if len(actual) != len(expected):
        raise ValueError(
            "prediction and truth contain different sample counts"
        )
    squared_error = sum(
        (float(a) - float(b)) ** 2
        for a, b in zip(actual, expected)
    )
    squared_truth = sum(float(value) ** 2 for value in expected)
    return sqrt(squared_error / max(squared_truth, 1.0e-30))
