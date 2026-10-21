"""Budget-adaptive initial refinement for GP macros with preserved heat."""
from __future__ import annotations
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
from initial_sampling import sample_initial
from projection import ProjectionHandle, low_initial, modes_to_grid, FIELDS
from source_sampler import sample_source_at_initial
from hybrid_initial_adapter import HybridInitialHandle
from derivative_forecast import (raw_moment_derivative, macros_and_derivative,
                                 GaussianDerivativeForecast)

HEAT_FORECAST = False
CENTRAL_NX = 96
TAU_BOUNDS = (.1, 1.)
GUARD_NT = 129
FINE_INITIAL_POINT_RATIO = 10.24
FINE_INITIAL_TIME_MARGIN = 1.25
FINE_RETURN_RESERVE = 380.


def pressure_minimum(values):
    xx, xy, yy = values[5], values[6], values[7]
    return .5*(xx+yy-np.hypot(xx-yy, 2*xy))


class Prepared:
    def __init__(self, initial, baseline, model, initial_macros, check_budget):
        self.initial = initial
        self.baseline = baseline
        self.model = model
        self.closed = False
        self.guard_times = np.linspace(*initial["interval"], GUARD_NT)
        self.guard_fractions = np.ones(GUARD_NT)
        fluctuation = modes_to_grid(model.fluctuation_modes)
        derivative = modes_to_grid(model.derivative_modes)
        density_floor = max(1e-10, .01*float(np.min(initial_macros[0])))
        pressure_floor = max(0., .01*float(np.min(pressure_minimum(initial_macros))))
        self.density_floor, self.pressure_floor = density_floor, pressure_floor

        def admissible(value):
            return (np.isfinite(value).all() and np.min(value[0]) >= density_floor
                    and np.min(pressure_minimum(value)) >= pressure_floor)

        baseline_valid = admissible(initial_macros)
        for i, t in enumerate(self.guard_times):
            elapsed = t-initial["interval"][0]
            correlation = np.exp(-.5*(elapsed/model.tau)**2)[:, None, None]
            correction = (correlation-1)*fluctuation+correlation*elapsed*derivative
            if not HEAT_FORECAST:
                correction[8:] = 0.
            if admissible(initial_macros+correction):
                continue
            if not baseline_valid:
                self.guard_fractions[i] = 0.
                continue
            lo, hi = 0., 1.
            for _ in range(24):
                fraction = .5*(lo+hi)
                if admissible(initial_macros+fraction*correction):
                    lo = fraction
                else:
                    hi = fraction
            self.guard_fractions[i] = .95*lo
            check_budget()
        self.diagnostics = {"forecast": model.diagnostics,
                            "guard_min": float(np.min(self.guard_fractions))}

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("prepared solution is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("bounded macro forecast accepts POINTS only")
        if (not query.fields or len(set(query.fields)) != len(query.fields)
                or not set(query.fields) <= set(FIELDS)):
            raise ValueError("unknown or duplicate transport field")
        full_query = AnswerQuery(kind=query.kind, fields=FIELDS, points=query.points)
        baseline_answer = self.baseline.evaluate(full_query)
        baseline = np.asarray([baseline_answer.fields[field] for field in FIELDS])
        points = np.asarray([(p["t"], p["x"], p["y"]) for p in query.points])
        correction = self.model.correction(points).T
        if not HEAT_FORECAST:
            correction[8:] = 0.
        strength = np.interp(points[:, 0], self.guard_times, self.guard_fractions)
        correction *= strength
        values = baseline+correction
        # Deterministic pointwise projection only protects against an off-grid
        # escape between the dense preparation checks. All ten correction
        # components share its fraction; it is independent of query neighbors.
        bad = ((values[0] < self.density_floor)
               | (pressure_minimum(values) < self.pressure_floor)
               | ~np.isfinite(values).all(axis=0))
        if np.any(bad):
            for index in np.flatnonzero(bad):
                lo, hi = 0., 1.
                for _ in range(28):
                    fraction = .5*(lo+hi)
                    trial = baseline[:, index]+fraction*correction[:, index]
                    if (np.isfinite(trial).all() and trial[0] >= self.density_floor
                            and pressure_minimum(trial) >= self.pressure_floor):
                        lo = fraction
                    else:
                        hi = fraction
                values[:, index] = baseline[:, index]+.95*lo*correction[:, index]
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite prepared macro projection")
        arrays = {field: tuple(map(float, values[i])) for i, field in enumerate(FIELDS)}
        return AnswerView(kind=query.kind, fields={field: arrays[field] for field in query.fields},
                          points=query.points, shape=(len(points),))

    def close(self):
        self.closed = True
        self.baseline.close()


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    if solve_budget is not None and (isinstance(solve_budget, bool)
            or not isinstance(solve_budget, (int, float))
            or not np.isfinite(solve_budget) or solve_budget <= 0):
        raise ValueError("solve_budget must be finite and positive")
    budget = 850. if solve_budget is None else float(solve_budget)

    def check_budget():
        if candidate_solve_clock()-started > budget-40.:
            raise TimeoutError("bounded forecast reached return reserve")

    def remaining():
        return budget-(candidate_solve_clock()-started)

    session = open_public_case_session(case)
    base_started = candidate_solve_clock()
    base_initial = sample_initial(case, degree=4, session=session, check_budget=check_budget)
    base_seconds = max(0., candidate_solve_clock()-base_started)
    estimated_fine_seconds = FINE_INITIAL_POINT_RATIO*base_seconds*FINE_INITIAL_TIME_MARGIN
    initial = base_initial
    refinement = {"base_initial_seconds": base_seconds,
                  "estimated_fine_seconds": estimated_fine_seconds,
                  "fine_reserve_seconds": FINE_RETURN_RESERVE,
                  "status": "skipped_estimate", "selected_nx": 32}
    if estimated_fine_seconds+FINE_RETURN_RESERVE < remaining():
        fine_started = candidate_solve_clock()

        def check_fine_budget():
            if remaining() < FINE_RETURN_RESERVE:
                raise TimeoutError("initial refinement reached source/return reserve")

        try:
            check_fine_budget()
            refined = sample_initial(case, degree=4, nx=64, nv=128, vmax=22.,
                                     session=session, check_budget=check_fine_budget)
            check_fine_budget()
            initial = refined
            refinement.update(status="completed", selected_nx=64)
        except TimeoutError:
            refinement["status"] = "fine_budget_timeout"
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            refinement["status"] = "fine_numerical_failure"
        refinement["fine_initial_seconds"] = max(0., candidate_solve_clock()-fine_started)
    # Successful GP predictions always use the original static initial anchor.
    # The .25 hybrid is a separate source/numerical failure fallback.
    gp_baseline = ProjectionHandle(initial)
    fallback = HybridInitialHandle(initial, damping_rate=.25)
    fallback.diagnostics["refinement"] = refinement
    try:
        check_budget()
        source, sampling = sample_source_at_initial(case, session, initial, check_budget)
        raw = low_initial(initial, source.shape[-1])
        question = case.public_data["question"]
        derivative = raw_moment_derivative(
            raw, source, initial["periods"], float(question["collision_nu"]),
            float(question["temperature"]), initial["center"])
        _, dmacros = macros_and_derivative(raw, derivative, initial["periods"], initial["center"])
        fine_raw = modes_to_grid(initial["modes"], CENTRAL_NX)
        fine_macros, _ = macros_and_derivative(
            fine_raw, np.zeros((10, CENTRAL_NX, CENTRAL_NX)),
            initial["periods"], initial["center"])
        model = GaussianDerivativeForecast(fine_macros, dmacros, initial["periods"],
                                            t0=initial["interval"][0], tau_bounds=TAU_BOUNDS)
        check_budget()
        result = Prepared(initial, gp_baseline, model, fine_macros, check_budget)
        result.diagnostics["sampling"] = sampling
        result.diagnostics["refinement"] = refinement
        check_budget()
        return result
    except (TimeoutError, FloatingPointError, ValueError, np.linalg.LinAlgError):
        return fallback
