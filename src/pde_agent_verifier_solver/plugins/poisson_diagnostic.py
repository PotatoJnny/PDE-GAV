"""CPU Poisson diagnostic through the existing Task and public-session SDK.

The manufactured fields stay in the Controller. Candidates see only a forcing
query and the public equation. This fixture does not train a Generator.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, CandidateEvaluationOutput, Case, CaseAccessKind,
    CaseEvaluation, CaseRecord, CaseSource, CaseVisibility, EvaluationPhase,
    EvaluationPlan, FrozenGuidanceCapability, GeneratorCheckpoint, GeneratorState,
    PanelCard, ProgramInfo, PublicOracleMethod, PublicProblemBundle, PublicResource,
    QueryKind, TaskKind, measure_prepare_solution, measure_point_query,
)
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_remaining_seconds
from ._evaluation import PluginTaskEvaluationProgram

CENTERS = np.linspace(0., 1., 16)
LENGTH_SCALE = .2
EXPECTED_RELATIVE_RMS = .1
MAX_POINTS = 4096


@lru_cache(maxsize=1)
def gaussian_scale():
    """One ensemble normalization; never rescale or select individual draws."""
    x = np.linspace(0., 1., 8193)
    kernel = np.exp(-.5*((x[:, None]-CENTERS)/LENGTH_SCALE)**2)
    variance = (x*(1-x))**2 * np.mean(kernel**2, axis=1)
    return EXPECTED_RELATIVE_RMS / np.sqrt(2*np.trapezoid(variance, x))


@dataclass(frozen=True)
class PoissonField:
    amplitude: float
    gaussian_coefficients: tuple[float, ...] = ()

    def _noise(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not self.gaussian_coefficients:
            return np.zeros_like(x), np.zeros_like(x)
        d = x[:, None]-CENTERS
        k = np.exp(-.5*(d/LENGTH_SCALE)**2)/np.sqrt(len(CENTERS))
        z = np.asarray(self.gaussian_coefficients)
        g = k@z
        gp = (-d/LENGTH_SCALE**2*k)@z
        gpp = ((d*d/LENGTH_SCALE**4-1/LENGTH_SCALE**2)*k)@z
        w = x*(1-x)
        scale = gaussian_scale()*abs(self.amplitude)
        return scale*w*g, scale*(-2*g+2*(1-2*x)*gp+w*gpp)

    def solution(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        noise, _ = self._noise(x)
        return self.amplitude*np.sin(np.pi*x)+noise

    def forcing(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        _, noise_second = self._noise(x)
        return self.amplitude*np.pi**2*np.sin(np.pi*x)-noise_second

    def answer(self, query):
        if query.fields != ('u',):
            raise ValueError('Poisson field only provides u')
        if query.kind is QueryKind.POINTS:
            x = np.asarray([p['x'] for p in query.points])
            return AnswerView(query.kind, {'u': tuple(self.solution(x))},
                              points=query.points, shape=(len(x),))
        x = np.linspace(0., 1., query.resolution['x'])
        return AnswerView(query.kind, {'u': tuple(self.solution(x))},
                          axes={'x': tuple(x)}, shape=(len(x),))


def public_data():
    return {'equation': '-u_xx = f(x)', 'domain': {'bounds': {'x': [0., 1.]}},
            'boundary_condition': {'kind': 'dirichlet', 'left': 0., 'right': 0.},
            'oracle_protocol': {'limits': {'max_points_per_request': MAX_POINTS}}}


def generate_records(seed, count, *, paired=False, hidden=True):
    rng = np.random.default_rng(seed)
    visibility = CaseVisibility.HIDDEN if hidden else CaseVisibility.DEVELOPMENT
    parents, children = [], []
    for index in range(count):
        a = float(rng.uniform(.5, 2.)*rng.choice([-1., 1.]))
        key = f'poisson-{seed}-{index}'
        parent = Case(key, CaseSource.GENERATED, visibility, public_data())
        parents.append(CaseRecord(parent, PoissonField(a)))
        if paired:
            # Separate random stream: requesting children cannot change parents.
            noise_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 7123]))
            noise = tuple(float(v) for v in noise_rng.standard_normal(len(CENTERS)))
            child = Case(key+'-child', CaseSource.PERTURBED, visibility, public_data(),
                         parent_case_id=key)
            children.append(CaseRecord(child, PoissonField(a, noise)))
    return tuple(parents+children)


class _ForcingSession:
    public_calls_thread_safe = True

    def __init__(self, field):
        self._field = field
        self._closed = False

    def query_forcing(self, points):
        if self._closed:
            raise RuntimeError('closed public session')
        x = np.asarray(points, dtype=np.float64)
        if (x.ndim != 2 or x.shape[1] != 1 or not 0 < len(x) <= MAX_POINTS
                or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1)):
            raise ValueError('query_forcing needs 1 to 4096 finite [x] rows in [0,1]')
        return self._field.forcing(x[:, 0])[:, None]

    def close(self):
        self._closed = True


class PoissonSessionProvider:
    """Only forcing access; no reference-query or online-acquisition capability."""
    def __init__(self, records):
        self._records = {r.case.case_id: r for r in records}
        if len(self._records) != len(records):
            raise ValueError('duplicate case identity')
        identity = tuple(self._records)
        self.state = GeneratorState(GeneratorCheckpoint('poisson-frozen-bank', identity), identity)

    def case_session_methods(self, access):
        return ('query_forcing',) if access is CaseAccessKind.PUBLIC_PROBLEM else ()

    def open_case_session(self, state, case, access):
        if state != self.state or access is not CaseAccessKind.PUBLIC_PROBLEM:
            raise PermissionError('only the public forcing is available')
        row = self._records.get(case.case_id)
        if row is None or row.case != case:
            raise ValueError('case identity changed')
        return _ForcingSession(row.truth)


def private_query(case, settings, seed):
    del case
    n = int(settings.get('development_point_count', 128))
    rng = np.random.default_rng(seed)
    return AnswerQuery.at_points(fields=('u',), points=tuple({'x': float(x)} for x in rng.uniform(0., 1., n)))


class PoissonSolverTask:
    info = ProgramInfo('poisson-1d-solver', 'Solve one-dimensional Poisson',
                       'Return the continuous Dirichlet solution for a supplied forcing.')
    kind = TaskKind.SOLVER

    def setup_fields(self):
        return ()

    def agent_brief(self, settings):
        return ('Solve -u_xx=f(x) for 0<x<1 with u(0)=u(1)=0. '
                'Implement prepare_solution(case, solve_budget=None) returning a handle '
                'with evaluate(query) -> AnswerView of field u. Prepare your solution before '
                'the hidden scoring points are revealed. The score is relative L2 error of u. '
                'Use open_public_case_session(case).query_forcing(points) for public inputs: '
                'points is a float64 (N,1) array, returned forcing is (N,1), at most 4096 rows '
                'per request. Multiple requests are allowed. Reference solutions are private.')

    def create_evaluation(self, settings):
        return PluginTaskEvaluationProgram.from_settings(program_id='poisson-1d-evaluation',
            settings=settings, private_development_query=private_query,
            execute_candidate=self.execute_candidate, score_candidate=self.score_candidate)

    def execute_candidate(self, artifact, case, plan):
        prepared = measure_prepare_solution(artifact.program, case,
                                            solve_budget=candidate_solve_remaining_seconds())
        queried = measure_point_query(prepared.handle, plan.query)
        return CandidateEvaluationOutput(queried.answer, {
            'candidate_solve_wall_seconds': prepared.candidate_solve_wall_seconds,
            'candidate_query_wall_seconds': queried.candidate_query_wall_seconds,
            'candidate_solve_queue_wait_seconds': prepared.candidate_solve_queue_wait_seconds,
            'candidate_query_queue_wait_seconds': queried.candidate_query_queue_wait_seconds,
            'candidate_query_point_count': float(queried.point_count),
            'candidate_query_wall_seconds_per_point': queried.candidate_query_wall_seconds_per_point,
            'candidate_formal_two_phase': 1.,
        })

    def score_candidate(self, artifact, case, truth, plan, output):
        del artifact
        prediction = output.prediction
        y, ref = np.asarray(prediction.fields['u']), np.asarray(truth.fields['u'])
        if (prediction.kind is not QueryKind.POINTS or prediction.points != plan.query.points
                or y.shape != ref.shape or not np.isfinite(y).all()):
            raise ValueError('candidate must return finite u at every requested point in order')
        error = float(np.linalg.norm(y-ref)/max(np.linalg.norm(ref), 1e-14))
        return CaseEvaluation(case.case_id, error,
                              {'u_relative_l2': error, **dict(output.metrics)}, prediction)

    def evaluate(self, artifact, case, truth, query):
        plan = EvaluationPlan(EvaluationPhase.DEVELOPMENT, query)
        return self.score_candidate(artifact, case, truth, plan,
                                    self.execute_candidate(artifact, case, plan))

    def select(self, candidates, incumbent):
        return min(tuple(candidates)+(() if incumbent is None else (incumbent,)), key=lambda r: r.primary_error)

    def result_card(self, evaluation):
        return PanelCard(evaluation.artifact.name, 'Poisson solution error',
                         {'relative_l2': evaluation.primary_error})


def poisson_public_problem_bundle():
    text = (PoissonSolverTask().agent_brief({})+'\n\n'
            'Example input query: oracle = open_public_case_session(case)\n'
            'values = oracle.query_forcing(np.asarray([[0.2], [0.5]], dtype=float))\n'
            'Return AnswerView(kind=QueryKind.POINTS, fields={"u": tuple(values)}, '
            'points=query.points, shape=(len(query.points),)) from evaluate.\n'
            'The complete supplied case collection is your development material. '
            'You may freely choose a solution method, grid and query locations.\n')
    return PublicProblemBundle('poisson-public-v1', 'dirichlet_poisson_1d',
        'One-dimensional forced Poisson problem on the unit interval.',
        (PublicResource('README.md', text.encode(), 'text/markdown', 'Equation and solver interface'),),
        oracle_methods=(PublicOracleMethod('query_forcing', 'Evaluate f at chosen x coordinates',
            {'arguments': [{'name': 'points', 'dtype': 'float64', 'shape': ['N', 1], 'coordinate_order': ['x']}]},
            {'dtype': 'float64', 'shape': ['N', 1], 'field': 'forcing'},
            'evaluated_point', {'max_points_per_request': MAX_POINTS}),),
        guidance_capability=FrozenGuidanceCapability(supported=False))
