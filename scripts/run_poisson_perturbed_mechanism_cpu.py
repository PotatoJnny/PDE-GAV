#!/usr/bin/env python3
"""Small CPU-only Poisson Gaussian-perturbation experiment using the existing Controller SDK.

One common development stage, two fresh conversations with identical incumbent
and research records, then a frozen independent parent/child evaluation. No
learned Generator training, GPU, or changes to the main experiment protocol.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram, SubprocessAgentExecutionService
from pde_agent_verifier_solver.engine.candidate_execution import SubprocessCandidateExecutionService, PythonTaskWorkerSpec, CandidateExecutionError, CandidateExecutionInfrastructureError
from pde_agent_verifier_solver.engine.campaign import AgentSpec, CaseCounts, DevelopmentSamplingPolicy, EvaluatorPolicy, RoundPlan
from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
from pde_agent_verifier_solver.engine.common_evaluation import json_value
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator, HiddenVerifier
from pde_agent_verifier_solver.engine.memory import AgentRoundLog, CampaignRoundLog, CampaignMemorySnapshot
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.engine.runtime import AgentContext
from pde_agent_verifier_solver.plugins.poisson_diagnostic import (
    PoissonField, PoissonSessionProvider, PoissonSolverTask, generate_records,
    poisson_public_problem_bundle, public_data, gaussian_scale,
    EXPECTED_RELATIVE_RMS, LENGTH_SCALE, CENTERS,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Case, CaseRecord, CaseSource, CaseVisibility, EvaluationPhase,
    EvaluationPlan, HiddenSuiteSpec, PublicResource, SampleSource, SamplingStrategy, TaskKind,
)


FIELDS = ('u',)
QUERY = AnswerQuery.at_points(fields=FIELDS, points=({'x': .3},))
SETTINGS = {'default_query': QUERY, 'development_point_count': 128,
            'official_point_count': 512, 'terminal_point_count': 512}
TASK = PoissonSolverTask()
CODEC = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
STAGES = ('common', 'without_p', 'with_p')
ENV = {'PYTHONPATH': str(ROOT/'src'), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}


def save(path, value):
    def finite(v):
        if isinstance(v, dict): return {k: finite(x) for k, x in v.items()}
        if isinstance(v, list): return [finite(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v): return None
        if isinstance(v, Path): return str(v)
        return v
    atomic_json(Path(path), finite(json_value(value)))


def close_if_supported(service):
    close = getattr(service, 'close', None)
    if callable(close): close()


class FixedEvaluation:
    def __init__(self, evaluation, plans):
        self.evaluation, self.plans = evaluation, plans

    def __getattr__(self, name):
        return getattr(self.evaluation, name)

    def make_plan(self, case, phase, settings, seed, request=None):
        plan = self.plans[case.case_id]
        if phase is not plan.phase or request is not None:
            raise ValueError('Frozen evaluation cannot change points or phase')
        return plan


class PoissonBank:
    def __init__(self, records, plans):
        self.records = records
        self.plans = plans
        self.strata = {r.case.case_id: ('base_perturbed' if r.case.parent_case_id else 'base') for r in records}
        self.generator = PoissonSessionProvider(records)
        self.generator_state = self.generator.state

    def fixed_evaluation(self, evaluation):
        return FixedEvaluation(evaluation, self.plans)


def make_bank(path, seed, count, paired=False, role='development_storage'):
    rows = generate_records(seed, count, paired=paired)
    rng = np.random.default_rng(seed+2000000)
    plans = {r.case.case_id: [float(x) for x in rng.uniform(0., 1., 512)] for r in rows[:count]}
    for parent, child in zip(rows[:count], rows[count:]):
        plans[child.case.case_id] = plans[parent.case.case_id]
    save(path, {'schema': 'poisson_gaussian_bank_v1', 'role': role, 'seed': seed,
        'length_scale': LENGTH_SCALE, 'centers': CENTERS.tolist(), 'global_scale': gaussian_scale(),
        'expected_relative_rms': EXPECTED_RELATIVE_RMS,
        'cases': [{'case': json_value(r.case), 'amplitude': r.truth.amplitude,
                   'gaussian_coefficients': r.truth.gaussian_coefficients,
                   'test_x': plans[r.case.case_id]} for r in rows]})


def prepare(root):
    data = root/'data'
    data.mkdir(parents=True, exist_ok=True)
    if (root/'prepared.json').exists():
        return
    for name, seed, count, paired, role in (
        ('initial', 931101, 8, False, 'development_storage'),
        ('additional', 931201, 8, True, 'development_storage'),
        ('extra_base', 931301, 8, False, 'development_storage'),
        ('terminal', 941101, 8, True, 'independent_test'),
        ('control', 951101, 2, True, 'numerical_qualification')):
        target = data/(name+'.json')
        if not target.exists():
            make_bank(target, seed, count, paired, role)
        bank(root, name)
    save(root/'prepared.json', {'status': 'completed', 'gpu_used': False, 'generator_training': False,
        'equation': '-u_xx=f; x in (0,1); u(0)=u(1)=0',
        'private_base': 'a*sin(pi*x), random signed amplitude in [0.5,2]',
        'private_perturbation': 'smooth Gaussian field times x*(1-x), then recompute -u_xx',
        'expected_relative_rms': EXPECTED_RELATIVE_RMS, 'length_scale': LENGTH_SCALE,
        'global_scale': gaussian_scale(), 'per_realization_normalization': False,
        'branch_context': 'fresh conversations with identical incumbent and common public history',
        'agent_seconds_per_stage': 900, 'solver_seconds_per_case': 10,
        'development_points': 128, 'terminal_points': 512,
        'initial_cases': 8, 'branch_cases': 24, 'terminal_cases': 16,
        'agent_model': 'gpt-5.6-sol', 'reasoning_effort': 'max'})


def bank(root, name):
    data = json.loads((root/'data'/(name+'.json')).read_text())
    if data['schema'] != 'poisson_gaussian_bank_v1':
        raise ValueError('Unexpected Poisson bank schema')
    if (data['length_scale'] != LENGTH_SCALE or data['centers'] != CENTERS.tolist()
            or data['global_scale'] != gaussian_scale()):
        raise ValueError('Frozen Gaussian field parameters changed')
    rows, plans = [], {}
    for item in data['cases']:
        c = item['case']
        case = Case(c['case_id'], CaseSource(c['source']), CaseVisibility(c['visibility']),
                    c['public_data'], parent_case_id=c['parent_case_id'])
        field = PoissonField(item['amplitude'], tuple(item['gaussian_coefficients']))
        rows.append(CaseRecord(case, field))
        query = AnswerQuery.at_points(fields=FIELDS, points=tuple({'x': x} for x in item['test_x']))
        plans[case.case_id] = EvaluationPlan(EvaluationPhase.TERMINAL, query)
    return PoissonBank(tuple(rows), plans)


def development(root, stage):
    names = ['initial'] + ([] if stage=='common' else ['additional']) + (['extra_base'] if stage=='without_p' else [])
    rows = []
    for name in names:
        b = bank(root, name)
        selected = b.records[:8] if name=='additional' and stage=='without_p' else b.records
        rows.extend(CaseRecord(replace(r.case, visibility=CaseVisibility.DEVELOPMENT), r.truth) for r in selected)
    return tuple(rows), PoissonSessionProvider(tuple(rows))


def candidate_service(directory, workers=2):
    return SubprocessCandidateExecutionService(directory, task=PythonTaskWorkerSpec.from_task_type(PoissonSolverTask),
        evaluation_settings=SETTINGS, max_workers=workers, oracle_max_workers=workers,
        case_hard_wall_seconds=10, development_batch_hard_wall_seconds=300,
        environment={"PYTHONPATH": str(ROOT/"src")})


def bind(locator, root, key):
    return CODEC.bind_existing(locator, storage_root=root/'frozen'/key, artifact_id=key, name=key,
        summary='Frozen program for the CPU mechanism experiment.', usage='prepare_solution')


def report_data(report, strata):
    cases = [{'case_id': c.case_id, 'stratum': strata[c.case_id], 'error': c.primary_error,
              'metrics': dict(c.metrics)} for c in report.cases]
    return {'mean_error': report.primary_error, 'metrics': dict(report.metrics), 'cases': cases,
        'by_source': {s: float(np.mean([c['error'] for c in cases if c['stratum']==s])) for s in sorted(set(strata.values()))}}


def evaluate(artifact, b, directory, *, records=None, workers=2):
    records = b.records if records is None else records
    execution = candidate_service(directory, workers=workers)
    started = time.monotonic()
    try:
        verifier = HiddenVerifier(b.fixed_evaluation(TASK.create_evaluation(SETTINGS)), QUERY, SETTINGS,
            phase=EvaluationPhase.TERMINAL, generator=b.generator, generator_state=b.generator_state,
            candidate_execution=execution)
        result = report_data(verifier.evaluate(artifact, records),
            {r.case.case_id: b.strata[r.case.case_id] for r in records})
        result['wall_seconds'] = time.monotonic()-started
        save(directory/'result.json', result)
        return result
    finally:
        save(directory/'audits.json', execution.audits)
        close_if_supported(execution)


def control_source(intervals):
    # Author-owned numerical qualification only. Never the Agent's starting code.
    return f'''import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session
class Prepared:
    def __init__(self, spline): self.spline = spline
    def evaluate(self, query):
        x = np.asarray([p['x'] for p in query.points])
        return AnswerView(QueryKind.POINTS, {{'u': tuple(self.spline(x))}}, points=query.points, shape=(len(x),))
def prepare_solution(case, solve_budget=None):
    n = {intervals}
    x = np.linspace(0., 1., n+1)
    oracle = open_public_case_session(case)
    f = np.asarray(oracle.query_forcing(x[1:-1, None])).reshape(-1)
    ab = np.zeros((3,n-1)); ab[0,1:] = -1.; ab[1] = 2.; ab[2,:-1] = -1.
    u = np.zeros(n+1); u[1:-1] = solve_banded((1,1), ab, f/n**2)
    return Prepared(CubicSpline(x, u))
'''


def construction_check(root):
    # All preselected control draws retained, with no error-based rejection.
    rows = generate_records(951201, 16, paired=True)
    x = np.linspace(0., 1., 4097)
    probes = np.linspace(.001, .999, 257)
    output = []
    for p, c in zip(rows[:16], rows[16:]):
        field = c.truth
        h = 1e-4
        second = (-field.solution(probes+2*h)+16*field.solution(probes+h)
                  -30*field.solution(probes)+16*field.solution(probes-h)-field.solution(probes-2*h))/(12*h*h)
        residual = float(np.linalg.norm(-second-field.forcing(probes))/np.linalg.norm(field.forcing(probes)))
        rel = float(np.linalg.norm(field.solution(x)-p.truth.solution(x))/np.linalg.norm(p.truth.solution(x)))
        shortcut = float(np.linalg.norm(field.forcing(x)/np.pi**2-field.solution(x))/np.linalg.norm(field.solution(x)))
        boundary = float(np.max(np.abs(field.solution(np.array([0., 1.])))))
        output.append({'case_id': c.case.case_id, 'relative_solution_change': rel,
                       'relative_mms_residual': residual, 'boundary_error': boundary,
                       'author_shortcut_relative_error': shortcut})
    passed = all(r['relative_mms_residual'] < 1e-6 and r['boundary_error'] < 1e-14 for r in output)
    result = {'passed': passed, 'cases': output, 'agent_results': False,
              'criterion': 'Independent five-point derivative residual <1e-6; boundary <1e-14',
              'expected_relative_rms': EXPECTED_RELATIVE_RMS, 'no_selection_by_solver_error': True}
    save(root/'construction_check.json', result)
    return result


def qualify(root):
    if (root/'qualification.json').exists():
        if not json.loads((root/'qualification.json').read_text())['passed']:
            raise RuntimeError('Numerical qualification failed; preserve evidence')
        return
    construction = construction_check(root)
    b = bank(root, 'control')
    result = {'construction_passed': construction['passed']}
    for label, n in (('coarse', 32), ('fine', 128)):
        target = root/'controls'/label/'submission.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(control_source(n))
        result[label] = evaluate(bind(target, root, 'control-'+label), b, root/'qualification'/label)
    passed = (construction['passed'] and result['fine']['mean_error'] < result['coarse']['mean_error']
              and all(c['error'] is not None and np.isfinite(c['error']) and c['error'] < .001
                      for c in result['fine']['cases']))
    result.update(passed=bool(passed), criterion='MMS check; FD refinement improves mean; every fine error <0.1%.', gpu_used=False)
    save(root/'qualification.json', result)
    if not passed:
        raise RuntimeError('Numerical qualification failed; Agent stages have not started')


def common_records(root):
    public = root/'common'/'public_history.json'
    payload = json.loads(public.read_text())
    return PublicResource('prior_research_record.json', json.dumps(payload, ensure_ascii=False).encode(),
                          'application/json', 'Identical prior development record supplied to each continuation.')


def run_agent(root, stage, *, codex_binary, auth_file):
    if not json.loads((root/'qualification.json').read_text())['passed']:
        raise RuntimeError('Numerical qualification must pass before model calls')
    folder = root/stage
    folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():
        previous = json.loads((folder/'result.json').read_text())
        if previous['status'] == 'completed':
            return
        raise RuntimeError('A failed Agent stage needs diagnosis before retrying')
    if (folder/'started.json').exists():
        raise RuntimeError('An interrupted stage is preserved; do not silently rerun its full budget')
    rows, provider = development(root, stage)
    previous = None if stage=='common' else json.loads((root/'common'/'result.json').read_text())
    locator = ROOT/'configs/candidates/poisson_zero.py' if previous is None else Path(previous['candidate'])
    initial = bind(locator, folder, 'starting-program')
    bundle = poisson_public_problem_bundle()
    memory = CampaignMemorySnapshot()
    if previous is not None:
        bundle = replace(bundle, resources=(*bundle.resources, common_records(root)))
        log = previous['round_log'] or 'Previous stage saved code and public development records.'
        memory = CampaignMemorySnapshot((CampaignRoundLog(1,
            (AgentRoundLog(1, 'poisson-researcher', 'Poisson researcher', log),),
            'The previous development program is the shared starting point.', log),))
    prompt = ('Develop a reusable accurate solver for the declared one-dimensional Poisson problem. '
        'Use the available public inputs and numerical feedback to investigate and improve your method. '
        'You have CPU resources only. The per-Case charged solve limit is 10 seconds, including Oracle computation. '
        'The development wall-clock budget is 900 seconds including every tool call and wait. '
        'Use your research time actively, keep an executable submission saved, and finish with exactly one final program. '
        'You may choose any numerical approach and any public-input query locations. '
        'All available development Cases are supplied in this stage; no further acquisition or solution-truth access is available. '
        'Final evaluation uses independent Poisson Cases. Research only the issued Cases through their public interfaces. '
        'Do not inspect private Controller files, generator implementations, or hidden reference fields. '
        'Before your final submission, evaluate your final program on every supplied development Case at least once. '
        'This coverage check is part of your 900-second budget. '
        'Record your numerical assumptions, experiments and code changes in ROUND_LOG.md. '
        + ('The supplied prior research record and incumbent are your common starting material. '
           'Continue investigating with the additional Cases available in this stage. ' if previous else '')
        + TASK.agent_brief(SETTINGS))
    case_execution = candidate_service(folder/'candidate_execution')
    execution = SubprocessAgentExecutionService(folder/'agent_turns', hard_wall_seconds=900)
    runtime = folder/'agent_runtime'
    runtime.mkdir(mode=0o700, exist_ok=True); runtime.chmod(0o700)
    program = CommandAgentProgram((sys.executable, str(ROOT/'scripts/run_canonical_codex_agent.py'),
        '--codex-binary', str(codex_binary), '--auth-file', str(auth_file), '--runtime-root', str(runtime), '--cpu-only'),
        CODEC.decode_nomination, environment=ENV, incumbent_encoder=CODEC.encode_incumbent,
        worker_settings={'model':'gpt-5.6-sol', 'reasoning_effort':'max', 'nomination_policy':'latest', 'transient_resume_max_attempts':3},
        gpu_devices=())
    evaluator = AgentEvaluator(rows, TASK, EvaluatorPolicy(QUERY, allow_grid_truth=False,
        allow_point_truth=False, development_point_count=128), SETTINGS, seed=101,
        generator=provider, generator_state=provider.state, candidate_execution=case_execution)
    aliases = {c.case_id: r for c, r in zip(evaluator.cases, rows)}
    save(folder/'case_manifest.json', [
        {'public_case_id': key, 'private_case_id': r.case.case_id,
         'stratum': 'base_perturbed' if r.case.parent_case_id else 'base'}
        for key, r in aliases.items()])
    role = AgentSpec('poisson-researcher', 'Poisson researcher', 'Investigate the numerical problem and improve the solver.')
    counts = CaseCounts(base=len(rows)-(8 if stage=='with_p' else 0), base_perturbed=8 if stage=='with_p' else 0)
    policy = DevelopmentSamplingPolicy(strategy=SamplingStrategy.FIXED_RECIPE,
        max_cases_per_agent_per_round=len(rows), allowed_sources=(SampleSource.BASE,), fixed_recipe=counts)
    context = AgentContext(RoundPlan(1 if stage=='common' else 2, (role,), 900, policy,
        HiddenSuiteSpec(base=8, base_perturbed=8)), role, prompt, evaluator, initial, memory,
        seed=101, public_problem_bundle=bundle, candidate_solve_budget_seconds=10)
    save(folder/'started.json', {'stage': stage, 'job_id': os.environ.get('SLURM_JOB_ID'),
        'time': time.time(), 'gpu_used': False, 'case_count':len(rows), 'budget_seconds':900})
    try:
        outcome = execution.execute(program, context, 900)
        if len(outcome.nominations)!=1 or outcome.execution_status=='failed':
            raise RuntimeError(f'Agent did not complete an eligible stage: {outcome.execution_status}, nominations={len(outcome.nominations)}')
        candidate = outcome.nominations[0]
        save(folder/'result.json', {'status':'completed', 'agent_execution_status':outcome.execution_status,
            'candidate':str(candidate.program.bundle_locator), 'round_log':outcome.round_log,
            'model':'gpt-5.6-sol', 'branch_context':'fresh conversation with shared code and prior public record',
            'gpu_used':False})
        if stage=='common':
            audit = execution.audits[-1]
            # The stdout contains only model events and public tool interaction;
            # the canonical worker keeps credentials and Controller answers private.
            stdout = Path(audit.workspace)/'stdout.log'
            if not stdout.exists():
                stdout = Path(audit.stdout_path)
            events=[]
            for line in stdout.read_text(errors='replace').splitlines():
                try: value=json.loads(line)
                except json.JSONDecodeError: continue
                if value.get('type') in ('item.completed','item.started','item.updated'):
                    events.append(value)
            save(folder/'public_history.json', {'round_log':outcome.round_log, 'public_events':events,
                'note':'Prior tool request tokens and paths are historical; use the current agent_tool session.'})
    except Exception as exc:
        save(folder/'result.json', {'status':'failed', 'error':f'{type(exc).__name__}: {exc}', 'gpu_used':False})
        raise
    finally:
        save(folder/'agent_execution_audits.json', execution.audits)
        save(folder/'candidate_execution_audits.json', case_execution.audits)
        successful = {a.case_id for a in case_execution.audits if a.status == 'completed'}
        attempted = {a.case_id for a in case_execution.audits if a.execution_started}
        coverage = [{'public_case_id': key, 'private_case_id': r.case.case_id,
                     'stratum': 'base_perturbed' if r.case.parent_case_id else 'base',
                     'attempted': key in attempted, 'completed': key in successful}
                    for key, r in aliases.items()]
        save(folder/'development_coverage.json', {
            'issued': len(rows), 'attempted': len(attempted), 'completed': len(successful),
            'all_cases_evaluated': all(c['completed'] for c in coverage),
            'perturbed_completed': sum(c['completed'] and c['stratum']=='base_perturbed' for c in coverage),
            'note': 'Distinct cases with completed candidate executions, across all development candidates; not a claim of final-program coverage.',
            'cases': coverage})
        close_if_supported(case_execution)



def evaluate_terminal(artifact, b, directory):
    """Keep every held-out Case, including solver failures, without peer cancellation."""
    started=time.monotonic()
    def one(index_record):
        index, record=index_record
        target=directory/f'case-{index:03d}'
        cached=target/'case_result.json'
        if cached.exists():
            row=json.loads(cached.read_text())
            if row['status'] in ('completed','solver_failed'):
                return row
        row={'case_id':record.case.case_id, 'stratum':b.strata[record.case.case_id]}
        try:
            result=evaluate(artifact,b,target,records=(record,),workers=1)
            row.update(result['cases'][0],status='completed',wall_seconds=result['wall_seconds'])
            if row['error'] is None or not math.isfinite(row['error']):
                row.update(status='solver_failed',error=None,failure='nonfinite score')
        except CandidateExecutionInfrastructureError as exc:
            row.update(status='infrastructure_failed',error=None,failure=str(exc))
        except CandidateExecutionError as exc:
            row.update(status='solver_failed',error=None,failure=f'{type(exc).__name__}: {exc}')
        except Exception as exc:
            row.update(status='infrastructure_failed',error=None,failure=f'{type(exc).__name__}: {exc}')
        save(cached,row)
        return row
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(one,enumerate(b.records)))
    def mean(group):
        return float(np.mean([r['error'] for r in group])) if all(r['status']=='completed' for r in group) else None
    def failure_rate(group):
        return sum(r['status']=='solver_failed' for r in group)/len(group)
    groups={s:[r for r in rows if r['stratum']==s] for s in sorted(set(b.strata.values()))}
    result={'mean_error':mean(rows), 'by_source':{s:mean(g) for s,g in groups.items()},
        'failure_rate':failure_rate(rows), 'failure_rate_by_source':{s:failure_rate(g) for s,g in groups.items()},
        'infrastructure_failures':sum(r['status']=='infrastructure_failed' for r in rows),
        'cases':rows,'wall_seconds':time.monotonic()-started,
        'failed_mean_policy':'A cohort with any failure has no finite mean; failed Cases are never discarded.'}
    save(directory/'result.json',result)
    return result


def terminal(root):
    b = bank(root,'terminal')
    results={}
    for stage in STAGES:
        item=json.loads((root/stage/'result.json').read_text())
        if item['status']!='completed':
            raise RuntimeError('All development stages must complete before final evaluation')
        cached=root/'terminal_evaluation'/stage/'result.json'
        results[stage] = json.loads(cached.read_text()) if cached.exists() else evaluate_terminal(bind(item['candidate'], root/'terminal_evaluation', stage), b, cached.parent)
    complete=all(r['infrastructure_failures']==0 for r in results.values())
    save(root/'results.json', {'status':'completed' if complete else 'incomplete_infrastructure', 'gpu_used':False, 'results':results,
        'scope':'One constructed mechanism trajectory and two controlled continuations; no claim of occurrence frequency.'})
    if not complete:
        raise RuntimeError('Terminal infrastructure failures are retained separately; investigate before retry')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','qualify',*STAGES,'terminal'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--codex-binary',type=Path,default=Path(os.environ.get('CODEX_BINARY', 'codex')))
    parser.add_argument('--auth-file',type=Path,default=Path(os.environ.get('CODEX_AUTH_FILE', os.path.expanduser('~/.codex/auth.json'))))
    args=parser.parse_args()
    print(json.dumps({'stage':args.stage, 'event':'starting', 'gpu_used':False}), flush=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):
        raise RuntimeError('CPU-only experiment requires an empty CUDA device assignment')
    root=args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.stage=='prepare': prepare(root)
    elif args.stage=='qualify': qualify(root)
    elif args.stage=='terminal': terminal(root)
    else: run_agent(root,args.stage,codex_binary=args.codex_binary,auth_file=args.auth_file)
    print(json.dumps({'stage':args.stage, 'event':'completed', 'gpu_used':False}), flush=True)


if __name__=='__main__':
    main()
