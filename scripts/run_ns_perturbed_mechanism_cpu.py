#!/usr/bin/env python3
"""Small CPU-only NS mechanism experiment using the existing Controller SDK.

One common development stage, two fresh conversations with identical incumbent
and research records, then a frozen independent parent/child evaluation. No
learned Generator training, GPU, or changes to the main experiment protocol.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'legacy/ns/src')]
import numpy as np
import torch
from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram, SubprocessAgentExecutionService
from pde_agent_verifier_solver.engine.candidate_execution import SubprocessCandidateExecutionService, PythonTaskWorkerSpec, CandidateExecutionError, CandidateExecutionInfrastructureError
from pde_agent_verifier_solver.engine.campaign import AgentSpec, CaseCounts, DevelopmentSamplingPolicy, EvaluatorPolicy, RoundPlan
from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
from pde_agent_verifier_solver.engine.common_evaluation import json_value
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator, HiddenVerifier
from pde_agent_verifier_solver.engine.memory import AgentRoundLog, CampaignRoundLog, CampaignMemorySnapshot
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.engine.runtime import AgentContext
from pde_agent_verifier_solver.plugins.ns.case_bank import NSBankSessionProvider, load_ns_continuous_case_bank, write_ns_continuous_case_bank
from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSManufacturedCaseSource, LegacyStructuredNSSmoothPerturber, _NSVelocityTruth
from pde_agent_verifier_solver.plugins.ns.public_problem import ns_public_problem_bundle
from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.sdk import AnswerQuery, CaseRecord, CaseVisibility, EvaluationPhase, GenerationKind, HiddenSuiteSpec, PublicResource, SampleSource, SamplingStrategy, TaskKind

from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest

FIELDS = ('velocity_x', 'velocity_y')
QUERY = AnswerQuery.at_points(fields=FIELDS, points=({'t': .05, 'x': .3, 'y': .4},))
SETTINGS = {'default_query': QUERY, 'include_pressure': False, 'development_point_count': 128,
            'official_point_count': 512, 'terminal_point_count': 512}
TASK = NSSolverTask()
CODEC = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
AMP = 0.003955665760443581
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


def sampler_module():
    p = ROOT/'docs/research/evidence/ns_gp_diagnostics_20260922/check_restricted_representation.py'
    spec = importlib.util.spec_from_file_location('restricted_ns_fixture', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source():
    module = sampler_module()
    return StructuredNSManufacturedCaseSource(generator_id='synthetic-ns-mechanism-v1',
        artifact_sha256=sha256(Path(module.__file__).read_bytes()).hexdigest(),
        clean_sampler=lambda seed: module.parent(seed)[0],
        perturbation_sampler=LegacyStructuredNSSmoothPerturber(amplitude=AMP, spectral_max_mode=5))


def make_bank(path, seed, count, paired=False, role='development_storage'):
    src = source()
    roots = src.sample(SampleRequest(1, CohortKind.TERMINAL, seed=seed, count=count))
    children = (src.sample(SampleRequest(1, CohortKind.TERMINAL, seed=seed+1000000, count=count,
                 kind=GenerationKind.PERTURBED, anchors=tuple(r.case for r in roots))) if paired else ())
    rows = roots + children
    evaluation = TASK.create_evaluation(SETTINGS)
    plans = {r.case.case_id: evaluation.make_plan(r.case, EvaluationPhase.TERMINAL, SETTINGS, seed+2000000+i)
             for i, r in enumerate(roots)}
    for p, c in zip(roots, children):
        plans[c.case.case_id] = plans[p.case.case_id]
    return write_ns_continuous_case_bank(path, bank_id=path.stem, purpose='terminal', records=rows,
        plans=plans, strata={r.case.case_id: ('base' if r in roots else 'base_perturbed') for r in rows},
        provenance={'scientific_role': role, 'parent_seed': seed, 'paired': paired,
            'constructed_distribution': True, 'generator_training': False, 'gpu_used': False})


def prepare(root):
    data = root/'data'
    data.mkdir(parents=True, exist_ok=True)
    if (root/'prepared.json').exists():
        return
    for name, seed, count, paired, role in (
        ('initial', 930101, 8, False, 'development_storage'),
        ('additional', 930201, 8, True, 'development_storage'),
        ('extra_base', 930301, 8, False, 'development_storage'),
        ('terminal', 940101, 8, True, 'independent_test'),
        ('control', 950101, 2, True, 'numerical_qualification')):
        target = data/(name+'.json')
        if target.exists():
            load_ns_continuous_case_bank(target, expected_purpose='terminal')
        else:
            make_bank(target, seed, count, paired, role)
    save(root/'prepared.json', {'status': 'completed', 'gpu_used': False, 'banks': 5,
        'hidden_storage_note': 'Development fields are privately archived by the existing continuous-bank codec; only the designated development banks are projected to development visibility.',
        'branch_context': 'fresh conversations with identical frozen incumbent and complete common public research record',
        'agent_seconds_per_stage': 3600, 'solver_seconds_per_case': 300,
        'development_points': 128, 'terminal_points': 512, 'amplitude': AMP,
        'initial_cases': 8, 'branch_cases': 24, 'terminal_cases': 16,
        'agent_model': 'gpt-5.6-sol', 'reasoning_effort': 'max'})


def bank(root, name):
    return load_ns_continuous_case_bank(root/'data'/(name+'.json'), expected_purpose='terminal', device='cpu')


def development(root, stage):
    names = ['initial'] + ([] if stage=='common' else ['additional']) + (['extra_base'] if stage=='without_p' else [])
    views = []
    for name in names:
        b = bank(root, name)
        selected = b.records[:8] if name=='additional' and stage=='without_p' else b.records
        rows = tuple(CaseRecord(replace(r.case, visibility=CaseVisibility.DEVELOPMENT), _NSVelocityTruth(b._answers[r.case.case_id])) for r in selected)
        views.append(SimpleNamespace(records=rows, _answers=b._answers, manifest_sha256=b.manifest_sha256))
    provider = NSBankSessionProvider(views)
    return tuple(r for view in views for r in view.records), provider


def candidate_service(directory, workers=2):
    return SubprocessCandidateExecutionService(directory, task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=SETTINGS, max_workers=workers, oracle_max_workers=workers,
        case_hard_wall_seconds=300, development_batch_hard_wall_seconds=1800,
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


def qualify(root):
    if (root/'qualification.json').exists():
        result = json.loads((root/'qualification.json').read_text())
        if not result['passed']:
            raise RuntimeError('Numerical qualification previously failed; preserve evidence and investigate')
        return
    b = bank(root, 'control')
    template = (ROOT/'docs/paper/iclr_results/presentation_20260914/solvers/full_r3/taylor.py').read_text()
    result = {}
    for label, n, nt in (('coarse', 49, 321), ('fine', 81, 641)):
        target = root/'controls'/label/'submission.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        code = template.replace('n = 65 if available >= 180 else (49 if available >= 80 else 33)', f'n = {n}')
        code = code.replace('nt = 81 if available >= 80 else 41', f'nt = {nt}')
        if f'n = {n}' not in code or f'nt = {nt}' not in code:
            raise RuntimeError('Frozen numerical-control grid configuration was not applied')
        target.write_text(code)
        cached = root/'qualification'/label/'result.json'
        result[label] = json.loads(cached.read_text()) if cached.exists() else evaluate(bind(target, root, 'control-'+label), b, cached.parent)
    coarse = result['coarse']['mean_error']; fine = result['fine']['mean_error']
    passed = bool(isinstance(coarse, (int, float)) and isinstance(fine, (int, float))
                  and np.isfinite(coarse) and np.isfinite(fine) and fine < coarse
                  and all(c['error'] is not None and math.isfinite(c['error']) and c['error'] < .03 for c in result['fine']['cases']))
    result.update(passed=passed, criterion='Finite on every Case, finer grid reduces mean error, every fine error below 3%.', gpu_used=False)
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
    locator = ROOT/'configs/candidates/ns_hold_initial_bundle' if previous is None else Path(previous['candidate'])
    initial = bind(locator, folder, 'starting-program')
    bundle = ns_public_problem_bundle(guidance_supported=False)
    memory = CampaignMemorySnapshot()
    if previous is not None:
        bundle = replace(bundle, resources=(*bundle.resources, common_records(root)))
        log = previous['round_log'] or 'Previous stage saved code and public development records.'
        memory = CampaignMemorySnapshot((CampaignRoundLog(1,
            (AgentRoundLog(1, 'ns-researcher', 'NS researcher', log),),
            'The previous development program is the shared starting point.', log),))
    prompt = ('Develop a reusable accurate solver for the declared unsteady forced incompressible Navier--Stokes problem. '
        'Use the available public inputs and numerical feedback to investigate and improve your method. '
        'You have CPU resources only. The per-Case charged solve limit is 300 seconds, including Oracle computation. '
        'The development wall-clock budget is 3600 seconds including every tool call and wait. '
        'Use your research time actively, keep an executable submission saved, and finish with exactly one final program. '
        'You may choose any numerical approach and any public-input query locations. '
        'All available development Cases are supplied in this stage; no further acquisition or solution-truth access is available. '
        'Final evaluation uses independent unsteady NS Cases. Research only the issued Cases through their public interfaces. '
        'Do not inspect private Controller files, generator implementations, or hidden reference fields. '
        'Record your numerical assumptions, experiments and code changes in ROUND_LOG.md. '
        + ('The supplied prior research record and incumbent are your common starting material. '
           'Continue investigating with the additional Cases available in this stage. ' if previous else '')
        + TASK.agent_brief(SETTINGS))
    case_execution = candidate_service(folder/'candidate_execution')
    execution = SubprocessAgentExecutionService(folder/'agent_turns', hard_wall_seconds=3600)
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
    role = AgentSpec('ns-researcher', 'NS researcher', 'Investigate the numerical problem and improve the solver.')
    counts = CaseCounts(base=len(rows)-(8 if stage=='with_p' else 0), base_perturbed=8 if stage=='with_p' else 0)
    policy = DevelopmentSamplingPolicy(strategy=SamplingStrategy.FIXED_RECIPE,
        max_cases_per_agent_per_round=len(rows), allowed_sources=(SampleSource.BASE,), fixed_recipe=counts)
    context = AgentContext(RoundPlan(1 if stage=='common' else 2, (role,), 3600, policy,
        HiddenSuiteSpec(base=8, base_perturbed=8)), role, prompt, evaluator, initial, memory,
        seed=101, public_problem_bundle=bundle, candidate_solve_budget_seconds=300)
    save(folder/'started.json', {'stage': stage, 'job_id': os.environ.get('SLURM_JOB_ID'),
        'time': time.time(), 'gpu_used': False, 'case_count':len(rows), 'budget_seconds':3600})
    try:
        outcome = execution.execute(program, context, 3600)
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
    torch.set_num_threads(1)
    root=args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.stage=='prepare': prepare(root)
    elif args.stage=='qualify': qualify(root)
    elif args.stage=='terminal': terminal(root)
    else: run_agent(root,args.stage,codex_binary=args.codex_binary,auth_file=args.auth_file)
    print(json.dumps({'stage':args.stage, 'event':'completed', 'gpu_used':False}), flush=True)


if __name__=='__main__':
    main()
