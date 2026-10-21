"""Run one bounded numerical-solver shard using the existing common evaluator."""
import argparse,hashlib,json,os,time
from pathlib import Path
from dataclasses import replace
from pde_agent_verifier_solver.engine.common_evaluation import CommonProgram,CommonCase,evaluate_common_programs,json_value
from pde_agent_verifier_solver.engine.candidate_execution import SubprocessCandidateExecutionService,PythonTaskWorkerSpec
from pde_agent_verifier_solver.engine.checkpoint_store import run_writer_lock,RunSegmentBudget
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.plugins.ns.case_bank import load_ns_continuous_case_bank
from pde_agent_verifier_solver.plugins.ns.tasks.solver import NSSolverTask
from pde_agent_verifier_solver.sdk import AnswerQuery,TaskKind,EvaluationPhase
from pde_agent_verifier_solver.sdk.candidate_bundle import inspect_candidate_bundle
ROOT=Path(__file__).resolve().parents[2]/'.paper_runs/ns_numerical_baselines_20260924'
METRICS=('velocity_relative_l2','velocity_x_relative_l2','velocity_y_relative_l2')

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def run(job):
    output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    cohorts=json.loads((ROOT/'control/cohorts.json').read_text());cohort=cohorts[job['task']+'_'+job['stage']]
    case_ids=job['case_ids']
    if not set(case_ids)<=set(cohort['case_ids']):raise ValueError('shard changed cohort')
    qualification=json.loads((ROOT/'qualification'/f"{job['method']}.json").read_text())
    if qualification['status']!='passed':raise RuntimeError('numerical qualification failed')
    bank=load_ns_continuous_case_bank(cohort['manifest'],expected_purpose=cohort['purpose'],device='cpu',case_ids=case_ids)
    # The common static-program evaluator executes TERMINAL-phase plans.
    # Calibration/validation retain their frozen queries and references; only
    # this in-memory invocation tag changes, never a bank or scientific split.
    original_phases={k:p.phase.value for k,p in bank.plans.items()}
    bank.plans={k:replace(p,phase=EvaluationPhase.TERMINAL) for k,p in bank.plans.items()}
    if job['task']=='obstacle':
        from ns_d0_local_test.public_cpu import ExactCachedBankProvider
        bank.generator=ExactCachedBankProvider((bank,));bank.generator_state=bank.generator.state
    settings={'default_query':AnswerQuery.at_points(fields=('velocity_x','velocity_y'),points=({'t':0.,'x':0.,'y':0.},)),
              'primary_metric':'velocity_relative_l2','include_pressure':False,'terminal_point_count':128}
    identity={'job':job,'original_plan_phases':original_phases,'bank':bank.manifest_sha256,'seconds':900,'workers':4,'oracle':'same96_exact_cpu_v6' if job['task']=='obstacle' else 'frozen_bank_public_v5',
              'runner':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with run_writer_lock(output):
        resume=(output/'input_identity.json').exists()
        if resume:
            if json.loads((output/'input_identity.json').read_text())!=identity:raise ValueError('resume configuration changed')
        else:atomic(output/'input_identity.json',identity)
        codec=PythonCandidateCodec(task_kind=TaskKind.SOLVER);programs=[]
        for profile in job['profiles']:
            row=json.loads((ROOT/'control/profiles.json').read_text())[profile]
            artifact=codec.bind_existing(row['bundle'],storage_root=output/'programs'/profile,artifact_id=profile,
                name=profile,summary='Classical numerical NS baseline',usage='prepare_solution')
            programs.append(CommonProgram(profile,'classical_numerical_20260924',20260924,profile,artifact,artifact.program.tree_sha256))
        cases=[CommonCase(job['task']+'_'+job['stage'],bank.bank_id,r,bank.plans[r.case.case_id]) for r in bank.records]
        execution=SubprocessCandidateExecutionService(output/'execution',task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
            evaluation_settings=settings,max_workers=4,case_hard_wall_seconds=900,oracle_max_workers=4)
        budget=RunSegmentBudget(10200,flush_seconds=600);execution.allocation_cancellation_event=budget.cancel
        try:
            with budget.signals():
                summary=evaluate_common_programs(task_id='ns-full-field-solver',programs=programs,cases=cases,
                    evaluation=bank.fixed_evaluation(NSSolverTask().create_evaluation(settings)),evaluation_settings=settings,
                    candidate_execution=execution,metrics=METRICS,output_directory=output,max_workers=4,
                    generator=bank.generator,generator_state=bank.generator_state,resume=resume,segment_budget=budget)
            atomic(output/f'execution_segment_{time.time_ns()}.json',{'job_id':os.environ.get('SLURM_JOB_ID'),'audits':[json_value(a) for a in execution.audits]})
            if not all(a.process_group_cleaned and not a.controller_truth_shared for a in execution.audits):
                raise RuntimeError('execution isolation or cleanup failed')
            print(json.dumps({'status':summary['status'],'output':str(output),'groups':summary.get('groups',[])}),flush=True)
            return 75 if summary['status']=='paused' else 0
        finally:budget.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('numerical evaluations require a compute allocation')
    return run(json.loads(a.jobs.read_text())[a.index])
if __name__=='__main__':raise SystemExit(main())
