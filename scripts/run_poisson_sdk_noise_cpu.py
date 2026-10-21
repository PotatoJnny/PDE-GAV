#!/usr/bin/env python3
"""Two CPU Agent continuations using the SDK's actual perturbation sampler."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import numpy as np
import run_poisson_perturbed_mechanism_cpu as core
from pde_agent_verifier_solver.plugins.poisson_diagnostic import PoissonSessionProvider, PoissonSolverTask, public_data
from pde_agent_verifier_solver.plugins.poisson_sdk_noise import SDKPoissonField, perturb, LENGTH_SCALES, SOLUTION_RANGE, MAX_SOURCE_CHANGE
from pde_agent_verifier_solver.sdk import Case, CaseRecord, CaseSource, CaseVisibility, AnswerQuery, EvaluationPlan, EvaluationPhase


class ResearchTask(PoissonSolverTask):
    def agent_brief(self, settings):
        return super().agent_brief(settings)+(
            ' Commit a valid nomination early. After an evaluated improvement, update the nomination immediately; '
            'an improved file that was not submitted will not be selected when the wall clock expires. '
            'Checkpoint ROUND_LOG.md through the provided agent_tool during the work. '
            'Before the deadline, stop optional comparisons early enough to submit your selected program and log.')


def write_bank(path, seed, count, *, paired, role):
    rng=np.random.default_rng(seed)
    rows=[];children=[]
    for i in range(count):
        a=float(rng.uniform(.5,2)*rng.choice([-1.,1.]))
        parent=SDKPoissonField(a)
        key=f'poisson-sdk-{seed}-{i}'
        case=Case(key,CaseSource.GENERATED,CaseVisibility.HIDDEN,public_data())
        test_x=np.random.default_rng(seed+100000+i).uniform(0.,1.,512).tolist()
        rows.append({'case':core.json_value(case),'field':parent.to_payload(),'test_x':test_x,'perturbation':None})
        if paired:
            child,report=perturb(parent,seed=seed+1000000+i,length_scale=LENGTH_SCALES[i%len(LENGTH_SCALES)])
            child_case=Case(key+'-child',CaseSource.PERTURBED,CaseVisibility.HIDDEN,public_data(),parent_case_id=key)
            children.append({'case':core.json_value(child_case),'field':child.to_payload(),
                             'test_x':test_x,'perturbation':report})
    core.save(path,{'schema':'poisson_sdk_noise_bank_v1','seed':seed,'role':role,'cases':rows+children})


def bank(root, name):
    item=json.loads((root/'data'/(name+'.json')).read_text())
    if item['schema']!='poisson_sdk_noise_bank_v1':raise ValueError('wrong SDK bank schema')
    rows=[];plans={}
    for data in item['cases']:
        c=data['case']
        case=Case(c['case_id'],CaseSource(c['source']),CaseVisibility(c['visibility']),c['public_data'],parent_case_id=c['parent_case_id'])
        field=SDKPoissonField.from_payload(data['field'])
        rows.append(CaseRecord(case,field))
        query=AnswerQuery.at_points(fields=('u',),points=tuple({'x':x} for x in data['test_x']))
        plans[case.case_id]=EvaluationPlan(EvaluationPhase.TERMINAL,query)
    return core.PoissonBank(tuple(rows),plans)


def development(root,stage):
    b=bank(root,'shared')
    rows=list(b.records if stage=='with_p' else b.records[:8])
    if stage=='without_p':rows.extend(bank(root,'extra_base').records)
    if stage not in ('without_p','with_p'):raise ValueError('only two continuation arms')
    rows=tuple(CaseRecord(replace(r.case,visibility=CaseVisibility.DEVELOPMENT),r.truth) for r in rows)
    return rows,PoissonSessionProvider(rows)


def prepare(root):
    if (root/'prepared.json').exists():return
    source=root/'input'
    if not (source/'starting_program/candidate_bundle.json').exists():
        raise FileNotFoundError('The archived Agent starting program must be staged before preparation')
    (root/'data').mkdir(exist_ok=True)
    for name,seed,count,paired,role in (
        ('shared',962101,8,True,'development'),
        ('extra_base',962201,8,False,'development'),
        ('terminal',972101,8,True,'independent_test'),
        ('control',982101,4,True,'numerical_qualification')):
        path=root/'data'/(name+'.json')
        if not path.exists():write_bank(path,seed,count,paired=paired,role=role)
        bank(root,name)
    (root/'common').mkdir(exist_ok=True)
    archive=json.loads((source/'shared_history.json').read_text())
    core.save(root/'common/public_history.json',archive)
    core.save(root/'common/result.json',{
        'status':'completed','agent_execution_status':'archived_starting_program_no_new_agent_call',
        'candidate':str(source/'starting_program/candidate_bundle.json'),
        'round_log':'Controller summary of archived work: the supplied program is an Agent-developed 257-node Chebyshev integration solver. It was developed using the public Poisson equation and Base development cases. Both continuations receive identical code and archived public research records.',
        'gpu_used':False,'source_record':str(source/'parent_provenance.json')})
    reports=[r['perturbation'] for name in ('shared','terminal','control')
             for r in json.loads((root/'data'/(name+'.json')).read_text())['cases'] if r['perturbation']]
    core.save(root/'prepared.json',{'status':'completed','gpu_used':False,'generator_training':False,
        'noise_sampler':'sdk.draw_smooth_random_field',
        'perturbation_sampler':'sdk.sample_multiplicative_mms_perturbation',
        'composition':'SDK epsilon-shifted multiplication; Poisson x*(1-x) boundary envelope; recompute f=-u_xx',
        'length_scales':LENGTH_SCALES,'solution_relative_change_range':SOLUTION_RANGE,
        'maximum_source_relative_change':MAX_SOURCE_CHANGE,'term_count':32,'maximum_frequency_factor':4.,
        'epsilon_fraction':.001,'candidate_based_rejection':False,
        'agent_seconds_per_arm':900,'solver_seconds_per_case':10,'case_count_per_arm':16,
        'with_p_sources':{'base':8,'perturbed':8},'without_p_sources':{'base':16},
        'terminal_sources':{'base':8,'perturbed':8},
        'final_candidates':['archived_start','without_p','with_p'],
        'all_physical_checks_passed':True,
        'maximum_observed_source_change':max(r['physical_checks']['source_relative_change'] for r in reports),
        'maximum_observed_mms_residual':max(r['physical_checks']['mms_residual'] for r in reports)})


def qualify(root):
    if (root/'qualification.json').exists():
        if not json.loads((root/'qualification.json').read_text())['passed']:raise RuntimeError('Existing failed qualification must be investigated')
        return
    b=bank(root,'control');result={}
    for label,n in (('coarse',512),('fine',4096)):
        p=root/'controls'/label/'submission.py';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(core.control_source(n))
        result[label]=core.evaluate(core.bind(p,root,'control-'+label),b,root/'qualification'/label)
    passed=bool(result['fine']['mean_error']<result['coarse']['mean_error'] and
                all(r['error'] is not None and np.isfinite(r['error']) and r['error']<1e-6 for r in result['fine']['cases']))
    result.update(passed=passed,gpu_used=False,criterion='MMS checked on every generated child; fine FD improves mean and every fine error <1e-6.')
    core.save(root/'qualification.json',result)
    if not passed:raise RuntimeError('Numerical qualification failed before any Agent call')


def configure_driver():
    # Reuse the already verified CPU Agent/Verifier driver. These bindings are
    # local to this separate process; neither source nor old runs are changed.
    core.bank=bank
    core.development=development
    core.TASK=ResearchTask()


def terminal(root):
    core.terminal(root)
    p=root/'results.json';result=json.loads(p.read_text())
    result['scope']='Two controlled Agent continuations from one archived Base-developed program; actual SDK noise and new held-out cases.'
    result['common_label']='archived_starting_solver'
    result['development_coverage']={stage:json.loads((root/stage/'development_coverage.json').read_text()) for stage in ('without_p','with_p')}
    metadata=json.loads((root/'data/terminal.json').read_text())
    lengths={row['case']['case_id']:row['perturbation']['noise_settings']['length_scales'][0]
             for row in metadata['cases'] if row['perturbation']}
    for values in result['results'].values():
        values['by_noise_length']={str(length):float(np.mean([row['error'] for row in values['cases']
            if lengths.get(row['case_id'])==length])) for length in LENGTH_SCALES
            if all(row['status']=='completed' for row in values['cases'] if lengths.get(row['case_id'])==length)}
    core.save(p,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','qualify','without_p','with_p','terminal'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--codex-binary',type=Path,default=Path(os.environ.get('CODEX_BINARY', 'codex')))
    p.add_argument('--auth-file',type=Path,default=Path(os.environ.get('CODEX_AUTH_FILE', os.path.expanduser('~/.codex/auth.json'))))
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):raise RuntimeError('CPU-only execution required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    configure_driver()
    print(json.dumps({'event':'starting','stage':args.stage,'gpus':0}),flush=True)
    if args.stage=='prepare':prepare(root)
    elif args.stage=='qualify':qualify(root)
    elif args.stage=='terminal':terminal(root)
    else:core.run_agent(root,args.stage,codex_binary=args.codex_binary,auth_file=args.auth_file)
    print(json.dumps({'event':'completed','stage':args.stage,'gpus':0}),flush=True)

if __name__=='__main__':main()
