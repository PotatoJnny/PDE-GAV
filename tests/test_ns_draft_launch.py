from copy import deepcopy
import csv
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from pde_agent_verifier_solver.sdk import AnswerQuery, Case, CaseSource, CaseVisibility

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    spec=importlib.util.spec_from_file_location('ns_paper_test_'+path.stem,path)
    module=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode=previous
    return module


def test_common_s0_uses_only_public_initial_arrays():
    module=load(ROOT/'configs/paper/ns_draft_v1/s0_hold_initial_bundle/submission.py')
    x=np.array([0.,.5,1.]); y=x.copy()
    xx,yy=np.meshgrid(x,y,indexing='ij')
    case=Case(case_id='original',source=CaseSource.INITIAL,visibility=CaseVisibility.DEVELOPMENT,
              public_data={'question':{'time':{'t0':0.,'t1':.12}},
                           'arrays':{'x':x,'y':y,'initial_velocity':np.stack((xx+2*yy,3*xx-yy))}})
    handle=module.prepare_solution(case)
    points=({'t':.01,'x':.2,'y':.3},{'t':.11,'x':.2,'y':.3})
    answer=handle.evaluate(AnswerQuery.at_points(fields=('velocity_x','velocity_y'),points=points))
    assert answer.fields['velocity_x']==pytest.approx((.8,.8))
    assert answer.fields['velocity_y']==pytest.approx((.3,.3))


def test_common_s0_preserves_generated_oracle_path(monkeypatch):
    module=load(ROOT/'configs/paper/ns_draft_v1/s0_hold_initial_bundle/submission.py')
    class Session:
        def query_initial_velocity(self,points):
            return np.column_stack((points[:,0]+2*points[:,1],3*points[:,0]-points[:,1]))
    monkeypatch.setattr(module,'open_public_case_session',lambda case:Session())
    case=Case(case_id='generated',source=CaseSource.GENERATED,visibility=CaseVisibility.DEVELOPMENT,
              public_data={'domain':{'time_interval':[0.,.12],'bounds':{'x':[0.,1.],'y':[0.,1.]}},
                           'oracle_protocol':{'limits':{'max_points_per_request':128}}})
    query=AnswerQuery.at_points(fields=('velocity_x','velocity_y'),points=({'t':.11,'x':.2,'y':.3},))
    answer=module.prepare_solution(case).evaluate(query)
    assert answer.fields['velocity_x']==pytest.approx((.8,))
    assert answer.fields['velocity_y']==pytest.approx((.3,))


def test_first_batch_freezes_approved_protocol():
    from pde_agent_verifier_solver.plugins.ns.baselines.search import NSBaselineConfig
    runner=load(ROOT/'scripts/run_canonical_ns_campaign.py')
    directory=ROOT/'configs/paper/ns_draft_v1'
    for method in ('base','full'):
        config,path=runner._load_config(directory/f'{method}_s101.json')
        runner._validate_config(config)
        assert config['master_seed']==101
        assert config['round_count']==5 and len(config['agents'])==2
        assert 'agent_active_budget_seconds' not in config['execution']
        assert config['execution']['agent_hard_wall_seconds']==14400
        assert config['development']['max_cases_per_agent_per_round']==512
        assert config['development']['strategy']=='agent_adaptive'
        assert config['generator']['guidance_enabled']==(method=='full')
        assert config['generator']['perturbations_enabled']==(method=='full')
        assert config['official']==dict(original=0,base=1024,guided=0,base_perturbed=0,guided_perturbed=0)
    fixed=json.loads((directory/'fixed_s101.json').read_text())
    c=NSBaselineConfig(method=fixed['method'],output_directory='/tmp/not-created',
                       agent_budget_seconds=fixed['agent']['hard_wall_seconds'],**fixed['search'])
    assert c.turn_count==10
    assert 'active_budget_seconds' not in fixed['agent']
    assert fixed['agent']['hard_wall_seconds']==14400
    assert len(fixed['agent']['roles'])==2
    assert fixed['agent']['worker_settings']['controller_rpc_timeout_seconds']==7200


def test_current_manifest_uses_only_single_seed_hard_wall_protocol():
    directory=ROOT/'configs/paper/ns_draft_v1'
    manifest=json.loads((directory/'first_batch.json').read_text())
    assert manifest['seed']==101
    assert manifest['time_budget_mode']=='hard_wall'
    assert manifest['agent_hard_wall_seconds']==14400
    assert manifest['execution_status']=='paused_not_submitted'
    with (ROOT/'docs/paper/SUBMISSION_EXECUTION_RUNS.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    indexed={row['run_id']:row for row in rows}
    assert len(indexed)==len(rows)
    assert {row['seed'] for row in rows}=={'101'}
    assert all(row['time_budget_mode']=='hard_wall' for row in rows)
    assert all(row['hard_wall_minutes_per_agent_per_unit']=='240' for row in rows)
    # The run ledger changes as jobs are submitted/completed; its scientific
    # seed and budget invariants above remain fixed across those states.
    assert all(row['execution_status'].strip() for row in rows)
    assert {entry['method'] for entry in manifest['entries']}=={'fixed','base','full'}
    for entry in manifest['entries']:
        row=indexed[entry['csv_run_id']]
        assert row['budget_revision']==manifest['protocol_revision']
        assert row['agents']=='2' and row['units']=='5'
        assert float(row['agent_wall_hours_cap'])==40
