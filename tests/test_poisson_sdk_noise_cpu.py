"""Verify direct SDK noise use, physics, data separation and two CPU workers."""
from pathlib import Path
import importlib.util
import json
import shutil
import sys
import numpy as np
import pytest
from pde_agent_verifier_solver.plugins.poisson_sdk_noise import (
    SDKPoissonField, perturb, LENGTH_SCALES, SOLUTION_RANGE,
)
from pde_agent_verifier_solver.sdk.perturbation import (
    SmoothRandomFieldSettings, draw_smooth_random_field,
    epsilon_shifted_multiplicative_value,
)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_poisson_sdk_noise_cpu as m


def make_input(root):
    data=root/'input';data.mkdir()
    candidate=m.core.bind(ROOT/'configs/candidates/poisson_zero.py',root,'local-test-start')
    shutil.copytree(candidate.program.bundle_locator.parent,data/'starting_program')
    m.core.save(data/'shared_history.json',{'public_events':[],'note':'Local test, no prior model calls.'})
    m.core.save(data/'parent_provenance.json',{'test_only':True})


def test_noise_and_composition_are_existing_sdk_and_mms_is_correct():
    for i,length in enumerate(LENGTH_SCALES):
        p=SDKPoissonField(-.9 if i%2 else .9)
        c,r=perturb(p,seed=961501+i,length_scale=length)
        drawn=draw_smooth_random_field(SmoothRandomFieldSettings((length,)),seed=r['random_field_seed'])
        assert c.noise==drawn
        assert SOLUTION_RANGE[0]<=r['relative_solution_change']<=SOLUTION_RANGE[1]
        assert r['physical_checks']['source_relative_change']<=1
        assert r['physical_checks']['mms_residual']<1e-5
        x=np.linspace(0,1,71)
        for order,value in enumerate(c.random_derivatives(x)):
            np.testing.assert_allclose(value,[drawn.derivative((v,),(order,)) for v in x],rtol=1e-12,atol=1e-9)
        expected=[epsilon_shifted_multiplicative_value(float(p.solution([v])[0]),drawn((v,))*v*(1-v),alpha=c.alpha,epsilon_fraction=c.epsilon_fraction,reference_scale=abs(p.amplitude)) for v in x]
        np.testing.assert_allclose(c.solution(x),expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(c.solution(x),SDKPoissonField.from_payload(c.to_payload()).solution(x))


def test_new_banks_matched_exposure_and_private_noise(tmp_path):
    make_input(tmp_path);m.prepare(tmp_path)
    no,_=m.development(tmp_path,'without_p');yes,_=m.development(tmp_path,'with_p')
    ids=lambda rs:{r.case.case_id for r in rs}
    assert len(no)==len(yes)==16 and len(ids(no)&ids(yes))==8
    assert sum(r.case.parent_case_id is not None for r in yes)==8
    terminal=m.bank(tmp_path,'terminal')
    assert len(terminal.records)==16 and not ids(terminal.records)&(ids(no)|ids(yes))
    for parent,child in zip(terminal.records[:8],terminal.records[8:]):
        assert terminal.plans[parent.case.case_id]==terminal.plans[child.case.case_id]
    public=json.dumps([r.case.public_data for r in yes])
    for name in ('frequencies','phases','weights','alpha','seed','amplitude'):
        assert name not in public
    status=json.loads((tmp_path/'prepared.json').read_text())
    assert status['all_physical_checks_passed'] and status['noise_sampler']=='sdk.draw_smooth_random_field'
    before=(tmp_path/'data/terminal.json').read_bytes()
    m.prepare(tmp_path)
    assert before==(tmp_path/'data/terminal.json').read_bytes()


def test_real_isolated_cpu_qualification(tmp_path):
    make_input(tmp_path);m.prepare(tmp_path);m.qualify(tmp_path)
    result=json.loads((tmp_path/'qualification.json').read_text())
    assert result['passed'] and len(result['fine']['cases'])==8


def test_both_continuations_use_real_controller_with_stub(tmp_path,monkeypatch):
    from tests.test_poisson_perturbed_mechanism_cpu import stub_codex
    make_input(tmp_path);m.prepare(tmp_path)
    m.core.save(tmp_path/'qualification.json',{'passed':True,'test_only':'worker wiring'})
    monkeypatch.setattr(m.core,'bank',m.bank)
    monkeypatch.setattr(m.core,'development',m.development)
    monkeypatch.setattr(m.core,'TASK',m.ResearchTask())
    executable,auth=stub_codex(tmp_path)
    executable.write_text(executable.read_text().replace('in (8,24)','in (16,)'))
    for stage in ('without_p','with_p'):
        m.core.run_agent(tmp_path,stage,codex_binary=executable,auth_file=auth)
        result=json.loads((tmp_path/stage/'result.json').read_text())
        assert result['status']=='completed' and result['round_log']
        coverage=json.loads((tmp_path/stage/'development_coverage.json').read_text())
        assert coverage['issued']==16 and coverage['completed']==1
        audit=json.loads((tmp_path/stage/'agent_execution_audits.json').read_text())[0]
        prompt=(Path(audit['workspace'])/'CANONICAL_AGENT_PROMPT.md').read_text()
        assert 'update the nomination immediately' in prompt
        assert '10 seconds' in prompt and '900 seconds' in prompt
        assert 'length_scales' not in prompt


def test_cpu_launcher_freezes_sdk_source_and_only_two_agents(tmp_path,monkeypatch):
    import submit_poisson_sdk_noise_cpu as launch
    monkeypatch.setattr(launch,'stage_parent',make_input)
    receipt=launch.submit(tmp_path/'dry-run',dry_run=True)
    data=json.loads(receipt.read_text())
    assert list(data['jobs'])==['qualify','without_p','with_p','terminal']
    assert data['jobs']['without_p']['depends_on']==['qualify']
    assert data['jobs']['with_p']['depends_on']==['qualify']
    assert data['jobs']['terminal']['depends_on']==['without_p','with_p']
    assert all('--gres=gpu:0' in v['command'] for v in data['jobs'].values())
    frozen=Path(data['source'])
    assert (frozen/'src/pde_agent_verifier_solver/sdk/perturbation.py').read_bytes()==(ROOT/'src/pde_agent_verifier_solver/sdk/perturbation.py').read_bytes()
    assert 'run_poisson_sdk_noise_cpu.py' in (receipt.parent/'run_stage.sh').read_text()
