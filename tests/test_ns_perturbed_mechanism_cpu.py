"""Small real-bank checks before the CPU-only mechanism experiment is launched."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cpu_mechanism', ROOT/'scripts/run_ns_perturbed_mechanism_cpu.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_frozen_banks_branch_exposure_and_real_cpu_scoring(tmp_path):
    m.torch.set_num_threads(1)
    m.prepare(tmp_path)
    common, _ = m.development(tmp_path,'common')
    no, no_provider = m.development(tmp_path,'without_p')
    yes, yes_provider = m.development(tmp_path,'with_p')
    assert (len(common),len(no),len(yes))==(8,24,24)
    assert {r.case.case_id for r in common}<={r.case.case_id for r in no}&{r.case.case_id for r in yes}
    no_ids={r.case.case_id for r in no}; yes_ids={r.case.case_id for r in yes}
    assert len(no_ids & yes_ids)==16
    assert sum(r.case.parent_case_id is not None for r in no)==0
    assert sum(r.case.parent_case_id is not None for r in yes)==8
    assert all(r.case.visibility is m.CaseVisibility.DEVELOPMENT for r in yes)
    terminal=m.bank(tmp_path,'terminal')
    assert not {r.case.case_id for r in terminal.records}&(no_ids|yes_ids)
    roots=terminal.records[:8]; children=terminal.records[8:]
    for p,c in zip(roots, children):
        assert c.case.parent_case_id==p.case.case_id
        assert terminal.plans[c.case.case_id].query==terminal.plans[p.case.case_id].query
    public=json.dumps([m.json_value(r.case.public_data) for r in no+yes])
    assert all(x not in public for x in ('base_slope_private','rank_one','linear_scaling','coefficients.psi'))
    # Smallest real candidate/Oracle worker path; no LLM and no GPU.
    one=m.load_ns_continuous_case_bank(tmp_path/'data/terminal.json',expected_purpose='terminal',case_ids=(roots[0].case.case_id,))
    initial=m.bind(ROOT/'configs/candidates/ns_hold_initial_bundle',tmp_path,'s0')
    result=m.evaluate(initial,one,tmp_path/'test_execution')
    assert np.isfinite(result['mean_error']) and len(result['cases'])==1
    assert result['cases'][0]['metrics']['velocity_relative_l2']>=0


def test_three_real_controller_stages_with_local_stub(tmp_path):
    from tests.test_ns_baseline_canonical_worker import _stub_codex
    m.torch.set_num_threads(1)
    m.prepare(tmp_path)
    m.save(tmp_path/'qualification.json', {'passed':True, 'test_only':'Controller wiring; no numerical qualification claim'})
    executable, auth = _stub_codex(tmp_path)
    executable.write_text(executable.read_text().replace('assert len(cases) == 1','assert len(cases) in (8, 24)'))
    for stage in m.STAGES:
        m.run_agent(tmp_path,stage,codex_binary=executable,auth_file=auth)
        result=json.loads((tmp_path/stage/'result.json').read_text())
        assert result['status']=='completed' and result['gpu_used'] is False
        audits=json.loads((tmp_path/stage/'agent_execution_audits.json').read_text())
        request=json.loads(Path(audits[0]['request_path']).read_text())
        prompt=(Path(audits[0]['workspace'])/'CANONICAL_AGENT_PROMPT.md').read_text()
        assert 'no GPU is assigned' in prompt
        entered=json.loads((tmp_path/('entered-'+request['execution_id']+'.json')).read_text())
        assert entered['case_count']==(8 if stage=='common' else 24)
    shared=json.loads((tmp_path/'common/public_history.json').read_text())
    assert 'round_log' in shared and 'public_events' in shared
    entered=[json.loads(p.read_text()) for p in tmp_path.glob('entered-*.json')]
    assert len(entered)==3 and all(row['cuda_visible_devices']=='' and row['auth_removed'] for row in entered)
    # Fresh conversations must receive byte-identical common public history.
    resources=[]
    for stage in ('without_p','with_p'):
        audit=json.loads((tmp_path/stage/'agent_execution_audits.json').read_text())[0]
        resources.append((Path(audit['workspace'])/'public_resources/prior_research_record.json').read_bytes())
    assert resources[0]==resources[1]


def test_terminal_keeps_failed_cases_and_reuses_completed_work(tmp_path,monkeypatch):
    from types import SimpleNamespace
    rows=tuple(SimpleNamespace(case=SimpleNamespace(case_id=str(i))) for i in range(3))
    b=SimpleNamespace(records=rows,strata={'0':'base','1':'base','2':'base_perturbed'})
    calls=[]
    def run(artifact,bank,directory,*,records,workers):
        case_id=records[0].case.case_id
        calls.append(case_id)
        assert workers==1 and len(records)==1
        if case_id=='0':
            raise m.CandidateExecutionError('solver failed on this held-out Case')
        return {'cases':[{'case_id':case_id,'stratum':b.strata[case_id],'error':.01,'metrics':{}}],
                'wall_seconds':1.}
    monkeypatch.setattr(m,'evaluate',run)
    result=m.evaluate_terminal(None,b,tmp_path/'terminal')
    assert sorted(calls)==['0','1','2'] and len(result['cases'])==3
    assert result['mean_error'] is None and result['by_source']['base'] is None
    assert result['by_source']['base_perturbed']==pytest.approx(.01)
    assert result['failure_rate']==pytest.approx(1/3) and result['infrastructure_failures']==0
    m.evaluate_terminal(None,b,tmp_path/'terminal')
    assert len(calls)==3
