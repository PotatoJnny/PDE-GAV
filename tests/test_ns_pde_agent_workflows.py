"""New PDE-Agent policies use real NS evaluation and existing continuation."""
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl, CampaignPaused
from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
from pde_agent_verifier_solver.plugins.ns.baselines.pde_workflows import validate_sharp_selection
from tests.test_ns_baselines import _artifact, _policy, _records
from tests.test_ns_literature_baselines import _options


def options(**kwargs):
    return _options(autopde_attempts=1, autopde_review_seconds=2, sharp_samples=3,
                    sharp_judges=2, sharp_cycles=1, sharp_hybrid_rounds=1, **kwargs)


def run(root, method, agent, **kwargs):
    development, selection = _records()
    return run_ns_baseline(NSBaselineConfig(method, root, seed=101, agent_budget_seconds=20,
        literature_options=options()), development_records=development, selection_records=selection,
        evaluator_policy=_policy(), task_settings={"official_point_count":4}, agent_program=agent,
        agent_execution=InProcessAgentExecutionService(), **kwargs)


def context_info(context):
    resources = {r.relative_path:r.data for r in context.public_problem_bundle.resources}
    return resources, json.loads(resources.get('baseline_context.json', b'{}'))


class PDEAgent:
    def __init__(self, artifacts):
        self.artifacts, self.contexts, self.genesis = artifacts, [], 0
    def run(self, context):
        self.contexts.append(context)
        phase=context.agent.name
        if phase=='autopde diagnosis':
            value={k:'NS diagnosis' for k in ('equation_type','coupling','time_dependence','boundary_conditions','dominant_physics')}
        elif phase=='autopde method':
            value={k:'NS method' for k in ('formulation','spatial_method','time_method','algebraic_solver','stabilization','boundary_treatment')}
        elif phase.startswith('sharp analysis '):
            value={'finding':'equation analysis','consequences':'numerical choices'}
        elif phase=='sharp judge':
            resources,info=context_info(context)
            refs=info['seed_implementations']
            assert refs and all(ref['entrypoint'] in resources for ref in refs)
            hashes=[r['tree_sha256'] for r in refs]
            value={'selected':hashes[:math.ceil(len(hashes)/2)],'nominee':hashes[0],'rationale':'judge rationale'}
        else:
            if phase=='sharp genesis':
                result=self.artifacts[self.genesis];self.genesis+=1
            elif phase in ('sharp hybrid','autopde review'):
                result=self.artifacts[-1]
            else:
                result=self.artifacts[0]
            return AgentOutcome(result,round_log='METHOD updated after an actual pilot; numerical plan retained.')
        return AgentOutcome(round_log=json.dumps(value))


def test_autopde_cards_precede_code_and_fresh_review_receives_program_and_feedback(tmp_path):
    agent=PDEAgent([_artifact(tmp_path,.8,'develop'),_artifact(tmp_path,.95,'review')])
    result=run(tmp_path/'run','autopde_ns_adaptation',agent)
    assert [c.agent.name for c in agent.contexts]==['autopde diagnosis','autopde method','autopde develop','autopde review']
    _,coding=context_info(agent.contexts[2]);_,review=context_info(agent.contexts[3])
    assert set(coding['numerical_plan'])=={'DIAGNOSIS','METHOD'}
    assert agent.contexts[2].incumbent is None
    assert agent.contexts[3].incumbent.program.tree_sha256==agent.artifacts[0].program.tree_sha256
    assert review['feedback']['error']==pytest.approx(.2)
    assert 'METHOD updated after an actual pilot' in agent.contexts[3].prompt
    assert 'do not redesign' in agent.contexts[3].prompt
    assert result.selection_error==pytest.approx(.05)
    turns=json.loads((result.output_directory/'baseline_turns.json').read_text())
    assert len({t['branch'] for t in turns})==1
    assert turns[-1]['hard_wall_limit_seconds']==2
    assert turns[2]['hard_wall_limit_seconds']<=18


def test_autopde_resume_between_development_and_review_does_not_repeat_first_pass(tmp_path):
    agent=PDEAgent([_artifact(tmp_path,.8,'develop'),_artifact(tmp_path,.95,'review')])
    store=CheckpointStore(tmp_path/'checkpoint',identity={'test':'autopde'})
    fired=False
    def should_pause(stage,required):
        nonlocal fired
        if stage=='literature_agent' and len(agent.contexts)==3 and not fired:
            fired=True;return True
        return False
    pause=CampaignPauseControl(store.save,should_pause,agent_start_margin_seconds=0,hidden_start_margin_seconds=0)
    with pytest.raises(CampaignPaused):
        run(tmp_path/'run','autopde_ns_adaptation',agent,pause_control=pause)
    saved=store.load();elapsed=saved['literature']['agent_wall_seconds']
    assert len(saved['turns'])==3 and elapsed>0
    result=run(tmp_path/'run','autopde_ns_adaptation',agent,continuation=saved,pause_control=pause)
    assert len(agent.contexts)==4 and result.selection_error==pytest.approx(.05)
    assert saved['literature']['agent_wall_seconds']>=elapsed


def test_sharp_judges_see_full_unranked_pool_and_only_nominees_receive_outer_scores(tmp_path):
    artifacts=[_artifact(tmp_path,x,f'solver-{i}') for i,x in enumerate((.7,.8,.99,.95))]
    agent=PDEAgent(artifacts)
    result=run(tmp_path/'run','pde_sharp_ns_adaptation',agent)
    names=[c.agent.name for c in agent.contexts]
    assert names[:5]==['sharp analysis '+s for s in ('classification','analytic','transformation','decomposition','stability')]
    judges=[c for c in agent.contexts if c.agent.name=='sharp judge']
    assert len(judges)==2
    for c in judges:
        _,info=context_info(c)
        assert len(info['seed_implementations'])==3
        assert all(r['validation_error'] is None for r in info['seed_implementations'])
    hybrids=[c for c in agent.contexts if c.agent.name=='sharp hybrid']
    assert len(hybrids)==2
    feedback=[context_info(c)[1]['feedback'] for c in hybrids]
    assert feedback[0]==feedback[1] and len(feedback[0])==1
    assert all(c.incumbent.program.tree_sha256==artifacts[0].program.tree_sha256 for c in hybrids)
    reports=json.loads((result.output_directory/'literature_evaluations.json').read_text()).values()
    assert artifacts[2].program.tree_sha256 not in {r['tree_sha256'] for r in reports}
    assert result.selection_error==pytest.approx(.05)  # un-nominated .01 candidate is not test-selected


def test_sharp_resume_after_judges_does_not_repeat_analysis_or_genesis(tmp_path):
    agent=PDEAgent([_artifact(tmp_path,x,f'solver-{i}') for i,x in enumerate((.7,.8,.99,.95))])
    store=CheckpointStore(tmp_path/'checkpoint',identity={'test':'sharp'})
    fired=False
    def should_pause(stage,required):
        nonlocal fired
        if stage=='literature_agent' and len(agent.contexts)==10 and not fired:
            fired=True;return True
        return False
    pause=CampaignPauseControl(store.save,should_pause,agent_start_margin_seconds=0,hidden_start_margin_seconds=0)
    with pytest.raises(CampaignPaused):
        run(tmp_path/'run','pde_sharp_ns_adaptation',agent,pause_control=pause)
    saved=store.load()
    result=run(tmp_path/'run','pde_sharp_ns_adaptation',agent,continuation=saved,pause_control=pause)
    assert len(agent.contexts)==12 and agent.genesis==3 and result.selection_error==pytest.approx(.05)



def test_sharp_round_records_are_snapshots_and_next_cycle_rejudges_expanded_pool(tmp_path):
    artifacts=[_artifact(tmp_path,x,f'solver-{i}') for i,x in enumerate((.7,.8,.99,.95))]
    agent=PDEAgent(artifacts)
    development,selection=_records()
    result=run_ns_baseline(NSBaselineConfig('pde_sharp_ns_adaptation',tmp_path/'run',
        agent_budget_seconds=20,literature_options={**options(),'sharp_cycles':2,'sharp_hybrid_rounds':2}),
        development_records=development,selection_records=selection,evaluator_policy=_policy(),
        task_settings={'official_point_count':4},agent_program=agent,
        agent_execution=InProcessAgentExecutionService())
    records=json.loads((result.output_directory/'literature_search.json').read_text())['records']
    first=records['sharp-cycle-1/round-1'];second=records['sharp-cycle-1/round-2']
    assert all(len(history)==2 for history in first['judge_histories'])
    assert all(len(history)==3 for history in second['judge_histories'])
    judges=[context_info(c)[1] for c in agent.contexts if c.agent.name=='sharp judge']
    assert [len(c['seed_implementations']) for c in judges]==[3,3,4,4]
    assert result.selection_error==pytest.approx(.05)



def test_sharp_failed_nominee_remains_available_to_judges_for_repair(tmp_path):
    from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
    from pde_agent_verifier_solver.sdk import TaskKind
    path=tmp_path/'bad.py'
    path.write_text('def prepare_solution(case, solve_budget=None):\n    raise ValueError("repair me")\n')
    bad=PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(path,storage_root=tmp_path/'bad-bundle',
        artifact_id='bad',name='bad',summary='bad',usage='test')
    artifacts=[bad,*[_artifact(tmp_path,x,f'solver-{i}') for i,x in enumerate((.8,.99,.95))]]
    agent=PDEAgent(artifacts)
    result=run(tmp_path/'run','pde_sharp_ns_adaptation',agent)
    hybrids=[context_info(c)[1] for c in agent.contexts if c.agent.name=='sharp hybrid']
    assert len(hybrids)==2
    for info in hybrids:
        failed=next(r for r in info['seed_implementations'] if r['tree_sha256']==bad.program.tree_sha256)
        assert failed['validation_error'] is None and failed['validation_status']=='failed'
        assert 'repair me' in failed['validation_failure']
        assert info['feedback'][0]['status']=='failed'
    assert result.selection_error==pytest.approx(.05)


def test_sharp_rejects_unknown_or_duplicate_programs_before_nominee_execution():
    for payload in ({'selected':['a','a'],'nominee':'a','rationale':'bad'},
                    {'selected':['a','missing'],'nominee':'a','rationale':'bad'},
                    {'selected':['a','b'],'nominee':'c','rationale':'bad'}):
        with pytest.raises(ValueError):
            validate_sharp_selection(payload,('a','b','c'))


@pytest.mark.parametrize('name',['codepde','autonumerics','autopde','pde_sharp'])
def test_all_eight_templates_preserve_pairing_and_known_methods(tmp_path,name):
    from pde_agent_verifier_solver.plugins.ns.baselines.preparation import _replace,METHOD_KEYS
    root=Path(__file__).resolve().parents[1]/'configs/paper/ns_literature_v1'
    values={'@STARTER@':'starter','@SELECTION@':'selection','@INITIAL@':'s0','@INITIAL_SHA256@':'a'*64,
            '@CODEX@':'codex','@AUTH@':'auth','@GPU@':'0','@WORKERS@':4,'@GENERATOR@':{}}
    fixed=_replace(json.loads((root/f'{name}_fixed_sampled.template.json').read_text()),values)
    online=_replace(json.loads((root/f'{name}_continuous_online_base.template.json').read_text()),values)
    for k in ('agent','evaluation','search','prompt','initial_candidate','seed'):
        assert fixed[k]==online[k]
    assert fixed['method']==online['method']==METHOD_KEYS[name]
    for data in (fixed,online):
        cfg=NSBaselineConfig(data['method'],tmp_path,agent_budget_seconds=data['agent']['hard_wall_seconds'],**data['search'])
        assert cfg.turn_count>0
        assert data['evaluation']['allow_point_truth'] is False
    changed={k for k in set(fixed['dataset'])|set(online['dataset']) if fixed['dataset'].get(k)!=online['dataset'].get(k)}
    assert changed=={'arm','new_cases_per_branch','online_generator'}


def test_eight_arm_terminal_export_requires_all_four_real_pairs(tmp_path):
    from hashlib import sha256
    from pde_agent_verifier_solver.plugins.ns.baselines.preparation import export_terminal_manifest,METHOD_KEYS,ENVIRONMENT_ARMS
    initial=_artifact(tmp_path,0.,'s0');final=_artifact(tmp_path,1.,'result')
    terminal=tmp_path/'terminal.json';terminal.write_text('{}')
    runs=[]
    for name,method in METHOD_KEYS.items():
        for arm in ENVIRONMENT_ARMS:
            label=f'{name}_{arm}';root=tmp_path/label;root.mkdir()
            config=root/'config.json';config.write_text(json.dumps({'method':method,'arm':arm}))
            digest=sha256(config.read_bytes()).hexdigest()
            (root/'baseline_input.json').write_text(json.dumps({'config_sha256':digest}))
            (root/'baseline_result.json').write_text(json.dumps({'status':'completed','method':method,
                'selected_bundle':str(final.program.bundle_locator),'selected_tree_sha256':final.program.tree_sha256,
                'selected_artifact_id':final.artifact_id}))
            runs.append({'label':label,'method':method,'arm':arm,'output':str(root),'config':str(config),'config_sha256':digest})
    plan={'schema':'ns_literature_comparisons_v1','methods':list(METHOD_KEYS),'runs':runs,'seed':101,
          'terminal_manifest':str(terminal),'terminal_manifest_sha256':sha256(terminal.read_bytes()).hexdigest(),
          'initial_bundle':str(initial.program.bundle_locator),'initial_tree_sha256':initial.program.tree_sha256,
          'candidate_case_workers':4}
    path=tmp_path/'comparisons.json';path.write_text(json.dumps(plan))
    result=json.loads(export_terminal_manifest(path,tmp_path/'programs.json').read_text())
    assert len(result['programs'])==9
    plan['runs'][-1]=plan['runs'][-2]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='both distinct predeclared arms'):
        export_terminal_manifest(path,tmp_path/'invalid.json')
    assert not (tmp_path/'invalid.json').exists()
