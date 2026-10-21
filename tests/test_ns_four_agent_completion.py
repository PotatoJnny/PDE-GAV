"""Agreed AutoPDE/SHARP behavior through the actual NS host and CPU evaluator."""
import json
from pathlib import Path

import pytest

from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl, CampaignPaused
from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
from tests.test_ns_baselines import _artifact, _records, _policy, SequenceAgent


def settings():
    return dict(protocol='completion_v2', total_wall_seconds=None, branch_wall_seconds=None,
                planning_call_seconds=None, autopde_attempts=1, autopde_review_seconds=None,
                sharp_samples=4, sharp_judges=3, sharp_cycles=1, sharp_hybrid_rounds=2)


def run(root, method, agent, **kw):
    development, selection = _records()
    return run_ns_baseline(NSBaselineConfig(method, root, agent_budget_seconds=None,
        literature_options=settings()), development_records=development, selection_records=selection,
        evaluator_policy=_policy(), task_settings={'official_point_count':4}, agent_program=agent,
        agent_execution=InProcessAgentExecutionService(), **kw)


def info(context, path='baseline_context.json'):
    return json.loads(next(r.data for r in context.public_problem_bundle.resources if r.relative_path==path))


class Environment:
    description={'arm':'continuous_online_base'}
    def bind(self, state): self.progress=state
    def records(self, branch): return ()
    def sampler(self, branch, evaluator, seed):
        return type('Sampler', (), {'remaining':None})()


@pytest.mark.parametrize('online',[False, True])
def test_autopde_delivers_review_even_when_first_pass_and_s0_score_better(tmp_path, online):
    first, review, s0 = [_artifact(tmp_path, v, n) for v,n in ((.95,'first'),(.8,'review'),(1.,'s0'))]
    agent=SequenceAgent([first,review])
    result=run(tmp_path/'run','autopde_ns_adaptation',agent,initial_candidate=s0,
               development_environment=Environment() if online else None)
    assert result.selected.program.tree_sha256==review.program.tree_sha256
    assert result.selection_error==pytest.approx(.2)
    assert [c.agent.name for c in agent.contexts]==['autopde develop','autopde review']
    assert agent.contexts[0].incumbent is None
    assert agent.contexts[1].incumbent.program.tree_sha256==first.program.tree_sha256
    assert info(agent.contexts[1])['feedback']['tree_sha256']==first.program.tree_sha256
    assert info(agent.contexts[1])['feedback']['error']==pytest.approx(.05)
    assert 'do not redesign' in agent.contexts[1].prompt
    assert len(info(agent.contexts[1], 'baseline_branch_history.json'))==1
    assert all(c.round_plan.agent_budget_seconds is None for c in agent.contexts)
    assert all((c.sampler is not None)==online for c in agent.contexts)
    if online:
        assert all(c.round_plan.development_sampling.max_cases_per_agent_per_round is None for c in agent.contexts)
    report=json.loads((result.output_directory/'literature_search.json').read_text())
    assert report['records']['autopde_delivery']['selection']=='single_review_output_not_validation_argmin'


def test_missing_autopde_review_is_failure_not_unreviewed_or_s0_success(tmp_path):
    class Agent(SequenceAgent):
        def run(self, context):
            if context.agent.name=='autopde review':
                self.contexts.append(context)
                return AgentOutcome(round_log='No submission')
            return super().run(context)
    agent=Agent([_artifact(tmp_path,.95,'first')])
    with pytest.raises(RuntimeError,match='no candidate'):
        run(tmp_path/'run','autopde_ns_adaptation',agent,initial_candidate=_artifact(tmp_path,1.,'s0'))
    records=json.loads((tmp_path/'run/literature_search.json').read_text())['records']
    assert records['autopde_delivery']['failure']=='missing_review_submission'
    assert len(agent.contexts)==2


class SharpAgent:
    def __init__(self, root, duplicate=False):
        values=(.5,.6,.7,.99,.95,.9,.85,.8,.82,.84)
        self.artifacts=[_artifact(root,v,f'p{i}') for i,v in enumerate(values)]
        self.contexts=[]
        self.codes=self.judges=0
        self.duplicate=duplicate
    def run(self, context):
        self.contexts.append(context)
        phase=context.agent.name
        if phase=='sharp analysis':
            return AgentOutcome(round_log=json.dumps({k:'Applicable NS analysis' for k in
                ('classification','analytic','transformation','decomposition','stability')}))
        if phase=='sharp judge':
            pool=info(context)['seed_implementations']
            assert len(pool)==4
            assert all(r['validation_error'] is None for r in pool)
            hashes=[r['tree_sha256'] for r in pool]
            i=0 if self.duplicate else self.judges
            self.judges+=1
            return AgentOutcome(round_log=json.dumps({'selected':[hashes[i],hashes[(i+1)%4]],
                'nominee':hashes[i],'rationale':'Inspect the full code pool'}))
        artifact=self.artifacts[self.codes];self.codes+=1
        return AgentOutcome(artifact,round_log='Recorded numerical change and its rationale')


@pytest.mark.parametrize('online',[False,True])
def test_sharp_unranked_pool_feedback_barrier_and_best_eligible_version(tmp_path,online):
    agent=SharpAgent(tmp_path)
    result=run(tmp_path/'run','pde_sharp_ns_adaptation',agent,
               initial_candidate=_artifact(tmp_path,1.,'perfect-s0'),
               development_environment=Environment() if online else None)
    assert len(agent.contexts)==14
    assert agent.contexts[0].agent.name=='sharp analysis'
    genesis=[c for c in agent.contexts if c.agent.name=='sharp genesis']
    assert len(genesis)==4 and all(c.incumbent is None for c in genesis)
    assert all(not info(c)['seed_implementations'] for c in genesis)
    hybrids=[c for c in agent.contexts if c.agent.name=='sharp hybrid']
    assert len(hybrids)==6
    feedback=[info(c)['feedback'] for c in hybrids]
    assert feedback[0]==feedback[1]==feedback[2] and len(feedback[0])==3
    assert feedback[3]==feedback[4]==feedback[5] and len(feedback[3])==6
    assert all(c.round_plan.agent_budget_seconds is None for c in agent.contexts)
    assert all((c.sampler is not None)==online for c in agent.contexts)
    # Unnominated Genesis (.99), perfect S0, and the worse last round cannot win.
    assert result.selected.program.tree_sha256==agent.artifacts[4].program.tree_sha256
    assert result.selection_error==pytest.approx(.05)
    reports=json.loads((result.output_directory/'literature_evaluations.json').read_text()).values()
    candidate_reports=[r for r in reports if r['tree_sha256']!=result.report.get('initial_tree_sha256')]
    hashes={r['tree_sha256'] for r in reports}
    assert agent.artifacts[3].program.tree_sha256 not in hashes
    assert len(hashes-{_artifact(tmp_path,1.,'perfect-s0').program.tree_sha256})==9


def test_sharp_repeated_nominees_reuse_validation(tmp_path):
    agent=SharpAgent(tmp_path,duplicate=True)
    result=run(tmp_path/'run','pde_sharp_ns_adaptation',agent)
    reports=json.loads((result.output_directory/'literature_evaluations.json').read_text())
    assert len(reports)==7  # one nominee and six different hybrids
    assert result.selection_error==pytest.approx(.05)


@pytest.mark.parametrize('method,completed_calls',[('autopde_ns_adaptation',1),('pde_sharp_ns_adaptation',9)])
def test_completion_method_resume_keeps_committed_roles_and_feedback(tmp_path,method,completed_calls):
    if method.startswith('autopde'):
        agent=SequenceAgent([_artifact(tmp_path,.95,'first'),_artifact(tmp_path,.8,'review')])
        expected_calls, error=2,.2
    else:
        agent=SharpAgent(tmp_path)
        expected_calls,error=14,.05
    store=CheckpointStore(tmp_path/'checkpoint',identity={'method':method})
    paused=False
    def should_pause(stage,seconds):
        nonlocal paused
        if stage=='literature_agent' and len(agent.contexts)==completed_calls and not paused:
            paused=True
            return True
        return False
    control=CampaignPauseControl(store.save,should_pause,agent_start_margin_seconds=0,hidden_start_margin_seconds=0)
    with pytest.raises(CampaignPaused):
        run(tmp_path/'run',method,agent,pause_control=control)
    saved=store.load()
    assert len(saved['turns'])==completed_calls
    result=run(tmp_path/'run',method,agent,pause_control=control,continuation=saved)
    assert len(agent.contexts)==expected_calls
    assert result.selection_error==pytest.approx(error)


def test_autopde_completion_rejects_silent_old_limits(tmp_path):
    with pytest.raises(ValueError,match='one attempt'):
        NSBaselineConfig('autopde_ns_adaptation',tmp_path,agent_budget_seconds=None,
            literature_options={**settings(),'autopde_attempts':6})


def test_prepared_arm_entry_preserves_paths_limits_and_resume(tmp_path):
    import importlib.util
    from hashlib import sha256
    path=Path(__file__).resolve().parents[1]/'scripts/run_ns_literature_arm.py'
    spec=importlib.util.spec_from_file_location('four_agent_arm_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=tmp_path/'config with spaces.json';config.write_text('{}')
    plan=tmp_path/'comparisons.json'
    plan.write_text(json.dumps({'schema':'ns_literature_comparisons_v1','protocol':'completion_v2','runs':[
        {'label':'autopde_fixed_sampled','config':str(config),'config_sha256':sha256(config.read_bytes()).hexdigest(),
         'output':str(tmp_path/'run with spaces')}]}))
    calls=[]
    def execute(args):
        calls.append(args)
        return 75
    assert module.run_arm(plan,'autopde_fixed_sampled',gpu_token='GPU-allocated',resume=True,baseline_main=execute)==75
    args=calls[0]
    assert args[args.index('--config')+1]==str(config)
    assert args[args.index('--gpu-token')+1]=='GPU-allocated'
    assert args[args.index('--segment-seconds')+1]=='10800'
    assert '--resume' in args
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='configuration changed'):
        module.run_arm(plan,'autopde_fixed_sampled',gpu_token='0',baseline_main=execute)
    assert len(calls)==1


@pytest.mark.parametrize('status,code',[('paused',75),('completed',0)])
def test_portable_terminal_entry_delegates_to_resumable_frozen_lane(tmp_path,status,code):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/run_ns_literature_terminal.py'
    spec=importlib.util.spec_from_file_location('four_agent_terminal_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan=tmp_path/'plan.json'
    calls=[]
    def runner(actual,lane,*,budget,evaluation_devices):
        calls.append((actual,lane,evaluation_devices))
        assert budget is not None
        return {'status':status,'lane':lane}
    assert module.run_terminal(plan,2,runner=runner)==code
    assert calls==[(plan,2,('cuda:0',))]
