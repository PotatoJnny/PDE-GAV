"""Real subprocess boundary tests with synthetic metadata; no model or CUDA calls."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    CommandAgentProgram,SubprocessAgentExecutionService,UnavailableAgentCapabilityFacade)
from pde_agent_verifier_solver.engine.campaign import AgentSpec,DevelopmentSamplingPolicy,RoundPlan
from pde_agent_verifier_solver.engine.memory import CampaignMemorySnapshot
from pde_agent_verifier_solver.engine.runtime import AgentContext
from pde_agent_verifier_solver.sdk import HiddenSuiteSpec,SampleSource

ROOT = Path(__file__).resolve().parents[1]
WORKER = '''
import json, os
from pde_agent_verifier_solver.engine.agent_execution import write_agent_worker_response
write_agent_worker_response(observations=(json.dumps({
    'pid':os.getpid(),
    'job':os.environ.get('SLURM_JOB_ID'),
    'order':os.environ.get('CUDA_DEVICE_ORDER'),
    'visible':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'unrelated_secret_inherited':'UNRELATED_PRIVATE_TOKEN' in os.environ,
    'scheduler_internal_inherited':'SLURM_JOB_NODELIST' in os.environ,
}),))
'''


@pytest.mark.parametrize('profile,parent,override,expected',[
    ('gpu',True,False,('424242','PCI_BUS_ID')),
    ('gpu',True,True,('424242','PCI_BUS_ID')),
    ('gpu',False,True,(None,None)),
    ('gpu',False,False,(None,None)),
    ('cpu',True,False,(None,None)),
    ('cpu',True,True,('declared-legacy-job','FASTEST_FIRST')),
    ('not-required',True,False,(None,None)),
    ('wrong-protocol',True,False,(None,None)),
])
def test_actual_worker_gets_only_current_opt_in_allocation_context(
        tmp_path,monkeypatch,profile,parent,override,expected):
    # These are clearly synthetic test inputs. No qualifier or NVIDIA command
    # is called, and these values are never supplied to a real experiment.
    for name,value in (('SLURM_JOB_ID','424242'),('CUDA_DEVICE_ORDER','PCI_BUS_ID')):
        if parent: monkeypatch.setenv(name,value)
        else: monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('UNRELATED_PRIVATE_TOKEN','synthetic-private-marker')
    monkeypatch.setenv('SLURM_JOB_NODELIST','synthetic-controller-only-hosts')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','controller-device-must-not-leak')
    environment = {'PYTHONPATH':str(ROOT/'src')}
    if override: environment.update(SLURM_JOB_ID='declared-legacy-job',CUDA_DEVICE_ORDER='FASTEST_FIRST')
    settings = {}
    if profile != 'cpu':
        settings['agent_gpu_execution'] = {'protocol':'shared_gpu_v1',
            'required':True,'sandbox_backend':'bwrap-nvidia-v1','runtime_sha256':'a'*64}
        if profile == 'not-required': settings['agent_gpu_execution']['required'] = False
        elif profile == 'wrong-protocol': settings['agent_gpu_execution']['protocol'] = 'unknown'
    def no_nomination(*_args): raise AssertionError('this worker never nominates')
    program = CommandAgentProgram((sys.executable,'-c',WORKER),no_nomination,
        environment=environment,worker_settings=settings,gpu_devices=('0',))
    agent = AgentSpec(agent_id='allocation-metadata',name='Synthetic metadata worker')
    context = AgentContext(round_plan=RoundPlan(round_index=1,agents=(agent,),agent_budget_seconds=8,
        development_sampling=DevelopmentSamplingPolicy(max_cases_per_agent_per_round=1,
            allowed_sources=(SampleSource.BASE,)),round_hidden=HiddenSuiteSpec(base=1)),
        agent=agent,prompt='Read only selected environment metadata.',evaluator=SimpleNamespace(cases=()),
        incumbent=None,memory=CampaignMemorySnapshot(),seed=1,candidate_solve_budget_seconds=900)
    service = SubprocessAgentExecutionService(tmp_path/'turns',hard_wall_seconds=8,
        enforce_active_budget=False,capability_facade=UnavailableAgentCapabilityFacade(),poll_interval_seconds=.01)
    outcome = service.execute(program,context,8)
    assert outcome.execution_status == 'completed' and len(outcome.observations) == 1
    received = json.loads(outcome.observations[0])
    assert (received['job'],received['order']) == expected
    assert received['pid'] != os.getpid()
    assert received['visible'] == '0'
    assert received['unrelated_secret_inherited'] is False
    assert received['scheduler_internal_inherited'] is False
    assert service.audits[-1].process_group_cleaned is True
