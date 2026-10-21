"""Unlimited cumulative sampling still counts Cases and respects access/time rules."""
from dataclasses import replace
from pathlib import Path

import pytest

from pde_agent_verifier_solver.app import ProductApplication, RunStatus
from pde_agent_verifier_solver.engine.agent_execution import SubprocessAgentExecutionService
from pde_agent_verifier_solver.engine.campaign import CampaignCompiler
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.sdk import (
    GeneratorCapabilities, PairedSampleRequest, PerturbCasesRequest,
    SampleCasesRequest, SampleSource,
)
from tests import test_case_sampler as sampler_fixtures
from tests import test_campaign_compiler as compiler_fixtures
from tests.test_agent_capability import _command_program, _run_values
from tests.test_round_development_recovery import _controller, _recovery
from tests.test_canonical_codex_agent import WORKER


def test_unlimited_compiler_and_sampler_pass_512_with_all_generated_sources():
    recipe = compiler_fixtures.CampaignCompilerTests().recipe()
    recipe = replace(recipe, default_round=replace(
        recipe.default_round,
        development_sampling=replace(recipe.default_round.development_sampling,
                                     max_cases_per_agent_per_round=None),
    ))
    compiled = CampaignCompiler().compile(recipe, GeneratorCapabilities(online=True))
    assert compiled.rounds[0].development_count is None
    assert compiled.rounds[0].case_mix_summary()['sample_limit'] is None
    generator = sampler_fixtures._Generator()
    sampler = sampler_fixtures.RoundCaseSamplerTests().sampler(max_cases=None, generator=generator)
    try:
        base = sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 257))
        guided = sampler.sample_cases(SampleCasesRequest(SampleSource.GUIDED, 256))
        children = sampler.perturb_cases(PerturbCasesRequest((base.cases[0].case_id,)))
        paired = sampler.sample_paired_cases(PairedSampleRequest(SampleSource.GUIDED, 1))
        assert all(batch.remaining is None for batch in (base, guided, children, paired))
        assert sampler.remaining is None and sampler.issued_count == 516
        assert len(sampler.generated_case_ids) == 516
        # Removing the cumulative quota does not fabricate more finite originals.
        with pytest.raises(ValueError):
            sampler.sample_cases(SampleCasesRequest(SampleSource.ORIGINAL, 5))
        assert sampler.issued_count == 516
        # Unowned parents remain inaccessible.
        with pytest.raises(ValueError, match='not issued'):
            sampler.perturb_cases(PerturbCasesRequest(('someone-elses-case',)))
    finally:
        sampler.close()
    assert not generator.live_case_ids
    with pytest.raises(RuntimeError, match='closed'):
        sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))


def test_unlimited_failed_sampling_keeps_atomic_accounting():
    generator = sampler_fixtures._Generator()
    sampler = sampler_fixtures.RoundCaseSamplerTests().sampler(max_cases=None, generator=generator)
    try:
        generator.return_wrong_count = True
        with pytest.raises(ValueError):
            sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 2))
        assert sampler.issued_count == 0 and sampler.remaining is None
        generator.return_wrong_count = False
        assert sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 2)).count == 2
    finally:
        sampler.close()


def test_unlimited_recovery_accepts_actual_counts_above_old_cap():
    controller, _ = _controller(rounds=1)
    plan = controller.campaign.rounds[0]
    controller.campaign = replace(controller.campaign, rounds=(replace(
        plan, development_sampling=replace(plan.development_sampling,
                                          max_cases_per_agent_per_round=None),
    ),))
    result = controller.run(recovery=_recovery(agent_case_counts={'agent-a': {'base': 513}}))
    assert result.rounds[0].agent_case_counts['agent-a']['base'] == 513


def test_unlimited_prompt_is_explicit_and_keeps_time_budget():
    prompt = WORKER._research_prompt({
        'agent': {'prompt': 'Develop a solver.'},
        'round': {'round_index': 1, 'hard_wall_seconds': 14400,
                  'development_sampling': {'max_cases': None, 'allowed_sources': ['base', 'guided']}},
    })
    assert 'no cumulative Development Case-count limit' in prompt
    assert 'Case cap is None' not in prompt
    assert '14400 seconds of total elapsed time' in prompt
    assert 'counts toward this same deadline' in prompt


def test_unlimited_panel_runs_with_toy_agent():
    panel = build_toy_panel()
    launched = panel.create_and_launch({
        'round_count': 1, 'max_sampled_cases_per_agent_per_round': None,
        'agent_budget_seconds': 30,
    })
    panel.app.wait(launched['run_id'], timeout=10)
    page = panel.run_page(launched['run_id'])
    assert page['status'] == 'completed', page.get('error')
    assert page['plan'][0]['sample_limit'] is None
    assert page['rounds'][0]['agent_case_counts']


def test_unlimited_crosses_rpc_boundary_and_records_actual_usage(tmp_path):
    worker = tmp_path / 'sample_worker.py'
    worker.write_text('''from pde_agent_verifier_solver.engine.agent_execution import load_agent_worker_request, submit_agent_nomination, write_agent_worker_response
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
from pde_agent_verifier_solver.sdk import SampleSource
request = load_agent_worker_request()
assert request['round']['development_sampling']['max_cases'] is None
client = ControllerAgentCapabilityClient.from_request(request)
base = client.sample_base(513)
assert base.remaining is None
paired = client.sample_paired_cases(SampleSource.BASE, 1)
assert paired.remaining is None
child = client.perturb_cases((base.cases[0].case_id,))
assert child.remaining is None
assert len(client.current_cases()) == 517
response = client.evaluate({'mode': 2, 'suffix': 'probe'}, case_ids=(base.cases[0].case_id,))
assert response.cases[0].truth is None
submit_agent_nomination({'mode': 3, 'suffix': 'final'}, sequence=1)
write_agent_worker_response(observations=('Sampled above 512 without a cumulative quota.',))
''')
    service = SubprocessAgentExecutionService(tmp_path / 'turns', hard_wall_seconds=30)
    panel = build_toy_panel()
    panel.programs.agents['unlimited-agent'] = _command_program(worker)
    panel.app = ProductApplication.local_development(agent_execution=service)
    values = {**_run_values('unlimited-agent', allow_point_truth=False),
              'max_sampled_cases_per_agent_per_round': None, 'agent_budget_seconds': 30}
    launched = panel.create_and_launch(values)
    session = panel.app.wait(launched['run_id'], timeout=40)
    audit = service.audits[-1]
    stderr = Path(audit.stderr_path).read_text()
    assert session.status is RunStatus.COMPLETED, (session.error, stderr)
    assert audit.status == 'completed', stderr
    counts = session.result.rounds[0].agent_case_counts['unlimited-agent']
    assert sum(counts.values()) == 517 and counts['base_perturbed'] == 2
    assert audit.capability_revoked and audit.capability_drained
