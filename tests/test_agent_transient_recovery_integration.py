"""Two native worker fixtures recover inside one real Controller research turn.

No model service or physical PDE assets are involved. The only timing shortcut
is replacing the launcher's retry backoff in a private test entry point.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityFacade
from pde_agent_verifier_solver.engine.agent_execution import SubprocessAgentExecutionService
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignCompiler, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.plugins.toy import ToyPoissonGenerator, ToyPoissonProblem, ToySolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, DatasetInput, DatasetKind, HiddenSuiteSpec, OfficialEvaluationSpec, SampleSource,
    TerminalCohortSpec, TerminalEvaluationSpec, TerminalMetricSpec,
)
from tests.test_canonical_codex_agent import command_program, write_auth
from tests.test_canonical_codex_resume import _write_resumable_codex


ROOT = Path(__file__).resolve().parents[1]


def _workers(root, failures):
    native = _write_resumable_codex(root, failures=failures)
    source = native.read_text()
    source = source.replace(
        'prompt = sys.stdin.read()',
        '''prompt = sys.stdin.read()
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
client = ControllerAgentCapabilityClient.from_request(request)
cases_before = [case.case_id for case in client.current_cases()]
case_journal = workspace / 'retained-cases.json'
if resumed:
    assert set(json.loads(case_journal.read_text())) <= set(cases_before)
if attempt == 1 or attempt == FINAL_ATTEMPT:
    batch = client.sample_base(1)
    remaining = batch.remaining
else:
    remaining = None
cases_after = [case.case_id for case in client.current_cases()]
case_journal.write_text(json.dumps(cases_after))
with (workspace / 'case-allowance.jsonl').open('a') as stream:
    stream.write(json.dumps({'before': cases_before, 'after': cases_after,
        'remaining': remaining, 'hard_wall_seconds': request['round']['hard_wall_seconds'],
        'at_monotonic': time.monotonic()}) + '\\n')'''.replace('FINAL_ATTEMPT', str(len(failures) + 1)),
    )
    source = source.replace(
        f'failures = {list(failures)!r}',
        f"failures = {list(failures)!r} if request['agent']['agent_id'] == 'recovering' else []",
    )
    source = source.replace(
        "print(json.dumps({'type':'turn.completed','usage':{}}), flush=True)",
        "(workspace / 'native-completed.json').write_text(json.dumps({'attempt': attempt}))\n"
        "print(json.dumps({'type':'turn.completed','usage':{}}), flush=True)",
    )
    native.write_text(source)
    entry = root / 'fast-backoff-worker.py'
    entry.write_text(
        'import sys\n'
        f'sys.path.insert(0, {str(ROOT / "scripts")!r})\n'
        'import run_canonical_codex_agent as worker\n'
        'worker._resume_backoff_seconds = lambda attempt, failure: 2.0 if attempt == 1 else 0.02\n'
        'raise SystemExit(worker.main())\n'
    )
    auth, original_auth = write_auth(root)
    runtime = root / 'runtime'
    runtime.mkdir()
    programs = {}
    for name in ('recovering', 'peer'):
        program = command_program(native, auth, runtime, gpu_devices=())
        programs[name] = replace(program,
            command=(sys.executable, str(entry), *program.command[2:], '--cpu-only'),
            worker_settings={'nomination_policy': 'latest', 'transient_resume_max_attempts': 1, 'transient_resume_until_deadline': True})
    return programs, auth, original_auth


def _json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_controller_waits_for_same_session_recovery_without_restarting_peer_or_case_budget(tmp_path):
    # Four successive failures exceeds the old three-attempt limit. A committed
    # nomination on each failed attempt must remain work in progress throughout.
    failures = ('Selected model is at capacity', 'stream disconnected',
                'service unavailable', 'network error')
    programs, auth, original_auth = _workers(tmp_path, failures)
    wall = 180
    service = SubprocessAgentExecutionService(tmp_path / 'turns', hard_wall_seconds=wall,
        capability_facade=ControllerAgentCapabilityFacade(), poll_interval_seconds=.01,
        termination_grace_seconds=1.)
    problem, generator = ToyPoissonProblem(), ToyPoissonGenerator()
    dataset = problem.prepare_dataset(DatasetInput('retry-integration', DatasetKind.NUMERICAL,
        settings={'development_cases': 4, 'hidden_cases': 4}))
    recipe = CampaignRecipe(name='retry-integration', round_count=1,
        agents=tuple(AgentSpec(name, name) for name in programs), master_seed=101,
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=False),
        evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=('u',), x=17)),
        terminal_evaluation=TerminalEvaluationSpec(
            cohorts=(TerminalCohortSpec('clean', HiddenSuiteSpec(base=2)),),
            ranking=TerminalMetricSpec('clean', 'base')),
        default_round=RoundSettings(agent_budget_seconds=wall,
            development_sampling=DevelopmentSamplingPolicy(max_cases_per_agent_per_round=4,
                allowed_sources=(SampleSource.BASE,), starter=CaseCounts(base=1)),
            official_evaluation=OfficialEvaluationSpec(HiddenSuiteSpec(base=2))))
    controller = CampaignController(CampaignCompiler().compile(recipe, generator.capabilities),
        dataset, problem, ToySolverTask(), generator, programs, agent_execution=service)
    official = controller._official_reports
    official_calls = []

    def after_native_completion(*args, **kwargs):
        audits = {audit.agent_id: audit for audit in service.audits}
        assert set(audits) == {'recovering', 'peer'}
        assert all(audit.status == 'completed' for audit in audits.values()), {
            name: (audit.status, audit.error) for name, audit in audits.items()}
        for audit in audits.values():
            assert (Path(audit.workspace) / 'native-completed.json').is_file()
        official_calls.append(kwargs['seed_namespace'])
        return official(*args, **kwargs)

    controller._official_reports = after_native_completion
    result = controller.run()
    assert official_calls[0] == 'official_round:1'
    assert len(official_calls) == 2  # Round Official and terminal cohort.
    audits = {audit.agent_id: audit for audit in service.audits}
    assert len(service.audits) == 2
    recovering = Path(audits['recovering'].workspace)
    peer = Path(audits['peer'].workspace)
    starts = _json_lines(recovering / 'native-attempt-starts.jsonl')
    peer_starts = _json_lines(peer / 'native-attempt-starts.jsonl')
    assert len(starts) == 5 and len(peer_starts) == 1
    assert [row['resumed'] for row in starts] == [False, True, True, True, True]
    assert len({row['execution_id'] for row in starts}) == 1
    assert len({row['thread_id'] for row in starts}) == 1
    assert [row['nomination_count_before'] for row in starts] == list(range(5))
    recovery = json.loads((recovering / 'codex-recovery.json').read_text())
    assert recovery['status'] == 'completed'
    assert [row['decision'] for row in recovery['attempts']] == ['retry'] * 4 + ['completed']
    remaining = [row['remaining_seconds'] for row in recovery['attempts']]
    assert all(0 < later < earlier <= wall for earlier, later in zip(remaining, remaining[1:]))
    assert audits['recovering'].hard_wall_seconds == wall
    assert audits['recovering'].wall_seconds < wall
    cases = _json_lines(recovering / 'case-allowance.jsonl')
    assert [len(row['before']) for row in cases] == [1, 2, 2, 2, 2]
    assert [row['remaining'] for row in cases] == [2, None, None, None, 1]
    assert all(row['hard_wall_seconds'] == wall for row in cases)
    assert result.rounds[0].agent_case_counts['recovering']['base'] == 3
    assert result.rounds[0].agent_case_counts['peer']['base'] == 2
    nominees = {row.artifact.agent_id: row.artifact for row in result.rounds[0].candidate_evaluations}
    assert nominees['recovering'].metadata['max_resolved_mode'] == 5
    assert nominees['peer'].metadata['max_resolved_mode'] == 1
    events = controller.events.snapshot()
    finished = [event for event in events if event.event_type == 'agent_finished']
    assert len(finished) == 2 and {event.payload['agent_id'] for event in finished} == set(programs)
    assert not any(event.event_type == 'agent_failed' for event in events)
    recovery_events = [event for event in events
                       if event.event_type in {'agent_recovering', 'agent_resumed'}]
    assert {event.event_type for event in recovery_events} == {'agent_recovering', 'agent_resumed'}
    assert {event.payload['agent_id'] for event in recovery_events} == {'recovering'}
    assert all(event.payload['attempt'] > 1 for event in recovery_events
               if event.event_type == 'agent_resumed')
    assert auth.read_text() == original_auth
    assert not list((tmp_path / 'runtime').iterdir())
