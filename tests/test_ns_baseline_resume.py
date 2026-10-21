"""The fixed-data control preserves completed Agent turns across processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_fixture(root, *, resume=False, stop=None, initial_factor=0.0):
    from tests.test_ns_baselines import _artifact, _policy, _records
    from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl, CampaignPaused
    from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
    from pde_agent_verifier_solver.engine.runtime import AgentOutcome
    from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
    from pde_agent_verifier_solver.sdk import CaseRecord
    from dataclasses import replace

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    development, selection = _records()
    selection = tuple(CaseRecord(replace(selection[0].case, case_id=f'selection-{i}'),
                                 type(selection[0].truth)()) for i in range(4))
    candidates = [_artifact(root, value, f'candidate-{i}') for i, value in enumerate((initial_factor, .3, .7, .85, .95, 1.))]
    calls = root / 'agent_calls.jsonl'
    class Agent:
        def run(self, context):
            index = (context.round_plan.round_index - 1) * 2 + (context.agent.agent_id == 'ns-critical-refiner')
            with calls.open('a') as stream:
                stream.write(json.dumps({'round': context.round_plan.round_index,
                                         'agent': context.agent.agent_id,
                                         'budget': context.round_plan.agent_budget_seconds}) + '\n')
            return AgentOutcome(candidates[index], round_log=f'notes-{index}')
    store = CheckpointStore(root / 'checkpoints', identity={'fixture': 'fixed-v1'})
    state = store.load() if resume else None
    current = {'value': state}
    def save(value):
        current['value'] = value
        store.save(value)
    def should_pause(stage, required):
        state = current['value'] or {}
        progress = state.get('competition', {})
        if stop == 'development_done':
            return stage == 'baseline_selection' and progress.get('phase') == 'development_done'
        if stop == 'round_done':
            return stage == 'baseline_development' and progress.get('next_round') == 2
        if stop == 'partial_selection' and progress.get('phase') == 'development_done':
            scored = state.get('hidden_progress', {}).get('cases', {})
            return stage == 'baseline_selection' and any(0 < len(rows) < len(selection) for rows in scored.values())
        return False
    control = CampaignPauseControl(save, should_pause, hidden_batch_size=2)
    try:
        result = run_ns_baseline(
            NSBaselineConfig('competitive_refinement', root / 'result', seed=101,
                             competition_rounds=3, agent_budget_seconds=14400),
            development_records=development, selection_records=selection, evaluator_policy=_policy(),
            agent_program=Agent(), agent_execution=InProcessAgentExecutionService(),
            initial_candidate=candidates[0], continuation=state, pause_control=control,
        )
        (root / 'outcome.json').write_text(json.dumps(dict(result.report)))
    except CampaignPaused:
        (root / 'outcome.json').write_text(json.dumps({'status': 'paused'}))


def _child(root, **options):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).parent) + os.pathsep + env.get('PYTHONPATH', '')
    reply = subprocess.run([sys.executable, '-c',
        'import sys,json; from test_ns_baseline_resume import run_fixture; run_fixture(sys.argv[1],**json.loads(sys.argv[2]))',
        str(root), json.dumps(options)], env=env, capture_output=True, text=True)
    assert reply.returncode == 0, reply.stdout + reply.stderr


@pytest.mark.parametrize('stop', ['development_done', 'partial_selection', 'round_done'])
def test_fixed_competition_new_process_resume_does_not_repeat_agent_turns(tmp_path, stop):
    baseline, resumed = tmp_path / 'continuous', tmp_path / 'resumed'
    _child(baseline)
    _child(resumed, stop=stop)
    assert json.loads((resumed / 'outcome.json').read_text())['status'] == 'paused'
    _child(resumed, resume=True)
    expected = json.loads((baseline / 'outcome.json').read_text())
    actual = json.loads((resumed / 'outcome.json').read_text())
    for field in ('status', 'selected_tree_sha256', 'selection_error', 'completed_agent_turns', 'unique_scored_candidates'):
        assert actual[field] == expected[field]
    assert actual['completed_agent_turns'] == 6
    def calls(path):
        return sorted(
            (json.loads(line)['round'], json.loads(line)['agent'], json.loads(line)['budget'])
            for line in (path / 'agent_calls.jsonl').read_text().splitlines())
    assert calls(baseline) == calls(resumed)
    for name in ('campaign_memory.json', 'candidate_history.json', 'baseline_turns.json', 'selection_results.json'):
        assert json.loads((baseline / 'result' / name).read_text()) == json.loads((resumed / 'result' / name).read_text())


def test_fixed_competition_rejects_interrupted_agent_budget(tmp_path):
    from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
    root = tmp_path / 'unsafe'
    _child(root, stop='development_done')
    store = CheckpointStore(root / 'checkpoints', identity={'fixture': 'fixed-v1'})
    state = store.load()
    state['competition']['phase'] = 'development_running'
    store.save(state)
    with pytest.raises(RuntimeError, match='interrupted Agent development budget'):
        run_fixture(root, resume=True)


@pytest.mark.parametrize('kind', ['direct', 'chained', 'attached_cleanup'])
def test_baseline_service_failure_is_terminal_and_cannot_resume(tmp_path, monkeypatch, kind):
    from tests.test_ns_baselines import _artifact, _policy, _records, SequenceAgent
    from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
    from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError
    from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
    from pde_agent_verifier_solver.plugins.ns.baselines import search
    failure = CandidateExecutionInfrastructureError('Controller Oracle failed')
    if kind != 'direct':
        wrapper = RuntimeError('outer error')
        if kind == 'chained':
            wrapper.__cause__ = failure
        else:
            wrapper._candidate_cleanup_errors = (failure,)
        failure = wrapper
    def evaluate(self, *args, **kwargs):
        raise failure
    monkeypatch.setattr(search.HiddenVerifier, 'evaluate', evaluate)
    development, selection = _records()
    initial = _artifact(tmp_path, 0., 'initial')
    agent = SequenceAgent([])
    root = tmp_path / 'failed'
    kwargs = dict(config=NSBaselineConfig('competitive_refinement', root, competition_rounds=1),
                  development_records=development, selection_records=selection,
                  evaluator_policy=_policy(), agent_program=agent,
                  agent_execution=InProcessAgentExecutionService(), initial_candidate=initial)
    with pytest.raises(type(failure)):
        run_ns_baseline(**kwargs)
    assert json.loads((root / 'baseline_result.json').read_text())['status'] == 'failed'
    assert not (root / 'selection_results.json').exists()
    assert not agent.contexts
    with pytest.raises(RuntimeError, match='terminal failed baseline'):
        run_ns_baseline(**kwargs, continuation={'initial_key': initial.program.tree_sha256})


def test_fixed_resume_rejects_changed_initial_program(tmp_path):
    root = tmp_path / 'changed_initial'
    _child(root, stop='development_done')
    before = (root / 'agent_calls.jsonl').read_bytes()
    with pytest.raises(ValueError, match='initial Candidate hash differs'):
        run_fixture(root, resume=True, initial_factor=.1)
    assert (root / 'agent_calls.jsonl').read_bytes() == before


def test_cli_resume_costs_include_all_segments_and_completed_resume_is_idempotent(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from threading import Event
    from tests.test_ns_baselines import SOURCE, _write_numerical_data
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    from pde_agent_verifier_solver.plugins.ns.baselines.cli import run_from_config

    dataset = _write_numerical_data(tmp_path / 'data')
    worker = tmp_path / 'worker.py'
    worker.write_text('''from pathlib import Path
from pde_agent_verifier_solver.engine.agent_execution import submit_agent_nomination, write_agent_worker_response
Path("submission.py").write_text(SOURCE)
submit_agent_nomination({"bundle_path": "submission.py"}, sequence=1)
write_agent_worker_response(observations=["Local worker, no model call."])
'''.replace('SOURCE', repr(SOURCE.replace('FACTOR', '1.0'))))
    config = tmp_path / 'cli.json'
    config.write_text(json.dumps({
        'schema': 'ns_fixed_data_baseline_v1', 'method': 'competitive_refinement',
        'dataset': {'manifest': str(dataset), 'development_split': 'train', 'selection_split': 'validation'},
        'search': {'competition_rounds': 2},
        'agent': {'command': ['{python}', str(worker)], 'hard_wall_seconds': 20},
        'evaluation': {'selection_point_count': 4, 'candidate_case_workers': 1,
                       'candidate_case_hard_wall_seconds': 10},
    }))
    class Segment:
        def __init__(self, stop):
            self.stop = stop
            self.cancel = Event()
            self.round_starts = 0
        def should_pause(self, stage, required_seconds=0):
            if stage == 'baseline_development':
                self.round_starts += 1
                return self.stop and self.round_starts == 2
            return False
        @contextmanager
        def signals(self):
            yield
    output = tmp_path / 'cli_output'
    from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
    original_save = CheckpointStore.save
    observed = []
    def checked_save(store, value):
        if value.get('competition', {}).get('phase') == 'development_done':
            journal = output / 'execution_audits.jsonl'
            persisted = [json.loads(line) for line in journal.read_text().splitlines()]
            snapshot = output / 'execution_audits.json'
            old_rows = json.loads(snapshot.read_text()) if snapshot.exists() else []
            observed.append((len(persisted), len(old_rows)))
            assert len(persisted) >= 2
        return original_save(store, value)
    monkeypatch.setattr(CheckpointStore, 'save', checked_save)
    with pytest.raises(CampaignPaused):
        run_from_config(config, output, segment_budget=Segment(True))
    first_audits = json.loads((output / 'execution_audits.json').read_text())
    assert len(first_audits) == 2
    assert json.loads((output / 'costs.json').read_text())['agent_executions'] == 2
    result = run_from_config(config, output, resume=True, segment_budget=Segment(False))
    assert result.report['completed_agent_turns'] == 4
    audits = json.loads((output / 'execution_audits.json').read_text())
    candidate_audits = json.loads((output / 'candidate_execution_audits.json').read_text())
    costs = json.loads((output / 'costs.json').read_text())
    assert audits[:2] == first_audits
    assert (2, 0) in observed and (4, 2) in observed  # fsynced before checkpoint, ahead of finally JSON.
    assert [json.loads(line) for line in (output / 'execution_audits.jsonl').read_text().splitlines()] == audits
    assert [json.loads(line) for line in (output / 'candidate_execution_audits.jsonl').read_text().splitlines()] == candidate_audits
    assert len(audits) == costs['agent_executions'] == 4
    assert len(candidate_audits) == costs['candidate_executions'] == 1
    assert costs['agent_wall_seconds_sum'] == pytest.approx(sum(row['wall_seconds'] for row in audits))
    assert costs['candidate_wall_seconds_sum'] == pytest.approx(sum(row['wall_seconds'] for row in candidate_audits))
    again = run_from_config(config, output, resume=True, segment_budget=Segment(False))
    assert dict(again.report) == dict(result.report)
    assert json.loads((output / 'execution_audits.json').read_text()) == audits
    assert json.loads((output / 'costs.json').read_text()) == costs


def test_refinement_does_not_swallow_wrapped_controller_failure_as_bad_proposal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.test_ns_baselines import _artifact, _policy, _records
    from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
    from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError
    from pde_agent_verifier_solver.engine.runtime import AgentOutcome
    from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
    from pde_agent_verifier_solver.plugins.ns.baselines import search
    candidate = _artifact(tmp_path, 1., 'candidate')
    calls = []
    def propose(context):
        calls.append(context)
        return AgentOutcome(candidate)
    def evaluate(self, *args, **kwargs):
        try:
            raise CandidateExecutionInfrastructureError('Controller failed')
        except CandidateExecutionInfrastructureError as error:
            raise ValueError('wrapped service exception') from error
    monkeypatch.setattr(search.HiddenVerifier, 'evaluate', evaluate)
    development, selection = _records()
    root = tmp_path / 'refinement_failed'
    with pytest.raises(CandidateExecutionInfrastructureError, match='Controller fixed-bank'):
        run_ns_baseline(
            NSBaselineConfig('refinement', root, iterations=2),
            development_records=development, selection_records=selection,
            evaluator_policy=_policy(), agent_program=SimpleNamespace(run=propose),
            agent_execution=InProcessAgentExecutionService(),
        )
    assert len(calls) == 1
    assert not (root / 'selection_results.json').exists()
    assert json.loads((root / 'baseline_result.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('genuine_peer_cleanup_failure', [False, True])
def test_fixed_selection_peer_cancellation_preserves_failure_boundary(
    tmp_path, monkeypatch, genuine_peer_cleanup_failure,
):
    from threading import Event
    from types import SimpleNamespace
    from tests.test_ns_baselines import _artifact, _policy, _records
    from pde_agent_verifier_solver.engine import candidate_execution as execution
    from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl
    from pde_agent_verifier_solver.engine.runtime import AgentOutcome
    from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline, search

    initial = _artifact(tmp_path, .5, 'initial')
    rejected = _artifact(tmp_path, .1, 'timed-out')
    later = _artifact(tmp_path, 1., 'later-good')
    timeout = execution.CandidateExecutionTimeout('Candidate Case execution exceeded its hard wall')
    cancel = Event()
    cancel.set()
    with pytest.raises(execution.CandidateExecutionError) as captured:
        execution._raise_if_candidate_execution_cancelled(cancel, None, None)
    peer = captured.value
    if genuine_peer_cleanup_failure:
        execution._attach_execution_error(
            peer, execution.CandidateExecutionInfrastructureError('actual peer cleanup failed'),
        )
    execution._attach_execution_error(timeout, peer)
    calls = []
    original_evaluate = search.HiddenVerifier.evaluate_incremental

    def evaluate(verifier, artifact, *args, **kwargs):
        calls.append(artifact.program.tree_sha256)
        if artifact.program.tree_sha256 == rejected.program.tree_sha256:
            raise timeout
        return original_evaluate(verifier, artifact, *args, **kwargs)

    monkeypatch.setattr(search.HiddenVerifier, 'evaluate_incremental', evaluate)
    def propose(context):
        candidate = rejected if context.agent.agent_id == 'ns-broad-explorer' else later
        return AgentOutcome(candidate)

    development, selection = _records()
    output = tmp_path / 'result'
    state = {}
    control = CampaignPauseControl(lambda value: state.update(value), lambda *_: False)
    kwargs = dict(
        config=NSBaselineConfig('competitive_refinement', output, competition_rounds=1),
        development_records=development, selection_records=selection, evaluator_policy=_policy(),
        agent_program=SimpleNamespace(run=propose), agent_execution=InProcessAgentExecutionService(),
        initial_candidate=initial, pause_control=control,
    )
    if genuine_peer_cleanup_failure:
        with pytest.raises(execution.CandidateExecutionInfrastructureError, match='Controller fixed-bank'):
            run_ns_baseline(**kwargs)
        assert json.loads((output / 'baseline_result.json').read_text())['status'] == 'failed'
        assert later.program.tree_sha256 not in calls
    else:
        result = run_ns_baseline(**kwargs)
        assert result.selected.program.tree_sha256 == later.program.tree_sha256
        assert result.report['status'] == 'completed'
        assert calls == [initial.program.tree_sha256, rejected.program.tree_sha256,
                         later.program.tree_sha256]
        results = json.loads((output / 'selection_results.json').read_text())
        assert len(results) == 3
        failed = next(row for row in results if row['tree_sha256'] == rejected.program.tree_sha256)
        assert failed['failure'].startswith('CandidateExecutionTimeout:')
        assert failed['case_errors'] == []
        assert result.report['completed_agent_turns'] == 2
