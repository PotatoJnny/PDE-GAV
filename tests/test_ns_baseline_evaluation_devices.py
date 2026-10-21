"""Fixed selection borrows GPUs without changing plans or Agent ownership."""
from contextlib import contextmanager
import json
import sys

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    AgentInfrastructureError, CommandAgentProgram, InProcessAgentExecutionService,
    SubprocessAgentExecutionService,
)
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused, CampaignPauseControl
from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
from pde_agent_verifier_solver.plugins.ns.baselines.cli import allocated_evaluation_devices
from pde_agent_verifier_solver.sdk.evaluation_devices import current_evaluation_device_pool
from tests.test_ns_baselines import SequenceAgent, _artifact, _policy, _records


def test_mapping_uses_only_allocated_visible_devices_without_reordering_environment():
    env = {'CUDA_VISIBLE_DEVICES': 'GPU-extra,GPU-right,GPU-left'}
    before = dict(env)
    assert allocated_evaluation_devices(('GPU-left', 'GPU-right'), environment=env) == {
        'GPU-left': 'cuda:2', 'GPU-right': 'cuda:1'}
    assert env == before


@pytest.mark.parametrize('tokens,visible', [
    ((), '0,1'), (('0', '0'), '0,1'), (('0',), ''), (('0',), '-1'),
    (('0', '1'), '0'), (('0',), '0,0'), (('0',), '0, 1'),
])
def test_invalid_or_invisible_devices_are_rejected(tokens, visible):
    env = {'CUDA_VISIBLE_DEVICES': visible}
    with pytest.raises(ValueError):
        allocated_evaluation_devices(tokens, environment=env)
    assert env == {'CUDA_VISIBLE_DEVICES': visible}


class LocalAgentsWithRealLease(InProcessAgentExecutionService):
    def __init__(self, root, *, broken_cleanup=False):
        self.owner = SubprocessAgentExecutionService(
            root, hard_wall_seconds=30, evaluation_device_map={'GPU-left': 'cuda:0', 'GPU-right': 'cuda:1'})
        self.evaluation_device_map = self.owner.evaluation_device_map
        self.programs = tuple(CommandAgentProgram(
            (sys.executable, '-c', 'pass'), lambda *args: None, gpu_devices=(token,))
            for token in self.evaluation_device_map)
        self.pools = []
        self.broken_cleanup = broken_cleanup

    @contextmanager
    def reserve_evaluation_devices(self, programs):
        with self.owner.reserve_evaluation_devices(self.programs) as pool:
            self.pools.append(pool)
            if self.broken_cleanup:
                class BrokenResource:
                    def close(self):
                        raise RuntimeError('test cleanup failed')
                pool.get_resource('broken', BrokenResource)
            yield pool


def test_selection_captures_pool_reuses_truth_and_returns_devices_before_agent(tmp_path, monkeypatch):
    development, selection = _records()
    initial = _artifact(tmp_path, 0., 's0')
    candidates = [_artifact(tmp_path, .8, 'first'), _artifact(tmp_path, .9, 'second')]
    service = LocalAgentsWithRealLease(tmp_path / 'leases')
    captured = []
    original = HiddenVerifier.evaluate_incremental
    def evaluate(self, *args, **kwargs):
        pool = current_evaluation_device_pool()
        assert pool is self._case_sessions._evaluation_device_pool
        assert pool.devices == ('cuda:0', 'cuda:1')
        captured.append(pool)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(HiddenVerifier, 'evaluate_incremental', evaluate)
    class Agent(SequenceAgent):
        def run(self, context):
            assert current_evaluation_device_pool() is None
            assert all(pool.cleanup_confirmed for pool in service.pools)
            assert not service.owner._reserved_gpu_devices
            return super().run(context)
    result = run_ns_baseline(
        NSBaselineConfig('refinement', tmp_path / 'result', iterations=2),
        development_records=development, selection_records=selection,
        evaluator_policy=_policy(), agent_program=Agent(candidates),
        agent_execution=service, initial_candidate=initial,
        task_settings={'official_point_count': 4})
    assert result.selection_error == pytest.approx(.1)
    assert result.selected.program.tree_sha256 == candidates[-1].program.tree_sha256
    assert len(captured) == 3 and len({id(pool) for pool in captured}) == 3
    assert len(selection[0].truth.queries) == 1  # GPU scopes must not repeat truth queries.
    assert all(pool.cleanup_confirmed for pool in captured)
    assert current_evaluation_device_pool() is None
    events = [json.loads(line) for line in (tmp_path / 'result/evaluation_device_events.jsonl').read_text().splitlines()]
    assert [e['event_type'] for e in events] == ['evaluation_devices_borrowed', 'evaluation_devices_returned'] * 3
    assert all(e['cleanup_confirmed'] for e in events[1::2])


@pytest.mark.parametrize('failure', ['pause', 'cleanup'])
def test_selection_pause_or_cleanup_failure_releases_scope_and_stops(tmp_path, failure):
    development, selection = _records()
    service = LocalAgentsWithRealLease(tmp_path / 'leases', broken_cleanup=failure == 'cleanup')
    saved = []
    control = (CampaignPauseControl(saved.append, lambda *args: True)
               if failure == 'pause' else None)
    expected = CampaignPaused if failure == 'pause' else AgentInfrastructureError
    with pytest.raises(expected):
        run_ns_baseline(
            NSBaselineConfig('competitive_refinement', tmp_path / 'result', competition_rounds=1),
            development_records=development, selection_records=selection,
            evaluator_policy=_policy(), agent_program=SequenceAgent([]), agent_execution=service,
            initial_candidate=_artifact(tmp_path, 0., 'initial'), pause_control=control)
    assert current_evaluation_device_pool() is None
    assert not service.owner._reserved_gpu_devices
    assert service.pools[0].cleanup_confirmed is (failure == 'pause')
    if failure == 'cleanup':
        assert set(service.owner.quarantined_gpu_devices) == {'GPU-left', 'GPU-right'}
        assert json.loads((tmp_path / 'result/baseline_result.json').read_text())['status'] == 'failed'
    else:
        assert saved
