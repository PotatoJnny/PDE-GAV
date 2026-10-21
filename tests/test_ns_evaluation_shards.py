"""Real CPU solver execution, frozen plans and exact coverage; no Slurm or GPU."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def shards():
    return module(ROOT / 'scripts/ns_evaluation_shards.py', 'ns_shards_test')


@pytest.fixture
def fixture():
    return module(Path(__file__).with_name('test_ns_common_evaluation.py'), 'ns_common_fixture')


class Unbounded:
    def __init__(self):
        self.cancel = Event()

    def should_pause(self, stage, required_seconds=0):
        return False


class PauseOne(Unbounded):
    def __init__(self):
        super().__init__()
        self.started = 0

    def should_pause(self, stage, required_seconds=0):
        if stage == 'common_evaluation':
            self.started += 1
        return self.started > 1


def science(rows):
    return sorted([{key: value for key, value in row.items() if not key.endswith('_seconds')}
                   for row in rows], key=lambda row: (row['artifact_id'], row['case_id'], row['metric']))


def test_shards_reuse_pause_resume_merge_matches_uninterrupted(tmp_path, fixture, shards, monkeypatch):
    manifest, spec = fixture._manifest(tmp_path)
    spec['budget']['max_workers'] = 1
    fixture._write(manifest, spec)
    old = tmp_path / 'old'
    original = shards.run_ns_common_evaluation(manifest, old, segment_budget=PauseOne())
    assert original['n_completed_program_cases'] == 1
    old_checkpoint = (old / 'checkpoint.json').read_bytes()
    old_rows = fixture._rows(old)
    uninterrupted = tmp_path / 'uninterrupted'
    reference = shards.run_ns_common_evaluation(manifest, uninterrupted)
    output = tmp_path / 'split'
    # Preparation must never invoke a solver or Slurm.
    import subprocess
    from pde_agent_verifier_solver.engine.candidate_execution import SubprocessCandidateExecutionService
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('unexpected external command in prepare'))
        patch.setattr(SubprocessCandidateExecutionService, 'execute_cases', lambda *a, **k: pytest.fail('solver in prepare'))
        prepared = shards.prepare(manifest, output, reuse=old, shard_size=1, lanes=3, workers=2)
    assert prepared['n_reused_program_cases'] == 1
    assert sorted(i for s in prepared['shards'] for i in s['indices']) == list(range(1, 6))
    assert shards.checkpoint(output / 'aggregate')[0].rows == {0: old_rows}
    commands = shards.submit_commands(output / 'plan.json', ROOT / '.venv/bin/python')
    assert len(commands) == 3
    assert all(command[0] == 'sbatch' for command in commands)
    with pytest.raises((ValueError, FileNotFoundError)):
        shards.merge(output / 'plan.json')
    paused = shards.run_lane(output / 'plan.json', 0, budget=PauseOne())
    assert paused['status'] == 'paused' and paused['n_completed_program_cases'] == 1
    first = output / 'shards' / '0000' / 'result' / 'completed_cases' / '00000000.json'
    committed = first.read_bytes()
    # Concurrent lanes must use disjoint outputs. A paused lane resumes too.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(shards.run_lane, output / 'plan.json', lane, budget=Unbounded())
                   for lane in range(3)]
        assert all(f.result()['status'] == 'completed' for f in futures)
    assert first.read_bytes() == committed
    with monkeypatch.context() as patch:
        patch.setattr(SubprocessCandidateExecutionService, 'execute_cases', lambda *a, **k: pytest.fail('solver rerun in merge'))
        result = shards.merge(output / 'plan.json')
        again = shards.merge(output / 'plan.json')
    assert result['groups'] == reference['groups'] == again['groups']
    assert result['n_failed_program_cases'] == 1
    assert science(fixture._rows(output / 'aggregate')) == science(fixture._rows(uninterrupted))
    assert (old / 'checkpoint.json').read_bytes() == old_checkpoint
    audits = list(output.glob('shards/*/result/execution/candidate_execution_audits.jsonl'))
    assert sum(len(path.read_text().splitlines()) for path in audits) == 5


def test_multiple_cohorts_preserve_numerical_seed_offsets(tmp_path, fixture, shards):
    manifest, spec = fixture._manifest(tmp_path, include_partial=False)
    spec['programs'] = spec['programs'][1:]
    spec['datasets'].append({**spec['datasets'][0], 'cohort': 'second', 'dataset_id': 'second'})
    fixture._write(manifest, spec)
    expected = tmp_path / 'reference'
    baseline = shards.run_ns_common_evaluation(manifest, expected)
    output = tmp_path / 'split'
    shards.prepare(manifest, output, shard_size=1, lanes=1, workers=1)
    shards.run_lane(output / 'plan.json', 0, budget=Unbounded())
    result = shards.merge(output / 'plan.json')
    assert result['groups'] == baseline['groups']
    assert science(fixture._rows(output / 'aggregate')) == science(fixture._rows(expected))


def test_cached_failure_is_preserved_without_rerun(tmp_path, fixture, shards, monkeypatch):
    manifest, spec = fixture._manifest(tmp_path)
    old = tmp_path / 'old'
    original = shards.run_ns_common_evaluation(manifest, old)
    output = tmp_path / 'all_reused'
    plan = shards.prepare(manifest, output, reuse=old)
    assert plan['shards'] == [] and plan['lanes'] == 0
    assert shards.submit_commands(output / 'plan.json', ROOT / '.venv/bin/python') == []
    from pde_agent_verifier_solver.engine.candidate_execution import SubprocessCandidateExecutionService
    monkeypatch.setattr(SubprocessCandidateExecutionService, 'execute_cases', lambda *a, **k: pytest.fail('reused case ran'))
    result = shards.merge(output / 'plan.json')
    assert result['groups'] == original['groups'] and result['n_failed_program_cases'] == 1


@pytest.mark.parametrize('change', ['row_bytes', 'case_id', 'plan_sha256', 'timeout', 'source'])
def test_reuse_rejects_corruption_and_scientific_changes(tmp_path, fixture, shards, change):
    manifest, spec = fixture._manifest(tmp_path, include_partial=False)
    spec['budget']['max_workers'] = 1
    fixture._write(manifest, spec)
    old = tmp_path / 'old'
    shards.run_ns_common_evaluation(manifest, old, segment_budget=PauseOne())
    cp = shards.read(old / 'checkpoint.json')
    if change in {'row_bytes', 'case_id', 'plan_sha256'}:
        row = old / cp['completed']['0']['path']
        values = shards.read(row)
        values[0]['case_id' if change == 'row_bytes' else change] = 'wrong'
        fixture._write(row, values)
        if change != 'row_bytes':
            cp['completed']['0']['sha256'] = sha256(row.read_bytes()).hexdigest()
    elif change == 'source':
        cp['identity']['evaluator_source_sha256'] = '0' * 64
    else:
        spec['budget']['case_hard_wall_seconds'] += 1
        fixture._write(manifest, spec)
    fixture._write(old / 'checkpoint.json', cp)
    with pytest.raises(ValueError, match='hash differs|cached program|identity differs'):
        shards.prepare(manifest, tmp_path / 'bad', reuse=old)


def test_changed_old_progress_and_duplicate_work_block_dispatch(tmp_path, fixture, shards):
    manifest, _ = fixture._manifest(tmp_path, include_partial=False)
    old = tmp_path / 'old'
    shards.run_ns_common_evaluation(manifest, old, segment_budget=shards.PrepareOnly())
    output = tmp_path / 'split'
    shards.prepare(manifest, output, reuse=old, shard_size=1, lanes=1)
    oldcp = old / 'checkpoint.json'
    before = oldcp.read_bytes()
    oldcp.write_bytes(before + b'\n')
    with pytest.raises(ValueError, match='advanced'):
        shards.run_lane(output / 'plan.json', 0, budget=Unbounded())
    oldcp.write_bytes(before)
    plan = shards.read(output / 'plan.json')
    plan['shards'][1]['indices'] = plan['shards'][0]['indices']
    fixture._write(output / 'plan.json', plan)
    with pytest.raises(ValueError, match='overlap'):
        shards.load_plan(output / 'plan.json')


def test_drain_stops_new_work_without_cancelling_inflight(shards):
    from pde_agent_verifier_solver.engine.checkpoint_store import RunSegmentBudget
    clock = [0.0]
    budget = RunSegmentBudget(10800, flush_seconds=600, monotonic=lambda: clock[0])
    try:
        drain = shards.DrainBudget(budget, 900)
        clock[0] = 9200
        assert not drain.should_pause('common_evaluation')
        clock[0] = 9240
        assert drain.should_pause('common_evaluation')
        assert not budget.cancel.is_set()
    finally:
        budget.close()


def test_interrupted_case_is_not_imported_as_solver_failure(tmp_path, fixture, shards):
    manifest, spec = fixture._manifest(tmp_path, include_partial=False)
    spec['programs'] = [spec['programs'][1]]
    spec['programs'][0]['bundle'] = str(fixture._bundle(tmp_path / 'sleeper', sleep=True))
    fixture._write(manifest, spec)
    output = tmp_path / 'split'
    shards.prepare(manifest, output, lanes=1, workers=1)
    from threading import Timer
    budget = Unbounded()
    timer = Timer(1.0, budget.cancel.set)
    timer.start()
    try:
        result = shards.run_lane(output / 'plan.json', 0, budget=budget)
    finally:
        timer.cancel()
        timer.join()
    assert result['status'] == 'paused'
    assert result['n_completed_program_cases'] == 0
    rows = list(output.glob('shards/*/result/completed_cases/*.json'))
    assert rows == []


def test_lane_binds_allocated_oracle_pool(tmp_path, fixture, shards, monkeypatch):
    manifest, spec = fixture._manifest(tmp_path, include_partial=False)
    spec['programs'] = spec['programs'][1:]
    fixture._write(manifest, spec)
    output = tmp_path / 'split'
    shards.prepare(manifest, output, lanes=1)
    from pde_agent_verifier_solver.sdk.evaluation_devices import current_evaluation_device_pool
    real = shards.run_ns_common_evaluation
    observed = []
    def run(*args, **kwargs):
        pool = current_evaluation_device_pool()
        assert pool.devices == ('qualification-device',)
        observed.append(pool)
        return real(*args, **kwargs)
    monkeypatch.setattr(shards, 'run_ns_common_evaluation', run)
    shards.run_lane(output / 'plan.json', 0, budget=Unbounded(), evaluation_devices=('qualification-device',))
    assert observed and all(pool.cleanup_confirmed for pool in observed)
    assert current_evaluation_device_pool() is None


@pytest.mark.parametrize('progress', [0, 1])
def test_slurm_runner_accounts_for_setup_and_stops_zero_progress(tmp_path, fixture, shards, monkeypatch, progress):
    import importlib
    import time
    import torch
    from pde_agent_verifier_solver.engine.checkpoint_store import RunSegmentBudget
    manifest, _ = fixture._manifest(tmp_path, include_partial=False)
    output = tmp_path / 'split'
    shards.prepare(manifest, output, lanes=1)
    monkeypatch.setenv('SLURM_JOB_ID', '301')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'allocated-gpu')
    monkeypatch.setenv('SLURM_JOB_END_TIME', str(int(time.time()) + 7200))
    for name in ('NS_CONTINUATION_RUN_ROOT', 'NS_CONTINUATION_SEGMENT_INDEX', 'NS_CONTINUATION_INTENT_ID'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: None)
    seconds = []
    def budget(duration, **kwargs):
        seconds.append(duration)
        return RunSegmentBudget(duration, **kwargs)
    monkeypatch.setattr(shards, 'RunSegmentBudget', budget)
    def lane(plan, index, **kwargs):
        assert kwargs['evaluation_devices'] == ('cuda:0',)
        result = {'status': 'paused', 'n_completed_program_cases': progress}
        shards.atomic_json(output / 'lanes/0/checkpoint.json', result)
        return result
    monkeypatch.setattr(shards, 'run_lane', lane)
    continuation = importlib.import_module('ns_slurm_continuation')
    commands = []
    def submit(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='302;cluster\n', stderr='')
    monkeypatch.setattr(continuation.subprocess, 'run', submit)
    if progress:
        result = shards.run_slurm(output / 'plan.json', 0, auto_continue=True)
        assert result['continuation_status'] == 'submitted'
        assert len(commands) == 1
        partitions = ','.join(f'gpubase_l40s_b{i}' for i in range(1, 6))
        initial, = shards.submit_commands(output / 'plan.json', ROOT / '.venv/bin/python')
        for item in ('--time=03:00:00', f'--partition={partitions}', '--gres=gpu:l40s:1',
                     '--cpus-per-task=8', '--mem=64G'):
            assert item in initial
            assert item in commands[0]
        for item in ('--dependency=afterok:301',):
            assert item in commands[0]
    else:
        with pytest.raises(RuntimeError, match='no completed Case'):
            shards.run_slurm(output / 'plan.json', 0, auto_continue=True)
        assert commands == []
        assert continuation.read_state(output / 'lanes/0')['status'] == 'failed'
    assert len(seconds) == 1 and 7190 < seconds[0] <= 7200


def test_resumed_lane_rejects_changed_plan_before_solver_execution(tmp_path, fixture, shards):
    manifest, _ = fixture._manifest(tmp_path, include_partial=False)
    output = tmp_path / 'split'
    shards.prepare(manifest, output, lanes=1, workers=1, shard_size=1)
    result = shards.run_lane(output / 'plan.json', 0, budget=PauseOne())
    assert result['status'] == 'paused'
    audits = list(output.glob('shards/*/result/execution/candidate_execution_audits.jsonl'))
    before = {p: p.read_bytes() for p in audits}
    plan = shards.read(output / 'plan.json')
    plan['shard_size'] += 1
    fixture._write(output / 'plan.json', plan)
    with pytest.raises(ValueError, match='resume plan changed'):
        shards.run_lane(output / 'plan.json', 0, budget=Unbounded())
    assert {p: p.read_bytes() for p in audits} == before


def test_submission_preserves_virtualenv_python_symlink(tmp_path, shards, monkeypatch):
    import sys
    venv_python = tmp_path / 'venv' / 'bin' / 'python'
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setattr(shards, 'load_plan', lambda path: (tmp_path, {
        'lanes': 1, 'max_segments': 12, 'resources': {'profile': 'evaluation_3h_compact'}}))
    command, = shards.submit_commands(tmp_path / 'plan.json', venv_python)
    assert command[-1] == str(venv_python)
    assert Path(command[-1]).resolve() != Path(command[-1])


@pytest.mark.parametrize('profile,workers,cpus', [
    ('evaluation_3h_compact', 4, 8), ('evaluation_3h', 8, 16),
    ('evaluation_3h_parallel', 14, 16),
])
def test_resources_reject_overcommit_and_undersized_allocation(
        tmp_path, fixture, shards, monkeypatch, profile, workers, cpus):
    manifest, _ = fixture._manifest(tmp_path, include_partial=False)
    output = tmp_path / 'split'
    with pytest.raises(ValueError, match='workers for'):
        shards.prepare(manifest, output, workers=workers + 1, resource_profile=profile)
    assert not output.exists()
    plan = shards.prepare(manifest, output, workers=workers, resource_profile=profile)
    assert plan['resources']['cpus'] == cpus
    assert plan['workers'] == workers
    monkeypatch.setenv('SLURM_JOB_ID', '301')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'allocated-gpu')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', str(cpus - 1))
    monkeypatch.setattr(shards, 'run_lane', lambda *a, **k: pytest.fail('must reject before execution'))
    with pytest.raises(ValueError, match=f'requires {cpus} allocated CPUs'):
        shards.run_slurm(output / 'plan.json', 0)
    assert not (output / 'lanes/0/slurm_continuation.json').exists()


def test_batch_wrapper_uses_frozen_source_when_slurm_copies_script(tmp_path):
    import os
    import subprocess
    import sys
    source = tmp_path / 'frozen source'
    (source / 'scripts').mkdir(parents=True)
    (source / 'scripts/ns_evaluation_shards.py').write_text(
        'import json, os, sys\n'
        'print(json.dumps({"cwd":os.getcwd(),"args":sys.argv[1:],'
        '"omp":os.environ["OMP_NUM_THREADS"],"pythonpath":os.environ["PYTHONPATH"]}))\n')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'source_root': str(source)}))
    spool = tmp_path / 'slurm spool'
    spool.mkdir()
    wrapper = spool / 'slurm_script'
    wrapper.write_bytes((ROOT / 'scripts/slurm_ns_evaluate_short.sh').read_bytes())
    env = {**os.environ, 'SLURM_JOB_ID': '301', 'SLURM_TMPDIR': str(tmp_path / 'scratch'),
           'CUDA_VISIBLE_DEVICES': 'allocated-gpu'}
    proc = subprocess.run(['bash', str(wrapper), str(plan), '0', sys.executable],
                          cwd=spool, env=env, capture_output=True, text=True, check=True)
    result = json.loads(proc.stdout)
    assert result['cwd'] == str(source)
    assert result['args'] == ['run', '--plan', str(plan), '--lane', '0', '--auto-continue']
    assert result['pythonpath'] == f'{source}/src:{source}/legacy/ns/src'
    assert result['omp'] == '1'
