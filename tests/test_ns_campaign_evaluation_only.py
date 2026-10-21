from contextlib import nullcontext
from hashlib import sha256
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ns_campaign_evaluation_only as compact
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignContinuation, CampaignPauseControl, CampaignPaused
from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore, source_identity, scientific_progress
from tests.test_campaign_continuation import _controller, _normalize


def test_guard_rejects_agent_work_and_partial_final_development():
    value = SimpleNamespace(state={'phase': 'round_prepare', 'round_results': [SimpleNamespace(round_index=1)]})
    with pytest.raises(ValueError, match='still needs Agent'):
        compact.require_evaluation_tail(value, 3, ['a', 'b'])
    value.state.update(phase='official_ready', round_results=[SimpleNamespace(round_index=1), SimpleNamespace(round_index=2)],
                       round_data={'outcomes': [SimpleNamespace(execution_status='completed')], 'agent_logs': []})
    with pytest.raises(ValueError, match='still needs Agent'):
        compact.require_evaluation_tail(value, 3, ['a', 'b'])
    for service in (compact.InactiveAgent().run, compact.EvaluationOnlyAgents().execute, compact.EvaluationOnlyAgents().execute_batch):
        with pytest.raises(RuntimeError, match='cannot run an Agent'):
            service(None)


@pytest.mark.parametrize('pause_stage', ['official_sampled', 'before_terminal_guidance', 'terminal_completed', 'before_hidden_batch'])
def test_evaluation_tail_resume_matches_original_no_agent_repeat(tmp_path, pause_stage):
    original = _controller(tmp_path / 'original')
    reference = original.run()
    controller = _controller(tmp_path / 'resumed')
    store = CheckpointStore(tmp_path / 'checkpoint', identity={'test': 1}, generator=controller.generator)
    def pause(stage, _seconds):
        state = controller._continuation_state
        completed = len(state.get('round_results', ()))
        if pause_stage == 'official_sampled':
            return completed == 2 and stage == pause_stage
        if pause_stage == 'before_hidden_batch':
            return state.get('phase') == 'terminal_ready' and stage == pause_stage
        return completed == 3 and stage == pause_stage
    with pytest.raises(CampaignPaused):
        controller.run(pause_control=CampaignPauseControl(store.save, pause, hidden_batch_size=2))
    saved = store.load()
    compact.require_evaluation_tail(saved, 3, ['careful', 'explorer'])
    before = [json.loads(l) for l in (tmp_path / 'resumed/work.jsonl').read_text().splitlines()]
    resumed = _controller(tmp_path / 'resumed')
    resumed.agents = {k: compact.InactiveAgent() for k in resumed.agents}
    service = compact.EvaluationOnlyAgents()
    service.reserve_evaluation_devices = lambda _programs: nullcontext(None)
    resumed.agent_execution = service
    result = resumed.run(continuation=saved, pause_control=CampaignPauseControl(store.save, lambda *_: False, hidden_batch_size=4))
    after = [json.loads(l) for l in (tmp_path / 'resumed/work.jsonl').read_text().splitlines()]
    assert [x for x in before if x['kind'] == 'agent'] == [x for x in after if x['kind'] == 'agent']
    assert _normalize(reference.selected) == _normalize(result.selected)
    assert _normalize(reference.terminal_round_reports) == _normalize(result.terminal_round_reports)
    assert _normalize(reference.rounds) == _normalize(result.rounds)
    evaluations = [(r['artifact'], r['case'], r['phase']) for r in after if r['kind'] == 'evaluate']
    original_evaluations = [(r['artifact'], r['case'], r['phase']) for r in
                            map(json.loads, (tmp_path / 'original/work.jsonl').read_text().splitlines()) if r['kind'] == 'evaluate']
    from collections import Counter
    assert Counter(evaluations) == Counter(original_evaluations)


def test_resource_transfer_preserves_cases_inputs_and_controller_rng(tmp_path, monkeypatch):
    old_source, new_source = tmp_path / 'old_source', tmp_path / 'new_source'
    for path in (old_source, new_source):
        (path / 'scripts').mkdir(parents=True)
        (path / 'scripts/x.py').write_text('# ' + path.name)
    monkeypatch.setattr(compact, 'ROOT', new_source)
    config = {'round_count': 1, 'agents': [{'agent_id': 'a'}]}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    identity = {'source_sha256': source_identity(old_source), 'config_sha256': sha256(json.dumps({**config, '_config_path': str(config_path)}, sort_keys=True).encode()).hexdigest(), 'generator_inputs': {'flow': 'unchanged'}}
    old = tmp_path / 'old'
    old.mkdir(); (old / 'objects').mkdir()
    token = ('pickle', {'name': 'case.pkl', 'sha256': 'same-reference'})
    state = {'phase': 'terminal_ready', 'round_results': [SimpleNamespace(round_index=1)],
             'hidden_progress': {('terminal', 'terminal'): {'base': {'cases': {'solver': {'case': compact._Reference(token)}}}}}}
    rng = {'python': (1, 2), 'torch': np.array([6], dtype=np.uint8), 'cuda': [np.array([i], dtype=np.uint8) for i in range(3)]}
    value = CampaignContinuation(state=state, controller_state={'events': []})
    encoded = compact.encode_references({'value': value, 'rng': rng})
    hashed = sha256(encoded).hexdigest()
    (old / 'objects' / (hashed + '.pkl')).write_bytes(encoded)
    (old / 'objects/case.pkl').write_bytes(b'not decoded during migration')
    manifest = {'schema': 'campaign_checkpoint_v1', 'identity': identity, 'sequence': 7,
                'payload': {'name': hashed + '.pkl', 'sha256': hashed}, 'generator_runtime': {'same': True}, 'progress': scientific_progress(value)}
    (old / 'checkpoint.json').write_text(json.dumps(manifest))
    before = (old / 'checkpoint.json').read_bytes()
    new = tmp_path / 'new/checkpoints'
    receipt = compact.transfer_checkpoint(old, new, old_source=old_source, config=config_path)
    assert (old / 'checkpoint.json').read_bytes() == before
    after = compact.read(new / 'checkpoint.json')
    payload = compact.decode_references((new / 'objects' / after['payload']['name']).read_bytes())
    assert after['sequence'] == 7 and after['progress'] == manifest['progress']
    assert after['generator_runtime'] == manifest['generator_runtime']
    assert receipt['cuda_streams_before'] == 3 and receipt['cuda_streams_after'] == 1
    np.testing.assert_array_equal(payload['rng']['cuda'][0], rng['cuda'][0])
    assert len(payload['rng']['cuda']) == 1
    assert compact.encode_references(payload['value']) == compact.encode_references(value)
    assert after['identity']['generator_inputs'] == identity['generator_inputs']
    assert (new / 'objects/case.pkl').stat().st_ino == (old / 'objects/case.pkl').stat().st_ino
    with pytest.raises(ValueError, match='new destination'):
        compact.transfer_checkpoint(old, new, old_source=old_source, config=config_path)


def test_drain_reserves_full_case_budget():
    from threading import Event
    calls = []
    budget = SimpleNamespace(cancel=Event(), should_pause=lambda stage, seconds: calls.append((stage, seconds)) or False)
    wrapped = compact.DrainBudget(budget, 900)
    wrapped.should_pause('next_case')
    wrapped.should_pause('long_stage', 7200)
    assert calls == [('next_case', 1020), ('long_stage', 7200)]


@pytest.mark.parametrize('workers', [4, 12, 14])
def test_canonical_evaluation_composition_uses_requested_workers_without_auth_or_agent_process(tmp_path, monkeypatch, workers):
    import run_canonical_ns_campaign as canonical
    from pde_agent_verifier_solver.engine import GeneratorRecipe
    from pde_agent_verifier_solver.plugins.toy import ToyPoissonProblem, ToyPoissonGenerator
    from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind
    config_path = Path(__file__).resolve().parents[1] / 'configs/ns_canonical_structured_5round.json'
    config, config_path = canonical._load_config(config_path)
    # This fixture exercises CPU evaluation; the shipped template also offers
    # per-Agent CUDA routing and explicit Oracle parallelism.
    config["execution"].update(candidate_device="cpu", development_solver_placement="shared")
    config["execution"].pop("gpu_execution", None)
    config["execution"].pop("oracle_max_workers", None)
    config["execution"].pop("solver_batch_size", None)
    before = json.dumps(config, sort_keys=True)
    problem, generator = ToyPoissonProblem(), ToyPoissonGenerator()
    dataset = problem.prepare_dataset(DatasetInput('fixture', DatasetKind.NUMERICAL,
        settings={'development_cases': 2, 'hidden_cases': 2}))
    monkeypatch.setattr(canonical, '_compose_numerical', lambda *_: (dataset, problem, generator, GeneratorRecipe(), None, {}))
    monkeypatch.setattr(canonical, 'SubprocessAgentExecutionService', lambda *_a, **_kw: pytest.fail('Agent process composition in evaluation-only mode'))
    root = tmp_path / 'composition'; root.mkdir()
    composition = canonical._compose(config=config, config_path=config_path, run_root=root,
        codex_binary=Path('/missing/codex'), auth_file=Path('/missing/auth'), worker_gpus=(),
        controller_gpu='0', evaluation_workers=workers)
    try:
        assert composition.candidate_execution.max_workers == workers
        assert composition.candidate_execution.oracle_max_workers == workers
        assert composition.candidate_execution.case_hard_wall_seconds == 900
        assert isinstance(composition.execution, compact.EvaluationOnlyAgents)
        assert all(isinstance(a, compact.InactiveAgent) for a in composition.agents.values())
        assert json.dumps(config, sort_keys=True) == before
    finally:
        close = getattr(composition.candidate_execution, 'close', None)
        if callable(close):
            close()


def test_final_guidance_reserves_its_declared_training_and_sampling_windows():
    from threading import Event
    calls=[]
    budget=SimpleNamespace(cancel=Event(),should_pause=lambda stage, seconds: calls.append((stage,seconds)) or False)
    drain=compact.DrainBudget(budget,900,guidance_training_seconds=7200,guidance_sampling_seconds=1800)
    drain.should_pause('before_terminal_guidance')
    drain.should_pause('terminal_guidance_ready')
    assert calls==[('before_terminal_guidance',7320),('terminal_guidance_ready',1920)]


def test_guidance_training_counts_as_progress_without_a_new_case(tmp_path):
    directory=tmp_path/'objects';directory.mkdir()
    checkpoint=tmp_path/'checkpoint.json'
    progress=[]
    for ready in (False,True):
        state={'phase':'terminal_prepare','final_guidance_ready':ready,'round_results':[]}
        data=compact.encode_references({'value':SimpleNamespace(state=state)})
        hashed=sha256(data).hexdigest();name=hashed+'.pkl';(directory/name).write_bytes(data)
        checkpoint.write_text(json.dumps({'schema':'campaign_checkpoint_v1','progress':'same-original-case-count',
                                         'payload':{'name':name,'sha256':hashed}}))
        progress.append(compact.evaluation_progress(checkpoint,'scientific'))
    assert progress[0]!=progress[1]
    assert compact.evaluation_progress(checkpoint,'scientific')==progress[1]


@pytest.mark.parametrize('workers', [1, 12, 14])
def test_parallel_evaluation_resources_keep_controller_headroom(workers):
    plan = {'workers': workers, 'resources': {'profile': 'evaluation_3h_parallel',
            'cpus': 16, 'gpus': 1, 'memory': '64G', 'wall_seconds': 10800, 'flush_seconds': 600}}
    compact.validate_resources(plan)
    for invalid in (15, 16, True):
        with pytest.raises(ValueError, match='workers'):
            compact.validate_resources({**plan, 'workers': invalid})
    with pytest.raises(ValueError, match='cpus'):
        compact.validate_resources({**plan, 'resources': {**plan['resources'], 'cpus': 8}})
