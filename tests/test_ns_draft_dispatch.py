"""Check actual GPU/path assignments before handing a job to existing runners."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def driver():
    spec = importlib.util.spec_from_file_location('ns_draft_dispatch_test', ROOT/'scripts/run_ns_draft_job.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_competition_keeps_distinct_allocated_agent_gpus():
    module = driver()
    path = ROOT/module.CONFIG_PATHS['fixed']
    payload = module._baseline_payload(path, ('GPU-left', 'GPU-right'))
    assert [role['gpu_devices'] for role in payload['agent']['roles']] == [['GPU-left'], ['GPU-right']]
    assert Path(payload['dataset']['manifest']).is_file()
    assert Path(payload['initial_candidate']).is_dir()
    assert payload['agent']['hard_wall_seconds'] == 14400
    with pytest.raises(ValueError):
        module._baseline_payload(path, ('GPU-only',))


@pytest.mark.parametrize('mode', ['bon', 'aide'])
def test_serial_search_uses_one_allocated_gpu_and_preserves_search(mode):
    module = driver()
    path = ROOT/module.CONFIG_PATHS[mode]
    original = json.loads(path.read_text())
    payload = module._baseline_payload(path, ('GPU-search',))
    assert payload['agent']['gpu_devices'] == ['GPU-search']
    assert 'roles' not in payload['agent']
    assert payload['search'] == original['search']
    assert payload['agent']['hard_wall_seconds'] == 14400
    assert module.BASELINE_GPUS[mode] == 1
    with pytest.raises(ValueError, match='one allocated'):
        module._baseline_payload(path, ('GPU-search', 'GPU-extra'))


def test_separate_hidden_bank_is_resolved_from_frozen_config(tmp_path):
    module = driver()
    original = json.loads((ROOT/module.CONFIG_PATHS['bon']).read_text())
    original['dataset']['selection_manifest'] = '../../control/shared/fixed_selection.json'
    config = tmp_path/'source/configs/bon.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(original))
    payload = module._baseline_payload(config, ('GPU-allocated',))
    assert Path(payload['dataset']['selection_manifest']) == (config.parent/'../../control/shared/fixed_selection.json').resolve()
    assert payload['dataset']['selection_manifest'] != payload['dataset']['manifest']


def test_dispatch_only_authorizes_first_two_batches():
    module = driver()
    assert module.AUTHORIZED_METHODS == ('fixed', 'base', 'full', 'base_guided', 'base_perturbed')
    for method in module.AUTHORIZED_METHODS:
        assert Path(module.CONFIG_PATHS[method]).parent.name == 'ns_bgp_v3'
        assert (ROOT/module.CONFIG_PATHS[method]).is_file()
    with pytest.raises(SystemExit) as exc:
        module.main(['--mode', 'aide', '--batch-root', '/tmp/not-used'])
    assert exc.value.code == 2


def _prepared_layout(directory):
    from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
    directory.mkdir()
    def bank(name, counts, purpose):
        data = {'strata_counts': counts, 'case_count': sum(counts.values()), 'purpose': purpose,
                'cases': [{'case': {'case_id': f'{name}-{i}'}} for i in range(sum(counts.values()))]}
        data['content_sha256'] = content_sha256(data)
        (directory/name).write_text(json.dumps(data))
    rows = []
    for index in range(1, 6):
        name = f'official_{index}.json'
        bank(name, {'base': 64, 'base_perturbed': 64}, 'promotion')
        rows.append({'round_index': index, 'manifest': name, 'delegate_kinds': ['guided']})
    bank('fixed_selection.json', {'base':64, 'guided':64, 'base_perturbed':64}, 'promotion')
    bank('terminal.json', {'base':256, 'guided':256, 'base_perturbed':256}, 'terminal')
    (directory/'comparison_suite.json').write_text(json.dumps({
        'status': 'frozen', 'master_seed':101, 'official_rounds':rows, 'terminal_manifest':'terminal.json'}))


def test_preflight_requires_correct_disjoint_case_inventories(tmp_path):
    from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
    module = driver()
    path = tmp_path/'prepared'
    _prepared_layout(path)
    checks = module.validate_prepared_banks(path)
    assert len(checks) == 7
    selection = path/'fixed_selection.json'
    data = json.loads(selection.read_text())
    data['cases'][0]['case']['case_id'] = 'official_1.json-0'
    data.pop('content_sha256')
    data['content_sha256'] = content_sha256(data)
    selection.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='repeated Cases'):
        module.validate_prepared_banks(path)


@pytest.mark.parametrize('field,value,match', [
    ('strata_counts', {'base':64, 'guided':63, 'base_perturbed':65}, 'approved Case counts'),
    ('purpose', 'terminal', 'purpose'),
])
def test_preflight_rejects_wrong_scientific_bank_even_if_hash_updated(tmp_path, field, value, match):
    from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
    module = driver()
    path = tmp_path/'prepared'
    _prepared_layout(path)
    selection = path/'fixed_selection.json'
    data = json.loads(selection.read_text())
    data[field] = value
    data.pop('content_sha256')
    data['content_sha256'] = content_sha256(data)
    selection.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        module.validate_prepared_banks(path)


def test_failed_s0_never_qualifies_mixed_selection():
    from types import SimpleNamespace as NS
    module = driver()
    records = (NS(case=NS(case_id='a')), NS(case=NS(case_id='b')))
    complete = NS(primary_error=1., cases=(NS(case_id='a', primary_error=1.), NS(case_id='b', primary_error=1.)))
    module._require_finite_selection(complete, records)
    with pytest.raises(RuntimeError, match='incomplete'):
        module._require_finite_selection(NS(primary_error=1., cases=complete.cases[:1]), records)
    complete.cases[1].primary_error = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'):
        module._require_finite_selection(complete, records)


def test_final_evaluation_exception_cannot_record_completed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import sys
    from types import SimpleNamespace
    module = driver()
    batch = tmp_path/'batch'
    source = batch/'source'
    source.mkdir(parents=True)
    control = batch/'control'
    control.mkdir()
    (control/'preflight_passed.json').write_text(json.dumps({
        'status':'passed','protocol_revision':module.PAPER_PROTOCOL}))
    monkeypatch.setattr(module, 'ROOT', source)
    monkeypatch.setattr(module, 'verify_snapshot', lambda: None)
    monkeypatch.setenv('NS_JOB_GPU_TOKENS', '0,1,2')
    monkeypatch.setenv('SLURM_JOB_ID', '500001')
    monkeypatch.setattr(module, 'allocation_remaining_seconds', lambda job: 43200)
    monkeypatch.setitem(sys.modules, 'run_canonical_ns_campaign', SimpleNamespace(main=lambda args, **kwargs: 0))
    def fail(*args, **kwargs):
        raise RuntimeError('final evaluation unavailable')
    monkeypatch.setattr(module, 'evaluate_final_bundle', fail)
    with pytest.raises(RuntimeError, match='final evaluation unavailable'):
        module.main(['--mode','base','--batch-root',str(batch)])
    state=json.loads((batch/'job_status/base-500001.json').read_text())
    assert state['status'] == 'failed'
    assert state['returncode'] != 0
    assert state['error'] == 'final evaluation unavailable'


@pytest.mark.parametrize('devices', [(), ('cuda:0', 'cuda:1', 'cuda:2')])
@pytest.mark.parametrize('fail_evaluation', [False, True])
@pytest.mark.parametrize('method', ['base', 'fixed'])
def test_common_final_binds_devices_and_releases_resources(tmp_path, monkeypatch, devices, fail_evaluation, method):
    from pde_agent_verifier_solver.plugins.ns import common_evaluation
    from pde_agent_verifier_solver.sdk.evaluation_devices import current_evaluation_device_pool
    module = driver()
    observed = []
    closed = []
    class Resource:
        def close(self):
            closed.append(True)
    def evaluate(manifest_path, output_directory):
        manifest = json.loads(manifest_path.read_text())
        # Scientific budgets and frozen bank source remain identical.
        assert manifest['query'] == {'point_count':128, 'include_pressure':False}
        assert manifest['budget'] == {'max_workers':28, 'case_hard_wall_seconds':900}
        assert manifest['datasets'][0]['manifest'] == str(tmp_path/'control/mixed_evaluation/terminal.json')
        pool = current_evaluation_device_pool()
        observed.append(pool)
        if devices:
            assert pool.devices == devices
            pool.get_resource('test-owned-resource', Resource)
        else:
            assert pool is None
        if fail_evaluation:
            raise RuntimeError('final evaluation failed')
    monkeypatch.setattr(common_evaluation, 'run_ns_common_evaluation', evaluate)
    def run():
        module.evaluate_final_bundle(method, tmp_path/'solver', tmp_path/'run', tmp_path,
                                     evaluation_devices=devices)
    if fail_evaluation:
        with pytest.raises(RuntimeError, match='final evaluation failed'):
            run()
    else:
        run()
    assert len(observed) == 1
    assert current_evaluation_device_pool() is None
    assert closed == ([True] if devices else [])
    if devices:
        assert observed[0].cleanup_confirmed


def test_new_preflight_rejects_old_full_selection_bank_even_with_valid_hash(tmp_path):
    from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
    module = driver()
    path = tmp_path/'prepared'
    _prepared_layout(path)
    selection = path/'fixed_selection.json'
    data = json.loads(selection.read_text())
    data.pop('content_sha256')
    data.update(strata_counts={'base':342, 'guided':341, 'base_perturbed':341}, case_count=1024)
    data['cases'] = [{'case': {'case_id': f'old-selection-{i}'}} for i in range(1024)]
    data['content_sha256'] = content_sha256(data)
    selection.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='approved Case counts'):
        module.validate_prepared_banks(path)


def test_new_dispatch_rejects_old_preflight_revision_before_launch(tmp_path, monkeypatch):
    module = driver()
    batch = tmp_path/'batch'
    source = batch/'source'
    source.mkdir(parents=True)
    (batch/'control').mkdir()
    (batch/'control/preflight_passed.json').write_text(json.dumps({
        'status':'passed', 'protocol_revision':'iclr_ns_bgp_1024_terminal_base_perturb_v4'}))
    monkeypatch.setattr(module, 'ROOT', source)
    monkeypatch.setattr(module, 'verify_snapshot', lambda: None)
    monkeypatch.setenv('NS_JOB_GPU_TOKENS', 'GPU-left,GPU-right')
    with pytest.raises(RuntimeError, match='preflight has not passed'):
        module.main(['--mode', 'fixed', '--batch-root', str(batch)])
