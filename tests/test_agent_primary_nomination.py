"""Explicit final choice survives later submissions, process failure and resume."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    AGENT_EXECUTION_ID_ENV, AGENT_NOMINATIONS_ENV, AGENT_REQUEST_ENV, AGENT_WORKSPACE_ENV,
    AGENT_WORKER_REQUEST_SCHEMA,
    SubprocessAgentExecutionService,
    UnavailableAgentCapabilityFacade, read_agent_primary_nomination,
    select_agent_primary_nomination,
)
from pde_agent_verifier_solver.engine.agent_tool import main as agent_tool
from tests.test_agent_latest_nomination import _journal, _load
from tests.test_agent_execution_service import _agent_context, _decode_toy_nomination, _worker_program

ROOT = Path(__file__).resolve().parents[1]


def choose(tmp_path, sequence):
    return select_agent_primary_nomination(sequence, directory=tmp_path, execution_id='test-execution')


def test_primary_is_explicit_can_go_back_and_does_not_decode_other_versions(tmp_path):
    first = _journal(tmp_path, 1, {'mode': 1})
    choose(tmp_path, 1)
    second = _journal(tmp_path, 2, {'mode': 2})
    # Even an undecodable research version is harmless unless chosen.
    third = _journal(tmp_path, 3, {'not_a_solver': True})
    saved = [p.read_bytes() for p in (first, second, third)]
    calls = []
    def decode(body, context, workspace):
        calls.append(body)
        return _decode_toy_nomination(body, context, workspace)
    def selected_mode():
        artifacts, rejected = _load(tmp_path, policy='single_primary_v1', decoder=decode)
        assert rejected == 0 and len(artifacts) == 1
        return artifacts[0].metadata['max_resolved_mode']
    assert selected_mode() == 1
    choose(tmp_path, 2)
    assert selected_mode() == 2
    choose(tmp_path, 1)
    assert selected_mode() == 1
    assert calls == [{'mode': 1}, {'mode': 2}, {'mode': 1}]
    assert [p.read_bytes() for p in (first, second, third)] == saved


@pytest.mark.parametrize('damage', ['missing', 'foreign_turn', 'boolean_sequence', 'unknown_sequence', 'invalid_json', 'invalid_bundle', 'symlink'])
def test_missing_or_invalid_primary_does_not_fall_back(tmp_path, damage):
    _journal(tmp_path, 1, {'mode': 1})
    _journal(tmp_path, 2, {'bad': True} if damage == 'invalid_bundle' else {'mode': 2})
    marker = choose(tmp_path, 2)
    if damage == 'missing': marker.unlink()
    elif damage == 'invalid_json': marker.write_text('{')
    elif damage == 'symlink':
        other = tmp_path / 'other.json'; other.write_bytes(marker.read_bytes())
        marker.unlink(); marker.symlink_to(other)
    elif damage != 'invalid_bundle':
        data = json.loads(marker.read_text())
        if damage == 'foreign_turn': data['execution_id'] = 'other-turn'
        else: data['sequence'] = True if damage == 'boolean_sequence' else 3
        marker.write_text(json.dumps(data))
    assert _load(tmp_path, policy='single_primary_v1') == ((), 1)


def test_failed_atomic_selection_keeps_previous_primary(tmp_path):
    _journal(tmp_path, 1, {'mode': 1}); _journal(tmp_path, 2, {'mode': 2})
    marker = choose(tmp_path, 1)
    saved = marker.read_bytes()
    with patch('pde_agent_verifier_solver.engine.agent_execution.os.replace', side_effect=OSError('write interrupted')):
        with pytest.raises(OSError): choose(tmp_path, 2)
    assert marker.read_bytes() == saved
    assert read_agent_primary_nomination(tmp_path, 'test-execution') == 1
    assert not list(tmp_path.glob('*.partial'))


@pytest.mark.parametrize('ending', ['completed', 'failed', 'timeout'])
def test_real_worker_keeps_primary_through_later_submission_and_interruption(tmp_path, ending):
    script = tmp_path / 'worker.py'
    script.write_text('\n'.join([
        'import time',
        'from pde_agent_verifier_solver.engine.agent_execution import (submit_agent_nomination, select_agent_primary_nomination, write_agent_worker_response)',
        "submit_agent_nomination({'mode': 1}, sequence=1)",
        'select_agent_primary_nomination(1)',
        "submit_agent_nomination({'mode': 2}, sequence=2)",
        'time.sleep(60)' if ending == 'timeout' else
        "write_agent_worker_response(status='failed', error='test interruption')" if ending == 'failed' else
        'write_agent_worker_response()',
    ]))
    service = SubprocessAgentExecutionService(tmp_path / 'turns', hard_wall_seconds=3,
        capability_facade=UnavailableAgentCapabilityFacade(), termination_grace_seconds=.1,
        poll_interval_seconds=.01)
    outcome = service.execute(_worker_program(script, worker_settings={'nomination_policy': 'single_primary_v1'}),
                              _agent_context(), 3)
    assert len(outcome.nominations) == 1
    assert outcome.nominations[0].metadata['max_resolved_mode'] == 1
    assert outcome.execution_status == {'completed':'completed', 'failed':'failed', 'timeout':'timed_out'}[ending]
    audit = service.audits[-1]
    assert audit.final_nomination_sequence == 1 and audit.nomination_journal_entry_count == 2
    assert audit.process_group_cleaned and audit.nomination_policy == 'single_primary_v1'
    assert read_agent_primary_nomination(Path(audit.workspace) / 'nominations', audit.execution_id) == 1


def test_cli_primary_uses_frozen_code_and_weights_and_reports_selection(tmp_path, monkeypatch, capsys):
    nominations = tmp_path / 'nominations'; nominations.mkdir()
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'schema': AGENT_WORKER_REQUEST_SCHEMA, 'execution_id': 'cli-test',
                                   'worker_settings': {'nomination_policy':'single_primary_v1'}}))
    for key, value in {AGENT_EXECUTION_ID_ENV:'cli-test', AGENT_NOMINATIONS_ENV:str(nominations),
                       AGENT_WORKSPACE_ENV:str(tmp_path), AGENT_REQUEST_ENV:str(request)}.items():
        monkeypatch.setenv(key, value)
    from pde_agent_verifier_solver.sdk import write_candidate_bundle_manifest
    source = tmp_path / 'solver'; source.mkdir()
    (source / 'submission.py').write_text('VALUE = 1\n')
    (source / 'weights.bin').write_bytes(b'first weights')
    write_candidate_bundle_manifest(source, entrypoint='submission.py', files=['submission.py','weights.bin'])
    def submit(sequence, primary=False):
        command = ['submit','solver','--sequence',str(sequence),'--name','test','--summary','test','--usage','test']
        assert agent_tool(command + (['--primary'] if primary else [])) == 0
        return json.loads(capsys.readouterr().out)
    first = submit(1, True)
    (source / 'submission.py').write_text('VALUE = 2\n'); (source / 'weights.bin').write_bytes(b'new weights')
    submit(2)
    assert read_agent_primary_nomination(nominations, 'cli-test') == 1
    assert agent_tool(['select-primary','2']) == 0; capsys.readouterr()
    assert agent_tool(['select-primary','1']) == 0
    assert json.loads(capsys.readouterr().out)['primary_nomination_sequence'] == 1
    saved = tmp_path / first['bundle_snapshot']
    assert (saved / 'weights.bin').read_bytes() == b'first weights'
    assert (saved / 'submission.py').read_text() == 'VALUE = 1\n'
    assert len(list(nominations.glob('nomination-*.json'))) == 2
    # A fresh client process uses the same durable choice; no in-memory fallback.
    from pde_agent_verifier_solver.engine import agent_tool as module
    class Client:
        def current_status(self): return (), []
    monkeypatch.setattr(module.ControllerAgentCapabilityClient, 'from_request',
                        classmethod(lambda cls, *args, **kwargs: Client()))
    assert agent_tool(['status']) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown['primary_nomination_sequence'] == 1
    assert shown['nomination_policy'] == 'single_primary_v1'
    request.write_text(json.dumps({'schema': AGENT_WORKER_REQUEST_SCHEMA, 'execution_id': 'cli-test',
                                   'worker_settings': {'nomination_policy':'latest'}}))
    with pytest.raises(SystemExit): agent_tool(['select-primary','2'])
    assert read_agent_primary_nomination(nominations, 'cli-test') == 1


def test_new_prompt_explains_explicit_choice_and_existing_config_stays_compatible():
    path = ROOT / 'scripts/run_canonical_codex_agent.py'
    spec = importlib.util.spec_from_file_location('primary_prompt_worker', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    text = module._research_prompt({'agent':{'prompt':'Solve the problem.'},
        'round':{'round_index':1,'hard_wall_seconds':14400,'time_budget_mode':'hard_wall',
                 'development_sampling':{'max_cases':None,'allowed_sources':['base']}},
        'worker_settings':{'nomination_policy':'single_primary_v1'},
        'candidate_execution':{'case_charged_wall_seconds':900}})
    assert 'select-primary N' in text and 'Later ordinary submissions do not replace' in text
    assert 'Only your last committed submission' not in text
    for name in ('ns_canonical_structured_5round.json','ns_canonical_piv_5round.json','ns_modular_prompt_example.json'):
        assert json.loads((ROOT/'configs'/name).read_text())['execution']['nomination_policy'] == 'single_primary_v1'


def test_panel_records_rule_for_each_run_without_mutating_bound_agents(tmp_path):
    from pde_agent_verifier_solver.panel.controller import build_toy_panel
    panel = build_toy_panel()
    with pytest.raises(ValueError, match='command Agents'):
        panel.create_and_launch({'nomination_policy':'single_primary_v1'})
    script = tmp_path/'worker.py'; script.write_text('pass')
    panel.programs.agents.update({k:_worker_program(script) for k in panel.programs.agents})
    assert panel.setup_page()['submission_selection_available'] is True
    ids = tuple(panel.programs.agents)
    specs = panel._prompt_agents({'agents':ids})
    chosen = panel._programs_for_run({'nomination_policy':'single_primary_v1'}, specs)
    assert all(p.worker_settings['nomination_policy']=='single_primary_v1' for p in chosen.agents.values())
    assert all(p.worker_settings.get('nomination_policy','all')=='all' for p in panel.programs.agents.values())
    assert set(panel._submission_policies(chosen).values()) == {'single_primary_v1'}


def test_panel_two_round_campaign_evaluates_only_primary_and_inherits_it(tmp_path):
    from pde_agent_verifier_solver.app.service import ProductApplication
    from pde_agent_verifier_solver.panel.controller import build_toy_panel
    from tests.test_agent_execution_service import _run_values
    script = tmp_path/'campaign-worker.py'
    script.write_text("\n".join([
        'import json',
        'from pathlib import Path',
        'from pde_agent_verifier_solver.engine.agent_execution import (load_agent_worker_request, submit_agent_nomination, select_agent_primary_nomination, write_agent_worker_response)',
        'r=load_agent_worker_request()',
        "assert r['worker_settings']['nomination_policy']=='single_primary_v1'",
        "Path('seen-incumbent.json').write_text(json.dumps(r['incumbent']))",
        "submit_agent_nomination({'mode': 1}, sequence=1)",
        'select_agent_primary_nomination(1)',
        "submit_agent_nomination({'mode': 8}, sequence=2)",
        'write_agent_worker_response()',
    ]))
    service = SubprocessAgentExecutionService(tmp_path/'turns', hard_wall_seconds=10,
        capability_facade=UnavailableAgentCapabilityFacade())
    panel = build_toy_panel()
    panel.app = ProductApplication.local_development(agent_execution=service)
    panel.programs.agents.update({name:_worker_program(script) for name in ('alpha','beta')})
    values = _run_values('alpha', budget_seconds=10)
    values.update(round_count=2, agents=['alpha','beta'], nomination_policy='single_primary_v1')
    launched = panel.create_and_launch(values)
    session = panel.app.wait(launched['run_id'], timeout=35)
    page = panel.run_page(launched['run_id'])
    assert page['status']=='completed', page.get('error')
    assert page['agent_submission_policies']['alpha']=='single_primary_v1'
    assert len(service.audits)==4
    for audit in service.audits:
        assert audit.final_nomination_sequence==1 and audit.nomination_journal_entry_count==2
        incumbent = json.loads((Path(audit.workspace)/'seen-incumbent.json').read_text())
        if audit.round_index==2:
            assert incumbent['artifact_id']==session.result.rounds[0].selected.artifact.artifact_id
    assert all(r.selected.artifact.metadata['max_resolved_mode']==1 for r in session.result.rounds)
    assert panel.programs.agents['alpha'].worker_settings.get('nomination_policy','all')=='all'


def test_native_same_session_resume_retains_primary_and_original_deadline(tmp_path):
    from tests.test_canonical_codex_resume import (
        _write_resumable_codex, write_auth, command_program, agent_context,
    )
    native = _write_resumable_codex(tmp_path)
    source = native.read_text()
    needle = "write_agent_round_log('# Same research turn"
    assert needle in source
    source = source.replace(needle,
        "from pde_agent_verifier_solver.engine.agent_execution import select_agent_primary_nomination, read_agent_primary_nomination\n"
        "if attempt == 1: select_agent_primary_nomination(1)\n"
        "else: assert read_agent_primary_nomination(workspace / 'nominations', request['execution_id']) == 1\n" + needle, 1)
    native.write_text(source)
    auth, original_auth = write_auth(tmp_path)
    runtime = tmp_path / 'runtime'; runtime.mkdir()
    program = replace(command_program(native, auth, runtime, gpu_devices=('0',)),
        worker_settings={'nomination_policy':'single_primary_v1', 'transient_resume_max_attempts':3})
    service = SubprocessAgentExecutionService(tmp_path/'turns', hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(), poll_interval_seconds=.01,
        termination_grace_seconds=.5)
    outcome = service.execute(program, agent_context(), 20)
    audit = service.audits[-1]
    assert outcome.execution_status == 'completed'
    assert len(outcome.nominations)==1 and outcome.nominations[0].metadata['max_resolved_mode']==1
    assert audit.final_nomination_sequence==1 and audit.nomination_journal_entry_count==2
    recovery = json.loads((Path(audit.workspace)/'codex-recovery.json').read_text())
    assert len(recovery['attempts'])==2
    assert recovery['attempts'][1]['remaining_seconds'] < recovery['attempts'][0]['remaining_seconds']
    assert auth.read_bytes()==original_auth.encode() and not list(runtime.iterdir())
