"""Explicit CPU mode must preserve the canonical worker and auth lifecycle."""
from dataclasses import replace
from pathlib import Path
import json
import os
import pytest
from tests.test_canonical_codex_agent import WORKER, agent_context, command_program, write_auth
from tests.test_ns_baseline_canonical_worker import _stub_codex
from pde_agent_verifier_solver.engine.agent_execution import SubprocessAgentExecutionService, UnavailableAgentCapabilityFacade


def test_cpu_environment_requires_empty_assignment(tmp_path, monkeypatch):
    for name in WORKER._CANONICAL_AGENT_ENVIRONMENT:
        monkeypatch.setenv(name, str(tmp_path/name))
    args = dict(repository_root=tmp_path, workspace=tmp_path, codex_home=tmp_path/'codex', agent_home=tmp_path/'agent')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert WORKER._codex_environment(**args, cpu_only=True)['CUDA_VISIBLE_DEVICES'] == ''
    with pytest.raises(ValueError, match='exactly one GPU'):
        WORKER._codex_environment(**args)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CPU-only'):
        WORKER._codex_environment(**args, cpu_only=True)
    assert WORKER._codex_environment(**args)['CUDA_VISIBLE_DEVICES'] == '0'


def test_real_worker_cpu_keeps_auth_cleanup_and_nomination(tmp_path, monkeypatch):
    from tests.test_canonical_codex_agent import write_fake_codex
    runtime = tmp_path/'runtime'; runtime.mkdir()
    capture = tmp_path/'capture.json'
    codex = write_fake_codex(tmp_path, capture_path=capture, exit_code=0, nomination_count=1)
    auth, original = write_auth(tmp_path)
    program = command_program(codex, auth, runtime, gpu_devices=())
    program = replace(program, command=(*program.command, '--cpu-only'), worker_settings={'nomination_policy':'latest'})
    service=SubprocessAgentExecutionService(tmp_path/'turns', hard_wall_seconds=15,
        capability_facade=UnavailableAgentCapabilityFacade())
    outcome=service.execute(program, agent_context(), 10)
    assert outcome.execution_status=='completed' and len(outcome.nominations)==1
    captured=json.loads(capture.read_text())
    assert captured['cuda_visible_devices']==''
    prompt=(Path(service.audits[-1].workspace)/'CANONICAL_AGENT_PROMPT.md').read_text()
    assert 'no GPU is assigned' in prompt
    assert 'may use its assigned GPU' not in prompt
    audit=json.loads((Path(service.audits[-1].workspace)/'codex-auth-lifecycle.json').read_text())
    assert audit['auth_removed_before_first_event_forwarded'] is True
    assert auth.read_text()==original
    assert not tuple(runtime.iterdir())
