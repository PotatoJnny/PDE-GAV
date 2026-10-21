from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from pde_agent_verifier_solver.sdk import AgentPromptSections, resolve_agent_prompt_files
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent

ROOT = Path(__file__).resolve().parents[1]


def test_file_sections_are_shared_but_roles_and_replay_are_independent(tmp_path):
    (tmp_path / 'TASK.md').write_text('Solve the specified equation.')
    (tmp_path / 'RESOURCES.md').write_text('Use the public inputs and feedback.')
    (tmp_path / 'explorer.md').write_text('Explore new methods.')
    (tmp_path / 'refiner.md').write_text('Improve the current method.')
    config = {'prompt_files': {'task': 'TASK.md', 'resources': 'RESOURCES.md'}, 'agents': [
        {'agent_id': name, 'name': name, 'role_file': name + '.md'}
        for name in ('explorer', 'refiner')]}
    snapshot = resolve_agent_prompt_files(config, relative_to=tmp_path)
    assert 'prompt' not in config['agents'][0]
    first, second = snapshot['agents']
    assert 'Explore new methods.' in first['prompt'] and 'Improve the current method.' not in first['prompt']
    assert 'Improve the current method.' in second['prompt']
    for path in tmp_path.glob('*.md'):
        path.unlink()
    assert resolve_agent_prompt_files(snapshot, relative_to=tmp_path) == snapshot


def test_legacy_inline_prompts_remain_unchanged():
    config = {'agents': [{'agent_id': 'one', 'prompt': 'Keep this exact prompt.'}]}
    assert resolve_agent_prompt_files(config, relative_to=Path('/unused')) == config


@pytest.mark.parametrize('value', [
    {'task': ''}, {'task': 'test', 'unknown': 'test'}, {'task': 3},
])
def test_invalid_sections_fail_before_launch(value):
    with pytest.raises((ValueError, TypeError)):
        AgentPromptSections.from_mapping(value)


def test_conflicting_inline_and_file_prompt_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='not both'):
        resolve_agent_prompt_files({'prompt_files': {'task': 'TASK.md'},
            'agents': [{'prompt': 'old', 'role_file': 'role.md'}]}, relative_to=tmp_path)


def test_real_ns_loader_snapshots_modular_prompt_and_runtime_adds_actual_limits():
    def module(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
        return result
    runner = module(ROOT / 'scripts/run_canonical_ns_campaign.py', 'modular_ns_runner')
    worker = module(ROOT / 'scripts/run_canonical_codex_agent.py', 'modular_ns_worker')
    config, _ = runner._load_config(ROOT / 'configs/ns_modular_prompt_example.json')
    runner._validate_config(config, inspect_initial_candidate=False)
    text = worker._research_prompt({'agent': config['agents'][0],
        'round': {'round_index': 2, 'hard_wall_seconds': 7200, 'time_budget_mode': 'hard_wall',
                  'development_sampling': {'max_cases': None, 'allowed_sources': ['base']}},
        'worker_settings': {'candidate_execution': {'device': 'cpu_or_cuda', 'cuda_available': True}},
        'candidate_execution': {'case_charged_wall_seconds': 300}})
    assert 'Navier–Stokes' in text and 'exploring new approaches' in text
    assert '7200 seconds' in text and '300-second' in text
    assert 'no cumulative Development Case-count limit' in text
    assert 'CPU and CUDA' in text and '512 total' not in text


def test_panel_preview_matches_the_instructions_actually_received_over_two_rounds():
    panel = build_toy_panel()
    received = []
    class Recording(ToyResearchAgent):
        def run(self, context):
            received.append((context.agent.agent_id, context.round_plan.round_index, context.prompt))
            return super().run(context)
    panel.programs.agents.update({'careful-agent': Recording(1), 'explorer-agent': Recording(2)})
    values = {'project_name': 'Markdown prompt test', 'round_count': 2,
        'agents': ['careful-agent', 'explorer-agent'],
        'agent_prompt_sections': {'task': 'Solve the public Poisson problems.',
                                  'resources': 'The supplied Cases and feedback are available.'},
        'agent_roles': {'careful-agent': 'Improve the current method.',
                        'explorer-agent': 'Explore new methods.'}}
    preview = {a['agent_id']: a['prompt'] for a in panel.prompt_preview(values)['agents']}
    launched = panel.create_and_launch(values)
    panel.app.wait(launched['run_id'], timeout=10)
    page = panel.run_page(launched['run_id'])
    assert page['status'] == 'completed', page.get('error')
    assert len(received) == 4
    assert all(prompt == preview[agent] for agent, _, prompt in received)
    assert 'Task output requirements' in preview['careful-agent']
    assert 'Explore new methods.' not in preview['careful-agent']


def test_panel_preview_http_uses_same_sections_without_starting_a_run():
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from pde_agent_verifier_solver.panel.server import _handler
    panel = build_toy_panel()
    server = ThreadingHTTPServer(('127.0.0.1', 0), _handler(panel))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        payload = {'agent_prompt_sections': {'task': 'Use the public task.'},
                   'agents': ['careful-agent']}
        request = Request(f'http://127.0.0.1:{server.server_port}/api/prompt-preview',
                          data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        assert len(result['agents']) == 1 and 'Use the public task.' in result['agents'][0]['prompt']
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
