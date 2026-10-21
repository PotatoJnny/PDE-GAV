import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import tomllib

import pytest


SPEC = importlib.util.spec_from_file_location(
    'shared_agent_gpu_bwrap_test_module',
    Path(__file__).resolve().parents[1] / 'scripts' / 'shared_agent_gpu.py',
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CANONICAL = '''approval_policy = "never"
sandbox_mode = "workspace-write"
web_search = "live"
allow_login_shell = false
[features]
guardian_approval = false
[sandbox_workspace_write]
network_access = true
[shell_environment_policy]
inherit = "all"
include_only = ["CUDA_VISIBLE_DEVICES", "PATH"]
'''
DEVICES = ['/dev/nvidia3', '/dev/nvidiactl', '/dev/nvidia-uvm']
MAPPING = {'visible_token': '0', 'uuid': 'GPU-abc-123', 'minor_number': 3,
           'device_path': '/dev/nvidia3'}


def request(runtime_hash):
    return {'worker_settings': {'agent_gpu_execution': {
        'protocol': 'shared_gpu_v1', 'sandbox_backend': 'bwrap-nvidia-v1',
        'required': True, 'runtime_sha256': runtime_hash}}}


def gate_fixture(tmp_path, monkeypatch, *, status='passed', report_patch=None):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    runtime = tmp_path / 'private-runtime'
    runtime.mkdir()
    binary = tmp_path / 'isolated-codex'
    binary.write_bytes(b'unit-test runtime fixture; never executed')
    config = tmp_path / 'canonical.toml'
    config.write_text(CANONICAL)
    runtime_hash = sha256(binary.read_bytes()).hexdigest()
    calls = []

    def fake_qualify(codex_binary, output, **kwargs):
        calls.append((codex_binary, output, kwargs))
        output.mkdir()
        return {'status': status, 'binary_sha256': runtime_hash,
                'gpu_mapping': dict(MAPPING), 'gpu_device_paths': list(DEVICES),
                'gpu_device_access_rule': 'write', 'probe': {'checks': {}, 'details': {}},
                **(report_patch or {})}

    monkeypatch.setitem(sys.modules, 'qualify_agent_gpu_sandbox', SimpleNamespace(
        qualify=fake_qualify, _gpu_device_paths=lambda requested: list(DEVICES)))
    monkeypatch.setattr(MODULE, '_allocated_nvidia_device', lambda: dict(MAPPING))
    monkeypatch.delenv('PDE_AGENT_DEADLINE_MONOTONIC', raising=False)
    kwargs = {'codex_binary': binary, 'config_source': config,
              'runtime_root': runtime, 'workspace': workspace}
    return request(runtime_hash), kwargs, calls


def test_new_backend_accepts_absent_or_historical_runtime_label():
    good = request('a' * 64)
    assert MODULE.agent_gpu_backend(good) == 'bwrap-nvidia-v1'
    for value in ('', 'a' * 63, 'A' * 64, None, True):
        assert MODULE.agent_gpu_backend(request(value)) == 'bwrap-nvidia-v1'
    del good['worker_settings']['agent_gpu_execution']['runtime_sha256']
    assert MODULE.agent_gpu_backend(good) == 'bwrap-nvidia-v1'


def test_config_preserves_science_options_and_only_adds_exact_device_grants():
    derived = tomllib.loads(MODULE._gpu_native_config(CANONICAL, DEVICES))
    original = tomllib.loads(CANONICAL)
    assert 'sandbox_mode' not in derived
    assert 'sandbox_workspace_write' not in derived
    assert derived.pop('default_permissions') == 'agent-gpu-v1'
    assert derived.pop('permissions') == {'agent-gpu-v1': {
        'extends': ':workspace', 'network': {'enabled': True},
        'filesystem': {path: 'write' for path in DEVICES}}}
    assert derived['features'].pop('use_legacy_landlock') is False
    del original['sandbox_mode'], original['sandbox_workspace_write']
    assert derived == original


@pytest.mark.parametrize('addition', [
    '\n[permissions.old]\nextends = ":workspace"\n',
    '\n[profiles.old]\nsandbox_mode = "danger-full-access"\n',
])
def test_config_rejects_preexisting_permission_profiles(addition):
    with pytest.raises(ValueError, match='preexisting'):
        MODULE._gpu_native_config(CANONICAL + addition, DEVICES)


@pytest.mark.parametrize('replacement', [
    'network_access = true\nwritable_roots = ["/outside"]',
    'network_access = true\nexclude_slash_tmp = true',
    'network_access = false',
])
def test_does_not_silently_drop_legacy_sandbox_settings(replacement):
    with pytest.raises(ValueError, match='exact canonical'):
        MODULE._gpu_native_config(CANONICAL.replace('network_access = true', replacement), DEVICES)


@pytest.mark.parametrize('paths', [[], ['/dev'], ['/dev/null'], ['/dev/nvidia*'],
                                   ['/dev/nvidia0', '/dev/nvidia1'],
                                   ['/dev/nvidia3', '/dev/nvidia3']])
def test_config_never_broadens_device_grants(paths):
    with pytest.raises(ValueError, match='exact single-GPU'):
        MODULE._gpu_native_config(CANONICAL, paths)


def test_allocated_minor_does_not_assume_cuda_index(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    seen = []

    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=(
            '<nvidia_smi_log><gpu><minor_number>3</minor_number>'
            '<uuid>GPU-abc-123</uuid></gpu></nvidia_smi_log>'))

    monkeypatch.setattr(MODULE.subprocess, 'run', fake_run)
    assert MODULE._allocated_nvidia_device() == MAPPING
    assert seen == [(['nvidia-smi', '-q', '-x', '-i', '0'],
                     {'capture_output': True, 'text': True, 'timeout': 10})]


@pytest.mark.parametrize('token', ['', '0,1', 'MIG-xyz', '*'])
def test_discovery_rejects_ambiguous_visible_devices(monkeypatch, token):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', token)
    monkeypatch.setattr(MODULE.subprocess, 'run', lambda *a, **kw: pytest.fail('must not invoke'))
    with pytest.raises(ValueError, match='exactly one'):
        MODULE._allocated_nvidia_device()


def test_passing_gate_produces_private_bound_launch_without_changing_original(tmp_path, monkeypatch):
    req, kwargs, calls = gate_fixture(tmp_path, monkeypatch)
    launch = MODULE.qualify_agent_gpu(req, **kwargs)
    assert isinstance(launch, MODULE.AgentGpuNativeLaunch)
    assert launch.sandbox_backend == 'bwrap-nvidia-v1'
    assert launch.cli_overrides == ('--disable', 'use_legacy_landlock')
    assert launch.gpu_uuid == MAPPING['uuid']
    assert calls[0][2] == {'mode': 'bwrap-gpu', 'gpu_device_paths': ['/dev/nvidia3'],
                           'gpu_device_access': 'write'}
    assert kwargs['config_source'].read_text() == CANONICAL
    assert not launch.config_source.is_relative_to(kwargs['workspace'])
    assert stat.S_IMODE(launch.config_source.stat().st_mode) == 0o400
    MODULE.validate_agent_gpu_native_launch(launch, codex_binary=kwargs['codex_binary'])
    receipt = json.loads((kwargs['workspace'] / 'AGENT_GPU_READINESS.json').read_text())
    assert receipt['status'] == 'passed'
    assert receipt['gpu_mapping'] == MAPPING
    assert receipt['native_config_sha256'] == launch.config_sha256
    assert receipt['config_sha256'] == sha256(CANONICAL.encode()).hexdigest()


def test_runtime_label_does_not_replace_actual_cuda_probe(tmp_path, monkeypatch):
    req, kwargs, calls = gate_fixture(tmp_path, monkeypatch)
    req['worker_settings']['agent_gpu_execution']['runtime_sha256'] = '0' * 64
    launch = MODULE.qualify_agent_gpu(req, **kwargs)
    assert launch.sandbox_backend == 'bwrap-nvidia-v1'
    assert len(calls) == 1


@pytest.mark.parametrize('status,patch', [
    ('failed', {}),
    ('passed', {'gpu_mapping': {**MAPPING, 'minor_number': 0}}),
    ('passed', {'gpu_device_access_rule': 'read'}),
    ('passed', {'gpu_device_paths': ['/dev/nvidia0']}),
])
def test_failure_or_mismatch_never_produces_native_launch(tmp_path, monkeypatch, status, patch):
    req, kwargs, calls = gate_fixture(tmp_path, monkeypatch, status=status, report_patch=patch)
    with pytest.raises(RuntimeError, match='CPU fallback forbidden'):
        MODULE.qualify_agent_gpu(req, **kwargs)
    assert len(calls) == 1
    assert not list(kwargs['runtime_root'].rglob('native.config.toml'))
    receipt = json.loads((kwargs['workspace'] / 'AGENT_GPU_READINESS.json').read_text())
    assert receipt['status'] == 'failed'


@pytest.mark.parametrize('target', ['runtime', 'config'])
def test_launch_uses_existing_files_without_reading_checksums(tmp_path, monkeypatch, target):
    req, kwargs, _ = gate_fixture(tmp_path, monkeypatch)
    launch = MODULE.qualify_agent_gpu(req, **kwargs)
    path = kwargs['codex_binary'] if target == 'runtime' else launch.config_source
    path.chmod(0o600)
    path.write_bytes(b'changed')
    MODULE.validate_agent_gpu_native_launch(launch, codex_binary=kwargs['codex_binary'])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        MODULE.validate_agent_gpu_native_launch(launch, codex_binary=kwargs['codex_binary'])
