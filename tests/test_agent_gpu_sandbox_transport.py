"""CPU-only transport tests; actual CUDA/policy qualification stays on Slurm."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import stat
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    'agent_gpu_sandbox_transport',
    Path(__file__).resolve().parents[1] / 'scripts' / 'qualify_agent_gpu_sandbox.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SERVER = r'''
import json, os, sys, time
behavior = sys.argv[1]
first = json.loads(sys.stdin.readline())
assert first['method'] == 'initialize'
assert first['params']['capabilities']['experimentalApi'] is True
print(json.dumps({'id': first['id'], 'result': {}}), flush=True)
initialized = json.loads(sys.stdin.readline())
assert initialized == {'method': 'initialized', 'params': {}}
request = json.loads(sys.stdin.readline())
assert request['method'] == 'command/exec'
params = request['params']
assert 'permissionProfile' not in params
assert params['sandboxPolicy'] == {
    'type': 'workspaceWrite', 'writableRoots': [], 'networkAccess': True,
    'excludeSlashTmp': False, 'excludeTmpdirEnvVar': False}
assert params['cwd'] == os.getcwd()
assert 0 < params['timeoutMs'] <= 10000
assert params['command'] == ['probe-python', 'probe.py']
print('transport stderr marker', file=sys.stderr, flush=True)
if behavior == 'hang':
    time.sleep(30)
elif behavior == 'eof':
    sys.exit(0)
elif behavior == 'rpc-error':
    print(json.dumps({'id': request['id'], 'error': {'code': -32602, 'message': 'bad policy'}}), flush=True)
elif behavior == 'server-request':
    print(json.dumps({'id': 99, 'method': 'approval/request', 'params': {}}), flush=True)
elif behavior == 'malformed':
    print('not-json', flush=True)
elif behavior == 'bad-result':
    print(json.dumps({'id': 2, 'result': {'exitCode': False, 'stdout': '', 'stderr': ''}}), flush=True)
else:
    print(json.dumps({'method': 'startup/notification', 'params': {}}), flush=True)
    print(json.dumps({'id': 2, 'result': {
        'exitCode': 0, 'stdout': '{"checks":{}}\n', 'stderr': ''}}), flush=True)
    time.sleep(30)
'''


def run_server(tmp_path, behavior, *, timeout=5):
    return MODULE._run_app_server_command(
        [sys.executable, '-u', '-c', SERVER, behavior], ['probe-python', 'probe.py'],
        workspace=tmp_path, environment=dict(os.environ), output=tmp_path, timeout=timeout)


def test_jsonl_handshake_command_and_notifications(tmp_path):
    result = run_server(tmp_path, 'success')
    assert result.returncode == 0
    assert json.loads(result.stdout) == {'checks': {}}
    assert 'startup/notification' in (tmp_path / 'app-server.jsonl').read_text()


@pytest.mark.parametrize(('behavior', 'error'), [
    ('rpc-error', 'RPC error'),
    ('server-request', 'server-initiated'),
    ('eof', 'closed stdout'),
    ('malformed', 'invalid JSONL'),
    ('bad-result', 'result shape'),
])
def test_jsonl_fails_closed(tmp_path, behavior, error):
    with pytest.raises(RuntimeError, match=error):
        run_server(tmp_path, behavior)


def test_transport_deadline_is_bounded(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_server(tmp_path, 'hang', timeout=0.5)


def test_zero_exit_without_complete_probe_checks_never_passes(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'qualification-unit-test')
    monkeypatch.setattr(MODULE.platform, 'node', lambda: 'compute-test')
    monkeypatch.setattr(MODULE, '_run_app_server_command', lambda *a, **k:
                        subprocess.CompletedProcess([], 0, '{"checks":{}}\n', ''))
    report = MODULE.qualify(Path(sys.executable), tmp_path / 'qualification')
    assert report['returncode'] == 0
    assert report['outside_unchanged'] is True
    assert report['protected_paths_unchanged'] is True
    assert report['status'] == 'failed'
    assert report['transport'] == 'app-server-command-exec-v1'
    assert report['model_invoked'] is False


@pytest.mark.parametrize('paths', [[], ['/dev'], ['/dev/nvidia*'], ['/dev/nvidiactl'],
                                  ['/dev/nvidia0/../nvidia1'], ['nvidia0'],
                                  ['/dev/nvidia0', '/dev/nvidia0']])
def test_gpu_device_grants_reject_broad_or_ambiguous_paths(paths):
    with pytest.raises(ValueError):
        MODULE._gpu_device_paths(paths)


def test_gpu_device_grants_reject_symlinks(monkeypatch):
    monkeypatch.setattr(Path, 'lstat', lambda self: SimpleNamespace(st_mode=stat.S_IFLNK))
    with pytest.raises(ValueError, match='non-symlink character'):
        MODULE._gpu_device_paths(['/dev/nvidia0'])


def test_gpu_device_grants_only_existing_nvidia_character_devices(monkeypatch):
    def metadata(path):
        if str(path) == '/dev/nvidia-uvm-tools':
            raise FileNotFoundError(str(path))
        return SimpleNamespace(st_mode=stat.S_IFCHR)
    monkeypatch.setattr(Path, 'lstat', metadata)
    assert MODULE._gpu_device_paths(['/dev/nvidia3']) == [
        '/dev/nvidia3', '/dev/nvidiactl', '/dev/nvidia-uvm']


def test_visible_zero_resolves_actual_minor_not_assumed_zero(monkeypatch):
    def query(command, **kwargs):
        assert command == ['nvidia-smi', '-q', '-x', '-i', '0']
        assert kwargs['timeout'] == 10
        return subprocess.CompletedProcess(command, 0,
            '<nvidia_smi_log><gpu><minor_number>3</minor_number>'
            '<uuid>GPU-test-uuid</uuid></gpu></nvidia_smi_log>', '')
    monkeypatch.setattr(MODULE.subprocess, 'run', query)
    result = MODULE._allocated_gpu_mapping(['/dev/nvidia3'], environment={'CUDA_VISIBLE_DEVICES': '0'})
    assert result['minor_number'] == 3
    assert result['uuid'] == 'GPU-test-uuid'
    with pytest.raises(ValueError, match='does not match'):
        MODULE._allocated_gpu_mapping(['/dev/nvidia0'], environment={'CUDA_VISIBLE_DEVICES': '0'})


@pytest.mark.parametrize('token', ['', '0,1', 'MIG-device', '../0', 'GPU-*'])
def test_visible_gpu_selector_fails_closed(token):
    with pytest.raises(ValueError, match='exactly one explicit'):
        MODULE._allocated_gpu_mapping(['/dev/nvidia0'], environment={'CUDA_VISIBLE_DEVICES': token})


@pytest.mark.parametrize('xml', ['not-xml', '<nvidia_smi_log/>',
    '<nvidia_smi_log><gpu><minor_number>unknown</minor_number><uuid>GPU-x</uuid></gpu></nvidia_smi_log>',
    '<nvidia_smi_log><gpu/><gpu/></nvidia_smi_log>'])
def test_mapping_rejects_ambiguous_or_missing_nvidia_metadata(monkeypatch, xml):
    monkeypatch.setattr(MODULE.subprocess, 'run', lambda *a, **k:
                        subprocess.CompletedProcess([], 0, xml, ''))
    with pytest.raises(RuntimeError):
        MODULE._allocated_gpu_mapping(['/dev/nvidia0'], environment={'CUDA_VISIBLE_DEVICES': '0'})


def test_explicit_gpu_uuid_cannot_silently_map_to_another_device(monkeypatch):
    monkeypatch.setattr(MODULE.subprocess, 'run', lambda *a, **k:
        subprocess.CompletedProcess([], 0,
            '<nvidia_smi_log><gpu><minor_number>3</minor_number>'
            '<uuid>GPU-other</uuid></gpu></nvidia_smi_log>', ''))
    with pytest.raises(ValueError, match='UUID does not match'):
        MODULE._allocated_gpu_mapping(['/dev/nvidia3'], environment={'CUDA_VISIBLE_DEVICES': 'GPU-requested'})


@pytest.mark.parametrize(('mode', 'access'), [('bwrap', 'read'), ('legacy-landlock', 'read'),
                                           ('bwrap-gpu', 'write')])
def test_device_deny_diagnostic_rejects_other_modes(tmp_path, monkeypatch, mode, access):
    monkeypatch.setenv('SLURM_JOB_ID', 'qualification-unit-test')
    monkeypatch.setattr(MODULE.platform, 'node', lambda: 'compute-test')
    with pytest.raises(ValueError, match='only a bwrap-gpu/read'):
        MODULE.qualify(Path(sys.executable), tmp_path / 'qualification', mode=mode,
                       gpu_device_access=access, deny_other_devices=True)


def test_device_deny_profile_is_exact_and_not_broad_write(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'qualification-unit-test')
    monkeypatch.setattr(MODULE.platform, 'node', lambda: 'compute-test')
    monkeypatch.setattr(MODULE, '_gpu_device_paths', lambda requested: ['/dev/nvidia3', '/dev/nvidiactl'])
    monkeypatch.setattr(MODULE, '_allocated_gpu_mapping', lambda requested:
                        {'uuid': 'GPU-test', 'device_path': '/dev/nvidia3'})
    original_lstat = Path.lstat
    def metadata(path):
        if str(path) in ('/dev/null', '/dev/zero', '/dev/random', '/dev/urandom'):
            return SimpleNamespace(st_mode=stat.S_IFCHR)
        return original_lstat(path)
    monkeypatch.setattr(Path, 'lstat', metadata)
    monkeypatch.setattr(MODULE, '_run_sandbox_command', lambda *a, **k:
                        subprocess.CompletedProcess([], 0, '{"checks":{}}\n', ''))
    output = tmp_path / 'qualification'
    report = MODULE.qualify(Path(sys.executable), output, mode='bwrap-gpu',
                            gpu_device_paths=['/dev/nvidia3'], deny_other_devices=True)
    config = (output / 'control' / 'config.toml').read_text()
    assert '"/dev" = "deny"' in config
    assert '"/dev/nvidia3" = "read"' in config
    assert '"/dev/null" = "read"' in config
    assert ' = "write"' not in config
    assert 'extends = ":workspace"' in config
    assert report['deny_other_devices'] is True
    assert report['status'] == 'failed'


def test_gpu_probe_uses_allocated_uuid_after_device_filtering(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'qualification-unit-test')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setattr(MODULE.platform, 'node', lambda: 'compute-test')
    monkeypatch.setattr(MODULE, '_gpu_device_paths', lambda requested: ['/dev/nvidia1', '/dev/nvidiactl'])
    mapping = {'visible_token': '1', 'uuid': 'GPU-allocated-one', 'device_path': '/dev/nvidia1'}
    monkeypatch.setattr(MODULE, '_allocated_gpu_mapping', lambda requested: mapping)
    seen = []
    def run_probe(*args, **kwargs):
        seen.append(kwargs['environment']['CUDA_VISIBLE_DEVICES'])
        return subprocess.CompletedProcess([], 0, '{"checks":{}}\n', '')
    monkeypatch.setattr(MODULE, '_run_sandbox_command', run_probe)
    report = MODULE.qualify(Path(sys.executable), tmp_path/'qualification', mode='bwrap-gpu',
                            gpu_device_paths=['/dev/nvidia1'], gpu_device_access='write')
    assert seen == ['GPU-allocated-one']
    assert report['gpu_mapping']['visible_token'] == '1'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
