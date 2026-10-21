#!/usr/bin/env python3
"""Slurm-only CUDA and filesystem qualification of an opt-in Codex sandbox.

No model or authentication is used. No host policy or installed binary changes.
This qualifies cooperative research execution, not hostile same-UID isolation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import stat
import subprocess
import sys
from threading import Thread
import time
import xml.etree.ElementTree as ET


PROBE = r'''
import errno, json, os
from pathlib import Path
from urllib.request import urlopen
import torch

settings = json.loads(Path('probe-settings.json').read_text())
checks, details = {}, {}
checks['cuda_available'] = torch.cuda.is_available()
if not checks['cuda_available']:
    details['device_nodes'] = [str(p) for p in Path('/dev').glob('nvidia*')]
    try:
        torch.cuda.init()
    except Exception as exc:
        details['cuda_initialization_error'] = type(exc).__name__ + ': ' + str(exc)
    print(json.dumps({'checks':checks,'details':details}), flush=True)
    raise SystemExit(2)
checks['exactly_one_visible_gpu'] = torch.cuda.device_count() == 1
torch.manual_seed(19)
model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
x = torch.randn(64, 2, device='cuda')
y = (x[:, :1] - 0.5*x[:, 1:])
initial = float(torch.mean((model(x)-y)**2).item())
for _ in range(24):
    optimizer.zero_grad()
    loss = torch.mean((model(x)-y)**2)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
final = float(torch.mean((model(x)-y)**2).item())
checks['gpu_backward_improves_loss'] = final < initial
torch.save(model.state_dict(), 'weights.pt')
restored = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).cuda()
restored.load_state_dict(torch.load('weights.pt', map_location='cuda', weights_only=True))
checks['checkpoint_restore'] = bool(torch.allclose(model(x), restored(x), atol=1e-6, rtol=1e-6))
torch.cuda.synchronize()
props = torch.cuda.get_device_properties(0)
details.update(device_name=props.name, device_uuid=str(getattr(props, 'uuid', '')),
               initial_loss=initial, final_loss=final,
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               torch_version=torch.__version__, peak_allocated_bytes=torch.cuda.max_memory_allocated())
if settings.get('expected_gpu_uuid'):
    # NVIDIA's XML includes GPU-, whereas torch properties may omit it.
    checks['allocated_gpu_uuid'] = (details['device_uuid'].removeprefix('GPU-').lower()
        == settings['expected_gpu_uuid'].removeprefix('GPU-').lower())
Path('inside-canary').write_text('inside')
checks['workspace_write'] = True
def must_deny(name, path):
    try:
        Path(path).write_text('UNEXPECTED')
        checks[name] = False
    except OSError as exc:
        checks[name] = exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOENT)
        details[name] = {'errno':exc.errno, 'error':str(exc)}
must_deny('outside_write_denied', settings['outside'])
for name in ('.git', '.codex', '.agents'):
    must_deny(name + '_write_denied', str(Path(name, 'canary')))
with urlopen(settings['loopback'], timeout=5) as response:
    checks['public_loopback'] = response.read() == b'gpu-qualification'
print(json.dumps({'checks':checks,'details':details}), flush=True)
raise SystemExit(0 if all(checks.values()) else 3)
'''


_REQUIRED_CHECKS = frozenset((
    'cuda_available', 'exactly_one_visible_gpu', 'gpu_backward_improves_loss',
    'checkpoint_restore', 'workspace_write', 'outside_write_denied',
    '.git_write_denied', '.codex_write_denied', '.agents_write_denied',
    'public_loopback',
))


def _stop_process_group(process: subprocess.Popen) -> None:
    """Reap only the private process group created for this qualification."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # The server may have exited before one of its command children. Clean the
    # private group even when wait() has already reaped the server itself.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def _run_sandbox_command(command, *, workspace: Path, environment: dict,
                         output: Path, timeout: float = 120) -> subprocess.CompletedProcess:
    """Bound the CLI, its helper group, and capture draining without using auth.

    The pinned helper remains in this private group. Its separate bubblewrap
    group/session is coupled to helper death by upstream PDEATHSIG and
    --die-with-parent; this is not containment of arbitrary hostile descendants.
    """
    process = subprocess.Popen(
        command, cwd=workspace, env=environment,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    stdout = stderr = ''

    def capture_text(value):
        # TimeoutExpired can carry bytes even for a text-mode Popen.
        return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''

    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = capture_text(exc.output), capture_text(exc.stderr)
            _stop_process_group(process)
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired as drain:
                stdout, stderr = capture_text(drain.output), capture_text(drain.stderr)
                raise RuntimeError(
                    'sandbox qualification timed out; capture pipes remained open after cleanup'
                ) from exc
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from exc
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        try:
            _stop_process_group(process)
        finally:
            # Never enter an unbounded communicate() on inherited capture pipes.
            process.stdout.close()
            process.stderr.close()
            (output / 'stdout.log').write_text(stdout)
            (output / 'stderr.log').write_text(stderr)


def _run_app_server_command(server_command, probe_command, *, workspace: Path,
                            environment: dict, output: Path,
                            timeout: float = 120) -> subprocess.CompletedProcess:
    """Use the pinned public command/exec ABI, without a thread or model turn.

    Unlike `sandbox -P`, this API accepts the legacy SandboxPolicy used by the
    actual Agent. No policy translation or full-access fallback is attempted.
    """
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        server_command, cwd=workspace, env=environment,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    stdout_buffer = b''
    total_output = 0
    output_cap = 8 * 1024 * 1024
    try:
        with (output / 'app-server.jsonl').open('wb') as protocol_log, \
                (output / 'app-server.stderr.log').open('wb') as stderr_log:
            selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
            selector.register(process.stderr, selectors.EVENT_READ, 'stderr')

            def send(message):
                process.stdin.write(json.dumps(message).encode('utf-8') + b'\n')
                process.stdin.flush()

            def response(request_id):
                nonlocal stdout_buffer, total_output
                while True:
                    while b'\n' in stdout_buffer:
                        line, stdout_buffer = stdout_buffer.split(b'\n', 1)
                        if not line.strip():
                            continue
                        try:
                            message = json.loads(line)
                        except (ValueError, UnicodeDecodeError) as exc:
                            raise RuntimeError('app-server emitted invalid JSONL') from exc
                        if not isinstance(message, dict):
                            raise RuntimeError('app-server message is not an object')
                        if 'method' in message:
                            if 'id' in message:
                                raise RuntimeError('qualification refuses server-initiated requests')
                            # Startup and command notifications carry no decision
                            # authority. Only the matching response determines success.
                            continue
                        if message.get('id') != request_id:
                            raise RuntimeError('app-server returned an unexpected response id')
                        if 'error' in message:
                            raise RuntimeError('app-server RPC error: ' + json.dumps(message['error']))
                        if not isinstance(message.get('result'), dict):
                            raise RuntimeError('app-server response has no object result')
                        return message['result']
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(server_command, timeout)
                    if not selector.get_map():
                        raise RuntimeError('app-server exited before its response')
                    for key, _ in selector.select(timeout=min(remaining, 0.5)):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            if key.data == 'stdout':
                                raise RuntimeError('app-server closed stdout before its response')
                            continue
                        total_output += len(chunk)
                        if total_output > output_cap:
                            raise RuntimeError('app-server qualification exceeded output cap')
                        if key.data == 'stderr':
                            stderr_log.write(chunk)
                            stderr_log.flush()
                        else:
                            protocol_log.write(chunk)
                            protocol_log.flush()
                            stdout_buffer += chunk

            send({'id': 1, 'method': 'initialize', 'params': {
                'clientInfo': {'name': 'gpu-qualification', 'version': '1'},
                'capabilities': {'experimentalApi': True},
            }})
            response(1)
            send({'method': 'initialized', 'params': {}})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(server_command, timeout)
            send({'id': 2, 'method': 'command/exec', 'params': {
                'command': list(probe_command), 'cwd': str(workspace),
                'sandboxPolicy': {
                    'type': 'workspaceWrite', 'writableRoots': [],
                    'networkAccess': True,
                    'excludeSlashTmp': False, 'excludeTmpdirEnvVar': False,
                },
                'timeoutMs': max(1, int(remaining * 1000)),
                'outputBytesCap': 1024 * 1024,
            }})
            result = response(2)
            if (type(result.get('exitCode')) is not int
                    or not isinstance(result.get('stdout'), str)
                    or not isinstance(result.get('stderr'), str)):
                raise RuntimeError('invalid command/exec result shape')
            return subprocess.CompletedProcess(
                probe_command, result['exitCode'], result['stdout'], result['stderr'])
    finally:
        selector.close()
        _stop_process_group(process)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def _gpu_device_paths(requested) -> list[str]:
    """Accept exact assigned NVIDIA character devices, never device directories."""
    requested = tuple(requested or ())
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('bwrap-gpu requires distinct explicit GPU device paths')
    paths = []
    for value in requested:
        if not isinstance(value, str) or re.fullmatch(r'/dev/nvidia[0-9]+', value) is None:
            raise ValueError('GPU path must be an exact /dev/nvidiaN device')
        # lstat deliberately rejects symlinks instead of following them to an
        # unexpected device or introducing wider path permissions.
        if not stat.S_ISCHR(Path(value).lstat().st_mode):
            raise ValueError('GPU path must be a non-symlink character device')
        paths.append(value)
    for value in ('/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools'):
        try:
            metadata = Path(value).lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISCHR(metadata.st_mode):
            raise ValueError('NVIDIA support path is not a non-symlink character device')
        paths.append(value)
    return paths


def _allocated_gpu_mapping(requested, *, environment=None) -> dict:
    """Resolve the one visible GPU's actual Linux minor via NVIDIA's XML API."""
    environment = os.environ if environment is None else environment
    token = environment.get('CUDA_VISIBLE_DEVICES', '').strip()
    if re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', token) is None:
        raise ValueError('GPU qualification requires exactly one explicit visible non-MIG GPU')
    command = ['nvidia-smi', '-q', '-x', '-i', token]
    process = subprocess.run(command, capture_output=True, text=True, timeout=10,
                             env=dict(environment))
    if process.returncode != 0:
        raise RuntimeError('nvidia-smi could not resolve the allocated GPU')
    try:
        devices = ET.fromstring(process.stdout).findall('gpu')
    except ET.ParseError as exc:
        raise RuntimeError('nvidia-smi did not return valid device XML') from exc
    if len(devices) != 1:
        raise RuntimeError('nvidia-smi did not resolve exactly one GPU')
    minor = (devices[0].findtext('minor_number') or '').strip()
    uuid = (devices[0].findtext('uuid') or '').strip()
    if re.fullmatch(r'[0-9]+', minor) is None or not uuid.startswith('GPU-'):
        raise RuntimeError('nvidia-smi returned no concrete GPU minor/UUID')
    device_path = '/dev/nvidia' + str(int(minor))
    if tuple(requested or ()) != (device_path,):
        raise ValueError('explicit GPU device path does not match the allocated GPU minor: ' + device_path)
    if token.startswith('GPU-') and token != uuid:
        raise ValueError('visible GPU UUID does not match NVIDIA device metadata')
    return {'visible_token': token, 'uuid': uuid, 'minor_number': int(minor),
            'device_path': device_path, 'query_command': command}


def qualify(codex_binary: Path, output: Path, *, mode: str = 'legacy-landlock',
            gpu_device_paths=None, gpu_device_access: str = 'read',
            deny_other_devices: bool = False) -> dict:
    if not os.environ.get('SLURM_JOB_ID') or 'login' in platform.node().split('.')[0].lower():
        raise RuntimeError('GPU qualification must run inside a compute allocation')
    binary = codex_binary.resolve(strict=True)
    if mode not in ('legacy-landlock', 'bwrap', 'bwrap-gpu'):
        raise ValueError('unknown sandbox mode')
    if mode != 'bwrap-gpu' and gpu_device_paths:
        raise ValueError('GPU device paths require explicit bwrap-gpu mode')
    if gpu_device_access not in ('read', 'write'):
        raise ValueError('unknown GPU device access rule')
    if deny_other_devices and (mode != 'bwrap-gpu' or gpu_device_access != 'read'):
        raise ValueError('deny-other-devices is only a bwrap-gpu/read diagnostic')
    exposed_devices = _gpu_device_paths(gpu_device_paths) if mode == 'bwrap-gpu' else []
    gpu_mapping = _allocated_gpu_mapping(gpu_device_paths) if mode == 'bwrap-gpu' else None
    basic_devices = []
    if deny_other_devices:
        for path in ('/dev/null', '/dev/zero', '/dev/random', '/dev/urandom'):
            if not stat.S_ISCHR(Path(path).lstat().st_mode):
                raise ValueError('basic device must be a non-symlink character device')
            basic_devices.append(path)
    output = output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    workspace, control = output / 'workspace', output / 'control'
    for path in (workspace, control, output / 'agent-home', workspace / 'tmp'):
        path.mkdir(mode=0o700)
    for name in ('.git', '.codex', '.agents'):
        (workspace / name).mkdir(mode=0o700)
        (workspace / name / 'canary').write_text('protected')
    outside = output / 'outside-canary'
    outside.write_text('outside')
    # Keep the real Agent's legacy workspace-write policy. Named permission
    # profiles force a different runtime path and cannot qualify that policy.
    if mode == 'legacy-landlock':
        # Named permission profiles mandate direct runtime enforcement and are
        # incompatible with the legacy backend. Use the actual canonical
        # Agent's legacy configuration vocabulary, with the same permissions.
        config = ('approval_policy = "never"\nsandbox_mode = "workspace-write"\n'
                  '[features]\nuse_legacy_landlock = true\n'
                  '[sandbox_workspace_write]\nnetwork_access = true\n')
    else:
        config = ('approval_policy = "never"\n'
                  'default_permissions = "gpu-qualification"\n'
                  '[features]\nuse_legacy_landlock = false\n'
                  '[permissions.gpu-qualification]\nextends = ":workspace"\n'
                  '[permissions.gpu-qualification.network]\nenabled = true\n')
        if exposed_devices:
            config += '[permissions.gpu-qualification.filesystem]\n'
            if deny_other_devices:
                config += '"/dev" = "deny"\n'
            config += ''.join(json.dumps(path) + ' = ' + json.dumps(gpu_device_access) + '\n'
                              for path in exposed_devices + basic_devices)
    (control / 'config.toml').write_text(config)
    (workspace / 'probe.py').write_text(PROBE)
    report = {'schema':'agent_gpu_sandbox_qualification_v1', 'status':'failed',
              'sandbox_backend':mode, 'hostname':platform.node(),
              'slurm_job_id':os.environ['SLURM_JOB_ID'], 'started_at_unix':time.time(),
              'binary':str(binary), 'binary_sha256':sha256(binary.read_bytes()).hexdigest(),
              'qualifier_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
              'probe_sha256':sha256(PROBE.encode()).hexdigest(),
              'authentication_read':False, 'model_invoked':False,
              'host_policy_modified':False, 'full_access_enabled':False}
    report['gpu_device_paths'] = exposed_devices
    report['gpu_device_access_rule'] = gpu_device_access if exposed_devices else None
    report['gpu_mapping'] = gpu_mapping
    report['deny_other_devices'] = deny_other_devices
    report['basic_device_read_paths'] = basic_devices
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'gpu-qualification')
        def log_message(self, *_args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval':0.1}, daemon=True)
    thread.start()
    (workspace / 'probe-settings.json').write_text(json.dumps({
        'outside':str(outside), 'loopback':f'http://127.0.0.1:{server.server_port}/',
        'expected_gpu_uuid': gpu_mapping['uuid'] if gpu_mapping else None}))
    environment = {key: value for key, value in os.environ.items() if key in (
        'PATH','LANG','LC_ALL','LD_LIBRARY_PATH','LIBRARY_PATH','VIRTUAL_ENV',
        'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER')}
    environment.update(CODEX_HOME=str(control), HOME=str(output/'agent-home'),
                       TMPDIR=str(workspace/'tmp'), PYTHONNOUSERSITE='1',
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NO_PROXY='127.0.0.1,localhost')
    if gpu_mapping is not None:
        # A device-node sandbox changes CUDA's available ordinal list. Keep the
        # allocated physical identity when that list contains only this GPU.
        environment['CUDA_VISIBLE_DEVICES'] = gpu_mapping['uuid']
    probe_command = [sys.executable, str(workspace / 'probe.py')]
    if mode == 'legacy-landlock':
        command = [str(binary), 'app-server', '--stdio', '--enable', 'use_legacy_landlock']
        report['transport'] = 'app-server-command-exec-v1'
    else:
        command = [str(binary), 'sandbox', '--permission-profile', 'gpu-qualification',
                   '-C', str(workspace), '--', *probe_command]
        report['transport'] = 'sandbox-cli-permission-profile-v1'
    report['command'] = command
    report['probe_command'] = probe_command
    try:
        if mode == 'legacy-landlock':
            process = _run_app_server_command(command, probe_command, workspace=workspace,
                                              environment=environment, output=output)
        else:
            process = _run_sandbox_command(command, workspace=workspace,
                                           environment=environment, output=output)
        (output/'stdout.log').write_text(process.stdout)
        (output/'stderr.log').write_text(process.stderr)
        report['returncode'] = process.returncode
        if process.stdout.strip():
            try:
                report['probe'] = json.loads(process.stdout.strip().splitlines()[-1])
            except ValueError:
                pass
        report['outside_unchanged'] = outside.read_text() == 'outside'
        report['protected_paths_unchanged'] = all(
            (workspace/name/'canary').read_text() == 'protected' for name in ('.git','.codex','.agents'))
        checks = report.get('probe', {}).get('checks', {})
        required_checks = _REQUIRED_CHECKS | ({'allocated_gpu_uuid'} if gpu_mapping else set())
        checks_pass = (isinstance(checks, dict)
                       and required_checks.issubset(checks)
                       and all(value is True for value in checks.values()))
        if (process.returncode == 0 and checks_pass
                and report['outside_unchanged'] and report['protected_paths_unchanged']):
            report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        report['finished_at_unix'] = time.time()
        (output/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('legacy-landlock','bwrap','bwrap-gpu'), default='legacy-landlock')
    parser.add_argument('--gpu-device-path', action='append', default=[],
                        help='Exact allocated /dev/nvidiaN character device; bwrap-gpu only')
    parser.add_argument('--gpu-device-access', choices=('read', 'write'), default='read',
                        help='Filesystem rule for explicit device nodes; read does not mean read-only GPU operations')
    parser.add_argument('--deny-other-devices', action='store_true',
                        help='Diagnostic only: deny /dev except exact GPU/basic read grants; not a general PTY qualification')
    args = parser.parse_args(argv)
    report = qualify(args.codex_binary, args.output, mode=args.mode,
                     gpu_device_paths=args.gpu_device_path, gpu_device_access=args.gpu_device_access,
                     deny_other_devices=args.deny_other_devices)
    print(json.dumps({'status':report['status'], 'report':str(args.output/'report.json')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
