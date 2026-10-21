"""Fail-closed native Agent GPU qualification, opt-in and equation independent."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import tomllib
from typing import Mapping, NamedTuple
import xml.etree.ElementTree as ET


class AgentGpuNativeLaunch(NamedTuple):
    """A Controller-prepared native launch, following a CUDA capability probe."""

    sandbox_backend: str
    config_source: Path
    cli_overrides: tuple[str, ...]
    runtime_sha256: str | None
    config_sha256: str
    gpu_uuid: str | None = None


def validate_agent_gpu_native_launch(launch: AgentGpuNativeLaunch, *, codex_binary: Path) -> None:
    """Check launch shape without rereading the binary or config for checksums."""
    if (not isinstance(launch, AgentGpuNativeLaunch)
            or launch.sandbox_backend != 'bwrap-nvidia-v1'
            or launch.cli_overrides != ('--disable', 'use_legacy_landlock')):
        raise ValueError('invalid qualified GPU native launch')
    if not codex_binary.is_file() or not launch.config_source.is_file():
        raise FileNotFoundError('GPU runtime or generated configuration is missing')


def agent_gpu_backend(request: Mapping[str, object]) -> str | None:
    settings = request.get('worker_settings', {})
    if not isinstance(settings, Mapping):
        raise TypeError('worker_settings must be an object')
    node = settings.get('agent_gpu_execution')
    if node is None:
        return None
    if not isinstance(node, Mapping):
        raise ValueError('Agent GPU execution needs an explicit qualified sandbox policy')
    backend = node.get('sandbox_backend')
    keys = {'protocol', 'sandbox_backend', 'required'}
    if ((set(node) - {'runtime_sha256'}) != keys
            or node.get('protocol') != 'shared_gpu_v1'
            or backend not in ('legacy-landlock', 'bwrap-nvidia-v1')
            or node.get('required') is not True):
        raise ValueError('Agent GPU execution needs the explicit shared_gpu_v1 qualified sandbox policy')
    return backend


def _allocated_nvidia_device() -> dict:
    """Discover Linux minor, which need not equal the CUDA visible index."""
    token = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', token) is None:
        raise ValueError('Agent GPU needs exactly one allocated non-MIG CUDA device')
    process = subprocess.run(['nvidia-smi', '-q', '-x', '-i', token],
                             capture_output=True, text=True, timeout=10)
    if process.returncode != 0:
        raise RuntimeError('cannot discover the allocated NVIDIA GPU')
    try:
        devices = ET.fromstring(process.stdout).findall('gpu')
    except ET.ParseError as exc:
        raise RuntimeError('invalid NVIDIA GPU metadata') from exc
    if len(devices) != 1:
        raise RuntimeError('NVIDIA metadata must identify exactly one GPU')
    minor = (devices[0].findtext('minor_number') or '').strip()
    uuid = (devices[0].findtext('uuid') or '').strip()
    if re.fullmatch(r'[0-9]+', minor) is None or re.fullmatch(r'GPU-[A-Za-z0-9-]+', uuid) is None:
        raise RuntimeError('NVIDIA metadata lacks a concrete minor/UUID')
    if token.startswith('GPU-') and token != uuid:
        raise ValueError('allocated GPU UUID does not match NVIDIA metadata')
    return {'visible_token': token, 'uuid': uuid, 'minor_number': int(minor),
            'device_path': '/dev/nvidia' + str(int(minor))}


def _gpu_native_config(source: str, device_paths: list[str]) -> str:
    """Replace only legacy sandbox syntax; prove all other TOML values unchanged."""
    config = tomllib.loads(source)
    if (config.get('sandbox_mode') != 'workspace-write'
            or config.get('approval_policy') != 'never'
            or config.get('sandbox_workspace_write') != {'network_access': True}):
        raise ValueError('GPU config requires the exact canonical workspace-write/never/network policy')
    if any(key in config for key in ('permissions', 'default_permissions', 'profile', 'profiles')):
        raise ValueError('GPU config must not include preexisting permission/profile selections')
    if (not device_paths or len(set(device_paths)) != len(device_paths)
            or sum(re.fullmatch(r'/dev/nvidia[0-9]+', path) is not None for path in device_paths) != 1
            or any(re.fullmatch(r'/dev/(?:nvidia[0-9]+|nvidiactl|nvidia-uvm|nvidia-uvm-tools)', path)
                   is None for path in device_paths)):
        raise ValueError('GPU config requires exact single-GPU device grants')
    expected = deepcopy(config)
    del expected['sandbox_mode']
    del expected['sandbox_workspace_write']
    expected['default_permissions'] = 'agent-gpu-v1'
    expected.setdefault('features', {})['use_legacy_landlock'] = False
    expected['permissions'] = {'agent-gpu-v1': {
        'extends': ':workspace', 'network': {'enabled': True},
        'filesystem': {path: 'write' for path in device_paths}}}
    lines = ['default_permissions = "agent-gpu-v1"\n']
    section = ''
    saw_features = False
    for line in source.splitlines(keepends=True):
        header = re.fullmatch(r'\s*\[([A-Za-z0-9_.-]+)\]\s*(?:#.*)?', line.rstrip('\n'))
        if header:
            section = header.group(1)
            if section == 'features':
                saw_features = True
                lines.extend([line, 'use_legacy_landlock = false\n'])
                continue
        if (section == 'sandbox_workspace_write'
                or (not section and re.match(r'\s*sandbox_mode\s*=', line))
                or (section == 'features' and re.match(r'\s*use_legacy_landlock\s*=', line))):
            continue
        lines.append(line)
    if not saw_features:
        lines.append('\n[features]\nuse_legacy_landlock = false\n')
    lines.append('\n[permissions.agent-gpu-v1]\nextends = ":workspace"\n'
                 '[permissions.agent-gpu-v1.network]\nenabled = true\n'
                 '[permissions.agent-gpu-v1.filesystem]\n')
    lines.extend(json.dumps(path) + ' = "write"\n' for path in device_paths)
    result = ''.join(lines)
    if tomllib.loads(result) != expected:
        raise ValueError('GPU config conversion did not preserve all non-sandbox configuration')
    return result


def qualify_agent_gpu(request, *, codex_binary: Path, config_source: Path,
                      runtime_root: Path, workspace: Path) -> str | AgentGpuNativeLaunch | None:
    backend = agent_gpu_backend(request)
    if backend is None:
        return None
    config_bytes = config_source.read_bytes()
    config = tomllib.loads(config_bytes.decode())
    if (config.get('sandbox_mode') != 'workspace-write'
            or config.get('approval_policy') != 'never'
            or config.get('sandbox_workspace_write', {}).get('network_access') is not True):
        raise ValueError('GPU development must preserve the canonical workspace-write/loopback policy')
    runtime_sha256 = None
    gpu_mapping = None
    if backend == 'bwrap-nvidia-v1':
        runtime_sha256 = request['worker_settings']['agent_gpu_execution'].get('runtime_sha256')
        if runtime_root.resolve().is_relative_to(workspace.resolve()):
            raise ValueError('GPU runtime configuration must remain outside the Agent workspace')
        # Validate policy shape before either the NVIDIA query or sandbox probe.
        _gpu_native_config(config_bytes.decode(), ['/dev/nvidia0'])
    raw_deadline = os.environ.get('PDE_AGENT_DEADLINE_MONOTONIC')
    if raw_deadline is not None:
        deadline = float(raw_deadline)
        # The new backend additionally performs two bounded device-metadata
        # queries (discovery and independent qualification verification).
        minimum_remaining = 150 if backend == 'bwrap-nvidia-v1' else 125
        if not isfinite(deadline) or deadline - time.monotonic() < minimum_remaining:
            raise RuntimeError('Agent round has insufficient remaining time for GPU qualification')
    # The probe lives outside the scientific workspace and never reads auth.
    # Retain failures for the allocation owner; never silently fall back to CPU.
    parent = Path(tempfile.mkdtemp(prefix='agent-gpu-gate-', dir=runtime_root))
    from qualify_agent_gpu_sandbox import qualify
    if backend == 'bwrap-nvidia-v1':
        gpu_mapping = _allocated_nvidia_device()
        report = qualify(codex_binary, parent/'qualification', mode='bwrap-gpu',
                         gpu_device_paths=[gpu_mapping['device_path']], gpu_device_access='write')
        returned_mapping = report.get('gpu_mapping') or {}
        if (any(returned_mapping.get(key) != value for key, value in gpu_mapping.items())
                or report.get('gpu_device_access_rule') != 'write'
                or config_source.read_bytes() != config_bytes):
            report['status'] = 'failed'
            report['launch_binding_error'] = 'runtime, allocation, device access, or source config changed'
    else:
        report = qualify(codex_binary, parent/'qualification', mode=backend)
    report['scientific_config_sha256'] = sha256(config_bytes).hexdigest()
    native_launch = None
    if backend == 'bwrap-nvidia-v1' and report['status'] == 'passed':
        from qualify_agent_gpu_sandbox import _gpu_device_paths
        device_paths = _gpu_device_paths([gpu_mapping['device_path']])
        if report.get('gpu_device_paths') != device_paths:
            report['status'] = 'failed'
            report['launch_binding_error'] = 'qualified device paths changed'
        else:
            derived = parent/'native.config.toml'
            derived_bytes = _gpu_native_config(config_bytes.decode(), device_paths).encode()
            with derived.open('xb') as handle:
                handle.write(derived_bytes)
            derived.chmod(0o400)
            derived_sha256 = sha256(derived_bytes).hexdigest()
            report['native_config_sha256'] = derived_sha256
            report['native_config_source'] = str(derived)
            native_launch = AgentGpuNativeLaunch(
                backend, derived, ('--disable', 'use_legacy_landlock'),
                runtime_sha256, derived_sha256, gpu_mapping['uuid'])
    report_path = parent/'qualification'/'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    public = {'schema':'agent-gpu-readiness-v1', 'status':report['status'],
              'protocol':'shared_gpu_v1', 'sandbox_backend':backend,
              'binary_sha256':report['binary_sha256'],
              'config_sha256':report['scientific_config_sha256'],
              'checks':report.get('probe', {}).get('checks', {}),
              'device':report.get('probe', {}).get('details', {}),
              'report_sha256':sha256(report_path.read_bytes()).hexdigest()}
    if native_launch is not None:
        public['native_config_sha256'] = native_launch.config_sha256
        public['gpu_mapping'] = gpu_mapping
    receipt = workspace/'AGENT_GPU_READINESS.json'
    with receipt.open('x') as handle:
        json.dump(public, handle, indent=2, sort_keys=True)
        handle.write('\n')
    receipt.chmod(0o400)
    if report['status'] != 'passed':
        raise RuntimeError(f'Agent GPU qualification failed; CPU fallback forbidden; retained report: {report_path}')
    return native_launch if native_launch is not None else backend
