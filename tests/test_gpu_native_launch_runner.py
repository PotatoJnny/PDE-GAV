"""Native GPU handoff tests: no real auth, subprocess, model, or CUDA calls."""
from hashlib import sha256
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


class FakeAuthSource:
    def __init__(self):
        self.reads = 0

    def read_bytes(self):
        self.reads += 1
        return b"credential-free unit-test fixture"


class MockRuntimeReached(Exception):
    """Raised before any supervisor/native process can actually start."""


@pytest.fixture
def runner_setup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    gpu = importlib.import_module("shared_agent_gpu")
    spec = importlib.util.spec_from_file_location(
        "_gpu_native_launch_runner_test", ROOT / "scripts/run_canonical_codex_agent.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    workspace, runtime = tmp_path / "workspace", tmp_path / "runtime"
    workspace.mkdir()
    runtime.mkdir()
    binary = tmp_path / "fake-codex"
    binary.write_bytes(b"isolated binary fixture; never executed")
    original = tmp_path / "original.toml"
    original.write_text('sandbox_mode = "workspace-write"\n')
    private = tmp_path / "native.config.toml"
    private.write_text('default_permissions = "agent-gpu-v1"\n'
                       '[features]\nuse_legacy_landlock = false\n'
                       '[permissions.agent-gpu-v1]\nextends = ":workspace"\n')
    private.chmod(0o400)
    launch = gpu.AgentGpuNativeLaunch(
        "bwrap-nvidia-v1", private, ("--disable", "use_legacy_landlock"),
        sha256(binary.read_bytes()).hexdigest(), sha256(private.read_bytes()).hexdigest())
    auth = FakeAuthSource()
    kwargs = dict(workspace=workspace, prompt="Mock only", codex_binary=binary,
                  auth_source=auth, config_source=original, runtime_root=runtime,
                  final_message=workspace / "final.txt", auth_audit_copy=workspace / "audit.json")
    monkeypatch.delenv("PDE_AGENT_DEADLINE_MONOTONIC", raising=False)
    monkeypatch.setattr(worker, "_codex_environment", lambda **kw: {
        "CODEX_HOME": str(kw["codex_home"]), "HOME": str(kw["agent_home"])})
    calls = []

    def fake_run(command, **options):
        home = Path(options["env"]["CODEX_HOME"])
        calls.append({"command": tuple(command), "config": (home / "config.toml").read_bytes(), "env": dict(options["env"])})
        raise MockRuntimeReached()

    monkeypatch.setattr(worker, "subprocess", SimpleNamespace(run=fake_run))
    return worker, gpu, launch, auth, kwargs, calls


def test_descriptor_uses_private_config_and_disables_legacy_backend(runner_setup):
    worker, _, launch, auth, kwargs, calls = runner_setup
    with pytest.raises(MockRuntimeReached):
        worker._run_codex(**kwargs, gpu_sandbox_backend=launch)
    assert len(calls) == 1 and auth.reads == 1
    native = calls[0]["command"]
    assert calls[0]["config"] == launch.config_source.read_bytes()
    assert calls[0]["config"] != kwargs["config_source"].read_bytes()
    assert any(native[index:index + 2] == ("--disable", "use_legacy_landlock")
               for index in range(len(native) - 1))
    assert not any(native[index:index + 2] == ("--enable", "use_legacy_landlock")
                   for index in range(len(native) - 1))
    assert "--sandbox" not in native
    assert "danger-full-access" not in native
    assert list(kwargs["runtime_root"].iterdir()) == []


@pytest.mark.parametrize("target", ["binary", "config"])
def test_existing_descriptor_is_loaded_without_checksum_scan(runner_setup, target):
    worker, _, launch, auth, kwargs, calls = runner_setup
    path = kwargs["codex_binary"] if target == "binary" else launch.config_source
    path.chmod(0o600)
    path.write_bytes(b"tampered")
    with pytest.raises(MockRuntimeReached):
        worker._run_codex(**kwargs, gpu_sandbox_backend=launch)
    assert auth.reads == 1 and len(calls) == 1
    assert list(kwargs["runtime_root"].iterdir()) == []


@pytest.mark.parametrize("backend", ["bwrap-nvidia-v1", "danger-full-access", "bwrap", {}])
def test_unqualified_bare_backend_rejected_before_auth_read(runner_setup, backend):
    worker, _, _, auth, kwargs, calls = runner_setup
    with pytest.raises(ValueError, match="unqualified"):
        worker._run_codex(**kwargs, gpu_sandbox_backend=backend)
    assert auth.reads == 0 and calls == []
    assert list(kwargs["runtime_root"].iterdir()) == []


def test_changed_private_copy_is_rejected_before_native_subprocess(runner_setup, monkeypatch):
    worker, _, launch, auth, kwargs, calls = runner_setup

    def tamper_copy(**kw):
        (kw["codex_home"] / "config.toml").write_bytes(b"changed private copy")
        return {"CODEX_HOME": str(kw["codex_home"]), "HOME": str(kw["agent_home"])}

    monkeypatch.setattr(worker, "_codex_environment", tamper_copy)
    with pytest.raises(ValueError, match="private native GPU config changed"):
        worker._run_codex(**kwargs, gpu_sandbox_backend=launch)
    assert auth.reads == 1 and calls == []
    assert list(kwargs["runtime_root"].iterdir()) == []


@pytest.mark.parametrize("backend,expected", [(None, None), ("legacy-landlock", "--enable")])
def test_historical_cpu_and_legacy_cli_flags_stay_unchanged(runner_setup, backend, expected):
    worker, _, _, _, kwargs, calls = runner_setup
    with pytest.raises(MockRuntimeReached):
        worker._run_codex(**kwargs, gpu_sandbox_backend=backend)
    assert calls[0]["config"] == kwargs["config_source"].read_bytes()
    command = calls[0]["command"]
    if expected is None:
        assert "use_legacy_landlock" not in command
    else:
        assert any(command[index:index + 2] == (expected, "use_legacy_landlock")
                   for index in range(len(command) - 1))


def test_nonzero_gpu_uses_qualified_uuid_in_native_child(runner_setup, monkeypatch):
    worker, _, launch, _, kwargs, calls = runner_setup
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    launch = launch._replace(gpu_uuid='GPU-nonzero-allocated-card')
    with pytest.raises(MockRuntimeReached):
        worker._run_codex(**kwargs, gpu_sandbox_backend=launch)
    assert calls[0]['env']['CUDA_VISIBLE_DEVICES'] == launch.gpu_uuid
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
