import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "shared_agent_gpu_test_module", Path(__file__).resolve().parents[1]/"scripts"/"shared_agent_gpu.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def request(node):
    return {"worker_settings":{"agent_gpu_execution":node}}


def test_legacy_cpu_has_no_gpu_gate():
    assert MODULE.agent_gpu_backend({}) is None
    assert MODULE.agent_gpu_backend(request(None)) is None


def test_gpu_gate_requires_exact_policy():
    policy = {"protocol":"shared_gpu_v1", "sandbox_backend":"legacy-landlock", "required":True}
    assert MODULE.agent_gpu_backend(request(policy)) == "legacy-landlock"
    for patch in ({"protocol":"unknown"}, {"sandbox_backend":"danger-full-access"},
                  {"required":False}, {"required":1}, {"extra":True}):
        with pytest.raises(ValueError):
            MODULE.agent_gpu_backend(request({**policy, **patch}))


def test_gpu_gate_rejects_weakened_config_before_launch(tmp_path):
    config = tmp_path/"config.toml"
    config.write_text('sandbox_mode = "danger-full-access"\n')
    policy = {"protocol":"shared_gpu_v1", "sandbox_backend":"legacy-landlock", "required":True}
    with pytest.raises(ValueError, match="workspace-write"):
        MODULE.qualify_agent_gpu(request(policy), codex_binary=Path('/nonexistent'),
                                 config_source=config, runtime_root=tmp_path, workspace=tmp_path)
