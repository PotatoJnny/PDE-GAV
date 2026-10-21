"""No model calls or actual sandboxes: deterministic launch prerequisite tests."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("canonical_environment_test", ROOT / "scripts/check_canonical_codex_environment.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
CONFIG = ROOT / "scripts/canonical_codex.config.toml"
CHECKS = {"inside_write": True, "outside_write_denied": True, "loopback": True}


def _fake_process(monkeypatch, *, code=0, checks=CHECKS, error=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "codex-cli 0.149.0\n", "")
        assert command[1:4] == ["sandbox", "--permission-profile", "canonical-preflight"]
        assert "exec" not in command
        control = Path(kwargs["env"]["CODEX_HOME"])
        profile = tomllib.loads((control / "config.toml").read_text())
        assert profile["permissions"]["canonical-preflight"] == {
            "extends": ":workspace", "network": {"enabled": True},
        }
        assert profile["approval_policy"] == "never"
        assert not (control / "auth.json").exists()
        if error is not None:
            raise error
        return subprocess.CompletedProcess(command, code, json.dumps(checks), "exact sandbox error" if code else "")

    monkeypatch.setattr(PROBE.subprocess, "run", run)
    return calls


def test_preflight_maps_canonical_policy_without_auth_or_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-inherited")
    monkeypatch.setenv("PDE_AGENT_SECRET", "must-not-be-inherited")
    calls = _fake_process(monkeypatch)
    destination = tmp_path / "new"
    report = PROBE.check_environment(Path(sys.executable), CONFIG, destination)
    assert report["status"] == "passed"
    assert report["checks"] == CHECKS
    assert report["model_invoked"] is report["authentication_read"] is report["host_policy_modified"] is False
    assert len(calls) == 2
    assert "OPENAI_API_KEY" not in calls[-1][1]["env"]
    assert "PDE_AGENT_SECRET" not in calls[-1][1]["env"]
    assert (destination / "canonical-source.config.toml").read_bytes() == CONFIG.read_bytes()
    assert json.loads((destination / "report.json").read_text()) == report
    assert (destination.stat().st_mode & 0o777) == 0o700


def test_preflight_preserves_exact_failure_without_retry(tmp_path, monkeypatch):
    calls = _fake_process(monkeypatch, code=1)
    destination = tmp_path / "new"
    report = PROBE.check_environment(Path(sys.executable), CONFIG, destination)
    assert report["status"] == "failed"
    assert report["returncode"] == 1
    assert (destination / "sandbox.stderr.log").read_text() == "exact sandbox error"
    assert len(calls) == 2


@pytest.mark.parametrize("checks", [dict(CHECKS, outside_write_denied=False), dict(CHECKS, loopback=False), {}])
def test_preflight_fails_closed_when_boundary_missing(tmp_path, monkeypatch, checks):
    _fake_process(monkeypatch, checks=checks)
    assert PROBE.check_environment(Path(sys.executable), CONFIG, tmp_path / "new")["status"] == "failed"


def test_preflight_timeout_keeps_output(tmp_path, monkeypatch):
    _fake_process(monkeypatch, error=subprocess.TimeoutExpired("sandbox", 30, output=b"partial stdout", stderr=b"partial stderr"))
    destination = tmp_path / "new"
    report = PROBE.check_environment(Path(sys.executable), CONFIG, destination)
    assert report["status"] == "failed"
    assert "TimeoutExpired" in report["error"]
    assert (destination / "sandbox.stdout.log").read_text() == "partial stdout"
    assert (destination / "sandbox.stderr.log").read_text() == "partial stderr"


def test_preflight_thread_start_failure_does_not_block_cleanup(tmp_path, monkeypatch):
    _fake_process(monkeypatch)
    closed = []

    class Server:
        def __init__(self, *args):
            pass

        def shutdown(self):
            pytest.fail("shutdown would hang when serve_forever never started")

        def server_close(self):
            closed.append(True)

        def serve_forever(self, **kwargs):
            pass

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(PROBE, "HTTPServer", Server)
    monkeypatch.setattr(PROBE, "Thread", Thread)
    destination = tmp_path / "new"
    report = PROBE.check_environment(Path(sys.executable), CONFIG, destination)
    assert report["status"] == "failed"
    assert "can't start new thread" in report["error"]
    assert closed == [True]
    assert (destination / "report.json").is_file()


@pytest.mark.parametrize("old,new", [('sandbox_mode = "workspace-write"', 'sandbox_mode = "danger-full-access"'), ('approval_policy = "never"', 'approval_policy = "on-request"'), ('network_access = true', 'network_access = false')])
def test_preflight_rejects_changed_worker_policy(tmp_path, old, new):
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.read_text().replace(old, new))
    destination = tmp_path / "new"
    with pytest.raises(ValueError, match="unchanged canonical"):
        PROBE.check_environment(Path(sys.executable), config, destination)
    assert not destination.exists()


def test_preflight_never_reuses_evidence_directory(tmp_path):
    with pytest.raises(FileExistsError):
        PROBE.check_environment(Path(sys.executable), CONFIG, tmp_path)


def test_formal_launcher_is_full_single_job_and_fail_closed():
    script = (ROOT / "scripts/slurm_artemis_canonical_5round.sh").read_text()
    assert "#SBATCH --array" not in script
    assert "#SBATCH --gres=gpu:l40s:3" in script
    assert "#SBATCH --time=3-00:00:00" in script
    assert "set -euo pipefail" in script
    assert "module load scipy-stack pytest" in script
    assert script.index("scripts/check_canonical_codex_environment.py") < script.index("scripts/run_canonical_artemis_campaign.py")
    assert "--preflight-only" not in script
    assert "--sandbox danger-full-access" not in script
    subprocess.run(["bash", "-n", str(ROOT / "scripts/slurm_artemis_canonical_5round.sh")], check=True)
