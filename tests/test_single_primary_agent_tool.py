"""Plasma CLI compatibility with upstream immutable versions and explicit choice."""

import json
from pathlib import Path
import stat
from unittest.mock import patch

import pytest

from pde_agent_verifier_solver.engine import agent_tool
from pde_agent_verifier_solver.engine.agent_execution import (
    AGENT_EXECUTION_ID_ENV, AGENT_NOMINATIONS_ENV, AGENT_WORKSPACE_ENV,
    read_agent_primary_nomination,
)
from pde_agent_verifier_solver.sdk import write_candidate_bundle_manifest

VALID_SOURCE = (
    "raise RuntimeError('Candidate source must never be imported on submit')\n"
    "def prepare_solution(case, solve_budget=None):\n"
    "    return None\n"
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "submission.py").write_text(VALID_SOURCE, encoding="utf-8")
    write_candidate_bundle_manifest(candidate, files=("submission.py",))
    (tmp_path / "nominations").mkdir()
    monkeypatch.setenv(AGENT_WORKSPACE_ENV, str(tmp_path))
    monkeypatch.setenv(AGENT_NOMINATIONS_ENV, str(tmp_path / "nominations"))
    monkeypatch.setenv(AGENT_EXECUTION_ID_ENV, "primary-cli-test")
    return tmp_path


def request(policy="single_primary_v1"):
    return {"execution_id": "primary-cli-test", "round": {"round_index": 1},
            "worker_settings": {"nomination_policy": policy}}


def submit(sequence, *, primary=True):
    return ["submit", "candidate", "--sequence", str(sequence),
            "--name", "Preferred solver", "--summary", "Development evidence",
            "--usage", "Call prepare_solution", *(["--primary"] if primary else [])]


def payload(workspace, sequence):
    return json.loads((workspace / "nominations" / f"nomination-{sequence:08d}.json").read_text())["payload"]


def choice(workspace):
    return read_agent_primary_nomination(workspace / "nominations", "primary-cli-test")


def test_versions_are_immutable_and_primary_can_select_an_earlier_version(workspace, capsys):
    with (
        patch.object(agent_tool, "load_agent_worker_request", return_value=request()),
        patch.object(agent_tool, "ControllerAgentCapabilityClient") as capability,
    ):
        assert agent_tool.main(submit(1)) == 0
        assert json.loads(capsys.readouterr().out)["primary_nomination_sequence"] == 1
        first = payload(workspace, 1)
        assert "official_primary" not in first
        snapshot = workspace / first["bundle_path"]
        assert (snapshot / "submission.py").read_text() == VALID_SOURCE
        assert not snapshot.stat().st_mode & stat.S_IWUSR
        (workspace / "candidate" / "submission.py").write_text(VALID_SOURCE + "VERSION = 2\n")
        assert agent_tool.main(submit(2)) == 0
        capsys.readouterr()
        assert choice(workspace) == 2
        assert agent_tool.main(submit(3, primary=False)) == 0
        capsys.readouterr()
        assert choice(workspace) == 2
        assert agent_tool.main(["select-primary", "1"]) == 0
        assert json.loads(capsys.readouterr().out)["primary_nomination_sequence"] == 1
        assert choice(workspace) == 1
        assert (snapshot / "submission.py").read_text() == VALID_SOURCE
        assert len(tuple((workspace / "nominations").glob("nomination-*.json"))) == 3
    capability.assert_not_called()


@pytest.mark.parametrize("policy", ["all", "latest"])
def test_primary_commands_are_opt_in_and_plain_submit_remains_supported(workspace, policy):
    with patch.object(agent_tool, "load_agent_worker_request", return_value=request(policy)):
        with pytest.raises(SystemExit):
            agent_tool.main(submit(1))
        assert choice(workspace) is None
        assert agent_tool.main(submit(1, primary=False)) == 0
        with pytest.raises(SystemExit):
            agent_tool.main(["select-primary", "1"])
        assert choice(workspace) is None


def test_unknown_selection_does_not_replace_previous_primary(workspace):
    with patch.object(agent_tool, "load_agent_worker_request", return_value=request()):
        assert agent_tool.main(submit(1)) == 0
        with pytest.raises(SystemExit):
            agent_tool.main(["select-primary", "2"])
        assert choice(workspace) == 1


def test_status_reports_the_actual_shared_selection(workspace):
    with patch.object(agent_tool, "load_agent_worker_request", return_value=request()):
        assert agent_tool.main(submit(1)) == 0
    status = agent_tool._status(request(), ())
    assert status["nomination_policy"] == "single_primary_v1"
    assert status["primary_nomination_sequence"] == 1
    assert "official_nomination_policy" not in status
