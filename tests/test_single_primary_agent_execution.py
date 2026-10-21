"""Plasma uses the upstream explicit-choice journal, not a private policy."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    AgentExecutionError, CommandAgentProgram, SubprocessAgentExecutionService,
    select_agent_primary_nomination, submit_agent_nomination,
)
from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent
from tests.test_agent_execution_service import _agent_context, _decode_toy_nomination

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
SETTINGS = {"nomination_policy": "single_primary_v1"}


def _context(round_index=4, *, incumbent=True):
    context = _agent_context()
    context = replace(context, round_plan=replace(context.round_plan, round_index=round_index))
    if incumbent:
        context = replace(context, incumbent=ToyResearchAgent._artifact(context, 1, suffix="incumbent"))
    return context


def _program(source="pass", *, settings=None):
    return CommandAgentProgram(
        (sys.executable, "-c", source), _decode_toy_nomination,
        environment={"PYTHONPATH": str(SOURCE_ROOT)},
        worker_settings=SETTINGS if settings is None else settings,
    )


@pytest.mark.parametrize("policy", [None, True, {}, "single_primary", "primary"])
def test_invalid_shared_policy_fails_before_starting_worker(policy):
    with pytest.raises(ValueError):
        _program(settings={"nomination_policy": policy})


@pytest.mark.parametrize("round_index", [1, 4, 5])
def test_shared_policy_preserves_dynamic_task_prompt_and_handoff(tmp_path, round_index):
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=2)
    context = _context(round_index, incumbent=False)
    context = replace(context, prompt=context.agent.prompt + "\n\nDYNAMIC TASK BRIEF AND HANDOFF")
    request = service._build_request(_program(), context, "request", tmp_path, 2, {})
    assert request["agent"]["prompt"] == context.prompt
    assert request["worker_settings"]["nomination_policy"] == "single_primary_v1"
    assert "official_nomination_policy" not in request["worker_settings"]


@pytest.mark.parametrize("policy,selected,expected", [
    ("all", None, [1, 2, 3]),
    ("latest", None, [3]),
    ("single_primary_v1", 1, [1]),
    ("single_primary_v1", 2, [2]),
])
def test_only_explicit_choice_is_decoded_and_old_policies_are_unchanged(
    tmp_path, policy, selected, expected,
):
    journal = tmp_path / "nominations"
    journal.mkdir()
    for sequence in range(1, 4):
        submit_agent_nomination({"mode": sequence}, sequence=sequence,
                                directory=journal, execution_id="test")
    if selected is not None:
        select_agent_primary_nomination(selected, directory=journal, execution_id="test")
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=2)
    artifacts, rejected = service._load_nominations(
        journal, "test", _program(settings={"nomination_policy": policy}), _context(), tmp_path)
    assert rejected == 0
    assert [artifact.metadata["max_resolved_mode"] for artifact in artifacts] == expected
    assert len(tuple(journal.glob("nomination-*.json"))) == 3


def test_saved_versions_without_explicit_choice_do_not_fall_back(tmp_path):
    submit_agent_nomination({"mode": 1}, sequence=1, directory=tmp_path, execution_id="test")
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=2)
    assert service._load_nominations(tmp_path, "test", _program(), _context(), tmp_path) == ((), 1)


def test_legacy_inline_marker_is_not_a_primary_choice(tmp_path):
    submit_agent_nomination({"mode": 1, "official_primary": True}, sequence=1,
                            directory=tmp_path, execution_id="test")
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=2)
    assert service._load_nominations(tmp_path, "test", _program(), _context(), tmp_path) == ((), 1)
