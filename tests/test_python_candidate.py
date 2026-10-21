from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.python_candidate import (
    PythonCandidateCodec,
    PythonCandidateError,
    PythonCandidateProgram,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseSource,
    CaseVisibility,
    QueryKind,
    TaskKind,
    write_candidate_bundle_manifest,
)


def _candidate(root: Path, *, scale: float = 2.0) -> Path:
    root.mkdir()
    (root / "helper.py").write_text(
        f"SCALE = {scale!r}\n",
        encoding="utf-8",
    )
    (root / "submission.py").write_text(
        "\n".join(
            (
                "from .helper import SCALE",
                "from pde_agent_verifier_solver.sdk import AnswerView, QueryKind",
                "class Handle:",
                "    def evaluate(self, query):",
                "        return AnswerView(kind=QueryKind.POINTS, fields={'u': tuple(SCALE for _ in query.points)}, points=query.points, shape=(len(query.points),))",
                "def prepare_solution(case, solve_budget=None):",
                "    return Handle()",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    write_candidate_bundle_manifest(
        root,
        files=("submission.py", "helper.py"),
    )
    return root


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        round_plan=SimpleNamespace(round_index=3),
        agent=SimpleNamespace(agent_id="agent-a"),
        incumbent=SimpleNamespace(artifact_id="candidate-s2"),
    )


def test_codec_freezes_loads_and_snapshots_multifile_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "turn"
    workspace.mkdir()
    source = _candidate(workspace / "solver")
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)

    artifact = codec.decode_nomination(
        {
            "bundle_path": "solver",
            "name": "Candidate A",
            "summary": "two-phase test",
            "usage": "prepare then evaluate",
        },
        _context(),
        workspace,
    )

    assert artifact.name == "Candidate A"
    assert artifact.round_index == 3
    assert artifact.agent_id == "agent-a"
    assert artifact.parent_artifact_id == "candidate-s2"
    assert isinstance(artifact.program, PythonCandidateProgram)
    query = AnswerQuery.at_points(fields=("u",), points=({"x": 0.25}, {"x": 0.75}))
    case = Case(
        "case",
        CaseSource.INITIAL,
        CaseVisibility.DEVELOPMENT,
        {},
    )
    answer = artifact.program.prepare_solution(case, None).evaluate(query)
    assert answer.fields["u"] == (2.0, 2.0)

    snapshot = deepcopy(artifact.program)
    assert snapshot is not artifact.program
    assert snapshot.prepare_solution(case, None).evaluate(query).fields["u"] == (
        2.0,
        2.0,
    )

    # The controller snapshot is independent from later Agent edits.
    (source / "helper.py").write_text("SCALE = 99.0\n", encoding="utf-8")
    fresh_snapshot = deepcopy(artifact.program)
    assert fresh_snapshot.prepare_solution(case, None).evaluate(query).fields["u"] == (
        2.0,
        2.0,
    )


def test_codec_rejects_paths_outside_workspace_and_missing_abi(tmp_path: Path) -> None:
    workspace = tmp_path / "turn"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def prepare_solution(case, solve_budget=None): pass\n")
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)

    with pytest.raises(PythonCandidateError, match="safe relative path"):
        codec.decode_nomination(
            {"bundle_path": "../outside.py"},
            _context(),
            workspace,
        )

    invalid = workspace / "invalid.py"
    invalid.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(PythonCandidateError, match="prepare_solution"):
        codec.decode_nomination(
            {"bundle_path": "invalid.py"},
            _context(),
            workspace,
        )

    noncallable = workspace / "noncallable.py"
    noncallable.write_text("prepare_solution = 1\n", encoding="utf-8")
    with pytest.raises(PythonCandidateError, match="prepare_solution"):
        codec.decode_nomination(
            {"bundle_path": "noncallable.py"},
            _context(),
            workspace,
        )


def test_codec_validation_does_not_execute_nomination_entrypoint(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "turn"
    workspace.mkdir()
    candidate = workspace / "candidate.py"
    candidate.write_text(
        "raise RuntimeError('must remain deferred')\n"
        "def prepare_solution(case, solve_budget=None):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).decode_nomination(
        {"bundle_path": "candidate.py"},
        _context(),
        workspace,
    )

    assert isinstance(artifact.program, PythonCandidateProgram)
    assert artifact.program._loaded is None


def test_codec_validation_does_not_execute_initial_entrypoint(tmp_path: Path) -> None:
    candidate = tmp_path / "initial.py"
    candidate.write_text(
        "raise RuntimeError('must remain deferred')\n"
        "def prepare_solution(case, solve_budget=None):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        candidate,
        storage_root=tmp_path / "sealed",
        artifact_id="user-s0",
        name="Initial",
        summary="baseline",
        usage="prepare then evaluate",
    )

    assert isinstance(artifact.program, PythonCandidateProgram)
    assert artifact.program._loaded is None


@pytest.mark.parametrize(
    "source",
    (
        "async def prepare_solution(case, solve_budget=None):\n    return object()\n",
        "@staticmethod\ndef prepare_solution(case, solve_budget=None):\n"
        "    return object()\n",
        "def prepare_solution(case):\n    return object()\n",
        "def prepare_solution(case, solve_budget):\n    return object()\n",
        "def prepare_solution(case, solve_budget=None, extra=None):\n"
        "    return object()\n",
        "def prepare_solution(case, *, solve_budget=None):\n    return object()\n",
        "def prepare_solution(case, solve_budget=None):\n"
        "    return object()\n"
        "prepare_solution = None\n",
        "def prepare_solution(case, solve_budget=None):\n"
        "    return object()\n"
        "if True:\n"
        "    prepare_solution = None\n",
        "def prepare_solution(case, solve_budget=None):\n"
        "    return object()\n"
        "from replacement import prepare_solution\n",
    ),
)
def test_codec_rejects_malformed_or_rebound_static_abi(
    tmp_path: Path,
    source: str,
) -> None:
    workspace = tmp_path / "turn"
    workspace.mkdir()
    candidate = workspace / "candidate.py"
    candidate.write_text(source, encoding="utf-8")

    with pytest.raises(PythonCandidateError, match="prepare_solution"):
        PythonCandidateCodec(task_kind=TaskKind.SOLVER).decode_nomination(
            {"bundle_path": "candidate.py"},
            _context(),
            workspace,
        )


def test_existing_bundle_can_be_bound_and_staged_as_incumbent(tmp_path: Path) -> None:
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
    artifact = codec.bind_existing(
        _candidate(tmp_path / "initial"),
        storage_root=tmp_path / "sealed",
        artifact_id="user-s0",
        name="Initial",
        summary="baseline",
        usage="prepare then evaluate",
    )
    workspace = tmp_path / "turn"
    workspace.mkdir()

    encoded = codec.encode_incumbent(artifact, workspace)

    assert encoded["bundle_path"] == "incumbent"
    assert (workspace / "incumbent" / "candidate_bundle.json").is_file()
    assert encoded["tree_sha256"] == artifact.program.tree_sha256
