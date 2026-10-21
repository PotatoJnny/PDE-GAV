"""Deterministic tests of the retained AIDE policy and its NS callback boundary."""
import ast
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.ns.baselines import aide
from pde_agent_verifier_solver.plugins.ns.baselines._vendor import aide_core


def _run(root, keys, errors, *, initial=None, options=None, seed=17):
    calls = []
    sequence = iter(keys)
    histories = {}

    def propose(parent, history, phase):
        calls.append((parent, history, phase))
        key = next(sequence)
        if key is not None:
            histories[key] = (*history, (f"Design for {key}", errors[key]))
        return key

    result = aide.run_official_aide_search(
        SimpleNamespace(output_directory=root, prompt="Solve NS", seed=seed, iterations=len(keys)),
        initial_key=initial, options=options, propose=propose,
        score=lambda key: errors[key], history_for=lambda key: histories.get(key, ()),
        diagnostics_for=lambda key: f"Execution failure in {key}" if not math.isfinite(errors[key]) else "",
    )
    return result, calls, json.loads((root / "aide_journal.json").read_text())


def test_upstream_pin_hashes_and_search_node_journal_methods_are_preserved():
    metadata = aide._provenance()
    root = Path(aide_core.__file__).parent
    assert metadata["revision"] == "60b3978ddf65b71f86eb7c64506965048a1398cf"
    assert "Copyright (c) 2024 Weco AI Ltd" in (root / "AIDE_LICENSE").read_text()
    assert sha256(Path(aide_core.__file__).read_bytes()).hexdigest() == metadata["derived_core"]["sha256"]
    core_ast = ast.parse(Path(aide_core.__file__).read_text())
    retained = {node.name: node for node in core_ast.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    source_map = {entry["upstream_path"]: ast.parse((root / entry["local_path"]).read_text())
                  for entry in metadata["upstream_sources"] if entry["upstream_path"].endswith(".py")}
    # Whole retained definitions are identical after the documented removal of
    # the serialization-only mixin base. This includes tie and debug-depth rules.
    for path, names in (
        ("aide/journal.py", ("Node", "Journal")),
        ("aide/utils/metric.py", ("MetricValue", "WorstMetricValue")),
        ("aide/interpreter.py", ("ExecutionResult",)),
        ("aide/utils/response.py", ("trim_long_string",)),
    ):
        for name in names:
            expected = next(n for n in source_map[path].body if getattr(n, "name", None) == name)
            if isinstance(expected, ast.ClassDef):
                expected.bases = [base for base in expected.bases
                                  if not isinstance(base, ast.Name) or base.id != "DataClassJsonMixin"]
            assert ast.dump(retained[name]) == ast.dump(expected), name
    upstream_agent = next(n for n in source_map["aide/agent.py"].body
                          if isinstance(n, ast.ClassDef) and n.name == "Agent")
    for method in retained["Agent"].body:
        original = next(n for n in upstream_agent.body if getattr(n, "name", None) == method.name)
        assert ast.dump(method) == ast.dump(original), method.name


def test_draft_debug_and_greedy_improve_keep_distinct_parent_and_memory_semantics(tmp_path):
    errors = {"good": 4, "buggy": math.inf, "repaired": 2, "best": 1, "regressed": 5}
    selected, calls, rows = _run(
        tmp_path, list(errors), errors,
        options=aide.AIDESearchOptions(num_drafts=2, debug_prob=1, max_debug_depth=0),
    )
    assert selected == "best"
    assert [call[0] for call in calls] == [None, None, "buggy", "repaired", "best"]
    assert [row["stage"] for row in rows] == ["draft", "draft", "debug", "improve", "improve"]
    assert "Design for good" in calls[1][2]
    assert "Execution failure in buggy" in calls[2][2]
    assert "Design for good" not in calls[2][2]
    assert calls[2][1] == (("Design for buggy\nThe solver failed fixed validation or returned a nonfinite error.\nExecution failure in buggy", None),)
    assert "Design for repaired" in calls[3][2]
    assert "Design for buggy" not in calls[3][2]  # Good-node memory only.
    assert rows[1]["is_buggy"] and not rows[1]["is_leaf"]
    assert rows[2]["debug_depth"] == 1 and rows[2]["parent_node_id"] == rows[1]["node_id"]
    assert rows[4]["parent_node_id"] == rows[3]["node_id"]
    report = json.loads((tmp_path / "aide_adaptation.json").read_text())
    assert report["search_policy_method_modified"] is False
    assert report["paper_end_to_end_reproduction"] is False
    assert report["terminal_selection"] is False


def test_s0_is_an_existing_draft_and_new_turn_budget_is_exact(tmp_path):
    errors = {"S0": 6, "one": 5, "two": 4, "three": 3, "four": 2, "five": 1}
    selected, calls, rows = _run(tmp_path, list(errors)[1:], errors, initial="S0")
    assert selected == "five"
    assert len(calls) == 5 and len(rows) == 6
    assert [call[0] for call in calls] == [None, None, None, None, "four"]
    assert sum(row["stage"] == "draft" for row in rows) == 5
    assert json.loads((tmp_path / "aide_adaptation.json").read_text())["s0_counts_as_draft"]


def test_author_debug_depth_inclusive_bound_and_buggy_leaf_filter_are_retained(tmp_path):
    errors = {"S0": math.inf, "first": math.inf, "second": math.inf, "fresh": 2}
    _, calls, rows = _run(
        tmp_path, ["first", "second", "fresh"], errors, initial="S0",
        options=aide.AIDESearchOptions(num_drafts=1, debug_prob=1, max_debug_depth=1),
    )
    # Author code uses <=: the depth-1 parent can produce a depth-2 child.
    # Once that child is too deep, its non-leaf ancestors cannot be retried.
    assert [call[0] for call in calls] == ["S0", "first", None]
    assert [row["debug_depth"] for row in rows] == [0, 1, 2, 0]
    assert [row["stage"] for row in rows] == ["draft", "debug", "debug", "draft"]


def test_no_nomination_spends_a_step_and_keeps_failed_node_for_debug(tmp_path):
    selected, calls, rows = _run(
        tmp_path, [None, "recovered"], {"recovered": 0.1},
        options=aide.AIDESearchOptions(num_drafts=1, debug_prob=1),
    )
    assert selected == "recovered" and len(calls) == 2
    assert rows[0]["is_buggy"] and rows[0]["bundle_key"] is None
    assert rows[1]["stage"] == "debug" and rows[1]["parent_node_id"] == rows[0]["node_id"]
    assert calls[1][0] is None and "no recoverable parent source" in calls[1][2]


def test_seeded_debug_choices_do_not_touch_global_rng_and_reproduce(tmp_path):
    keys = [f"failed-{i}" for i in range(12)]
    errors = dict.fromkeys(keys, math.inf)
    options = aide.AIDESearchOptions(num_drafts=2, debug_prob=.75, max_debug_depth=2)
    state = random.getstate()
    first = _run(tmp_path / "one", keys, errors, options=options, seed=39)
    assert random.getstate() == state
    second = _run(tmp_path / "two", keys, errors, options=options, seed=39)
    assert random.getstate() == state
    assert first == second
    assert first[0] is None  # Invalid nodes can never become the final result.


def test_greedy_ties_retain_earlier_valid_node_and_worse_attempts_remain_in_journal(tmp_path):
    selected, calls, rows = _run(
        tmp_path, ["equal", "worse"], {"S0": 1, "equal": 1, "worse": 2}, initial="S0",
        options=aide.AIDESearchOptions(num_drafts=1, debug_prob=0),
    )
    assert selected == "S0" and [call[0] for call in calls] == ["S0", "S0"]
    assert [row["bundle_key"] for row in rows] == ["S0", "equal", "worse"]


def test_callback_failure_persists_completed_journal_and_propagates(tmp_path):
    def fail(*args):
        raise RuntimeError("agent infrastructure failed")

    with pytest.raises(RuntimeError, match="agent infrastructure failed"):
        aide.run_official_aide_search(
            SimpleNamespace(output_directory=tmp_path, prompt="NS", seed=0, iterations=1),
            initial_key="S0", propose=fail, score=lambda key: .5, history_for=lambda key: (),
            options=aide.AIDESearchOptions(num_drafts=1),
        )
    rows = json.loads((tmp_path / "aide_journal.json").read_text())
    assert len(rows) == 1 and rows[0]["bundle_key"] == "S0"


@pytest.mark.parametrize("kwargs", [{"debug_prob": math.nan}, {"debug_prob": 1.1},
                                    {"num_drafts": 0}, {"max_debug_depth": -1}])
def test_invalid_policy_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        aide.AIDESearchOptions(**kwargs)


def test_real_ns_frozen_bundles_are_debugged_and_selected_by_shared_hidden_verifier(tmp_path):
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
    from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
    from pde_agent_verifier_solver.sdk import (
        AnswerQuery, AnswerView, Case, CaseRecord, CaseSource, CaseVisibility,
        QueryKind, TaskKind,
    )

    class Truth:
        queries = 0

        def answer(self, query):
            self.queries += 1
            return AnswerView(QueryKind.POINTS, {
                "velocity_x": tuple(1 + p["t"] + p["x"] for p in query.points),
                "velocity_y": tuple(2 + p["t"] + p["y"] for p in query.points),
            }, points=query.points, shape=(len(query.points),))

    truth = Truth()
    selection = (CaseRecord(Case(
        "validation-only", CaseSource.INITIAL, CaseVisibility.HIDDEN,
        {"domain": {"bounds": {"x": [0, 1], "y": [0, 1]},
                    "time_interval": [0, 1], "geometry": {}}},
    ), truth),)
    verifier = HiddenVerifier(
        NSSolverTask(),
        AnswerQuery.at_points(fields=("velocity_x", "velocity_y"),
                              points=({"t": .3, "x": .4, "y": .5},)),
        {"official_point_count": 4}, seed=21,
    )
    source = '''from pde_agent_verifier_solver.sdk import AnswerView, QueryKind
class Solution:
    def evaluate(self, query):
        return AnswerView(QueryKind.POINTS, {
            "velocity_x": tuple(FACTOR * (1 + p["t"] + p["x"]) for p in query.points),
            "velocity_y": tuple(FACTOR * (2 + p["t"] + p["y"]) for p in query.points),
        }, points=query.points, shape=(len(query.points),))
def prepare_solution(case, solve_budget=None):
    BODY
'''
    candidates = {}
    keys = []
    for name, factor in (("buggy", None), ("repaired", .9), ("regressed", 1.2)):
        path = tmp_path / f"{name}.py"
        path.write_text(source.replace("FACTOR", repr(factor)).replace(
            "BODY", "raise RuntimeError('synthetic numerical failure')" if factor is None else "return Solution()",
        ))
        artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
            path, storage_root=tmp_path / "frozen", artifact_id=name, name=name,
            summary="deterministic test candidate", usage="NS two-phase solver",
        )
        key = artifact.program.tree_sha256
        candidates[key] = artifact
        keys.append(key)
    scores = {}
    failures = {}

    def score(key):
        if key not in scores:
            try:
                scores[key] = float(verifier.evaluate(candidates[key], selection).primary_error)
            except Exception as exc:
                scores[key] = math.inf
                failures[key] = f"{type(exc).__name__}: {exc}"
        return scores[key]

    calls = []
    sequence = iter(keys)

    def propose(parent, history, phase):
        if parent is not None:
            assert candidates[parent].program.bundle_locator.is_file()
        calls.append((parent, phase))
        return next(sequence)

    selected = aide.run_official_aide_search(
        SimpleNamespace(output_directory=tmp_path / "run", prompt="Solve NS", seed=17, iterations=3),
        initial_key=None, propose=propose, score=score,
        history_for=lambda key: ((f"Solver {candidates[key].name}", scores.get(key)),),
        diagnostics_for=lambda key: failures.get(key, ""),
        options=aide.AIDESearchOptions(num_drafts=1, debug_prob=1),
    )
    assert selected == keys[1]
    assert [parent for parent, _ in calls] == [None, keys[0], keys[1]]
    assert scores[selected] == pytest.approx(.1)
    assert math.isinf(scores[keys[0]]) and len(scores) == 3
    assert truth.queries == 1  # One private query plan and truth view shared across candidates.
    assert candidates[selected].program.bundle_locator.is_file()
