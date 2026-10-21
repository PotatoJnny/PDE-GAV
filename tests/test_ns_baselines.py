from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest, HiddenVerifier
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
from pde_agent_verifier_solver.plugins.ns.baselines.cli import run_from_config
from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, CaseRecord, CaseSource, CaseVisibility,
    QueryKind, TaskKind,
)

SOURCE = '''from pde_agent_verifier_solver.sdk import AnswerView, QueryKind
class Handle:
    def evaluate(self, query):
        return AnswerView(QueryKind.POINTS, {
            "velocity_x": tuple(FACTOR * (1 + p["t"] + p["x"]) for p in query.points),
            "velocity_y": tuple(FACTOR * (2 + p["t"] + p["y"]) for p in query.points),
        }, points=query.points, shape=(len(query.points),))
def prepare_solution(case, solve_budget=None):
    assert not hasattr(case, "truth")
    return Handle()
'''


class Truth:
    def __init__(self):
        self.queries = []

    def answer(self, query):
        self.queries.append(query)
        return AnswerView(QueryKind.POINTS, {
            "velocity_x": tuple(1 + p["t"] + p["x"] for p in query.points),
            "velocity_y": tuple(2 + p["t"] + p["y"] for p in query.points),
        }, points=query.points, shape=(len(query.points),))


def _records():
    public = {"equation": "2D incompressible Navier--Stokes", "domain": {
        "bounds": {"x": [0, 1], "y": [0, 1]}, "time_interval": [0, 1], "geometry": {},
    }}
    development = CaseRecord(Case("dev", CaseSource.INITIAL,
                                 CaseVisibility.DEVELOPMENT, public), Truth())
    selection = CaseRecord(Case("selection", CaseSource.INITIAL,
                               CaseVisibility.HIDDEN, public), Truth())
    return (development,), (selection,)


def _policy(truth=False):
    return EvaluatorPolicy(
        AnswerQuery.at_points(fields=("velocity_x", "velocity_y"),
                              points=({"t": .3, "x": .4, "y": .5},)),
        allow_point_truth=truth, development_point_count=3,
    )


def _artifact(root, factor, name):
    path = root / f"{name}.py"
    path.write_text(SOURCE.replace("FACTOR", repr(factor)))
    return PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        path, storage_root=root / "frozen", artifact_id=name, name=name,
        summary="test solver", usage="test two-phase NS solver",
    )


class SequenceAgent:
    def __init__(self, candidates, *, evaluate_truth=False):
        self.candidates = iter(candidates)
        self.contexts = []
        self.evaluate_truth = evaluate_truth

    def run(self, context):
        self.contexts.append(context)
        candidate = next(self.candidates)
        if candidate is None:
            return AgentOutcome(observations=("No valid proposal",))
        response = context.evaluator.evaluate(
            candidate, EvaluationRequest(tuple(case.case_id for case in context.cases),
                                         return_truth=self.evaluate_truth),
        )
        assert all((row.truth is not None) == self.evaluate_truth for row in response.cases)
        return AgentOutcome(candidate, round_log=f"Notes from turn {len(self.contexts)}")


def _run(tmp_path, method, factors, **kwargs):
    development, selection = _records()
    candidates = [_artifact(tmp_path, factor, f"candidate-{i}") if factor is not None else None
                  for i, factor in enumerate(factors)]
    initial = _artifact(tmp_path, 0.0, "initial")
    agent = SequenceAgent(candidates, evaluate_truth=True)
    config = NSBaselineConfig(method, tmp_path / "result", iterations=len(factors),
                              independent_candidates=2, repair_iterations=1, **kwargs)
    result = run_ns_baseline(
        config, development_records=development, selection_records=selection,
        evaluator_policy=_policy(True), agent_program=agent,
        agent_execution=InProcessAgentExecutionService(), initial_candidate=initial,
        task_settings={"official_point_count": 4},
    )
    return result, agent, candidates, initial, development, selection


def test_refinement_keeps_validation_best_and_exports_a_frozen_bundle(tmp_path):
    result, agent, candidates, initial, development, selection = _run(
        tmp_path, "refinement", [.8, .4, .9],
    )
    assert result.selection_error == pytest.approx(.1)
    assert [ctx.incumbent.program.tree_sha256 for ctx in agent.contexts] == [
        initial.program.tree_sha256, candidates[0].program.tree_sha256,
        candidates[0].program.tree_sha256,
    ]
    assert [len(ctx.memory.rounds) for ctx in agent.contexts] == [0, 1, 2]
    assert all(ctx.sampler is None and len(ctx.cases) == 1 for ctx in agent.contexts)
    # One frozen validation view is shared by all proposed programs.
    assert len(selection[0].truth.queries) == 1
    assert len(development[0].truth.queries) == 3
    assert result.selected.program.bundle_locator.is_file()
    assert result.selected.program.tree_sha256 == candidates[2].program.tree_sha256
    (tmp_path / "candidate-2.py").write_text("raise RuntimeError('edited source')")
    report = HiddenVerifier(NSSolverTask(), _policy().default_query,
                            {"official_point_count": 4}).evaluate(result.selected, selection)
    assert report.primary_error == pytest.approx(.1)
    assert result.report["terminal_score"] is None
    assert "terminal_records" not in inspect.signature(run_ns_baseline).parameters


def test_bon_branches_are_independent_and_repair_starts_from_validation_winner(tmp_path):
    result, agent, candidates, initial, _, _ = _run(tmp_path, "bon_repair", [.3, .8, .9])
    assert [ctx.incumbent.program.tree_sha256 for ctx in agent.contexts] == [
        initial.program.tree_sha256, initial.program.tree_sha256,
        candidates[1].program.tree_sha256,
    ]
    assert [len(ctx.memory.rounds) for ctx in agent.contexts] == [0, 0, 1]
    assert "Notes from turn 1" not in agent.contexts[2].memory.markdown
    assert result.selected.program.tree_sha256 == candidates[2].program.tree_sha256


def test_official_era_futs_is_pinned_and_invoked_with_real_ns_callbacks(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.ns.baselines._vendor import era_futs
    calls = []
    original = era_futs.search

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(era_futs, "search", observed)
    result, agent, _, _, _, selection = _run(
        tmp_path, "era_futs_ns_adaptation", [.3, .8, .9],
    )
    assert len(calls) == 1 and calls[0]["num_iterations"] == 3
    assert result.selection_error == pytest.approx(.1)
    assert len(agent.contexts) == 3 and len(selection[0].truth.queries) == 1
    metadata = json.loads((result.output_directory / "era_adaptation.json").read_text())
    assert metadata["search_modified"] is False
    assert metadata["paper_end_to_end_reproduction"] is False
    assert metadata["local_source_sha256"] == (
        "5cc827f3d609105c1bd7b1103e4c33e393043c6db882bd1f44b9e4ec4301b814"
    )


def test_aide_policy_is_connected_to_ns_bundles_and_stage_prompts(tmp_path):
    result, agent, candidates, initial, _, _ = _run(
        tmp_path, "aide_ns_adaptation", [.3, .8, .9], aide_num_drafts=2,
    )
    assert [None if ctx.incumbent is None else ctx.incumbent.program.tree_sha256
            for ctx in agent.contexts] == [
        None, candidates[0].program.tree_sha256, candidates[1].program.tree_sha256,
    ]
    assert 'aide_draft' in agent.contexts[0].prompt
    assert 'Good-node journal' in agent.contexts[0].prompt
    assert 'aide_improve' in agent.contexts[1].prompt
    assert result.selected.program.tree_sha256 == candidates[2].program.tree_sha256
    assert result.report['completed_agent_turns'] == 3
    assert len(json.loads((result.output_directory / 'aide_journal.json').read_text())) == 4


def test_failed_turns_spend_budget_and_do_not_replace_valid_initial(tmp_path):
    result, agent, _, initial, _, _ = _run(tmp_path, "refinement", [None, None])
    assert result.selected.program.tree_sha256 == initial.program.tree_sha256
    assert len(agent.contexts) == 2 and result.report["failed_turns"] == 2
    rows = json.loads((result.output_directory / "baseline_turns.json").read_text())
    assert all(row["status"] == "failed" for row in rows)


def test_overlapping_banks_rejected_before_any_agent_turn(tmp_path):
    development, selection = _records()
    bad_selection = (CaseRecord(replace(development[0].case, visibility=CaseVisibility.HIDDEN),
                                selection[0].truth),)
    agent = SequenceAgent([])
    with pytest.raises(ValueError, match="disjoint"):
        run_ns_baseline(
            NSBaselineConfig("refinement", tmp_path / "bad"),
            development_records=development, selection_records=bad_selection,
            evaluator_policy=_policy(), agent_program=agent,
            agent_execution=InProcessAgentExecutionService(),
        )
    assert not agent.contexts


def _write_numerical_data(root):
    root.mkdir()
    rows = []
    for label in ("dev", "validation", "test"):
        directory = root / label
        directory.mkdir()
        (directory / "public_question.json").write_text(json.dumps({"time": {"t0": 0, "t1": 1}}))
        (directory / "case_spec.json").write_text(json.dumps({"parameters": {}}))
        axis = np.array([0., 1.])
        np.savez(directory / "public_arrays.npz", x=axis, y=axis,
                 fluid_mask=np.ones((2, 2), dtype=bool))
        t, x, y = np.meshgrid(axis, axis, axis, indexing="ij")
        np.savez(directory / "hidden_truth_cache.npz",
                 u_true=np.stack((1 + t + x, 2 + t + y), axis=1),
                 pressure=np.zeros((2, 2, 2)), t=axis)
        rows.append({"case_id": label, "case_dir": label})
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema": "ns_d0_v1", "splits": {
        "train": [rows[0]], "validation": [rows[1]], "external_test": [rows[2]],
    }}))
    return path


@pytest.mark.parametrize('method,search', [
    ('competitive_refinement', {'competition_rounds': 2}),
    ('bon_repair', {'independent_candidates': 2, 'repair_iterations': 1}),
    ('era_futs_ns_adaptation', {'iterations': 3}),
    ('aide_ns_adaptation', {'iterations': 3, 'aide_num_drafts': 2}),
    ('codepde_repeated_sample_ns_adaptation', {
        'codepde_num_repeated_samples': 3, 'codepde_num_debugging_trials_per_sample': 2}),
])
def test_cli_runs_actual_command_agent_capability_and_candidate_subprocesses(tmp_path, method, search):
    dataset = _write_numerical_data(tmp_path / "data")
    worker = tmp_path / "worker.py"
    worker.write_text('''from pathlib import Path
from pde_agent_verifier_solver.engine.agent_execution import (
    load_agent_worker_request, submit_agent_nomination, write_agent_worker_response,
)
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
request = load_agent_worker_request()
assert request["capabilities"]["wire"]["inline_array_bytes"] == 256
client = ControllerAgentCapabilityClient.from_request(request)
cases = client.current_cases()
assert len(cases) == 1
assert "hidden_truth" not in repr(cases[0].public_data)
Path("submission.py").write_text(SOURCE)
response = client.evaluate({"bundle_path": "submission.py"},
                          case_ids=[case.case_id for case in cases], return_truth=False)
assert response.primary_error < 1e-10
submit_agent_nomination({"bundle_path": "submission.py"}, sequence=1)
write_agent_worker_response(observations=["Local deterministic worker; no LLM."])
'''.replace("SOURCE", repr(SOURCE.replace("FACTOR", "1.0"))))
    config = tmp_path / "baseline.json"
    config.write_text(json.dumps({
        "schema": "ns_fixed_data_baseline_v1", "method": method,
        "dataset": {"manifest": str(dataset), "development_split": "train",
                    "selection_split": "validation"},
        "search": search,
        "agent": {"command": ["{python}", str(worker)],
                  "hard_wall_seconds": 40},
        "evaluation": {"development_point_count": 3, "selection_point_count": 4,
                       "candidate_case_hard_wall_seconds": 20},
    }))
    result = run_from_config(config, tmp_path / "cli-result")
    assert result.selection_error == pytest.approx(0., abs=1e-14)
    expected_turns = 4 if method == "competitive_refinement" else 3
    assert result.report["completed_agent_turns"] == expected_turns
    agent_audits = json.loads((result.output_directory / "execution_audits.json").read_text())
    candidate_audits = json.loads(
        (result.output_directory / "candidate_execution_audits.json").read_text()
    )
    assert len(agent_audits) == expected_turns and len(candidate_audits) == expected_turns + 1
    assert all(row["time_budget_mode"] == "hard_wall" for row in agent_audits)
    assert all(row["active_budget_seconds"] is None for row in agent_audits)
    assert all(row["hard_wall_seconds"] == 40 for row in agent_audits)
    assert result.selected.program.bundle_locator.is_file()


    # The delivered baseline bundle enters the shared external evaluator
    # without any baseline-specific Candidate adapter or new selection step.
    from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation
    common = tmp_path / "common.json"
    common.write_text(json.dumps({
        "schema": "ns_common_evaluation_v1",
        "datasets": [{"dataset_id": "external", "manifest": str(dataset),
                      "split": "external_test", "cohort": "held_out"}],
        "programs": [{"method": method, "run_id": "local-test", "seed": 0,
                      "artifact_id": result.selected.artifact_id, "role": "final",
                      "bundle": str(result.selected.program.bundle_locator)}],
        "query": {"point_count": 4, "seed": 99},
        "budget": {"max_workers": 1, "case_hard_wall_seconds": 20},
    }))
    common_output = tmp_path / "common-output"
    summary = run_ns_common_evaluation(common, common_output)
    assert summary["n_failed_program_cases"] == 0
    external_rows = [json.loads(line) for line in
                     (common_output / "cases.jsonl").read_text().splitlines()]
    assert external_rows and all(row["error"] == pytest.approx(0., abs=1e-14)
                                 for row in external_rows)
    assert json.loads((result.output_directory / "baseline_result.json").read_text()
                      )["terminal_score"] is None
    assert (result.output_directory / "costs.json").is_file()

    # A fixed query must not silently become a single initial/boundary point.
    payload = json.loads(config.read_text())
    payload["evaluation"]["development_coordinate_mode"] = "user_fixed_query"
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="explicit coordinates"):
        run_from_config(config, tmp_path / "fixed-query-rejected")


def test_numerical_reference_retains_operator_without_advertising_missing_sessions():
    from pde_agent_verifier_solver.plugins.ns.baselines.public_problem import (
        numerical_ns_public_problem_bundle,
    )
    from pde_agent_verifier_solver.plugins.ns.public_problem import ns_public_problem_bundle
    bundle = numerical_ns_public_problem_bundle()
    assert bundle.oracle_methods == ()
    assert bundle.development_solution_oracle_methods == ()
    assert bundle.guidance_capability.supported is False
    resources = {item.relative_path: item.data for item in bundle.resources}
    original = {item.relative_path: item.data for item in ns_public_problem_bundle().resources}
    assert resources["ns_reference/ns_operator.py"] == original["ns_reference/ns_operator.py"]
    assert b"case_parameters" in resources["ns_reference/README.md"]


def test_codepde_rejects_s0_before_validation_or_agent_work(tmp_path):
    development, selection = _records()
    agent = SequenceAgent([])
    with pytest.raises(ValueError, match='cold start'):
        run_ns_baseline(
            NSBaselineConfig('codepde_repeated_sample_ns_adaptation', tmp_path / 'rejected'),
            development_records=development, selection_records=selection,
            evaluator_policy=_policy(), agent_program=agent,
            agent_execution=InProcessAgentExecutionService(),
            initial_candidate=_artifact(tmp_path, 0, 's0'),
        )
    assert not agent.contexts and not selection[0].truth.queries
    assert not (tmp_path / 'rejected').exists()


def test_fixed_competition_preserves_round_boundaries_portfolios_and_timed_out_nomination(tmp_path):
    development, selection = _records()
    initial = _artifact(tmp_path, 0., "initial")
    first = _artifact(tmp_path, .3, "first")
    winner1 = _artifact(tmp_path, .8, "winner1")
    winner2 = _artifact(tmp_path, .95, "winner2")
    worse = _artifact(tmp_path, .5, "worse")
    contexts = []

    class Agent:
        def run(self, context):
            contexts.append(context)
            assert context.sampler is None
            assert len(context.cases) == 1
            assert context.cases[0].visibility is CaseVisibility.DEVELOPMENT
            if context.round_plan.round_index == 1:
                candidate = first if context.agent.agent_id == "ns-broad-explorer" else winner1
                return AgentOutcome(candidate, round_log=f"notes-{context.agent.agent_id}")
            if context.agent.agent_id == "ns-broad-explorer":
                return AgentOutcome(artifacts=(winner2, worse), round_log="preserved before timeout",
                                    execution_status="timed_out")
            return AgentOutcome(observations=("No new proposal",), round_log="no nomination")

    result = run_ns_baseline(
        NSBaselineConfig("competitive_refinement", tmp_path / "competition", seed=101,
                         competition_rounds=2, agent_budget_seconds=7200),
        development_records=development, selection_records=selection,
        evaluator_policy=_policy(), agent_program=Agent(),
        agent_execution=InProcessAgentExecutionService(), initial_candidate=initial,
        task_settings={"official_point_count": 4},
    )
    assert [context.incumbent.program.tree_sha256 for context in contexts] == [
        initial.program.tree_sha256, initial.program.tree_sha256,
        winner1.program.tree_sha256, winner1.program.tree_sha256,
    ]
    assert [len(context.memory.rounds) for context in contexts] == [0, 0, 1, 1]
    assert contexts[2].memory == contexts[3].memory
    assert "notes-ns-broad-explorer" in contexts[2].memory.markdown
    assert "notes-ns-critical-refiner" in contexts[2].memory.markdown
    assert len(contexts[2].candidate_history.nodes) == 3
    assert len({context.seed for context in contexts}) == 4
    assert result.selected.program.tree_sha256 == winner2.program.tree_sha256
    assert result.selection_error == pytest.approx(.05)
    assert len(selection[0].truth.queries) == 1
    assert result.report["completed_agent_turns"] == 4
    root = result.output_directory
    plan = json.loads((root / "baseline_plan.json").read_text())
    assert plan["planned_agent_wall_seconds"] == 4 * 7200
    rounds = json.loads((root / "baseline_rounds.json").read_text())
    assert len(rounds) == 2 and len(rounds[1]["selection"]) == 3
    assert Path(rounds[1]["selected_bundle"]).is_file()
    turns = json.loads((root / "baseline_turns.json").read_text())
    assert turns[2]["status"] == "timed_out" and len(turns[2]["nominations"]) == 2
    assert turns[3]["failure"] == "no_nominations"
    nodes = json.loads((root / "candidate_history.json").read_text())["nodes"]
    assert len(nodes) == 5 and all(node["status"] == "eligible" for node in nodes)
    assert nodes[-2]["parent_node_id"] == nodes[2]["node_id"]


def test_competition_ties_and_agent_failures_keep_incumbent_and_other_agent_runs(tmp_path):
    from pde_agent_verifier_solver.engine.agent_execution import AgentExecutionError
    development, selection = _records()
    initial = _artifact(tmp_path, .5, "initial")
    calls = []

    class Agent:
        def run(self, context):
            calls.append(context.agent.agent_id)
            if context.agent.agent_id == "ns-broad-explorer":
                raise AgentExecutionError("isolated Agent failure")
            return AgentOutcome(initial, round_log="duplicate incumbent")

    result = run_ns_baseline(
        NSBaselineConfig("competitive_refinement", tmp_path / "result", competition_rounds=1),
        development_records=development, selection_records=selection,
        evaluator_policy=_policy(), agent_program=Agent(),
        agent_execution=InProcessAgentExecutionService(), initial_candidate=initial,
        task_settings={"official_point_count": 4},
    )
    assert len(calls) == 2 and result.report["failed_turns"] == 1
    assert result.selected.artifact_id == initial.artifact_id
    assert result.report["unique_scored_candidates"] == 1
    assert len(selection[0].truth.queries) == 1


def test_paper_competition_config_has_ten_turns_twenty_agent_hours():
    config = NSBaselineConfig("competitive_refinement", "/tmp/not-run",
                              competition_rounds=5, agent_budget_seconds=7200)
    assert config.turn_count == 10
    assert config.turn_count * config.agent_budget_seconds / 3600 == 20
    with pytest.raises(ValueError, match="uniquely"):
        replace(config, competition_roles=(config.competition_roles[0],) * 2)


@pytest.mark.parametrize("invalid", ["missing_case", "nonfinite_case"])
def test_competition_rejects_incomplete_or_nonfinite_validation(tmp_path, monkeypatch, invalid):
    development, selection = _records()
    initial = _artifact(tmp_path, 0., "initial")
    malformed = _artifact(tmp_path, .99, "malformed")
    evaluate = HiddenVerifier.evaluate

    def corrupt(self, artifact, records):
        report = evaluate(self, artifact, records)
        if artifact.program.tree_sha256 == malformed.program.tree_sha256:
            cases = (() if invalid == "missing_case" else
                     (replace(report.cases[0], primary_error=float("nan")),))
            return replace(report, primary_error=0., cases=cases)
        return report

    monkeypatch.setattr(HiddenVerifier, "evaluate", corrupt)
    result = run_ns_baseline(
        NSBaselineConfig("competitive_refinement", tmp_path / "result", competition_rounds=1),
        development_records=development, selection_records=selection,
        evaluator_policy=_policy(), agent_program=SequenceAgent([malformed, malformed]),
        agent_execution=InProcessAgentExecutionService(), initial_candidate=initial,
        task_settings={"official_point_count": 4},
    )
    assert result.selected.program.tree_sha256 == initial.program.tree_sha256
    selections = json.loads((result.output_directory / "selection_results.json").read_text())
    assert selections[-1]["failure"] == (
        "incomplete_selection_results" if invalid == "missing_case" else "nonfinite_selection_error"
    )
    assert selections[-1]["primary_error"] is None
