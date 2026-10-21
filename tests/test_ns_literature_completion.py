"""Completion protocol acceptance: local Agents, CPU Cases, no external calls."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.agent_execution import SubprocessAgentExecutionService, UnavailableAgentCapabilityFacade
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
from pde_agent_verifier_solver.plugins.ns.baselines.environment import load_development_bank, write_development_bank, OnlineBaseEnvironment
from pde_agent_verifier_solver.plugins.ns.baselines.workflows import LiteratureOptions, screen_plans, validate_final_choice
from pde_agent_verifier_solver.sdk import SampleCasesRequest, SampleSource, SampleBatch
from tests.test_ns_baselines import _artifact, _records, _policy, SequenceAgent
from tests.test_ns_literature_baselines import _options, _plans, _plan_artifact, PlanningAgent
from tests.test_canonical_codex_agent import agent_context, command_program, write_auth
from tests.test_canonical_codex_resume import _write_resumable_codex
from tests.test_ns_continuous_case_bank import field
from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService


def options(**kw):
    return _options(protocol="completion_v2", total_wall_seconds=None, branch_wall_seconds=None,
                    planning_call_seconds=None, **kw)


def run(root, method, agent, *, initial=None, environment=None):
    development, selection = _records()
    return run_ns_baseline(NSBaselineConfig(method, root, seed=101, agent_budget_seconds=None,
        literature_options=options()), development_records=development, selection_records=selection,
        evaluator_policy=_policy(), task_settings={"official_point_count": 4}, agent_program=agent,
        agent_execution=InProcessAgentExecutionService(), initial_candidate=initial,
        development_environment=environment)


def test_codepde_all_initial_references_exclude_s0_and_no_agent_deadline(tmp_path):
    initial = _artifact(tmp_path, 1., "s0-perfect")
    agent = SequenceAgent([_artifact(tmp_path, v, f"c{i}") for i, v in enumerate((.5, .6, .8))])
    result = run(tmp_path / "run", "codepde_refinement_ns_adaptation", agent, initial=initial)
    assert result.selection_error == pytest.approx(.2)  # S0 is reference/fallback, not winner.
    context = agent.contexts[-1]
    payload = json.loads(next(r.data for r in context.public_problem_bundle.resources if r.relative_path == "baseline_context.json"))
    refs = payload["seed_implementations"]
    assert len(refs) == 2
    assert initial.program.tree_sha256 not in {r["tree_sha256"] for r in refs}
    assert all(c.round_plan.agent_budget_seconds is None for c in agent.contexts)
    assert all("No Agent development time cap" in c.prompt for c in agent.contexts)


class CompletePlanner(PlanningAgent):
    def run(self, context):
        phase = context.agent.name
        if phase.endswith("final selection"):
            self.contexts.append(context)
            return AgentOutcome(round_log=json.dumps({"plan_id": "0", "rationale": "Successful fine-grid result."}))
        if phase.endswith("reasoner"):
            self.contexts.append(context)
            return AgentOutcome(round_log=json.dumps({"explanation": "Measured evidence, not a guarantee.", "limitations": ["One seed"]}))
        return super().run(context)


def test_autonumerics_complete_roles_actual_grids_and_frozen_delivery(tmp_path):
    agent = CompletePlanner(_plan_artifact(tmp_path))
    result = run(tmp_path / "run", "autonumerics_ns_adaptation", agent)
    assert result.selection_error == pytest.approx(0.)
    assert [c.agent.name for c in agent.contexts] == [
        "autonumerics formulator", "autonumerics planner", "autonumerics selector",
        "autonumerics coder", "autonumerics final selection", "autonumerics reasoner"]
    records = json.loads((result.output_directory / "literature_search.json").read_text())["records"]
    chosen = records["autonumerics_final"]
    assert chosen["selected"] == result.selected.program.tree_sha256
    assert not chosen["fallback_to_s0"]
    assert chosen["selector"]["plan_id"] == "0"
    assert chosen["successful_plans"][0]["report"]["plan"]["spatial_discretization"]["Nx"] == 64
    context = agent.contexts[-1]
    refs = json.loads(next(r.data for r in context.public_problem_bundle.resources if r.relative_path == "baseline_context.json"))
    assert refs["seed_implementations"][0]["tree_sha256"] == chosen["selected"]


def test_final_selector_must_use_valid_result_and_dedup_keeps_first_plan():
    plan = _plans()[0]
    unique, checks, duplicates = screen_plans([plan, {**plan, "plan_id": "copy", "description": "renamed"}])
    assert unique == [plan] and len(checks) == 1
    assert duplicates == [{"plan_id": "copy", "duplicate_of": "0"}]
    with pytest.raises(ValueError, match="successfully"):
        validate_final_choice({"plan_id": "failed", "rationale": "invented"}, [{"plan_id": "0"}])
    with pytest.raises(ValueError, match="null"):
        LiteratureOptions(protocol="completion_v2")


def test_nullable_sample_allowance_is_preserved_on_wire():
    from pde_agent_verifier_solver.engine.agent_capability import _sample_batch_to_payload, _sample_batch_from_payload
    from pde_agent_verifier_solver.engine.agent_wire import AgentWireCodec
    development, _ = _records()
    batch = SampleBatch((development[0].case,), None)
    codec = AgentWireCodec()
    restored = _sample_batch_from_payload(_sample_batch_to_payload(batch, codec), codec)
    assert restored.remaining is None and restored.count == 1


def test_one_native_stage_resumes_same_session_workspace_and_nomination_sequence(tmp_path):
    runtime = tmp_path / "private-runtime"
    runtime.mkdir()
    native = _write_resumable_codex(tmp_path, failures=(), hang_after_attempt=1)
    auth, original_auth = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=None,
        capability_facade=UnavailableAgentCapabilityFacade(), poll_interval_seconds=.01,
        termination_grace_seconds=5.)
    program = replace(command_program(native, auth, runtime, gpu_devices=("0",)),
        worker_settings={"nomination_policy": "latest", "transient_resume_max_attempts": 2, "continuation_backend": "canonical_codex_v1"})
    state, checkpoints = {}, []
    def save():
        checkpoints.append(json.loads(json.dumps(state)))
    # Use the worker's round log, atomically written after the first nomination.
    service.allocation_should_pause = lambda: bool(state) and (Path(state["workspace"]) / "ROUND_LOG.md").read_text().startswith("# Same research turn")
    with pytest.raises(CampaignPaused):
        service.execute_continuable(program, agent_context(), state, save)
    assert service.audits[-1].status == "allocation_paused"
    assert service.audits[-1].process_group_cleaned
    assert service.audits[-1].error is None
    assert service.audits[-1].nomination_count == 0  # partial work isn't a finished stage
    first = state.copy()
    assert checkpoints and checkpoints[0]["workspace"] == first["workspace"]
    marker = json.loads((Path(first["workspace"]) / "codex-continuation.json").read_text())
    assert not (Path(marker["runtime"]) / "codex-home" / "auth.json").exists()
    service2 = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=None,
        capability_facade=UnavailableAgentCapabilityFacade(), poll_interval_seconds=.01,
        termination_grace_seconds=5.)
    outcome = service2.execute_continuable(program, agent_context(), state, save)
    assert outcome.execution_status == "completed"
    assert state["workspace"] == first["workspace"] and state["execution_id"] == first["execution_id"]
    assert state["segment"] == 2
    starts = [json.loads(line) for line in (Path(state["workspace"]) / "native-attempt-starts.jsonl").read_text().splitlines()]
    assert len(starts) == 2 and not starts[0]["resumed"] and starts[1]["resumed"]
    assert starts[0]["thread_id"] == starts[1]["thread_id"]
    assert service2.audits[-1].final_nomination_sequence == 2
    assert "None seconds" not in starts[1]["prompt"]
    assert auth.read_bytes() == original_auth.encode()
    assert not (Path(marker["runtime"]) / "codex-home" / "auth.json").exists()


def test_selector_choice_is_used_even_when_not_the_minimum_error(tmp_path):
    class Agent(CompletePlanner):
        def run(self, context):
            if context.agent.name.endswith("final selection"):
                self.contexts.append(context)
                return AgentOutcome(round_log=json.dumps({"plan_id": "1", "rationale": "Choose plan 1 explicitly for this test."}))
            return super().run(context)
    development, selection = _records()
    agent = Agent(_plan_artifact(tmp_path))
    result = run_ns_baseline(NSBaselineConfig("autonumerics_ns_adaptation", tmp_path / "run",
        seed=101, agent_budget_seconds=None, literature_options=options(autonumerics_top_k=2)),
        development_records=development, selection_records=selection, evaluator_policy=_policy(),
        task_settings={"official_point_count": 4}, agent_program=agent,
        agent_execution=InProcessAgentExecutionService())
    assert result.selection_error == pytest.approx(1.)
    scores = json.loads((result.output_directory / "selection_results.json").read_text())
    assert min(r["primary_error"] for r in scores) == pytest.approx(0.)


def test_finite_development_answers_are_same_in_both_arms_without_truth_oracle(tmp_path, field):
    from tests.test_ns_continuous_case_bank import _bank
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSVelocityTruth
    from pde_agent_verifier_solver.plugins.ns.baselines.environment import PairedSessionProvider
    from pde_agent_verifier_solver.sdk import CaseVisibility, CaseAccessKind
    _, records, _, _ = _bank(tmp_path / "eval", field, purpose="promotion")
    record = replace(records[0], case=replace(records[0].case, case_id="starter", visibility=CaseVisibility.DEVELOPMENT),
                     truth=_NSVelocityTruth(records[0].truth._answer))
    path = write_development_bank(tmp_path / "starter.json", (record,), provenance={},
                                  grid_size=4, forcing_times=3, include_reference=True)
    fixed = load_development_bank(path, arm="fixed_sampled", require_reference=True)
    online = load_development_bank(path, arm="continuous_online_base", require_reference=True)
    ref = fixed.records[0].case.public_data["development_reference"]
    other = online.records[0].case.public_data["development_reference"]
    np.testing.assert_array_equal(ref["velocity"], other["velocity"])
    assert ref["velocity"].shape == (3, 2, 4, 4) and ref["velocity"].dtype == np.float64
    sample = np.array([[ref["t"][1], ref["x"][2], ref["y"][3]]])
    np.testing.assert_array_equal(ref["velocity"][1, :, 2, 3], online.records[0].truth._answer.evaluate_velocity(sample)[0])
    selection = type("Bank", (), {"records": records, "manifest_sha256": "sel", "_answers": {
        r.case.case_id: r.truth._answer for r in records}})()
    provider = PairedSessionProvider(online, selection, arm="continuous_online_base")
    with pytest.raises(PermissionError):
        provider.open_case_session(provider.state, online.records[0].case, CaseAccessKind.DEVELOPMENT_SOLUTION)
    assert "arrays" not in online.records[0].case.public_data
    old = write_development_bank(tmp_path / "old.json", (record,), provenance={}, grid_size=4, forcing_times=3)
    with pytest.raises(ValueError, match="reference arrays"):
        load_development_bank(old, arm="fixed_sampled", require_reference=True)


def test_online_unlimited_draws_restore_and_retain_cases_across_roles(tmp_path):
    from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator
    from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
    from pde_agent_verifier_solver.sdk import CaseSource
    dev, _ = _records()
    class Generator:
        def sample(self, state, request):
            return tuple(replace(dev[0], case=replace(dev[0].case, source=CaseSource.GENERATED,
                case_id=f"fresh-{request.seed}-{i}")) for i in range(request.count))
    env = OnlineBaseEnvironment(Generator(), object(), new_cases_per_branch=None)
    state = {}
    env.bind(state)
    evaluator = AgentEvaluator(dev, NSSolverTask(), _policy(), {}, seed=101)
    first = env.sampler("planning", evaluator, 101)
    assert first.sample_cases(SampleCasesRequest(SampleSource.BASE, 65)).remaining is None
    restored = OnlineBaseEnvironment(env.generator, env.generator_state, new_cases_per_branch=None)
    restored.bind(state)
    second = restored.sampler("planning", evaluator, 102)
    assert second.sample_cases(SampleCasesRequest(SampleSource.BASE, 65)).remaining is None
    assert len(restored.records("planning")) == 130
    with pytest.raises(PermissionError):
        second.sample_cases(SampleCasesRequest(SampleSource.GUIDED, 1))


def test_planning_roles_have_online_tools_and_case_cap_is_really_null(tmp_path):
    # Exercise the real host rather than just inspecting configuration literals.
    class Environment:
        description = {"arm": "continuous_online_base"}
        def bind(self, state): self.progress = state
        def records(self, branch): return ()
        def sampler(self, branch, evaluator, seed):
            return type("Sampler", (), {"remaining": None})()
    agent = CompletePlanner(_plan_artifact(tmp_path))
    run(tmp_path / "run", "autonumerics_ns_adaptation", agent, environment=Environment())
    assert all(c.sampler is not None for c in agent.contexts)
    assert all(c.round_plan.development_sampling.max_cases_per_agent_per_round is None for c in agent.contexts)


@pytest.mark.parametrize("method,phase,inflight", [
    ("codepde_refinement_ns_adaptation", "codepde generate", "sample-1/trial-0"),
    ("autopde_ns_adaptation", "autopde develop", "autopde/develop"),
])
def test_controller_checkpoint_resumes_native_action_without_replaying_method_steps(tmp_path, method, phase, inflight):
    from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityFacade
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl
    from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
    from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
    from pde_agent_verifier_solver.sdk import TaskKind
    native = _write_resumable_codex(tmp_path, failures=(), hang_after_attempt=1)
    source = native.read_text().replace("str(attempt / 2)", "'1.0'")
    source = source.replace("filename = 'solver-' + str(attempt) + '.py'", "filename = 'solver.py'")
    source = source.replace("if attempt == 1:\n    time.sleep(60)",
                            f"if attempt == 1 and request['agent']['name'] == {phase!r} and 'RESUME_TEST_PAUSE' in prompt:\n    time.sleep(60)")
    source = source.replace("prompt = sys.stdin.read()", """prompt = sys.stdin.read()
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
client = ControllerAgentCapabilityClient(request['capabilities'], workspace=workspace)
assert client.current_cases()
""")
    native.write_text(source)
    auth, _ = write_auth(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
    program = replace(command_program(native, auth, runtime, gpu_devices=("0",)),
        nomination_decoder=codec.decode_nomination, incumbent_encoder=codec.encode_incumbent,
        worker_settings={"nomination_policy": "latest", "continuation_backend": "canonical_codex_v1"})
    dev, selection = _records()
    outcomes = []
    for paused in (False, True):
        root = tmp_path / ("paused" if paused else "continuous")
        root.mkdir()
        config = NSBaselineConfig(method, root / "result",
            prompt="Develop the solver." + (" RESUME_TEST_PAUSE" if paused else ""),
            seed=101, agent_budget_seconds=None, literature_options=options(codepde_samples=1, autopde_attempts=1, autopde_review_seconds=None))
        store = CheckpointStore(root / "checkpoints", identity={"completion-test": 1})
        control = CampaignPauseControl(store.save, lambda *_: False,
                                      agent_start_margin_seconds=0, hidden_start_margin_seconds=0)
        service = SubprocessAgentExecutionService(root / "turns", hard_wall_seconds=None,
            capability_facade=ControllerAgentCapabilityFacade(), termination_grace_seconds=10.)
        if paused:
            service.allocation_should_pause = lambda: any(p.read_text().startswith("# Same research turn")
                for p in (root / "turns").glob("*/ROUND_LOG.md"))
        kwargs = dict(development_records=dev, selection_records=selection, evaluator_policy=_policy(),
            agent_program=program, task_settings={"official_point_count": 4}, pause_control=control)
        if paused:
            with pytest.raises(CampaignPaused):
                run_ns_baseline(config, agent_execution=service, **kwargs)
            saved = store.load()
            assert saved["literature"]["inflight_agent"] == inflight
            assert saved["turns"] == []
            service = SubprocessAgentExecutionService(root / "turns", hard_wall_seconds=None,
                capability_facade=ControllerAgentCapabilityFacade(), termination_grace_seconds=10.)
            result = run_ns_baseline(config, agent_execution=service, continuation=saved, **kwargs)
        else:
            result = run_ns_baseline(config, agent_execution=service, **kwargs)
        assert result.report["completed_agent_turns"] == 2
        outcomes.append((result.selected.program.tree_sha256, result.selection_error))
    assert outcomes[0] == outcomes[1]


def test_new_templates_match_completion_protocol_without_numeric_quota_sentinels(tmp_path):
    root = Path(__file__).resolve().parents[1] / "configs/paper/ns_literature_v2"
    for method in ("codepde", "autonumerics", "autopde", "pde_sharp"):
        configs = [json.loads((root / f"{method}_{arm}.template.json").read_text())
                   for arm in ("fixed_sampled", "continuous_online_base")]
        for config in configs:
            parsed = NSBaselineConfig(config["method"], tmp_path / "out", seed=config["seed"],
                agent_budget_seconds=config["agent"]["hard_wall_seconds"], **config["search"])
            assert parsed.agent_budget_seconds is None
            assert config["search"]["literature_options"]["protocol"] == "completion_v2"
            assert config["agent"]["worker_settings"]["continuation_backend"] == "canonical_codex_v1"
            assert config["evaluation"]["candidate_case_hard_wall_seconds"] == 300
            assert config["evaluation"]["development_batch_hard_wall_seconds"] is None
            assert not config["evaluation"]["allow_point_truth"]
        assert configs[0]["evaluation"] == configs[1]["evaluation"]
        assert configs[0]["agent"] == configs[1]["agent"]
        assert configs[0]["dataset"]["manifest"] == configs[1]["dataset"]["manifest"]
        assert configs[1]["dataset"]["new_cases_per_branch"] is None
