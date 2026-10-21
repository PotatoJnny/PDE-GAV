"""Method behavior, actual plan execution, budget/Case isolation and continuation."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl, CampaignPaused
from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.plugins.ns.baselines import NSBaselineConfig, run_ns_baseline
from pde_agent_verifier_solver.plugins.ns.baselines.environment import (
    OnlineBaseEnvironment, PairedSessionProvider, load_development_bank, write_development_bank,
)
from pde_agent_verifier_solver.plugins.ns.baselines.workflows import LiteratureOptions, coarse_plan, rank_plans
from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.sdk import CaseAccessKind, CaseSource, CaseVisibility, SampleCasesRequest, SampleSource, TaskKind
from tests.test_ns_baselines import SOURCE, _artifact, _policy, _records, SequenceAgent
from tests.test_ns_continuous_case_bank import field, _bank


def _options(**kwargs):
    return {"total_wall_seconds": 60, "branch_wall_seconds": 20, "planning_call_seconds": 5,
            "codepde_samples": 2, "codepde_trials": 2, "codepde_seed_pool_size": 2,
            "codepde_reference_counts": [2], "autonumerics_plans": 2, "autonumerics_top_k": 1,
            "autonumerics_fresh_starts": 1, "autonumerics_coarse_repairs": 1,
            "autonumerics_fine_repairs": 1, **kwargs}


def _run(root, method, agent, **kwargs):
    development, selection = _records()
    return run_ns_baseline(NSBaselineConfig(method, root, seed=101, agent_budget_seconds=20,
        literature_options=_options()), development_records=development, selection_records=selection,
        evaluator_policy=_policy(), task_settings={"official_point_count": 4}, agent_program=agent,
        agent_execution=InProcessAgentExecutionService(), **kwargs)


def test_codepde_refinement_reads_multiple_frozen_seed_programs_and_scores(tmp_path):
    artifacts = [_artifact(tmp_path, value, f"solver-{i}") for i, value in enumerate((.8, .6, .99))]
    agent = SequenceAgent(artifacts)
    result = _run(tmp_path / "run", "codepde_refinement_ns_adaptation", agent)
    assert result.selection_error == pytest.approx(.01)
    assert len(agent.contexts) == 3  # finite but imperfect programs do NOT trigger debugging
    resources = {r.relative_path: r.data for r in agent.contexts[2].public_problem_bundle.resources}
    refs = json.loads(resources["baseline_context.json"])["seed_implementations"]
    assert len(refs) == 2
    assert sorted(r["validation_error"] for r in refs) == pytest.approx([.2, .4])
    seeds = {a.program.tree_sha256: a for a in artifacts[:2]}
    for ref in refs:
        a = seeds[ref["tree_sha256"]]
        manifest = Path(a.program.bundle_locator)
        assert resources[ref["entrypoint"]].decode() == (manifest.parent / json.loads(manifest.read_text())["entrypoint"]).read_text()
    assert agent.contexts[2].incumbent is None
    report = json.loads((result.output_directory / "literature_search.json").read_text())
    assert report["records"]["codepde_seed_pool"]["frozen_before_refinement"]
    assert result.report["terminal_score"] is None


def test_codepde_nonfinite_failure_is_debugged_with_incumbent(tmp_path):
    # Finite syntax, but candidate execution fails: this is a valid debugging parent.
    path = tmp_path / "bad.py"
    path.write_text('def prepare_solution(case, solve_budget=None):\n    raise ValueError("repair me")\n')
    bad = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(path, storage_root=tmp_path / "bad-bundle",
        artifact_id="bad", name="bad", summary="bad", usage="test")
    good = _artifact(tmp_path, .9, "good")
    class Agent:
        contexts = []
        def run(self, context):
            self.contexts.append(context)
            return AgentOutcome(bad if len(self.contexts) == 1 else good)
    agent = Agent()
    result = _run(tmp_path / "run", "codepde_refinement_ns_adaptation", agent)
    assert result.selection_error == pytest.approx(.1)
    assert agent.contexts[1].incumbent.program.tree_sha256 == bad.program.tree_sha256
    assert "codepde debug" == agent.contexts[1].agent.name


def _plans():
    return [{"plan_id": str(i), "description": f"method {i}",
             "spatial_discretization": {"method": "finite difference", "Nx": n, "Ny": n},
             "time_stepping": {"method": "Euler", "dt": .001}} for i, n in enumerate((64, 128))]


def _plan_artifact(root):
    source = SOURCE.replace("FACTOR", "self.factor").replace("class Handle:\n", "class Handle:\n    def __init__(self, factor=1.): self.factor = factor\n")
    source += '\ndef prepare_solution_for_plan(case, plan, solve_budget=None):\n    return Handle(plan["spatial_discretization"]["Nx"] / 64.)\n'
    path = root / "plan.py"
    path.write_text(source)
    return PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(path, storage_root=root / "frozen-plan",
        artifact_id="plan", name="plan", summary="plan", usage="test")


class PlanningAgent:
    def __init__(self, artifact): self.artifact, self.contexts = artifact, []
    def run(self, context):
        self.contexts.append(context)
        phase = context.agent.name
        if phase.endswith("formulator"):
            payload = {k: "NS" for k in ("equation", "boundary_conditions", "input_interface", "numerical_features")}
        elif phase.endswith("planner"):
            payload = {"plans": _plans()}
        elif phase.endswith("selector"):
            payload = {"scores": [{"plan_id": p["plan_id"], "score": 100 - i, "verdict": "accept"}
                                   for i, p in enumerate(_plans())]}
        else:
            return AgentOutcome(self.artifact, round_log="Implemented selected plan.")
        return AgentOutcome(round_log=json.dumps(payload))


def test_autonumerics_plans_then_executes_real_coarse_and_fine_grid_arguments(tmp_path):
    agent = PlanningAgent(_plan_artifact(tmp_path))
    result = _run(tmp_path / "run", "autonumerics_ns_adaptation", agent)
    assert result.selection_error == pytest.approx(0.)
    assert len(agent.contexts) == 4
    reports = list(json.loads((result.output_directory / "literature_evaluations.json").read_text()).values())
    coarse = next(r for r in reports if r["fidelity"] == "coarse")
    fine = next(r for r in reports if r["fidelity"] == "fine")
    assert coarse["error"] == pytest.approx(.75)
    assert fine["error"] == pytest.approx(0.)
    assert coarse["plan"]["spatial_discretization"]["Nx"] == 16
    assert fine["plan"]["spatial_discretization"]["Nx"] == 64
    assert fine["tree_sha256"] == result.selected.program.tree_sha256
    assert coarse["tree_sha256"] != fine["tree_sha256"]


def test_autonumerics_rejects_malformed_json_with_bounded_retries(tmp_path):
    class Invalid:
        count = 0
        def run(self, context):
            self.count += 1
            return AgentOutcome(round_log='{"bad":NaN}')
    agent = Invalid()
    with pytest.raises(ValueError, match="bounded JSON"):
        _run(tmp_path / "run", "autonumerics_ns_adaptation", agent)
    assert agent.count == 2


def test_plan_selector_all_rejected_fallback_and_resolution_does_not_mutate_original():
    plans = _plans()
    selected, fallback = rank_plans(plans, {"scores": [
        {"plan_id": "0", "score": 5, "verdict": "reject"},
        {"plan_id": "1", "score": 9, "verdict": "reject"}]}, 1)
    assert fallback and selected[0]["plan_id"] == "1"
    assert coarse_plan(plans[0], 4)["spatial_discretization"]["Nx"] == 16
    assert plans[0]["spatial_discretization"]["Nx"] == 64
    with pytest.raises(ValueError):
        LiteratureOptions(autonumerics_top_k=11)


def test_completed_agent_actions_resume_without_replay_or_refunding_budget(tmp_path):
    artifacts = [_artifact(tmp_path, value, f"s-{i}") for i, value in enumerate((.8, .6, .99))]
    agent = SequenceAgent(artifacts)
    store = CheckpointStore(tmp_path / "checkpoint", identity={"test": 1})
    armed = {"paused": False}
    def should_pause(stage, required):
        if stage == "literature_agent" and len(agent.contexts) == 1 and not armed["paused"]:
            armed["paused"] = True
            return True
        return False
    pause = CampaignPauseControl(store.save, should_pause, agent_start_margin_seconds=0, hidden_start_margin_seconds=0)
    with pytest.raises(CampaignPaused):
        _run(tmp_path / "run", "codepde_refinement_ns_adaptation", agent, pause_control=pause)
    saved = store.load()
    elapsed = saved["literature"]["agent_wall_seconds"]
    assert elapsed > 0 and len(saved["turns"]) == 1
    result = _run(tmp_path / "run", "codepde_refinement_ns_adaptation", agent,
                  continuation=saved, pause_control=pause)
    assert len(agent.contexts) == 3
    assert result.selection_error == pytest.approx(.01)
    assert saved["literature"]["agent_wall_seconds"] >= elapsed
    assert len(saved["turns"]) == 3


def test_online_cases_are_branch_owned_and_quota_survives_repair_and_restore():
    dev, hidden = _records()
    class Generator:
        def sample(self, state, request):
            return tuple(replace(dev[0], case=replace(dev[0].case, source=CaseSource.GENERATED,
                case_id=f"fresh-{request.seed}-{i}")) for i in range(request.count))
    env = OnlineBaseEnvironment(Generator(), object(), new_cases_per_branch=2, reserved_case_ids=("dev", "selection"))
    progress = {}
    env.bind(progress)
    def evaluator(branch):
        return AgentEvaluator((*dev, *env.records(branch)), NSSolverTask(), _policy(), {}, seed=101)
    first = env.sampler("a", evaluator("a"), 101)
    batch = first.sample_cases(SampleCasesRequest(SampleSource.BASE, 2))
    assert batch.remaining == 0 and len(env.records("a")) == 2
    assert not env.records("b")
    restored = OnlineBaseEnvironment(env.generator, env.generator_state, new_cases_per_branch=2)
    restored.bind(progress)
    repair = restored.sampler("a", evaluator("a"), 999)
    with pytest.raises(ValueError, match="allowance"):
        repair.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))
    with pytest.raises(PermissionError):
        first.sample_cases(SampleCasesRequest(SampleSource.GUIDED, 1))
    second = env.sampler("b", evaluator("b"), 101)
    assert second.sample_cases(SampleCasesRequest(SampleSource.BASE, 2)).remaining == 0
    assert {r.case.case_id for r in env.records("a")}.isdisjoint(r.case.case_id for r in env.records("b"))


def test_same_continuous_fields_exported_to_finite_inputs_and_oracle_permissions(tmp_path, field):
    _, records, _, _ = _bank(tmp_path / "eval", field, purpose="promotion")
    starter_record = replace(records[0], case=replace(records[0].case, case_id="starter",
                            visibility=CaseVisibility.DEVELOPMENT))
    # writer requires arbitrary-coordinate DEVELOPMENT truth, not frozen selection queries.
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSVelocityTruth
    starter_record = replace(starter_record, truth=_NSVelocityTruth(starter_record.truth._answer))
    path = write_development_bank(tmp_path / "starter.json", (starter_record,), provenance={}, grid_size=4, forcing_times=3)
    finite = load_development_bank(path, arm="fixed_sampled", expected_cases=1)
    continuous = load_development_bank(path, arm="continuous_online_base", expected_cases=1)
    assert finite.manifest_sha256 == continuous.manifest_sha256
    arrays = finite.records[0].case.public_data["arrays"]
    points = np.array([[0., arrays["x"][0], arrays["y"][0]]])
    np.testing.assert_array_equal(arrays["initial_velocity"][:, 0, 0],
                                  continuous.records[0].truth._answer.evaluate_velocity(points)[0])
    assert arrays["forcing"].shape == (3, 2, 4, 4)
    assert arrays["forcing"].dtype == np.float64
    selection = type("Bank", (), {"records": records, "manifest_sha256": "selection", "_answers": {
        r.case.case_id: r.truth._answer for r in records}})()
    fixed_provider = PairedSessionProvider(finite, selection, arm="fixed_sampled")
    with pytest.raises(PermissionError):
        fixed_provider.open_case_session(fixed_provider.state, finite.records[0].case, CaseAccessKind.PUBLIC_PROBLEM)
    online_provider = PairedSessionProvider(continuous, selection, arm="continuous_online_base")
    session = online_provider.open_case_session(online_provider.state, continuous.records[0].case,
                                               CaseAccessKind.PUBLIC_PROBLEM)
    try:
        np.testing.assert_array_equal(session.query_initial_velocity(points[:, 1:]),
                                      arrays["initial_velocity"][:, 0, 0][None])
    finally:
        session.close()
    with pytest.raises(ValueError, match="count"):
        load_development_bank(path, arm="fixed_sampled", expected_cases=64)


def test_branch_and_total_wall_budgets_are_shared_by_debug_calls(tmp_path):
    from pde_agent_verifier_solver.plugins.ns.baselines.literature import LiteratureHost
    from pde_agent_verifier_solver.plugins.ns.baselines.workflows import DevelopmentBudgetExhausted
    dev, sel = _records()
    artifact = _artifact(tmp_path, 1., "good")
    moments = iter((0., 8., 10., 13.))
    config = NSBaselineConfig("codepde_refinement_ns_adaptation", tmp_path, agent_budget_seconds=20,
                              literature_options=_options(total_wall_seconds=11, branch_wall_seconds=10))
    candidates = {artifact.program.tree_sha256: artifact}
    agent = SequenceAgent((artifact, artifact))
    host = LiteratureHost(config, development=dev, selection=sel, task=NSSolverTask(), settings={},
        evaluator_policy=_policy(), agent_program=agent, agent_execution=InProcessAgentExecutionService(),
        public_problem_bundle=None, candidate_execution=None, generator=None, generator_state=None,
        initial_key=None, candidates=candidates, register=lambda a: a.program.tree_sha256,
        score=lambda k: 0., errors={}, turns=[], selections=[], state={}, checkpoint=lambda: None,
        clock=lambda: next(moments))
    host.program("generate", "generate", "branch", instructions="generate")
    host.program("debug", "debug", "branch", instructions="debug")
    assert [ctx.round_plan.agent_budget_seconds
            for ctx in agent.contexts] == [10, 2]
    assert host.state["agent_wall_seconds"] == 11
    with pytest.raises(DevelopmentBudgetExhausted):
        host.program("new", "generate", "other", instructions="generate")


@pytest.mark.parametrize("arm", ["fixed_sampled", "continuous_online_base"])
@pytest.mark.parametrize("method", ["codepde_refinement_ns_adaptation", "autonumerics_ns_adaptation",
                                    "autopde_ns_adaptation", "pde_sharp_ns_adaptation"])
def test_paired_cli_real_canonical_workers_and_cpu_candidate_processes(tmp_path, field, monkeypatch, arm, method):
    import sys
    from pde_agent_verifier_solver.plugins.ns.baselines import environment
    from pde_agent_verifier_solver.plugins.ns.baselines.cli import run_from_config
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSVelocityTruth
    from pde_agent_verifier_solver.plugins.ns.case_bank import NSBankSessionProvider
    from tests.test_ns_continuous_case_bank import ORACLE_SOLVER
    from types import SimpleNamespace
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    selection_path, rows, _, _ = _bank(tmp_path / "selection", field, purpose="promotion")
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer
    starter_field = type(field)(type(field.coefficients)(field.coefficients.psi * 1.1, field.coefficients.pressure),
        config=field.config, geometry_spec=field.geometry_spec, domain_bounds=field.domain_bounds,
        viscosity=field.viscosity, t0=field.t0, t1=field.t1)
    record = replace(rows[0], case=replace(rows[0].case, case_id="starter", visibility=CaseVisibility.DEVELOPMENT),
                     truth=_NSVelocityTruth(StructuredNSMMSAnswer(starter_field)))
    hashes = {"flow_checkpoint": "a" * 64, "pca_checkpoint": "b" * 64}
    starter = write_development_bank(tmp_path / "starter.json", (record,),
                                     provenance={"generator_artifact_sha256": hashes}, grid_size=3, forcing_times=2)
    class Fresh:
        def __init__(self): self.records = {}
        def sample(self, state, request):
            result = tuple(replace(record, case=replace(record.case, case_id=f"fresh-{request.seed}-{i}"))
                           for i in range(request.count))
            self.records.update({r.case.case_id: r for r in result})
            return result
        def case_session_methods(self, access):
            return ("query_initial_velocity", "query_forcing") if access is CaseAccessKind.PUBLIC_PROBLEM else ()
        def open_case_session(self, state, case, access):
            r = self.records[case.case_id]
            bank = SimpleNamespace(records=(r,), _answers={case.case_id: r.truth._answer}, manifest_sha256="fresh")
            provider = NSBankSessionProvider((bank,))
            return provider.open_case_session(provider.state, case, access)
    fresh = Fresh()
    monkeypatch.setattr(environment, "build_online_generator", lambda config, base: (fresh, object()))
    solver = ORACLE_SOLVER.replace('def prepare_solution(case, solve_budget=None):\n',
        'def prepare_solution(case, solve_budget=None):\n    if "arrays" in case.public_data:\n        return Handle(case.public_data["arrays"]["initial_velocity"][:, 0, 0])\n') + '\ndef prepare_solution_for_plan(case, plan, solve_budget=None):\n    return prepare_solution(case, solve_budget=solve_budget)\n'
    codex = tmp_path / "stub-codex"
    codex.write_text(f'''#!{sys.executable}
import json, os, sys, time
from pathlib import Path
from pde_agent_verifier_solver.engine.agent_execution import submit_agent_nomination, write_agent_round_log
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
print(json.dumps({{"type":"thread.started"}}), flush=True)
prompt = sys.stdin.read()
for _ in range(2000):
    if not (Path(os.environ['CODEX_HOME']) / 'auth.json').exists(): break
    time.sleep(.001)
request = json.loads(Path(os.environ['PDE_AGENT_REQUEST']).read_text())
phase = request['agent']['name']
client = ControllerAgentCapabilityClient.from_request(request)
before = len(client.current_cases())
assert before in (1, 2)
if phase == 'autopde diagnosis':
    result = {{k:'NS' for k in ('equation_type','coupling','time_dependence','boundary_conditions','dominant_physics')}}
elif phase == 'autopde method':
    result = {{k:'NS' for k in ('formulation','spatial_method','time_method','algebraic_solver','stabilization','boundary_treatment')}}
elif phase.startswith('sharp analysis '):
    result = {{'finding':'NS','consequences':'stable numerical method'}}
elif phase == 'sharp judge':
    refs = json.loads(Path('public_resources/baseline_context.json').read_text())['seed_implementations']
    selected = [r['tree_sha256'] for r in refs[:(len(refs)+1)//2]]
    result = {{'selected':selected,'nominee':selected[0],'rationale':'inspect candidate'}}
elif phase.endswith('formulator'):
    result = {{k:'NS' for k in ('equation','boundary_conditions','input_interface','numerical_features')}}
elif phase.endswith('planner'):
    result = {{'plans':{_plans()!r}}}
elif phase.endswith('selector'):
    result = {{'scores':[{{'plan_id':str(i),'score':100-i,'verdict':'accept'}} for i in range(2)]}}
else:
    if {arm!r} == 'continuous_online_base':
        assert len(client.sample_base(1).cases) == 1
        assert len(client.current_cases()) == before + 1
    else:
        assert 'arrays' in client.current_cases()[0].public_data
    Path('submission.py').write_text({solver!r})
    submit_agent_nomination({{'bundle_path':'submission.py'}}, sequence=1)
    result = None
if result is not None:
    assert request['worker_settings']['nomination_policy'] == 'all'
    write_agent_round_log(json.dumps(result))
else:
    assert request['worker_settings']['nomination_policy'] == 'latest'
    write_agent_round_log('Local CPU integration test.')
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('done')
''')
    codex.chmod(0o700)
    auth = tmp_path / "dummy-auth.json"
    auth.write_text('{"test_only":true}')
    auth.chmod(0o600)
    dataset = {"kind": "continuous_mms_paired", "manifest": str(starter), "arm": arm,
               "expected_cases": 1, "selection_manifest": str(selection_path)}
    if arm == "continuous_online_base":
        dataset.update(online_generator={"artifact_sha256": hashes}, new_cases_per_branch=2)
    config = {"schema": "ns_fixed_data_baseline_v1", "method": method, "dataset": dataset,
        "search": {"literature_options": _options(total_wall_seconds=300, branch_wall_seconds=60, planning_call_seconds=30,
            autopde_attempts=1, autopde_review_seconds=20, sharp_samples=2, sharp_judges=1,
            sharp_cycles=1, sharp_hybrid_rounds=1)},
        "agent": {"command": ["{python}", "{repo}/scripts/run_canonical_codex_agent.py",
            "--codex-binary", str(codex), "--auth-file", str(auth), "--runtime-root", "{output}/agent_runtime",
            "--config", "{repo}/scripts/canonical_codex.config.toml"], "hard_wall_seconds": 30, "gpu_devices": ["0"],
            "worker_settings": {"model": "offline-test", "reasoning_effort": "low"}},
        "evaluation": {"development_point_count": 3, "selection_point_count": 4,
                       "candidate_case_hard_wall_seconds": 20, "candidate_case_workers": 2}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = run_from_config(config_path, tmp_path / "run")
    assert result.report["status"] == "completed" and result.report["failed_turns"] == 0
    expected_turns = {"autonumerics_ns_adaptation":4, "codepde_refinement_ns_adaptation":3,
                      "autopde_ns_adaptation":4, "pde_sharp_ns_adaptation":9}
    assert result.report["completed_agent_turns"] == expected_turns[method]
    assert Path(result.report["selected_bundle"]).is_file()
    assert bool(fresh.records) == (arm == "continuous_online_base")
    audits = json.loads((result.output_directory / "candidate_execution_audits.json").read_text())
    assert audits  # actual CPU candidate workers ran, not just a mocked score callback


def test_plan_wrapper_preserves_relative_imports_and_legacy_solve_clock(tmp_path):
    from pde_agent_verifier_solver.plugins.ns.baselines.literature import bind_plan
    from pde_agent_verifier_solver.sdk.candidate_bundle import write_candidate_bundle_manifest
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    folder = tmp_path / "authored" / "solver"
    folder.mkdir(parents=True)
    (folder / "helper.py").write_text('SCALE = .5\n')
    code = SOURCE.replace("FACTOR", "SCALE") + '''
def prepare_solution_for_plan(case, plan, solve_budget=None):
    from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock
    assert perf_counter is candidate_solve_clock
    return prepare_solution(case, solve_budget)
'''
    (folder / "submission.py").write_text('from .helper import SCALE\nfrom time import perf_counter\n' + code)
    manifest = write_candidate_bundle_manifest(folder.parent, entrypoint="solver/submission.py",
        files=("solver/submission.py", "solver/helper.py"))
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(manifest,
        storage_root=tmp_path / "frozen", artifact_id="relative", name="relative", summary="relative", usage="test")
    planned = bind_plan(artifact, _plans()[0], tmp_path / "planned")
    _, selection = _records()
    result = HiddenVerifier(NSSolverTask(), _policy().default_query, {"official_point_count": 4}).evaluate(planned, selection)
    assert result.primary_error == pytest.approx(.5)


def test_paired_feedback_coordinates_are_identical_and_not_limited_to_array_nodes(tmp_path, field):
    from pde_agent_verifier_solver.plugins.ns.baselines.environment import PairedNSSolverTask
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSVelocityTruth
    _, records, _, _ = _bank(tmp_path, field)
    record = replace(records[0], case=replace(records[0].case, visibility=CaseVisibility.DEVELOPMENT),
                     truth=_NSVelocityTruth(records[0].truth._answer))
    path = write_development_bank(tmp_path / "dev.json", (record,), provenance={}, grid_size=4, forcing_times=3)
    banks = [load_development_bank(path, arm=arm) for arm in ("fixed_sampled", "continuous_online_base")]
    settings = {"default_query": _policy().default_query, "development_point_count": 4}
    evaluator = PairedNSSolverTask().create_evaluation(settings)
    plans = [evaluator.make_private_development_plan(bank.records[0].case, settings, 101) for bank in banks]
    assert plans[0] == plans[1]
    assert any(p["x"] not in banks[0].records[0].case.public_data["arrays"]["x"] for p in plans[0].query.points)


def test_autonumerics_resume_after_coder_reuses_planning_and_nomination(tmp_path):
    agent = PlanningAgent(_plan_artifact(tmp_path))
    store = CheckpointStore(tmp_path / "checkpoint", identity={"test": "auto"})
    armed = {"paused": False}
    def should_pause(stage, required):
        if stage == "literature_coarse" and not armed["paused"]:
            armed["paused"] = True
            return True
        return False
    pause = CampaignPauseControl(store.save, should_pause, agent_start_margin_seconds=0, hidden_start_margin_seconds=0)
    with pytest.raises(CampaignPaused):
        _run(tmp_path / "run", "autonumerics_ns_adaptation", agent, pause_control=pause)
    assert len(agent.contexts) == 4
    result = _run(tmp_path / "run", "autonumerics_ns_adaptation", agent,
                  continuation=store.load(), pause_control=pause)
    assert len(agent.contexts) == 4 and result.selection_error == pytest.approx(0.)


def test_comparison_templates_keep_paired_budgets_and_banks_identical(tmp_path):
    from pde_agent_verifier_solver.plugins.ns.baselines.preparation import _replace, prepare_comparisons
    templates = Path(__file__).resolve().parents[1] / "configs/paper/ns_literature_v1"
    values = {"@STARTER@": "same-starter", "@SELECTION@": "same-selection", "@INITIAL@": "same-s0",
              "@INITIAL_SHA256@": "a" * 64, "@CODEX@": "codex", "@AUTH@": "auth.json",
              "@GPU@": "0", "@WORKERS@": 4, "@GENERATOR@": {"artifact_sha256": {}}}
    for name in ("codepde", "autonumerics"):
        fixed = _replace(json.loads((templates / f"{name}_fixed_sampled.template.json").read_text()), values)
        online = _replace(json.loads((templates / f"{name}_continuous_online_base.template.json").read_text()), values)
        assert fixed["agent"] == online["agent"] and fixed["evaluation"] == online["evaluation"]
        assert fixed["search"] == online["search"]
        assert fixed["dataset"]["manifest"] == online["dataset"]["manifest"]
        assert fixed["dataset"]["selection_manifest"] == online["dataset"]["selection_manifest"]
        for config in (fixed, online):
            NSBaselineConfig(config["method"], tmp_path, agent_budget_seconds=config["agent"]["hard_wall_seconds"], **config["search"])
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"schema": "ns_literature_binding_v1", **{
        name: "missing-file" for name in ("starter_manifest", "selection_manifest", "terminal_manifest",
                                         "initial_bundle", "codex_binary", "auth_file")}}))
    with pytest.raises(FileNotFoundError):
        prepare_comparisons(binding, tmp_path / "must-not-publish")
    assert not (tmp_path / "must-not-publish").exists()


def test_terminal_export_requires_four_completed_unmodified_results(tmp_path):
    from pde_agent_verifier_solver.plugins.ns.baselines.preparation import export_terminal_manifest
    from hashlib import sha256
    from pde_agent_verifier_solver.sdk.candidate_bundle import inspect_candidate_bundle
    initial = _artifact(tmp_path, 0., "initial")
    final = _artifact(tmp_path, 1., "final")
    terminal = tmp_path / "terminal.json"
    terminal.write_text('{"test_fixture":true}')
    runs = []
    for index in range(4):
        root = tmp_path / str(index)
        root.mkdir()
        config = root / "config.json"
        config.write_text(json.dumps({"fixture": index}))
        digest = sha256(config.read_bytes()).hexdigest()
        label = f"method-{index}"
        (root / "baseline_input.json").write_text(json.dumps({"config_sha256": digest}))
        (root / "baseline_result.json").write_text(json.dumps({"status": "completed", "method": label,
            "selected_bundle": str(final.program.bundle_locator), "selected_tree_sha256": final.program.tree_sha256,
            "selected_artifact_id": final.artifact_id}))
        runs.append({"label": label, "method": label, "output": str(root), "config": str(config), "config_sha256": digest})
    plan = tmp_path / "comparisons.json"
    plan.write_text(json.dumps({"schema": "ns_literature_comparisons_v1", "runs": runs, "seed": 101,
        "terminal_manifest": str(terminal), "terminal_manifest_sha256": sha256(terminal.read_bytes()).hexdigest(),
        "initial_bundle": str(initial.program.bundle_locator), "initial_tree_sha256": initial.program.tree_sha256,
        "candidate_case_workers": 4}))
    exported = export_terminal_manifest(plan, tmp_path / "programs.json")
    result = json.loads(exported.read_text())
    assert len(result["programs"]) == 5 and result["require_initial"]
    assert all(inspect_candidate_bundle(row["bundle"]).tree_sha256 == row["expected_tree_sha256"] for row in result["programs"])
    unfinished = Path(runs[0]["output"]) / "baseline_result.json"
    payload = json.loads(unfinished.read_text())
    payload["status"] = "failed"
    unfinished.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="every arm must finish"):
        export_terminal_manifest(plan, tmp_path / "bad-programs.json")
    assert not (tmp_path / "bad-programs.json").exists()


def test_online_generator_case_assets_and_quota_survive_real_cpu_checkpoint(tmp_path, field):
    from types import SimpleNamespace
    from hashlib import sha256
    from tests.test_ns_resume import _artifacts, _request, _coefficients
    from pde_agent_verifier_solver.plugins.ns.baselines.environment import build_online_generator
    from pde_agent_verifier_solver.sdk import GenerationPurpose
    artifacts = tmp_path / "weights"
    _artifacts(artifacts)
    config = {"generator_id": "baseline-resume-test", "generator_sha256": "a" * 64,
        "flow_checkpoint": str(artifacts / "flow.pt"), "pca_checkpoint": str(artifacts / "pca.pt"),
        "artifact_sha256": {name: sha256((artifacts / file).read_bytes()).hexdigest() for name, file in
                            (("flow_checkpoint", "flow.pt"), ("pca_checkpoint", "pca.pt"))},
        "time_interval": [0., .12], "viscosity": .01, "perturbation_amplitude": 1e-5,
        "device": "cpu", "integration_steps": 2, "flow_state": "raw", "spectral_max_mode": 2}
    generator, state = build_online_generator(config, tmp_path)
    records = generator.sample(state, replace(_request(count=1, seed=1), visibility=CaseVisibility.DEVELOPMENT,
                                             purpose=GenerationPurpose.DEVELOPMENT))
    generator.release(state, tuple(r.case.case_id for r in records))
    starter = SimpleNamespace(records=records, _answers={r.case.case_id: r.truth._answer for r in records}, manifest_sha256="starter")
    _, hidden, _, _ = _bank(tmp_path / "hidden", field, purpose="promotion")
    selection = SimpleNamespace(records=hidden, _answers={r.case.case_id: r.truth._answer for r in hidden}, manifest_sha256="selection")
    provider = PairedSessionProvider(starter, selection, dynamic=generator, dynamic_state=state, arm="continuous_online_base")
    env = OnlineBaseEnvironment(generator, state, new_cases_per_branch=2,
                              reserved_case_ids=[r.case.case_id for r in (*records, *hidden)])
    env.bind({})
    evaluator = AgentEvaluator(records, NSSolverTask(), _policy(), {}, seed=1,
                               generator=provider, generator_state=provider.state)
    env.sampler("plan-1", evaluator, 101).sample_cases(SampleCasesRequest(SampleSource.BASE, 1))
    store = CheckpointStore(tmp_path / "checkpoint", identity={"test": "online"}, generator=provider)
    store.save({"environment": env.progress})
    original_coefficients = _coefficients(env.records("plan-1"))
    generator2, state2 = build_online_generator(config, tmp_path)
    provider2 = PairedSessionProvider(starter, selection, dynamic=generator2, dynamic_state=state2, arm="continuous_online_base")
    restored = CheckpointStore(tmp_path / "checkpoint", identity={"test": "online"}, generator=provider2).load()
    env2 = OnlineBaseEnvironment(generator2, state2, new_cases_per_branch=2)
    env2.bind(restored["environment"])
    assert _coefficients(env2.records("plan-1")) == original_coefficients
    evaluator2 = AgentEvaluator((*records, *env2.records("plan-1")), NSSolverTask(), _policy(), {}, seed=2,
                                generator=provider2, generator_state=provider2.state)
    for current, service in ((env, evaluator), (env2, evaluator2)):
        sampler = current.sampler("plan-1", service, 202)
        assert sampler.remaining == 1
        sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))
        assert sampler.remaining == 0
    assert _coefficients(env.records("plan-1")) == _coefficients(env2.records("plan-1"))
    record2 = env2.records("plan-1")[0]
    session = provider2.open_case_session(provider2.state, record2.case, CaseAccessKind.PUBLIC_PROBLEM)
    try:
        assert np.isfinite(session.query_initial_velocity([[.2, .3]])).all()
    finally:
        session.close()
