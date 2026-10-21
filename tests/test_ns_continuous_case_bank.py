"""Continuous-bank roundtrip, fixed plans, scoped oracles and existing CLI checks."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import stat

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
from pde_agent_verifier_solver.plugins.ns.case_bank import (
    load_ns_continuous_case_bank, write_ns_continuous_case_bank,
)
from pde_agent_verifier_solver.plugins.ns.continuous_backend import (
    LegacyStructuredNSSmoothPerturber, StructuredNSManufacturedCaseSource,
)
from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, CaseAccessKind, CaseVisibility, EvaluationPhase, GenerationKind,
    GenerationPurpose, write_candidate_bundle_manifest,
)
from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def field(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "legacy/ns/src"))
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        ContinuousStructuredNSMMS, StructuredNSCaseCoefficients, StructuredNSProjectionConfig,
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    random = torch.Generator().manual_seed(101)
    geometry = ContinuousGeometrySpec(domain_bounds=(0., 1., 0., 1.), obstacle_polygons=())
    result = ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(
            psi=torch.randn((3, 9), generator=random, dtype=torch.float64) * 1e-3,
            pressure=torch.randn((3, 9), generator=random, dtype=torch.float64) * 1e-3,
        ), config=StructuredNSProjectionConfig(spatial_mode_x=2, spatial_mode_y=2, time_order=2),
        geometry_spec=geometry, domain_bounds=geometry.domain_bounds, viscosity=.01, t0=0., t1=.12,
    )
    yield result
    torch.set_num_threads(previous)


def _bank(tmp_path, field, *, purpose="terminal", boundary_velocity_trace=False):
    source = StructuredNSManufacturedCaseSource(
        generator_id="test-frozen-flow", artifact_sha256="a" * 64,
        clean_sampler=lambda seed: field,
        perturbation_sampler=LegacyStructuredNSSmoothPerturber(amplitude=.01, spectral_max_mode=3),
        boundary_velocity_trace=boundary_velocity_trace,
    )
    base = source.sample(SampleRequest(1, CohortKind.SELECTION, seed=101, count=1))[0]
    guided = source.issue_answer_first_records(
        (field,), round_index=1, visibility=CaseVisibility.HIDDEN,
        case_id_prefix="guided", controller_seed=202,
        distribution_lineage={"reference_guidance_sha256": "b" * 64},
    )[0]
    perturbed = source.sample(SampleRequest(
        1, CohortKind.SELECTION, seed=303, count=1,
        kind=GenerationKind.PERTURBED, anchors=(base.case,),
    ))[0]
    records = (base, guided, perturbed)
    default = AnswerQuery.at_points(fields=("velocity_x", "velocity_y"),
                                   points=({"t": .01, "x": .2, "y": .3},))
    settings = {"default_query": default, "official_point_count": 4, "terminal_point_count": 4}
    evaluation = NSSolverTask().create_evaluation(settings)
    phase = EvaluationPhase.OFFICIAL if purpose == "promotion" else EvaluationPhase.TERMINAL
    plans = {record.case.case_id: evaluation.make_plan(record.case, phase, settings, 888 + index)
             for index, record in enumerate(records)}
    strata = {record.case.case_id: stratum for record, stratum in
              zip(records, ("base", "guided", "base_perturbed"))}
    path = write_ns_continuous_case_bank(
        tmp_path / f"{purpose}.json", bank_id=f"test-{purpose}", purpose=purpose,
        records=records, plans=plans, strata=strata,
        provenance={"reference_guidance_sha256": "b" * 64, "parent_policy": "base-only-test"},
        case_provenance={perturbed.case.case_id: {"perturbation_seed": 303}},
    )
    return path, records, plans, source


def test_roundtrip_preserves_actual_fields_fixed_truth_and_public_oracles(tmp_path, field):
    path, original, plans, source = _bank(tmp_path, field)
    bank = load_ns_continuous_case_bank(path, expected_purpose="terminal")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.rglob("public_arrays.npz"))
    assert bank.manifest["strata_counts"] == {"base": 1, "guided": 1, "base_perturbed": 1}
    for before, after in zip(original, bank.records):
        assert content_sha256(before.case) == content_sha256(after.case)
        plan = plans[before.case.case_id]
        assert content_sha256(bank.plans[after.case.case_id]) == content_sha256(plan)
        assert content_sha256(before.truth.answer(plan.query)) == content_sha256(after.truth.answer(plan.query))
        np.testing.assert_array_equal(before.truth._answer.raw_answer.coefficients.psi,
                                      after.truth._answer.raw_answer.coefficients.psi)
        original_session = source.public_session_factory(before.case)
        loaded_session = bank.generator.open_case_session(
            bank.generator_state, after.case, CaseAccessKind.PUBLIC_PROBLEM,
        )
        spatial, spacetime = np.array([[.2, .3], [.6, .7]]), np.array([[.03, .2, .3], [.1, .6, .7]])
        np.testing.assert_array_equal(original_session.query_initial_velocity(spatial),
                                      loaded_session.query_initial_velocity(spatial))
        np.testing.assert_array_equal(original_session.query_forcing(spacetime),
                                      loaded_session.query_forcing(spacetime))
        original_session.close(); loaded_session.close()
        with pytest.raises(ValueError, match="frozen evaluation query"):
            after.truth.answer(AnswerQuery.at_points(fields=plan.query.fields,
                points=({"t": .05, "x": .33, "y": .43},)))
    assert bank.records[2].case.parent_case_id == bank.records[0].case.case_id
    assert bank.records[2].truth._answer.raw_answer.config.spatial_mode_x == 3
    assert bank.generator.case_session_methods(CaseAccessKind.DEVELOPMENT_SOLUTION) == ()
    with pytest.raises(PermissionError, match="public problem"):
        bank.generator.open_case_session(bank.generator_state, bank.records[0].case,
                                         CaseAccessKind.DEVELOPMENT_SOLUTION)


def test_fixed_plan_wrapper_does_not_reseed_or_rechoose_points(tmp_path, field):
    path, _, _, _ = _bank(tmp_path, field, purpose="promotion")
    bank = load_ns_continuous_case_bank(path, expected_purpose="promotion")
    evaluation = bank.fixed_evaluation(NSSolverTask().create_evaluation({"default_query": next(iter(bank.plans.values())).query}))
    case = bank.records[0].case
    for seed in (0, 101, 999999):
        plan = evaluation.make_plan(case, EvaluationPhase.OFFICIAL, {}, seed)
        assert plan is bank.plans[case.case_id] and plan.seed == 888
    with pytest.raises(ValueError, match="purpose"):
        evaluation.make_plan(case, EvaluationPhase.TERMINAL, {}, 0)


@pytest.mark.parametrize("corruption", ["purpose", "manifest", "field"])
def test_bank_checks_purpose_but_ignores_historical_checksums(tmp_path, field, corruption):
    path, _, _, _ = _bank(tmp_path, field)
    if corruption == "purpose":
        with pytest.raises(ValueError, match="purpose"):
            load_ns_continuous_case_bank(path, expected_purpose="promotion")
        return
    payload = json.loads(path.read_text())
    if corruption == "manifest":
        payload["cases"][0]["reference"]["fields"]["velocity_x"][0] += 1
        path.write_text(json.dumps(payload))
    else:
        state = path.parent / payload["cases"][0]["state_path"]
        state.write_bytes(state.read_bytes() + b"changed")
    bank = load_ns_continuous_case_bank(path, expected_purpose="terminal")
    assert len(bank.records) == payload["case_count"]
    if corruption == "manifest":
        query = bank.plans[bank.records[0].case.case_id].query
        assert bank.records[0].truth.answer(query).fields["velocity_x"][0] == payload["cases"][0]["reference"]["fields"]["velocity_x"][0]


ORACLE_SOLVER = '''import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session
class Handle:
    def __init__(self, values): self.values = values
    def evaluate(self, query):
        return AnswerView(QueryKind.POINTS, {
            "velocity_x": tuple(float(self.values[0]) for p in query.points),
            "velocity_y": tuple(float(self.values[1]) for p in query.points),
        }, points=query.points, shape=(len(query.points),))
def prepare_solution(case, solve_budget=None):
    assert not hasattr(case, "truth")
    assert not any(key in repr(case.public_data) for key in ("hidden_continuous_case.pt", "state_sha256", "reference_sha256"))
    if "arrays" in case.public_data:
        return Handle([0., 0.])
    session = open_public_case_session(case)
    assert not hasattr(session, "query_completed_velocity")
    initial = session.query_initial_velocity([[.2, .3]])
    forcing = session.query_forcing([[.03, .2, .3]])
    assert np.isfinite(initial).all() and np.isfinite(forcing).all()
    return Handle(initial[0])
'''


def _solver(path):
    path.mkdir()
    (path / "submission.py").write_text(ORACLE_SOLVER)
    write_candidate_bundle_manifest(path, files=("submission.py",))
    return path


def test_common_evaluation_uses_frozen_continuous_plans_and_subprocess_oracles(tmp_path, field, monkeypatch):
    from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation
    from pde_agent_verifier_solver.engine import candidate_execution
    requests = []
    original_write = candidate_execution._write_private_json
    def observe_request(path, payload, **kwargs):
        if path.name == "request.json":
            requests.append(json.dumps(payload))
        return original_write(path, payload, **kwargs)
    monkeypatch.setattr(candidate_execution, "_write_private_json", observe_request)
    path, _, plans, _ = _bank(tmp_path, field)
    solver = _solver(tmp_path / "solver")
    spec = {"schema": "ns_common_evaluation_v1",
            "datasets": [{"dataset_id": "frozen-ns", "manifest": str(path), "cohort": "mixed"}],
            "programs": [{"method": method, "run_id": method, "seed": 101,
                          "artifact_id": method, "role": "final", "bundle": str(solver)}
                         for method in ("baseline", "canonical")],
            "query": {"point_count": 4},
            "budget": {"max_workers": 2, "case_hard_wall_seconds": 20}}
    manifest = tmp_path / "comparison.json"
    manifest.write_text(json.dumps(spec))
    result = run_ns_common_evaluation(manifest, tmp_path / "comparison")
    assert result["n_failed_program_cases"] == 0
    rows = [json.loads(line) for line in (tmp_path / "comparison/cases.jsonl").read_text().splitlines()]
    for case_id, plan in plans.items():
        selected = [row for row in rows if row["case_id"] == case_id]
        assert {row["plan_sha256"] for row in selected} == {content_sha256(plan)}
        assert len({row["reference_sha256"] for row in selected}) == 1
        for metric in ("velocity_relative_l2", "velocity_x_relative_l2", "velocity_y_relative_l2"):
            values = [row["error"] for row in selected if row["metric"] == metric]
            assert values[0] == values[1]
    assert len(requests) == 6
    for text in requests:
        assert "hidden_continuous_case.pt" not in text and "reference_sha256" not in text
    spec["query"]["plans_file"] = str(tmp_path / "comparison/plans.json")
    manifest.write_text(json.dumps(spec))
    replay = run_ns_common_evaluation(manifest, tmp_path / "replay")
    assert replay["n_failed_program_cases"] == 0
    assert json.loads((tmp_path / "replay/plans.json").read_text()) == json.loads(
        (tmp_path / "comparison/plans.json").read_text())
    spec["datasets"][0]["strata"] = ["perturbed"]
    # A subset cannot impersonate a previous complete-bank replay.
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="replayed plan cases"):
        run_ns_common_evaluation(manifest, tmp_path / "bad-subset")
    spec["query"].pop("plans_file")
    manifest.write_text(json.dumps(spec))
    subset = run_ns_common_evaluation(manifest, tmp_path / "perturbed-only")
    assert subset["n_cases"] == 1 and subset["n_failed_program_cases"] == 0
    subset_rows = [json.loads(line) for line in
                   (tmp_path / "perturbed-only/cases.jsonl").read_text().splitlines()]
    assert all(row["parent_case_id"] is not None for row in subset_rows)


@pytest.mark.parametrize("gpu_comparison", [False, True])
def test_baseline_cli_uses_generated_selection_bank_without_exposing_it_to_agent(tmp_path, field, monkeypatch, gpu_comparison):
    import os
    torch = pytest.importorskip("torch")
    if gpu_comparison and not torch.cuda.is_available():
        pytest.skip("requires scheduled CUDA allocation")
    from pde_agent_verifier_solver.plugins.ns.baselines.cli import run_from_config
    path, _, plans, _ = _bank(tmp_path, field, purpose="promotion")
    numerical = tmp_path / "numerical"
    numerical.mkdir()
    (numerical / "public_question.json").write_text(json.dumps({"time": {"t0": 0., "t1": 1.}}))
    (numerical / "case_spec.json").write_text(json.dumps({"parameters": {}}))
    axis = np.array([0., 1.])
    np.savez(numerical / "public_arrays.npz", x=axis, y=axis, fluid_mask=np.ones((2, 2), dtype=bool))
    np.savez(numerical / "hidden_truth_cache.npz", t=axis, u_true=np.ones((2, 2, 2, 2)),
             pressure=np.zeros((2, 2, 2)))
    data = tmp_path / "development.json"
    data.write_text(json.dumps({"schema": "ns_d0_v1", "splits": {"development": [
        {"case_id": "original-development", "case_dir": str(numerical)}]}}))
    worker = tmp_path / "local_worker.py"
    worker.write_text('''from pathlib import Path
from pde_agent_verifier_solver.engine.agent_execution import load_agent_worker_request, submit_agent_nomination, write_agent_worker_response
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
request = load_agent_worker_request()
assert not any(key in repr(request) for key in ("hidden_continuous_case", "reference_sha256", "selection_manifest"))
cases = ControllerAgentCapabilityClient.from_request(request).current_cases()
assert len(cases) == 1 and "arrays" in cases[0].public_data
Path("submission.py").write_text(SOURCE)
submit_agent_nomination({"bundle_path": "submission.py"}, sequence=1)
write_agent_worker_response()
'''.replace("SOURCE", repr(ORACLE_SOLVER)))
    initial = _solver(tmp_path / "initial")
    config = {"schema": "ns_fixed_data_baseline_v1", "method": "bon_repair", "seed": 101,
              "dataset": {"manifest": str(data), "development_split": "development",
                          "selection_manifest": str(path)},
              "initial_candidate": str(initial),
              "search": {"independent_candidates": 2, "repair_iterations": 0},
              "agent": {"command": ["{python}", str(worker)], "hard_wall_seconds": 20},
              "evaluation": {"selection_point_count": 4, "candidate_case_hard_wall_seconds": 20}}
    config_path = tmp_path / "baseline.json"
    config_path.write_text(json.dumps(config))
    result = run_from_config(config_path, tmp_path / "baseline")
    assert result.report["completed_agent_turns"] == 2
    assert np.isfinite(result.selection_error)
    plan = json.loads((tmp_path / "baseline/baseline_plan.json").read_text())
    assert plan["development_case_ids"] == ["original-development"]
    assert plan["selection_case_ids"] == list(plans)
    effective = json.loads((tmp_path / "baseline/baseline_input.json").read_text())
    assert effective["selection_reference_kind"] == "frozen_continuous_ns"

    if gpu_comparison:
        from pde_agent_verifier_solver.plugins.ns import evaluation_devices
        from pde_agent_verifier_solver.sdk.evaluation_devices import current_evaluation_device_pool
        original_answer = evaluation_devices._NSDeviceAnswers.answer
        routed = []
        def observe(self, case_id, answer):
            result = original_answer(self, case_id, answer)
            routed.append(str(result.raw_answer.coefficients.psi.device))
            assert current_evaluation_device_pool() is not None
            return result
        monkeypatch.setattr(evaluation_devices._NSDeviceAnswers, "answer", observe)
        # Preserve the allocation's real token order; never expand visibility.
        token = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
        config["agent"]["gpu_devices"] = [token]
        config["evaluation"]["borrow_agent_gpus"] = True
        config_path.write_text(json.dumps(config))
        manifest_before = path.read_bytes()
        gpu = run_from_config(config_path, tmp_path / "gpu-baseline")
        assert gpu.selected.program.tree_sha256 == result.selected.program.tree_sha256
        assert gpu.selection_error == pytest.approx(result.selection_error, rel=2e-10, abs=1e-9)
        cpu_rows = json.loads((tmp_path / "baseline/selection_results.json").read_text())
        gpu_rows = json.loads((tmp_path / "gpu-baseline/selection_results.json").read_text())
        assert len(cpu_rows) == len(gpu_rows)
        for cpu_row, gpu_row in zip(cpu_rows, gpu_rows):
            assert cpu_row["tree_sha256"] == gpu_row["tree_sha256"]
            np.testing.assert_allclose(cpu_row["case_errors"], gpu_row["case_errors"], rtol=2e-10, atol=1e-9)
        assert routed and set(routed) == {"cuda:0"}
        assert current_evaluation_device_pool() is None and path.read_bytes() == manifest_before
        events = [json.loads(line) for line in (tmp_path / "gpu-baseline/evaluation_device_events.jsonl").read_text().splitlines()]
        assert all(e["cleanup_confirmed"] for e in events if e["event_type"] == "evaluation_devices_returned")
        print("GPU Fixed-bank CLI parity: same solver, B/G/Base-P Cases, frozen queries, scores, and selected hash; routed cuda:0 and released.")


def test_fixed_measured_development_and_selection_keep_boundary_sessions(tmp_path, field):
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    from pde_agent_verifier_solver.plugins.ns.experimental_mms import prepare_experimental_mms_answers
    from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import save_prepared_mms_dataset
    from pde_agent_verifier_solver.plugins.ns.baselines.cli import run_from_config
    from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind
    geometry=ContinuousGeometrySpec(domain_bounds=field.domain_bounds, obstacle_polygons=(), outer_boundary="manufactured_trace")
    import torch
    coefficients=type(field.coefficients)(psi=field.coefficients.psi, pressure=torch.zeros_like(field.coefficients.pressure))
    opened=type(field)(coefficients, config=field.config, geometry_spec=geometry,
        domain_bounds=field.domain_bounds, viscosity=field.viscosity, t0=field.t0, t1=field.t1)
    bank_path,_,_,_= _bank(tmp_path,opened,purpose="promotion",boundary_velocity_trace=True)
    prepared=prepare_experimental_mms_answers(DatasetInput("measured",DatasetKind.EXPERIMENTAL),
        {"development":[opened]},training_summary={"input_mode":"measured"})
    data=save_prepared_mms_dataset(prepared,tmp_path/"development.json")
    bank=load_ns_continuous_case_bank(bank_path,expected_purpose="promotion")
    assert "query_boundary_velocity" in bank.generator.case_session_methods(CaseAccessKind.PUBLIC_PROBLEM)
    record=bank.records[0]
    session=bank.generator.open_case_session(bank.generator_state,record.case,CaseAccessKind.PUBLIC_PROBLEM)
    points=np.array([[.03,0.,.3],[.08,1.,.6]])
    np.testing.assert_allclose(session.query_boundary_velocity(points),record.truth._answer.evaluate_velocity(points))
    with pytest.raises(ValueError):session.query_boundary_velocity([[.03,.5,.5]])
    session.close()
    solver=ORACLE_SOLVER.replace('    return Handle(initial[0])',
        '    boundary = session.query_boundary_velocity([[.03, 0., .3]])\n    assert np.isfinite(boundary).all()\n    return Handle(initial[0])')
    initial=tmp_path/"initial";initial.mkdir();(initial/"submission.py").write_text(solver)
    write_candidate_bundle_manifest(initial,files=("submission.py",))
    worker=tmp_path/"worker.py"
    worker.write_text('from pathlib import Path\n'
        'from pde_agent_verifier_solver.engine.agent_execution import load_agent_worker_request, submit_agent_nomination, write_agent_worker_response\n'
        'from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient\n'
        'request=load_agent_worker_request(); client=ControllerAgentCapabilityClient.from_request(request)\n'
        'cases=client.current_cases(); assert len(cases)==1 and "arrays" not in cases[0].public_data\n'
        'assert "observations" not in repr(cases[0].public_data) and "coefficients" not in repr(cases[0].public_data)\n'
        'Path("submission.py").write_text('+repr(solver)+')\n'
        'client.evaluate({"bundle_path":"submission.py"},case_ids=[cases[0].case_id])\n'
        'submit_agent_nomination({"bundle_path":"submission.py"},sequence=1);write_agent_worker_response()\n')
    config={"schema":"ns_fixed_data_baseline_v1","method":"bon_repair","seed":101,
        "dataset":{"kind":"experimental_mms","manifest":str(data),"development_split":"development","selection_manifest":str(bank_path)},
        "initial_candidate":str(initial),"search":{"independent_candidates":1,"repair_iterations":0},
        "agent":{"command":["{python}",str(worker)],"hard_wall_seconds":40},
        "evaluation":{"selection_point_count":4,"candidate_case_hard_wall_seconds":30}}
    path=tmp_path/"config.json";path.write_text(json.dumps(config))
    result=run_from_config(path,tmp_path/"run")
    assert result.report["completed_agent_turns"]==1
    assert np.isfinite(result.selection_error)
    assert result.report["failed_turns"] == 0, result.report


def test_measured_preflight_calibration_keeps_development_private_view_and_public_rpc(
    tmp_path, field, monkeypatch,
):
    """S0 calibration accepts development inputs without changing their role."""
    import importlib.util
    from types import SimpleNamespace
    import torch
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    from pde_agent_verifier_solver.engine.candidate_execution import (
        PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
    )
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
    from pde_agent_verifier_solver.plugins.ns.case_bank import NSBankSessionProvider
    from pde_agent_verifier_solver.plugins.ns.experimental_mms import prepare_experimental_mms_answers
    from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind, TaskKind
    from pde_agent_verifier_solver.plugins.ns.tasks.experimental_mms import NSExperimentalMMSSolverTask

    spec = importlib.util.spec_from_file_location('measured_preflight', ROOT/'scripts/run_ns_measured_job.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    geometry = ContinuousGeometrySpec(domain_bounds=field.domain_bounds, obstacle_polygons=(),
                                      outer_boundary='manufactured_trace')
    coefficients = type(field.coefficients)(psi=field.coefficients.psi,
                                             pressure=torch.zeros_like(field.coefficients.pressure))
    opened = type(field)(coefficients, config=field.config, geometry_spec=geometry,
                         domain_bounds=field.domain_bounds, viscosity=field.viscosity,
                         t0=field.t0, t1=field.t1)
    prepared = prepare_experimental_mms_answers(
        DatasetInput('measured-calibration', DatasetKind.EXPERIMENTAL),
        {'development': [opened, opened]}, training_summary={'input_mode': 'measured'},
    )
    original = prepared.development_cases
    original_hashes = tuple(content_sha256(row.case) for row in original)
    rows = driver._private_calibration_records(original)
    assert len(rows) == len(original) == 2
    for before, after in zip(original, rows):
        assert before.case.visibility is CaseVisibility.DEVELOPMENT
        assert after.case.visibility is CaseVisibility.HIDDEN
        assert after.case.case_id == before.case.case_id and after.truth is before.truth
        assert replace(after.case, visibility=CaseVisibility.DEVELOPMENT) == before.case
    with pytest.raises(ValueError, match='original development'):
        driver._private_calibration_records(rows)
    provider = NSBankSessionProvider((SimpleNamespace(
        records=rows, manifest_sha256='c'*64,
        _answers={row.case.case_id: row.truth._answer for row in rows},
    ),))
    assert provider.case_session_methods(CaseAccessKind.DEVELOPMENT_SOLUTION) == ()
    calls = []
    open_session = provider.open_case_session
    def observe_session(state, case, access):
        calls.append((case.case_id, case.visibility, access))
        return open_session(state, case, access)
    monkeypatch.setattr(provider, 'open_case_session', observe_session)
    solver = tmp_path/'calibration_solver'
    solver.mkdir()
    (solver/'submission.py').write_text(ORACLE_SOLVER.replace(
        '    return Handle(initial[0])',
        '    boundary = session.query_boundary_velocity([[.03, 0., .3]])\n'
        '    assert np.isfinite(boundary).all()\n    return Handle(initial[0])',
    ))
    write_candidate_bundle_manifest(solver, files=('submission.py',))
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        solver, storage_root=tmp_path/'frozen', artifact_id='s0', name='S0',
        summary='Preflight regression', usage='prepare_solution',
    )
    query = AnswerQuery.at_points(fields=('velocity_x', 'velocity_y'),
                                  points=({'t': .01, 'x': .2, 'y': .3},))
    settings = {'default_query': query, 'official_point_count': 128}
    evaluation = NSSolverTask().create_evaluation(settings)
    # Production uses the standard NS evaluator only for calibration and the
    # measured task in the Candidate worker; its split checks remain intact.
    with pytest.raises(ValueError, match='Original must use verifier'):
        NSExperimentalMMSSolverTask().create_evaluation(settings).make_plan(
            rows[0].case, EvaluationPhase.OFFICIAL, settings, seed=12345,
        )
    execution = SubprocessCandidateExecutionService(
        tmp_path/'execution', task=PythonTaskWorkerSpec.from_task_type(NSExperimentalMMSSolverTask),
        evaluation_settings=settings, max_workers=1, case_hard_wall_seconds=30,
    )
    verifier = HiddenVerifier(evaluation, query, settings, seed=12345,
        generator=provider, generator_state=provider.state, candidate_execution=execution)
    with pytest.raises(ValueError, match='hidden Cases'):
        verifier.evaluate(artifact, original)
    result = verifier.evaluate(artifact, rows)
    assert tuple(row.case_id for row in result.cases) == tuple(row.case.case_id for row in original)
    assert np.isfinite(result.primary_error)
    assert all(np.isfinite(row.primary_error) for row in result.cases)
    for index, record in enumerate(rows):
        plan = verifier._plans[record.case.case_id]
        assert plan.seed == 12345 + index and len(plan.query.points) == 128
    assert calls == [(row.case.case_id, CaseVisibility.HIDDEN, CaseAccessKind.PUBLIC_PROBLEM)
                     for row in rows]
    assert tuple(content_sha256(row.case) for row in prepared.development_cases) == original_hashes
    assert not prepared.hidden_cases
    assert len(execution.audits) == len(rows)
    assert all(audit.status == 'completed' and not audit.controller_truth_shared
               and audit.process_group_cleaned for audit in execution.audits)


def test_selected_bank_loads_only_requested_fields_without_changing_inputs(tmp_path, field, monkeypatch):
    path, original, plans, source = _bank(tmp_path, field)
    import agent_verifier.neural_generator.structured_ns_answer_first as backend
    load = backend.load_structured_continuous_case
    calls = []
    def traced(*args, **kwargs):
        calls.append(args[0])
        return load(*args, **kwargs)
    monkeypatch.setattr(backend, 'load_structured_continuous_case', traced)
    key = original[2].case.case_id
    bank = load_ns_continuous_case_bank(path, expected_purpose='terminal', case_ids=[key])
    assert len(calls) == 1
    assert [row.case.case_id for row in bank.records] == [key]
    assert bank.manifest['case_count'] == 3
    assert bank.manifest['strata_counts'] == {'base': 1, 'guided': 1, 'base_perturbed': 1}
    assert content_sha256(bank.records[0].truth.answer(plans[key].query)) == content_sha256(original[2].truth.answer(plans[key].query))
    before = source.public_session_factory(original[2].case)
    after = bank.generator.open_case_session(bank.generator_state, bank.records[0].case, CaseAccessKind.PUBLIC_PROBLEM)
    try:
        np.testing.assert_array_equal(before.query_forcing([[.03, .2, .3]]), after.query_forcing([[.03, .2, .3]]))
    finally:
        before.close(); after.close()
    for ids in ([], [key, key], ['absent']):
        with pytest.raises(ValueError, match='unique subset'):
            load_ns_continuous_case_bank(path, expected_purpose='terminal', case_ids=ids)


def test_continuous_shards_preserve_reference_oracles_and_scores(tmp_path, field):
    import importlib.util
    from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation
    script = importlib.util.spec_from_file_location('continuous_shards_test', ROOT / 'scripts/ns_evaluation_shards.py')
    shards = importlib.util.module_from_spec(script)
    script.loader.exec_module(shards)
    bank, _, _, _ = _bank(tmp_path, field)
    solver = _solver(tmp_path / 'solver')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'ns_common_evaluation_v1',
        'datasets': [{'dataset_id': 'ns', 'cohort': 'mixed', 'manifest': str(bank)}],
        'programs': [{'method': 'full', 'artifact_id': 'full-r5', 'run_id': 'r', 'seed': 101,
                      'role': 'final', 'bundle': str(solver)}],
        'query': {'point_count': 4}, 'budget': {'max_workers': 1, 'case_hard_wall_seconds': 30}}))
    baseline = run_ns_common_evaluation(manifest, tmp_path / 'baseline')
    output = tmp_path / 'split'
    shards.prepare(manifest, output, shard_size=1, lanes=1, workers=2)
    from threading import Event
    from types import SimpleNamespace
    budget = SimpleNamespace(cancel=Event(), should_pause=lambda *a, **k: False)
    shards.run_lane(output / 'plan.json', 0, budget=budget)
    result = shards.merge(output / 'plan.json')
    assert result['groups'] == baseline['groups']
    assert result['n_cases'] == 3 and result['n_failed_program_cases'] == 0
    before = json.loads((tmp_path / 'baseline/plans.json').read_text())
    after = json.loads((output / 'aggregate/plans.json').read_text())
    assert before == after


def test_overlapping_continuous_cohorts_share_one_case_session_bank(tmp_path, field):
    from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation
    bank, records, _, _ = _bank(tmp_path, field)
    ids = [row.case.case_id for row in records]
    solver = _solver(tmp_path / 'solver')
    spec = {'schema': 'ns_common_evaluation_v1',
            'datasets': [{'dataset_id': 'ns', 'cohort': 'first', 'manifest': str(bank), 'case_ids': ids[:2]},
                         {'dataset_id': 'ns', 'cohort': 'second', 'manifest': str(bank), 'case_ids': ids[1:]}],
            'programs': [{'method': 'full', 'artifact_id': 'full', 'run_id': 'r', 'seed': 101,
                          'role': 'final', 'bundle': str(solver)}],
            'query': {'point_count': 4}, 'budget': {'max_workers': 2, 'case_hard_wall_seconds': 30}}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(spec))
    result = run_ns_common_evaluation(manifest, tmp_path / 'result')
    assert result['n_cases'] == 4 and result['n_failed_program_cases'] == 0
    rows = [json.loads(line) for line in (tmp_path / 'result/cases.jsonl').read_text().splitlines()]
    shared = [row for row in rows if row['case_id'] == ids[1] and row['metric'] == 'velocity_relative_l2']
    assert len(shared) == 2 and shared[0]['error'] == shared[1]['error']
