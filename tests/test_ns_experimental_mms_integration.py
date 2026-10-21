"""Experimental reconstruction reaches the normal Campaign and isolated solver worker."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle, RunStatus
from pde_agent_verifier_solver.engine import PythonCandidateCodec
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy, EvaluatorPolicy,
    GeneratorRecipe, RoundOverride, RoundSettings,
)
from pde_agent_verifier_solver.engine.candidate_case import candidate_facing_case
from pde_agent_verifier_solver.engine.candidate_execution import (
    PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.panel.controller import PanelController
from pde_agent_verifier_solver.plugins import get_plugin
from pde_agent_verifier_solver.plugins.ns.product import build_ns_plugin
from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import ContinuousPIVReconstruction
from pde_agent_verifier_solver.plugins.ns.tasks.experimental_mms import NSExperimentalMMSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, CaseVisibility, DatasetInput, DatasetKind, EvaluationPhase,
    EvaluationPlan, GenerationKind, HiddenSuiteSpec, OfficialEvaluationSpec,
    ReconstructionRecipe, SamplingStrategy, TaskKind, TerminalCohortSpec,
    TerminalEvaluationSpec, TerminalMetricSpec, write_candidate_bundle_manifest,
)


_QUERY = AnswerQuery.at_points(
    fields=("velocity_x", "velocity_y"),
    points=({"t": 0.12, "x": 0.2, "y": 0.1}, {"t": 0.25, "x": 0.7, "y": 0.3}),
)
_RECONSTRUCTION = {"rank": 3, "prior_precision": 1e-5}
from tests.test_ns_experimental_mms_backend import _assembly, _settings
_FLOW = _settings()


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_work():
    torch = pytest.importorskip("torch")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


# One actual, deliberately simple solver: prepare a public initial-velocity grid,
# retain it in time, and interpolate by nearest neighbor. Oracle probes exercise
# all three scientific inputs without giving the Candidate a private answer.
_SOLVER_SOURCE = '''
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session

_FORBIDDEN = {"observations", "coefficients", "public_values", "sensor_indices",
              "project_reconstruction", "trajectory_pod", "basis", "pod_modes"}

def check_inputs(value):
    if isinstance(value, dict):
        assert not (_FORBIDDEN & set(value))
        for item in value.values():
            check_inputs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_inputs(item)

class Handle:
    def __init__(self, points, velocity):
        self.points, self.velocity = points, velocity
    def evaluate(self, query):
        xy = np.array([[point["x"], point["y"]] for point in query.points])
        indices = np.argmin(np.sum((xy[:, None, :] - self.points[None, :, :])**2, axis=2), axis=1)
        values = self.velocity[indices]
        return AnswerView(kind=QueryKind.POINTS,
            fields={"velocity_x": tuple(values[:, 0]), "velocity_y": tuple(values[:, 1])},
            points=query.points, shape=(len(query.points),))

def prepare_solution(case, solve_budget=None):
    check_inputs(case.public_data)
    assert "observations" not in case.public_data
    assert case.public_data["boundary_condition"]["access"] == "query_boundary_velocity"
    assert case.public_data["forcing"]["access"] == "query_forcing"
    domain = case.public_data["domain"]
    x0, x1 = domain["bounds"]["x"]
    y0, y1 = domain["bounds"]["y"]
    t0, t1 = domain["time_interval"]
    xx, yy = np.meshgrid(np.linspace(x0, x1, 4), np.linspace(y0, y1, 4), indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    session = open_public_case_session(case)
    velocity = np.asarray(session.query_initial_velocity(points))
    trace = np.asarray(session.query_boundary_velocity([[t0, x0, y0], [t1, x1, y1]]))
    forcing = np.asarray(session.query_forcing([[(t0+t1)/2, (x0+x1)/2, (y0+y1)/2]]))
    assert np.isfinite(trace).all() and np.isfinite(forcing).all()
    np.testing.assert_allclose(trace[0], velocity[0])
    return Handle(points, velocity)
'''


def _solver():
    namespace = {}
    exec(_SOLVER_SOURCE, namespace)
    return SimpleNamespace(prepare_solution=namespace["prepare_solution"])


class PublicSolverAgent:
    def __init__(self):
        self.rounds = []

    def run(self, context):
        artifact = Artifact(
            artifact_id=f"public-initial-r{context.round_plan.round_index}", kind=TaskKind.SOLVER,
            name="Public initial-field solver", program=_solver(),
            summary="Hold the public initial field on a solver-owned grid.",
            usage="prepare_solution(case), then handle.evaluate(points)",
            round_index=context.round_plan.round_index, agent_id=context.agent.agent_id,
        )
        response = context.evaluator.evaluate(
            artifact, EvaluationRequest(tuple(case.case_id for case in context.cases)),
        )
        assert np.isfinite(response.primary_error)
        assert tuple(row.case_id for row in response.cases) == tuple(case.case_id for case in context.cases)
        assert all(row.truth is None and row.prediction is None for row in response.cases)
        self.rounds.append((context.round_plan.round_index, len(context.cases)))
        return AgentOutcome(artifact, observations=("Used initial, forcing and boundary public queries.",))


def _piv_fixture(tmp_path):
    x, y = np.meshgrid(np.linspace(-.7, 1.2, 5), np.linspace(-.4, .8, 5), indexing="ij")
    x, y, times = x.ravel(), y.ravel(), np.linspace(0., .3, 4)
    points = np.column_stack((np.repeat(times, len(x)), np.tile(x, len(times)), np.tile(y, len(times))))
    sensors, frames = np.array([0, 4, 12, 20, 24]), np.array([0, 3])
    counts = {"generator_training": 8, "generator_validation": 2, "development": 2,
              "verifier": 2, "final_test": 1, "reconstruction_validation": 1}
    manifest = {"schema": "pde_completed_case_dataset_v1", "problem": "ns",
                "geometry": {"label": "analytic measured window", "obstacle_polygons": []},
                "observation_layout": {"noise_standard_deviation": [.002, .003]}, "splits": {}}
    serial = 0
    for role, count in counts.items():
        rows = []
        for index in range(count):
            rng = np.random.default_rng(serial)
            # An independent divergence-free polynomial fixture, unrelated to
            # either the old or current continuous field implementation.
            a, b, c, d = rng.normal(size=4) * .05
            factor = 1 + a + b * times[:, None]
            velocity = np.stack((factor * (.8 + c*x + 2*d*y),
                                 -factor * (c*y + 2*a*x)), axis=1)
            public = velocity[np.ix_(frames, (0, 1), sensors)]
            path = tmp_path / f"{role}-{index}.npz"
            stored = velocity if role == "generator_training" else np.full_like(velocity, 9e6)
            np.savez(path, velocity=stored, x=x, y=y, time=times, public_values=public,
                     sensor_indices=sensors, observed_frames=frames)
            row = {"case_id": path.stem, "case_file": path.name,
                   "generator_target_kind": "measured_full_field" if role == "generator_training" else "reconstruction_mean",
                   "source_recording": "analytic-piv", "source_frames": [10*serial, 10*serial+4]}
            if role == "reconstruction_validation":
                reference = tmp_path / "private-validation.npz"
                np.savez(reference, velocity=velocity, x=x, y=y, time=times)
                row["measurement_reference_file"] = reference.name
            rows.append(row)
            serial += 1
        manifest["splits"][role] = rows
    path = tmp_path / "piv.json"
    path.write_text(json.dumps(manifest))
    return DatasetInput("integration-piv", DatasetKind.EXPERIMENTAL, str(path),
                        settings={"viscosity": .02, "coordinate_units": "nondimensional",
                                  "mms_geometry": {"obstacle_polygons": [], "outer_boundary": "manufactured_trace"},
                                  "projection_config": {"spatial_mode_x": 2, "spatial_mode_y": 2,
                                                        "time_order": 2, "spatial_extension": .25}})


def _project(tmp_path, app, agent):
    plugin = build_ns_plugin(task_id="experimental_mms_solver")
    programs = ProgramBundle.from_plugin(plugin, {"local": agent})
    project = app.create_project(
        "Experimental observations to NS solver", _piv_fixture(tmp_path), programs,
        reconstruction=ReconstructionRecipe(ContinuousPIVReconstruction(), method_settings=_RECONSTRUCTION),
    )
    assembly = _assembly(tmp_path / "generator", project.dataset)
    plugin = replace(plugin, generator=assembly.generator)
    programs = ProgramBundle.from_plugin(plugin, {"local": agent})
    project = app.create_prepared_project(project.name, project.dataset, programs,
                                         reconstruction_report=project.reconstruction_report)
    return plugin, programs, project, assembly


def _policy(*, guided):
    return DevelopmentSamplingPolicy(
        strategy=SamplingStrategy.FIXED_RECIPE, max_cases_per_agent_per_round=8,
        fixed_recipe=CaseCounts(original=1, base=2, guided=int(guided),
                                base_perturbed=1, guided_perturbed=int(guided)),
    )


def _recipe(checkpoint):
    return CampaignRecipe(
        name="Two-round experimental MMS integration", round_count=2,
        agents=(AgentSpec("local", "Local public-input solver"),),
        default_round=RoundSettings(
            agent_budget_seconds=30, development_sampling=_policy(guided=True),
            official_evaluation=OfficialEvaluationSpec(HiddenSuiteSpec(
                original=1, base=2, guided=1, base_perturbed=1, guided_perturbed=1)),
        ),
        round_overrides={1: RoundOverride(
            development_sampling=_policy(guided=False),
            official_evaluation=OfficialEvaluationSpec(HiddenSuiteSpec(original=1, base=2, base_perturbed=1)),
        )},
        generator=GeneratorRecipe(settings=_FLOW, checkpoint=checkpoint),
        evaluator=EvaluatorPolicy(_QUERY, development_point_count=4),
        task_settings={"official_point_count": 4, "terminal_point_count": 4},
        terminal_evaluation=TerminalEvaluationSpec(
            cohorts=(TerminalCohortSpec("test", HiddenSuiteSpec(original=1, base=2, base_perturbed=1)),),
            ranking=TerminalMetricSpec("test", "base"),
        ),
        master_seed=71,
    )


def test_new_ns_task_reconstructs_then_runs_two_rounds_and_reaches_panel(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = PublicSolverAgent()
        app = ProductApplication.local_development()
        plugin, programs, project, assembly = _project(tmp_path, app, agent)
        assert "experimental_mms_solver" not in get_plugin("ns").tasks
        assert plugin.default_task == "experimental_mms_solver"
        assert plugin.problem._mms_backend is not plugin.generator.backend
        assert isinstance(programs.task, NSExperimentalMMSSolverTask)
        assert project.reconstruction_report is not None
        panel = PanelController(app, programs, prepared_dataset=project.dataset, default_query=_QUERY)
        setup = panel.setup_page()
        assert setup["task"]["id"] == "ns-experimental-mms-solver"
        assert setup["backend"]["online"] and setup["backend"]["guidance"]
        assert any(field["name"] == "viscosity" for field in setup["forms"]["problem"])
        sampled = []
        generate = plugin.generator.backend.generate
        def record_generation(base, guidance, request, progress=None):
            records = generate(base, guidance, request, progress)
            sampled.append((request.kind, request.visibility, records))
            return records
        monkeypatch.setattr(plugin.generator.backend, "generate", record_generation)
        run = app.launch_run(project.project_id, _recipe(assembly.checkpoint))
        completed = app.wait(run.run_id, timeout=30)
        assert completed.status is RunStatus.COMPLETED, completed.error
        assert agent.rounds == [(1, 4), (2, 6)]
        assert len(completed.result.rounds) == len(completed.result.terminal_round_reports) == 2
        assert {kind for kind, _, _ in sampled} == {GenerationKind.BASE, GenerationKind.GUIDED, GenerationKind.PERTURBED}
        assert {visibility for _, visibility, _ in sampled} == {CaseVisibility.DEVELOPMENT, CaseVisibility.HIDDEN}
        for _, _, records in sampled:
            for record in records:
                public = record.case.public_data
                assert "observations" not in public and "project_reconstruction" not in public
                assert "coefficients" not in json.dumps(public)
                assert public["boundary_condition"]["access"] == "query_boundary_velocity"
        page = panel.run_page(run.run_id)
        assert page["status"] == "completed"
        assert len(page["rounds"]) == 2 and len(page["terminal_round_reports"]) == 2
        assert np.isfinite(page["result"]["primary_error"])
        assert page["result"]["artifact_id"]
        assert page["reconstruction"]["method_id"] == "ns-gappy-pod-completed-velocity"
        for distributions in completed.result.rounds[1].hidden_distribution_metrics.values():
            assert set(distributions) >= {"original", "base", "guided", "base_perturbed", "guided_perturbed"}
    finally:
        torch.set_num_threads(previous_threads)


def test_new_ns_task_executes_frozen_subprocess_solver_with_three_public_oracles(tmp_path):
    pytest.importorskip("torch")
    plugin, programs, project, assembly = _project(tmp_path, ProductApplication.local_development(), PublicSolverAgent())
    record = project.dataset.hidden_cases[0]
    case = candidate_facing_case(record.case, id_seed=39)
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "submission.py").write_text(_SOLVER_SOURCE)
    write_candidate_bundle_manifest(source, files=("submission.py",))
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        source, storage_root=tmp_path / "frozen", artifact_id="public-only",
        name="Public initial-field solver", summary="Public-only MMS solver", usage="prepare_solution",
    )
    evaluation = programs.task.create_evaluation({"default_query": _QUERY})
    plan = evaluation.make_plan(record.case, EvaluationPhase.TERMINAL,
                                {"terminal_point_count": 4, "default_query": _QUERY}, seed=9)
    public_session = plugin.generator.backend.public_session_factory(record.case)
    calls = []
    def invoke(case_id, method, args, kwargs):
        assert case_id == case.case_id
        calls.append(method)
        return getattr(public_session, method)(*args, **kwargs)
    service = SubprocessCandidateExecutionService(
        tmp_path / "worker", task=PythonTaskWorkerSpec.from_task_type(NSExperimentalMMSSolverTask),
        evaluation_settings={"default_query": _QUERY}, max_workers=1, case_hard_wall_seconds=30,
    )
    outputs = service.execute_cases(
        artifact, (case,), (plan,), evaluation,
        public_session_methods=plugin.generator.backend.public_session_methods,
        invoke_public_session=invoke, close_public_session=lambda _: public_session.close(),
    )
    assert set(calls) == {"query_initial_velocity", "query_boundary_velocity", "query_forcing"}
    assert len(outputs) == 1
    scored = evaluation.score_candidate(artifact, case, record.truth.answer(plan.query), plan, outputs[0])
    assert np.isfinite(scored.primary_error)
    assert outputs[0].metrics["candidate_formal_two_phase"] == 1.0
    assert service.audits[0].status == "completed"


def test_standard_ns_registry_cannot_select_a_task_without_its_scientific_components():
    ordinary = get_plugin("ns")
    assert ordinary.default_task == "solver"
    with pytest.raises(ValueError, match="unknown task"):
        ordinary.task("experimental_mms_solver")
    with pytest.raises(ValueError, match="unknown task"):
        ProgramBundle.from_plugin(ordinary, {"agent": PublicSolverAgent()},
                                  task_id="experimental_mms_solver")
    experimental = build_ns_plugin(task_id="experimental_mms_solver")
    programs = ProgramBundle.from_plugin(experimental, {"agent": PublicSolverAgent()})
    assert isinstance(programs.task, NSExperimentalMMSSolverTask)
    assert programs.generator.backend is None
    assert "query_boundary_velocity" in {
        method.name for method in programs.public_problem_bundle.oracle_methods
    }


def test_experimental_solver_form_exposes_only_implemented_score_settings():
    task = NSExperimentalMMSSolverTask()
    fields = {field.name for field in task.setup_fields()}
    assert {"official_point_count", "terminal_point_count"} <= fields
    assert not {"primary_metric", "include_pressure"} & fields
    evaluation = task.create_evaluation({"default_query": _QUERY,
                                         "primary_metric": "velocity_relative_l2", "include_pressure": False})
    assert evaluation.program_id == "ns-full-field-solver-evaluation-v1"


@pytest.mark.parametrize("settings,message", [
    ({"include_pressure": True}, "velocity only"),
    ({"include_pressure": 0}, "must be false"),
    ({"primary_metric": "pressure_relative_l2"}, "fixed to velocity_relative_l2"),
    ({"primary_metric": "velocity_rmse"}, "fixed to velocity_relative_l2"),
    ({"default_query": AnswerQuery.at_points(fields=("pressure",), points=({"t": 0., "x": 0., "y": 0.},))},
     "both velocity components only"),
])
def test_experimental_solver_rejects_unimplemented_score_options(settings, message):
    with pytest.raises(ValueError, match=message):
        NSExperimentalMMSSolverTask().create_evaluation(settings)


@pytest.mark.parametrize("split,phase", [
    ("final_test", EvaluationPhase.OFFICIAL),
    ("verifier", EvaluationPhase.TERMINAL),
    ("generator_training", EvaluationPhase.OFFICIAL),
    ("generator_validation", EvaluationPhase.TERMINAL),
])
def test_original_split_mismatch_stops_before_solver_or_truth(tmp_path, split, phase):
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError
    from pde_agent_verifier_solver.sdk import CaseRecord

    _, programs, project, _ = _project(tmp_path, ProductApplication.local_development(), PublicSolverAgent())
    records = (*project.dataset.hidden_cases, *project.dataset.generator_training_cases,
               *project.dataset.generator_validation_cases)
    record = next(r for r in records if r.case.public_data["distribution_lineage"]["split"] == split)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid Original split reached hidden truth or Candidate execution")
    record = CaseRecord(record.case, SimpleNamespace(answer=forbidden))
    artifact = Artifact(artifact_id="unused", kind=TaskKind.SOLVER, name="Must not execute",
                        program=SimpleNamespace(prepare_solution=forbidden), summary="",
                        usage="prepare_solution", round_index=1, agent_id="local")
    verifier = HiddenVerifier(programs.task, _QUERY, phase=phase)
    with pytest.raises(CandidateExecutionInfrastructureError, match="prepare the hidden") as failure:
        verifier.evaluate(artifact, (record,))
    assert failure.value._hidden_evaluation_stage == "prepare_hidden_inputs"
    assert isinstance(failure.value.__cause__, ValueError)
    assert "reserve the entire final_test split" in str(failure.value.__cause__)


def test_prepared_panel_browser_payload_launches_frozen_generator_to_result(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        app = ProductApplication.local_development()
        agent = PublicSolverAgent()
        plugin, programs, project, assembly = _project(tmp_path, app, agent)
        checkpoint = assembly.checkpoint
        def forbidden(*args, **kwargs):
            pytest.fail("prepared Panel with selected checkpoint retrained Generator")
        monkeypatch.setattr(plugin.generator.backend, "train_base", forbidden)
        panel = PanelController(app, programs, prepared_dataset=project.dataset,
                                generator_checkpoint=checkpoint, default_query=_QUERY)
        setup = panel.setup_page()
        defaults = setup["defaults"]
        # Match server.py's browser submission: the old bound fallback ['base']
        # excluded the Original starter Cases and rejected the launch.
        payload = {
            **defaults, "project_name": "Prepared experimental solver", "round_count": 1,
            "dataset_location": "", "problem_settings": {},
            "allowed_sources": defaults.get("allowed_sources", ["base"]),
            "sampling_strategy": "agent_adaptive", "starter_cases": {"original": 2, "base": 2},
            "round_hidden": {"original": 1, "base": 2, "guided": 0},
            "terminal_evaluation": {"cohorts": [{"id": "base_model", "guidance_source": "base",
                "counts": {"original": 1, "base": 1}}],
                "ranking": {"cohort_id": "base_model", "distribution": "original", "statistic": "mean"}},
            "generator_settings": _FLOW, "task_settings": {"official_point_count": 4, "terminal_point_count": 4},
            "development_point_count": 4, "agent_budget_seconds": 30,
        }
        assert set(payload["allowed_sources"]) == {"original", "base", "guided"}
        launched = panel.create_and_launch(payload)
        completed = app.wait(launched["run_id"], timeout=30)
        assert completed.status is RunStatus.COMPLETED, completed.error
        assert agent.rounds == [(1, 4)]
        result = panel.run_page(launched["run_id"])
        assert result["status"] == "completed" and result["result"]["artifact_id"]
        assert np.isfinite(result["result"]["primary_error"])
        assert any(event.event_type == "generator_checkpoint_selected" for event in completed.events.snapshot())
    finally:
        torch.set_num_threads(old_threads)
