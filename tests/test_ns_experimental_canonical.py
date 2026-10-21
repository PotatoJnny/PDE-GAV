from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent_verifier_solver.engine import (
    AgentSpec,
    CampaignCompiler,
    CampaignRecipe,
    DevelopmentSamplingPolicy,
    EvaluatorPolicy,
    PythonCandidateCodec,
    PythonTaskWorkerSpec,
    RoundSettings,
    SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.agent_tool import _clone_incumbent
from pde_agent_verifier_solver.plugins.ns.datasets import NSPointCloudTruth
from pde_agent_verifier_solver.plugins.ns.experimental_piv import (
    assemble_completed_piv,
)
from pde_agent_verifier_solver.plugins.ns.field_generator import (
    LatentFlowConfig,
    build_latent_flow_model,
)
from pde_agent_verifier_solver.plugins.ns.problem import NSProblem
from pde_agent_verifier_solver.plugins.ns.reconstruction.real_piv import (
    GappyPODPosterior,
    SparsePIVLayout,
    SparsePIVObservations,
    fit_trajectory_pod,
    load_trajectory_pod,
    save_trajectory_pod,
)
from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    CaseVisibility,
    DatasetInput,
    DatasetKind,
    EvaluationPhase,
    EvaluationPlan,
    GenerationKind,
    GenerationRequest,
    GeneratorMode,
    HiddenSuiteSpec,
    OfficialEvaluationSpec,
    SampleSource,
    SamplingStrategy,
    TaskKind,
)


def _write_completed_dataset(root: Path) -> Path:
    rng = np.random.default_rng(20260904)
    x = np.asarray([-0.4, 0.1, 0.8, 1.3], dtype=np.float64)
    y = np.asarray([0.2, -0.3, 0.4, -0.1], dtype=np.float64)
    time = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    sensors = np.asarray([0, 3], dtype=np.int64)
    frames = np.asarray([0, 2], dtype=np.int64)
    split_sizes = {
        "generator_training": 2,
        "generator_validation": 2,
        "development": 1,
        "verifier": 2,
        "final_test": 1,
    }
    splits: dict[str, list[dict[str, object]]] = {}
    complete_windows = []
    serial = 0
    for split, count in split_sizes.items():
        rows = []
        for index in range(count):
            velocity = rng.normal(
                loc=0.2 * serial,
                scale=0.5,
                size=(len(time), 2, len(x)),
            ).astype(np.float32)
            complete_windows.append(velocity)
            public_values = velocity[np.ix_(frames, (0, 1), sensors)]
            case_id = f"{split}-{index}"
            case_file = root / f"{case_id}.npz"
            np.savez_compressed(
                case_file,
                velocity=velocity,
                x=x,
                y=y,
                time=time,
                public_values=public_values,
                sensor_indices=sensors,
                observed_frames=frames,
            )
            measurement_file = root / f"{case_id}-measurement.npz"
            np.savez_compressed(
                measurement_file,
                velocity=velocity + 10.0,
                x=x,
                y=y,
                time=time,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "case_file": case_file.name,
                    "measurement_reference_file": measurement_file.name,
                    "source_frames": [serial * len(time), (serial + 1) * len(time)],
                }
            )
            serial += 1
        splits[split] = rows

    prior, _ = fit_trajectory_pod(np.stack(complete_windows), max_rank=3)
    representation = root / "trajectory_pod.npz"
    save_trajectory_pod(representation, prior)
    manifest = root / "completed_case_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "pde_completed_case_dataset_v1",
                "problem": "ns",
                "source_kind": "experimental",
                "reconstruction_method": "gappy_pod",
                "fields": ["u", "v"],
                "geometry": {
                    "label": "laboratory PIV plane",
                    "topology": "point_cloud",
                    "spatial_dimension": 2,
                    "coordinate_names": ["x", "y"],
                    "point_count": len(x),
                },
                "observation_layout": {
                    "noise_standard_deviation": [0.02, 0.03],
                },
                "artifacts": {"representation": representation.name},
                "selected_method": {
                    "rank": 2,
                    "prior_precision": 10.0,
                    "noise_scale": 1.0,
                },
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _prepared(manifest: Path):
    return NSProblem().prepare_dataset(
        DatasetInput(
            dataset_id="experimental-piv",
            kind=DatasetKind.EXPERIMENTAL,
            location=str(manifest),
        )
    )


def test_completed_piv_adapter_preserves_terminal_before_official_and_truth_roles(
    tmp_path: Path,
) -> None:
    prepared = _prepared(_write_completed_dataset(tmp_path))

    assert [record.case.case_id for record in prepared.hidden_cases] == [
        "final_test-0",
        "verifier-0",
        "verifier-1",
    ]
    assert prepared.training_summary["hidden_case_order"] == (
        "terminal_final_test",
        "official",
    )
    assert prepared.training_summary["terminal_final_test_cases"] == 1
    assert prepared.training_summary["official_hidden_cases"] == 2
    assert len(prepared.generator_training_cases) == 2
    assert len(prepared.generator_validation_cases) == 2

    record = prepared.development_cases[0]
    assert isinstance(record.truth, NSPointCloudTruth)
    assert isinstance(record.generator_target, NSPointCloudTruth)
    point = record.case.public_data["coordinates"]
    query = AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"),
        points=(
            {
                "t": point["time"][0],
                "x": point["x"][0],
                "y": point["y"][0],
            },
        ),
    )
    measurement = record.truth.answer(query)
    completed = record.generator_target.answer(query)
    assert np.isclose(
        measurement.fields["velocity_x"][0]
        - completed.fields["velocity_x"][0],
        10.0,
    )
    assert "case_file" not in record.case.public_data
    assert "measurement_reference_file" not in record.case.public_data


def test_compact_validated_checkpoint_loads_through_canonical_generator(
    tmp_path: Path,
) -> None:
    import torch

    manifest = _write_completed_dataset(tmp_path)
    config = LatentFlowConfig(feature_dim=2, width=8, depth=1, time_features=4)
    model = build_latent_flow_model(config)
    checkpoint_path = tmp_path / "generator.pt"
    torch.save(
        {
            "schema": "ns_dataset_latent_flow_v1",
            "model_config": {
                "feature_dim": config.feature_dim,
                "width": config.width,
                "depth": config.depth,
                "time_features": config.time_features,
            },
            "model_state": model.state_dict(),
            "training_history": [
                {
                    "step": 5,
                    "train_loss": 1.0,
                    "validation_loss": 1.1,
                    "selected": 1.0,
                }
            ],
        },
        checkpoint_path,
    )
    (tmp_path / "train_report.json").write_text(
        json.dumps({"training_cases": 2}),
        encoding="utf-8",
    )
    (tmp_path / "flow_selection.json").write_text(
        json.dumps(
            {
                "criterion": "fixed_held_out_generated_distribution_rank",
                "selected_step": 5,
            }
        ),
        encoding="utf-8",
    )

    assembly = assemble_completed_piv(
        manifest_path=manifest,
        generator_checkpoint_path=checkpoint_path,
        generator_settings={"controller_seed": 771},
    )
    recipe = assembly.generator_recipe()
    state = assembly.generator.load(recipe.checkpoint, recipe.settings)
    records = assembly.generator.sample(
        state,
        GenerationRequest(
            round_index=1,
            count=1,
            mode=GeneratorMode.OFFLINE,
            kind=GenerationKind.BASE,
            visibility=CaseVisibility.DEVELOPMENT,
            seed=991,
            settings={"device": "cpu", "integration_steps": 2},
        ),
    )
    guided_records = assembly.generator.sample(
        state,
        GenerationRequest(
            round_index=2,
            count=1,
            mode=GeneratorMode.ONLINE,
            kind=GenerationKind.GUIDED,
            visibility=CaseVisibility.DEVELOPMENT,
            seed=992,
            settings={"device": "cpu", "integration_steps": 2},
        ),
    )

    assert assembly.terminal_final_test_cases == 1
    assert assembly.official_hidden_cases == 2
    assert [
        record.case.case_id for record in assembly.prepared_dataset.hidden_cases
    ] == ["final_test-0", "verifier-0", "verifier-1"]
    assert recipe.checkpoint.location == str(checkpoint_path)
    assert "generator_checkpoint" not in recipe.settings
    assert state.base.validation_report["accepted"] is True
    assert state.base.validation_report["checkpoint_schema"] == (
        "ns_dataset_latent_flow_v1"
    )
    assert len(records) == 1
    prepared_records = (
        assembly.prepared_dataset.development_cases
        + assembly.prepared_dataset.hidden_cases
        + assembly.prepared_dataset.generator_training_cases
        + assembly.prepared_dataset.generator_validation_cases
    )
    assert all(
        row.case.public_data["project_reconstruction"]
        == assembly.project_reconstruction
        for row in prepared_records
    )
    assert records[0].case.public_data["geometry"]["topology"] == "point_cloud"
    assert records[0].case.public_data["generator"]["kind"] == "base"
    assert (
        records[0].case.public_data["project_reconstruction"]
        == assembly.project_reconstruction
    )
    assert guided_records[0].case.public_data["generator"]["kind"] == "guided"
    assert (
        guided_records[0].case.public_data["project_reconstruction"]
        == assembly.project_reconstruction
    )


def test_piv_generator_capabilities_compile_for_fresh_agent_adaptive_sampling(
    tmp_path: Path,
) -> None:
    import torch

    manifest = _write_completed_dataset(tmp_path)
    config = LatentFlowConfig(feature_dim=2, width=8, depth=1, time_features=4)
    checkpoint_path = tmp_path / "generator.pt"
    torch.save(
        {
            "schema": "ns_dataset_latent_flow_v1",
            "model_config": {
                "feature_dim": config.feature_dim,
                "width": config.width,
                "depth": config.depth,
                "time_features": config.time_features,
            },
            "model_state": build_latent_flow_model(config).state_dict(),
            "training_history": [{"step": 1, "selected": 1.0}],
        },
        checkpoint_path,
    )
    (tmp_path / "train_report.json").write_text(
        json.dumps({"training_cases": 2}),
        encoding="utf-8",
    )
    assembly = assemble_completed_piv(
        manifest_path=manifest,
        generator_checkpoint_path=checkpoint_path,
        generator_settings={"controller_seed": 19},
    )
    policy = DevelopmentSamplingPolicy(
        strategy=SamplingStrategy.AGENT_ADAPTIVE,
        max_cases_per_agent_per_round=8,
        allowed_sources=(SampleSource.BASE,),
    )
    recipe = CampaignRecipe(
        name="experimental PIV canonical compile",
        round_count=1,
        agents=(AgentSpec("agent", "Agent"),),
        default_round=RoundSettings(
            development_sampling=policy,
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(original=1),
                terminal=HiddenSuiteSpec(original=1),
            ),
        ),
        generator=assembly.generator_recipe(
            guidance_enabled=False,
            perturbations_enabled=False,
        ),
        evaluator=EvaluatorPolicy(
            AnswerQuery.at_points(
                fields=("velocity_x", "velocity_y"),
                points=({"t": 0.0, "x": 0.0, "y": 0.0},),
            )
        ),
        promotion={
            "preset": "minimize_metric",
            "settings": {
                "primary": {"distribution": "original", "statistic": "mean"}
            },
        },
        terminal_score_distribution="original",
    )

    compiled = CampaignCompiler().compile(
        recipe,
        assembly.generator.capabilities,
    )

    assert compiled.rounds[0].development_sampling.strategy is (
        SamplingStrategy.AGENT_ADAPTIVE
    )
    assert assembly.generator.capabilities.online is True
    assert assembly.generator.capabilities.trainable is False
    assert compiled.rounds[0].development_sampling.cumulative_fixed_cases is False


class _TruthHandle:
    def __init__(self, truth: NSPointCloudTruth) -> None:
        self.truth = truth

    def evaluate(self, query: AnswerQuery):
        return self.truth.answer(query)


class _PreparedPointCloudCandidate:
    def __init__(self, truth: NSPointCloudTruth) -> None:
        self.truth = truth
        self.prepared_without_query = False

    def prepare_solution(self, case, solve_budget=None):
        del case, solve_budget
        self.prepared_without_query = True
        return _TruthHandle(self.truth)


def test_point_cloud_official_plan_is_private_native_and_two_phase(
    tmp_path: Path,
) -> None:
    prepared = _prepared(_write_completed_dataset(tmp_path))
    record = prepared.hidden_cases[1]
    settings = {
        "default_query": AnswerQuery.grid(
            fields=("velocity_x", "velocity_y"),
            t=3,
            point=4,
        ),
        "development_point_count": 5,
        "official_point_count": 11,
        "terminal_point_count": 13,
    }
    task = NSSolverTask()
    evaluation = task.create_evaluation(settings)
    first = evaluation.make_plan(
        record.case,
        EvaluationPhase.OFFICIAL,
        settings,
        seed=71,
    )
    repeated = evaluation.make_plan(
        record.case,
        EvaluationPhase.OFFICIAL,
        settings,
        seed=71,
    )
    fresh = evaluation.make_plan(
        record.case,
        EvaluationPhase.OFFICIAL,
        settings,
        seed=72,
    )
    coordinates = record.case.public_data["coordinates"]
    native_space = set(zip(coordinates["x"], coordinates["y"], strict=True))

    assert first.query == repeated.query
    assert first.query != fresh.query
    assert len(first.query.points) == 11
    assert {
        (point["x"], point["y"]) for point in first.query.points
    }.issubset(native_space)
    assert {point["t"] for point in first.query.points}.issubset(
        set(coordinates["time"])
    )

    candidate = _PreparedPointCloudCandidate(record.truth)
    artifact = Artifact(
        artifact_id="piv-two-phase",
        kind=TaskKind.SOLVER,
        name="PIV two-phase Candidate",
        program=candidate,
        summary="test",
        usage="prepare_solution(case) then handle.evaluate(points)",
        round_index=1,
        agent_id="agent",
    )
    truth = record.truth.answer(first.query)
    result = task.evaluate(artifact, record.case, truth, first.query)

    assert candidate.prepared_without_query is True
    assert result.primary_error == 0.0
    assert result.metrics["candidate_formal_two_phase"] == 1.0


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REAL_PIV_MANIFEST = (
    _REPOSITORY_ROOT
    / "project_runs/ns_real_piv_gappy_product_v2/completed_case_manifest.json"
)
_REAL_PIV_REPRESENTATION = (
    _REPOSITORY_ROOT
    / "project_runs/ns_real_piv_gappy_product_v2/artifacts/trajectory_pod.npz"
)
_REAL_PIV_CHECKPOINT = (
    _REPOSITORY_ROOT
    / "project_runs/ns_real_piv_generator_selection_probe_v2_20260826/artifacts"
    / "base_generator/generator.pt"
)
_REAL_PIV_S0_BUNDLE = (
    _REPOSITORY_ROOT
    / "configs/candidates/ns_piv_project_gappy_pod_initial_bundle"
)
_REAL_PIV_INPUTS = (
    _REAL_PIV_MANIFEST,
    _REAL_PIV_REPRESENTATION,
    _REAL_PIV_CHECKPOINT,
    _REAL_PIV_S0_BUNDLE / "submission.py",
)


@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_PIV_INPUTS),
    reason="local completed PIV research artifacts are not installed",
)
def test_real_completed_piv_case_runs_the_frozen_two_phase_gappy_pod_s0(
    tmp_path: Path,
) -> None:
    """Exercise the real collaborator manifest, basis, checkpoint, and Case."""

    assembly = assemble_completed_piv(
        manifest_path=_REAL_PIV_MANIFEST,
        generator_checkpoint_path=_REAL_PIV_CHECKPOINT,
        dataset_settings={
            "generator_training_cases": 2,
            "generator_validation_cases": 2,
            "development_cases": 1,
            "hidden_cases": 1,
            "final_test_cases": 1,
        },
        generator_settings={"device": "cpu", "controller_seed": 1904},
    )
    assert assembly.representation_path.samefile(_REAL_PIV_REPRESENTATION)
    assert dict(assembly.reconstruction_settings) == {
        "rank": 64,
        "prior_precision": 10.0,
    }

    initial = assembly.bind_initial_candidate(
        bundle_path=_REAL_PIV_S0_BUNDLE,
        storage_root=tmp_path / "frozen-candidates",
    )
    record = assembly.prepared_dataset.development_cases[0]
    coordinates = record.case.public_data["coordinates"]
    query = AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"),
        points=(
            {
                "t": coordinates["time"][0],
                "x": coordinates["x"][0],
                "y": coordinates["y"][0],
            },
        ),
    )
    truth = record.truth.answer(query)
    evaluation = assembly.task.evaluate(initial, record.case, truth, query)

    evaluation_program = assembly.task.create_evaluation({"default_query": query})
    plan = EvaluationPlan(
        phase=EvaluationPhase.DEVELOPMENT,
        query=query,
        seed=2026,
    )
    candidate_execution = SubprocessCandidateExecutionService(
        tmp_path / "candidate-execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings={"default_query": query},
        max_workers=1,
        case_hard_wall_seconds=60,
    )
    (subprocess_output,) = candidate_execution.execute_cases(
        initial,
        (record.case,),
        (plan,),
        evaluation_program,
    )
    subprocess_evaluation = evaluation_program.score_candidate(
        initial,
        record.case,
        truth,
        plan,
        subprocess_output,
    )

    assert initial.round_index == 0
    assert initial.metadata["candidate_bundle_bytes"] < 64 * 1024
    assert initial.metadata["project_reconstruction_sha256"] == (
        assembly.project_reconstruction["sha256"]
    )
    assert (
        record.case.public_data["project_reconstruction"]
        == assembly.project_reconstruction
    )
    assert evaluation.metrics["candidate_formal_two_phase"] == 1.0
    assert np.isfinite(evaluation.primary_error)
    assert subprocess_evaluation.primary_error == pytest.approx(
        evaluation.primary_error
    )
    assert len(candidate_execution.audits) == 1
    assert candidate_execution.audits[0].status == "completed"

    public = record.case.public_data
    observations = public["observations"]
    coordinates = public["coordinates"]
    layout = SparsePIVLayout(
        sensor_indices=np.asarray(observations["sensor_indices"], dtype=np.int64),
        observed_frames=np.asarray(observations["observed_frames"], dtype=np.int64),
        window_length=len(coordinates["time"]),
        point_count=len(coordinates["x"]),
    )
    expected = GappyPODPosterior(
        rank=64,
        prior_precision=10.0,
    ).reconstruct_observations(
        load_trajectory_pod(_REAL_PIV_REPRESENTATION),
        SparsePIVObservations(
            values=np.asarray(observations["values"], dtype=np.float32),
            layout=layout,
            noise_standard_deviation=np.asarray(
                observations["noise_standard_deviation"],
                dtype=np.float64,
            ),
        ),
        include_uncertainty=False,
    )
    assert evaluation.prediction is not None
    assert np.isclose(
        evaluation.prediction.fields["velocity_x"][0],
        expected.velocity[0, 0, 0],
        rtol=2.0e-5,
        atol=1.0e-7,
    )


@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_PIV_INPUTS),
    reason="local completed PIV research artifacts are not installed",
)
def test_lightweight_piv_incumbent_clones_do_not_amplify_project_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regress the 279 author clones that formerly consumed 41.23 GiB."""

    workspace = tmp_path / "agent-turn"
    workspace.mkdir()
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
    initial = codec.bind_existing(
        _REAL_PIV_S0_BUNDLE,
        storage_root=tmp_path / "controller",
        artifact_id="piv-s0",
        name="PIV S0",
        summary="lightweight algorithm bundle",
        usage="read the Project reconstruction descriptor",
    )
    encoded = codec.encode_incumbent(initial, workspace)
    request = {
        "incumbent": {
            "program_available": True,
            "program": encoded,
        }
    }
    monkeypatch.setenv("PDE_AGENT_WORKSPACE", str(workspace))
    clone_count = 279
    for index in range(clone_count):
        _clone_incumbent(request, f"author-{index:03d}")

    files = tuple(path for path in workspace.rglob("*") if path.is_file())
    assert not any(
        path.suffix.lower() in {".npz", ".npy", ".h5", ".hdf5"}
        for path in files
    )
    staged_bytes = sum(path.stat().st_size for path in files)
    project_bytes = _REAL_PIV_REPRESENTATION.stat().st_size
    assert clone_count * project_bytes > 40 * 1024**3
    assert staged_bytes < project_bytes
