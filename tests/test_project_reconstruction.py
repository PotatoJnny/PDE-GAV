from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import Mapping

import numpy as np
import pytest

from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle, RunStatus
from pde_agent_verifier_solver.engine import (
    AgentSpec,
    CampaignRecipe,
    DevelopmentSamplingPolicy,
    EvaluatorPolicy,
    GeneratorRecipe,
    RoundSettings,
)
from pde_agent_verifier_solver.panel.controller import PanelController
from pde_agent_verifier_solver.reconstruction import (
    GappyPODLSProgram,
    PODCompletedData,
    PODFitData,
    PODObservation,
    pretrained_pod_basis,
)
from pde_agent_verifier_solver.plugins.toy import (
    SineTruth,
    ToyPoissonGenerator,
    ToyPoissonProblem,
    ToyResearchAgent,
    ToySolverTask,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    DatasetInput,
    DatasetKind,
    FormField,
    HiddenSuiteSpec,
    GeneratorCheckpoint,
    OfficialEvaluationSpec,
    PanelCard,
    PreparedDataset,
    PreparedReconstructionSource,
    ProgramInfo,
    ReconstructionEvaluation,
    ReconstructionInput,
    ReconstructionOutput,
    ReconstructionRecipe,
    ReconstructionReport,
    SampleSource,
)


def _dataset(kind: DatasetKind = DatasetKind.EXPERIMENTAL) -> DatasetInput:
    return DatasetInput(
        dataset_id=f"project-reconstruction-{kind.value}",
        kind=kind,
        location="memory://project-reconstruction",
        problem={"equation": "1D Poisson"},
        geometry={"x": [0.0, 1.0]},
        settings={"development_cases": 8, "hidden_cases": 4},
    )


def _programs(
    problem: ToyPoissonProblem,
    generator: ToyPoissonGenerator | None = None,
) -> ProgramBundle:
    return ProgramBundle(
        problem=problem,
        task=ToySolverTask(),
        generator=generator or ToyPoissonGenerator(),
        agents={"agent": ToyResearchAgent()},
    )


def _campaign() -> CampaignRecipe:
    return CampaignRecipe(
        name="project reconstruction reuse",
        round_count=1,
        agents=(AgentSpec("agent", "Agent"),),
        default_round=RoundSettings(
            agent_budget_seconds=10,
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=2,
                allowed_sources=(SampleSource.BASE,),
            ),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=1),
                terminal=HiddenSuiteSpec(base=1),
            ),
        ),
        generator=GeneratorRecipe(
            guidance_enabled=False,
            perturbations_enabled=False,
        ),
        evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=17)),
    )


class _RecordingReconstruction:
    info = ProgramInfo(
        "recording-reconstruction",
        "Recording reconstruction",
        "Test-only observation completion program.",
    )

    def __init__(self, events: list[str], *, wrong_output: bool = False) -> None:
        self.events = events
        self.wrong_output = wrong_output
        self.calls = 0
        self.received: list[ReconstructionInput] = []
        self.last_output: ReconstructionOutput | None = None

    def setup_fields(self) -> tuple[FormField, ...]:
        return ()

    def reconstruct(
        self,
        source: ReconstructionInput,
        settings: Mapping[str, object],
        seed: int,
        progress=None,
    ) -> ReconstructionOutput:
        assert progress is None
        assert seed == 23
        assert settings == {
            "rank": 1,
            "private_checkpoint_path": "/sealed/checkpoints/pod.npz",
        }
        self.events.append("program")
        self.calls += 1
        self.received.append(source)
        if self.wrong_output:
            return object()  # type: ignore[return-value]
        output = ReconstructionOutput(
            method_id="gappy_pod_ls",
            completed_data={"amplitude": 1.25},
            fields=("u",),
            case_counts={"generator_training": 1, "generator_validation": 1},
            selected_rank=1,
            regularization=None,
            diagnostics={
                "condition_number": 1.0,
                "split_validation_passed": source.metadata[
                    "split_validation_passed"
                ],
                "role_counts": source.metadata["role_counts"],
            },
        )
        self.last_output = output
        return output

    def result_card(self, report: ReconstructionReport) -> PanelCard:
        return PanelCard(
            title="Reconstruction",
            summary=report.method_id,
            values=dict(report.aggregate_metrics),
        )


class _TrainingRecordingGenerator(ToyPoissonGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.checkpoints: list[GeneratorCheckpoint] = []

    def train(self, dataset, settings, progress=None) -> GeneratorCheckpoint:
        checkpoint = super().train(dataset, settings, progress)
        self.checkpoints.append(checkpoint)
        return checkpoint


class _ReconstructionProblem(ToyPoissonProblem):
    def __init__(
        self,
        events: list[str],
        *,
        wrong_stage: str | None = None,
        split_overlap: str | None = None,
    ) -> None:
        self.events = events
        self.wrong_stage = wrong_stage
        self.split_overlap = split_overlap
        self.reconstruction_validation_reference = {
            "sealed_error": 0.125,
            "private_values": (9.0, 10.0),
        }
        self.program_input: ReconstructionInput | None = None
        self.evaluated_output: ReconstructionOutput | None = None
        self.materialized_output: ReconstructionOutput | None = None

    def prepare_reconstruction_source(
        self,
        dataset: DatasetInput,
        settings: Mapping[str, object],
        seed: int,
    ) -> PreparedReconstructionSource:
        self.events.append("source")
        assert settings == {
            "recording_split": "fixed",
            "private_source_path": "/sealed/recordings/piv.mat",
        }
        assert seed == 23
        if self.wrong_stage == "source":
            return object()  # type: ignore[return-value]
        calibration_frames = {"calibration-frame-1", "calibration-frame-2"}
        target_frames = {"target-frame-1"}
        downstream_frames = {
            *(f"toy-initial-dev-{index}" for index in range(1, 9)),
            *(f"toy-initial-hidden-{index}" for index in range(1, 5)),
            "generator-validation-reconstructed-0",
        }
        visible_coordinates = {"target-frame-1:x=0.25"}
        validation_coordinates = {"target-frame-1:x=0.75"}
        if self.split_overlap == "frame":
            target_frames.add("calibration-frame-1")
        if self.split_overlap == "coordinate":
            validation_coordinates = set(visible_coordinates)
        if calibration_frames & (target_frames | downstream_frames):
            raise ValueError(
                "calibration frames must be disjoint from reconstruction and "
                "downstream frames"
            )
        if target_frames & downstream_frames:
            raise ValueError(
                "reconstructed target frames must be disjoint from downstream frames"
            )
        if visible_coordinates & validation_coordinates:
            raise ValueError(
                "visible and reconstruction-validation coordinates must be disjoint"
            )
        self.program_input = ReconstructionInput(
            public_split_id="fixed-split-v1",
            fit_data={"snapshots": ((0.0, 1.0), (1.0, 0.0))},
            target_observations={"visible": (0.25,)},
            geometry=dataset.geometry,
            noise_model={"variance": 0.01},
            metadata={
                "source_dataset_id": dataset.dataset_id,
                "split_validation_passed": True,
                "role_counts": {
                    "calibration_frames": len(calibration_frames),
                    "target_frames": len(target_frames),
                    "reconstruction_validation_observations": len(
                        validation_coordinates
                    ),
                    "downstream_frames": len(downstream_frames),
                },
            },
        )
        return PreparedReconstructionSource(
            program_input=self.program_input,
            reconstruction_validation_reference=(
                self.reconstruction_validation_reference
            ),
        )

    def evaluate_reconstruction(
        self,
        output: ReconstructionOutput,
        reconstruction_validation_reference: object,
        settings: Mapping[str, object],
    ) -> ReconstructionEvaluation:
        self.events.append("evaluate")
        assert (
            reconstruction_validation_reference
            is self.reconstruction_validation_reference
        )
        assert settings == {
            "recording_split": "fixed",
            "private_source_path": "/sealed/recordings/piv.mat",
        }
        self.evaluated_output = output
        if self.wrong_stage == "evaluation":
            return object()  # type: ignore[return-value]
        reference = reconstruction_validation_reference
        assert isinstance(reference, Mapping)
        return ReconstructionEvaluation(
            aggregate_metrics={
                "reconstruction_validation_relative_l2": reference["sealed_error"]
            },
            per_target_metrics={"validation-target-1": {"relative_l2": 0.125}},
            counts={"reconstruction_validation_observations": 1},
        )

    def prepare_reconstructed_dataset(
        self,
        dataset: DatasetInput,
        output: ReconstructionOutput,
    ) -> PreparedDataset:
        self.events.append("prepare")
        assert self.evaluated_output is None
        self.materialized_output = output
        if self.wrong_stage == "dataset":
            return object()  # type: ignore[return-value]
        prepared = super().prepare_dataset(dataset)
        reconstructed_target = SineTruth(
            amplitude=float(output.completed_data["amplitude"]),
            frequency=1,
        )
        training_source = prepared.development_cases[0]
        training = replace(
            training_source,
            case=replace(
                training_source.case,
                case_id="generator-training-reconstructed-0",
            ),
            truth=SineTruth(
                amplitude=float(training_source.case.public_data["amplitude"]),
                frequency=int(training_source.case.public_data["frequency"]),
            ),
            generator_target=reconstructed_target,
        )
        validation_source = prepared.development_cases[1]
        validation = replace(
            validation_source,
            case=replace(
                validation_source.case,
                case_id="generator-validation-reconstructed-0",
            ),
            truth=SineTruth(
                amplitude=float(validation_source.case.public_data["amplitude"]),
                frequency=int(validation_source.case.public_data["frequency"]),
            ),
            generator_target=reconstructed_target,
        )
        return replace(
            prepared,
            generator_training_cases=(training,),
            generator_validation_cases=(validation,),
            training_summary={
                **dict(prepared.training_summary),
                "reconstruction_method": output.method_id,
            },
        )


class _BuiltinPODProblem(ToyPoissonProblem):
    """Small real adapter proving the built-in method crosses the Product seam."""

    def __init__(self) -> None:
        self.completed_data: PODCompletedData | None = None

    def prepare_reconstruction_source(
        self,
        dataset: DatasetInput,
        settings: Mapping[str, object],
        seed: int,
    ) -> PreparedReconstructionSource:
        assert settings == {}
        assert seed == 31
        basis = pretrained_pod_basis(
            mean=np.zeros(2),
            modes=np.array([[1.0], [0.0]]),
            modal_variances=np.array([1.0]),
        )
        return PreparedReconstructionSource(
            program_input=ReconstructionInput(
                public_split_id="synthetic-pod-split",
                fit_data=PODFitData(None, np.ones(2)),
                target_observations=(
                    PODObservation(
                        "private-target-frame",
                        np.array([0]),
                        np.array([1.25]),
                    ),
                ),
                geometry=dataset.geometry,
                pretrained_basis=basis,
                metadata={"fields": ["u"]},
            ),
            reconstruction_validation_reference=np.array([1.25, 0.0]),
        )

    def prepare_reconstructed_dataset(
        self,
        dataset: DatasetInput,
        output: ReconstructionOutput,
    ) -> PreparedDataset:
        assert isinstance(output.completed_data, PODCompletedData)
        self.completed_data = output.completed_data
        amplitude = float(output.completed_data.values[0, 0])
        prepared = super().prepare_dataset(dataset)
        target = SineTruth(amplitude=amplitude, frequency=1)

        def generator_record(index: int, role: str):
            source = prepared.development_cases[index]
            return replace(
                source,
                case=replace(source.case, case_id=f"{role}-pod-0"),
                generator_target=target,
            )

        return replace(
            prepared,
            generator_training_cases=(generator_record(0, "generator-training"),),
            generator_validation_cases=(
                generator_record(1, "generator-validation"),
            ),
        )

    def evaluate_reconstruction(
        self,
        output: ReconstructionOutput,
        reconstruction_validation_reference: object,
        settings: Mapping[str, object],
    ) -> ReconstructionEvaluation:
        assert isinstance(output.completed_data, PODCompletedData)
        expected = np.asarray(reconstruction_validation_reference, dtype=np.float64)
        error = float(np.linalg.norm(output.completed_data.values[0] - expected))
        return ReconstructionEvaluation(
            aggregate_metrics={"reconstruction_validation_l2": error},
            counts={"reconstruction_validation_observations": expected.size},
        )


def _reconstruction_recipe(program: _RecordingReconstruction) -> ReconstructionRecipe:
    return ReconstructionRecipe(
        program=program,
        data_settings={
            "recording_split": "fixed",
            "private_source_path": "/sealed/recordings/piv.mat",
        },
        method_settings={
            "rank": 1,
            "private_checkpoint_path": "/sealed/checkpoints/pod.npz",
        },
        seed=23,
    )


def test_reconstruction_recipe_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        ReconstructionRecipe(program=_RecordingReconstruction([]), seed=-1)


def test_reconstruction_output_detaches_mutable_field_list() -> None:
    fields = ["u"]
    output = ReconstructionOutput(
        method_id="test",
        completed_data={},
        fields=fields,  # type: ignore[arg-type]
        case_counts={},
    )

    fields.append("private")
    assert output.fields == ("u",)


def test_builtin_gappy_pod_runs_from_project_creation_through_campaign() -> None:
    problem = _BuiltinPODProblem()
    generator = _TrainingRecordingGenerator()
    application = ProductApplication.local_development()
    project = application.create_project(
        "Built-in POD project",
        _dataset(),
        _programs(problem, generator),
        reconstruction=ReconstructionRecipe(
            program=GappyPODLSProgram(),
            method_settings={"rank": 1},
            seed=31,
        ),
    )

    assert problem.completed_data is not None
    assert project.reconstruction_report is not None
    assert project.reconstruction_report.method_id == "gappy_pod_ls"
    assert project.reconstruction_report.aggregate_metrics == {
        "reconstruction_validation_l2": 0.0
    }
    session = application.wait(
        application.launch_run(project.project_id, _campaign()).run_id,
        timeout=5,
    )
    assert session.status is RunStatus.COMPLETED, session.error
    assert generator.checkpoints[-1].payload.mean_amplitude == pytest.approx(1.25)


@pytest.mark.parametrize("kind", [DatasetKind.NUMERICAL, DatasetKind.EXPERIMENTAL])
def test_no_reconstruction_recipe_preserves_the_existing_project_path(
    kind: DatasetKind,
) -> None:
    application = ProductApplication.local_development()
    project = application.create_project(
        "Unchanged project path",
        _dataset(kind),
        _programs(ToyPoissonProblem()),
    )

    assert project.reconstruction_report is None
    assert len(project.dataset.development_cases) == 8
    assert len(project.dataset.hidden_cases) == 4
    assert project.dataset.generator_training_cases == ()
    assert project.dataset.generator_validation_cases == ()


def test_reconstruction_is_frozen_once_and_reused_by_every_project_run() -> None:
    events: list[str] = []
    problem = _ReconstructionProblem(events)
    program = _RecordingReconstruction(events)
    generator = _TrainingRecordingGenerator()
    programs = _programs(problem, generator)
    application = ProductApplication.local_development()

    project = application.create_project(
        "Reconstructed experimental project",
        _dataset(),
        programs,
        reconstruction=_reconstruction_recipe(program),
    )

    assert events == ["source", "program", "prepare", "evaluate"]
    assert program.calls == 1
    assert program.received == [problem.program_input]
    assert problem.program_input is not None
    assert not hasattr(program.received[0], "reconstruction_validation_reference")
    assert all(
        value is not problem.reconstruction_validation_reference
        for value in vars(program.received[0]).values()
    )
    assert problem.evaluated_output is program.last_output
    assert problem.materialized_output is program.last_output

    report = project.reconstruction_report
    assert report is not None
    assert report.method_id == "gappy_pod_ls"
    assert report.aggregate_metrics == {
        "reconstruction_validation_relative_l2": 0.125
    }
    assert report.diagnostics["evaluation_counts"] == {
        "reconstruction_validation_observations": 1
    }
    assert report.resolved_settings == {
        "selected_rank": 1,
        "regularization": None,
    }
    with pytest.raises(FrozenInstanceError):
        report.method_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        report.aggregate_metrics[
            "reconstruction_validation_relative_l2"
        ] = 9.0  # type: ignore[index]
    with pytest.raises(TypeError):
        report.diagnostics["role_counts"]["target_frames"] = 9  # type: ignore[index]
    mutable_fields = ["u"]
    normalized_report = replace(report, fields=mutable_fields)
    mutable_fields.append("mutated")
    assert normalized_report.fields == ("u",)

    training = project.dataset.generator_training_cases[0]
    original = project.dataset.development_cases[0]
    assert training.case.case_id != original.case.case_id
    assert training.truth is not original.truth
    assert training.generator_target is not None
    assert training.generator_target is not training.truth
    query = AnswerQuery.grid(fields=("u",), x=5)
    assert training.truth.answer(query).fields != training.generator_target.answer(query).fields

    first = application.wait(
        application.launch_run(project.project_id, _campaign()).run_id,
        timeout=5,
    )
    second = application.wait(
        application.launch_run(project.project_id, _campaign()).run_id,
        timeout=5,
    )
    assert first.status is RunStatus.COMPLETED, first.error
    assert second.status is RunStatus.COMPLETED, second.error
    assert program.calls == 1
    assert len(generator.checkpoints) == 2
    assert all(
        checkpoint.payload.mean_amplitude == 1.25
        for checkpoint in generator.checkpoints
    )
    assert all(
        checkpoint.report["training_role"] == "generator_training_cases"
        for checkpoint in generator.checkpoints
    )
    assert application.get_project(project.project_id).reconstruction_report is report

    page = PanelController(application, programs).run_page(second.run_id)
    assert page["reconstruction"] == report.as_dict()
    page["reconstruction"]["diagnostics"]["role_counts"]["target_frames"] = 9
    assert report.diagnostics["role_counts"]["target_frames"] == 1
    serialized = json.dumps(page["reconstruction"], allow_nan=False)
    assert "private_values" not in serialized
    assert "/sealed/recordings/piv.mat" not in serialized
    assert "/sealed/checkpoints/pod.npz" not in serialized
    assert "not experimental truth" in serialized


def test_reconstruction_rejects_numerical_and_unsupported_projects_atomically() -> None:
    events: list[str] = []
    program = _RecordingReconstruction(events)
    recipe = _reconstruction_recipe(program)

    numerical = ProductApplication.local_development()
    with pytest.raises(ValueError, match="only be selected for an experimental"):
        numerical.create_project(
            "Invalid numerical reconstruction",
            _dataset(DatasetKind.NUMERICAL),
            _programs(_ReconstructionProblem(events)),
            reconstruction=recipe,
        )
    assert program.calls == 0
    assert numerical._projects == {}
    assert numerical._generator_run_locks == {}

    unsupported = ProductApplication.local_development()
    with pytest.raises(TypeError, match="does not support experimental reconstruction"):
        unsupported.create_project(
            "Unsupported reconstruction",
            _dataset(),
            _programs(ToyPoissonProblem()),
            reconstruction=recipe,
        )
    assert program.calls == 0
    assert unsupported._projects == {}
    assert unsupported._generator_run_locks == {}


@pytest.mark.parametrize("overlap", ["frame", "coordinate"])
def test_problem_adapter_rejects_reconstruction_split_leakage(overlap: str) -> None:
    events: list[str] = []
    program = _RecordingReconstruction(events)
    application = ProductApplication.local_development()

    with pytest.raises(ValueError, match="must be disjoint"):
        application.create_project(
            "Leaking reconstruction split",
            _dataset(),
            _programs(_ReconstructionProblem(events, split_overlap=overlap)),
            reconstruction=_reconstruction_recipe(program),
        )

    assert events == ["source"]
    assert program.calls == 0
    assert application._projects == {}


@pytest.mark.parametrize(
    ("wrong_stage", "wrong_output", "message"),
    [
        ("source", False, "must return PreparedReconstructionSource"),
        (None, True, "must return ReconstructionOutput"),
        ("evaluation", False, "must return ReconstructionEvaluation"),
        ("dataset", False, "must return PreparedDataset"),
    ],
)
def test_wrong_reconstruction_boundary_types_do_not_register_partial_projects(
    wrong_stage: str | None,
    wrong_output: bool,
    message: str,
) -> None:
    events: list[str] = []
    problem = _ReconstructionProblem(events, wrong_stage=wrong_stage)
    program = _RecordingReconstruction(events, wrong_output=wrong_output)
    application = ProductApplication.local_development()

    with pytest.raises(TypeError, match=message):
        application.create_project(
            "Boundary failure",
            _dataset(),
            _programs(problem),
            reconstruction=_reconstruction_recipe(program),
        )

    assert application._projects == {}
    assert application._generator_run_locks == {}
