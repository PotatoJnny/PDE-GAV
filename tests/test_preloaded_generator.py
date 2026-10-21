from __future__ import annotations

from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle, RunStatus
from pde_agent_verifier_solver.engine import (
    AgentSpec,
    CampaignRecipe,
    DevelopmentSamplingPolicy,
    EvaluatorPolicy,
    GeneratorRecipe,
    RoundSettings,
)
from pde_agent_verifier_solver.plugins.toy import (
    ToyBaseGeneratorState,
    ToyPoissonGenerator,
    ToyPoissonProblem,
    ToyResearchAgent,
    ToySolverTask,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    DatasetInput,
    DatasetKind,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    HiddenSuiteSpec,
    OfficialEvaluationSpec,
    SampleSource,
)


class _FrozenToyGenerator(ToyPoissonGenerator):
    capabilities = GeneratorCapabilities(
        offline=True,
        online=True,
        guidance=False,
        perturbations=False,
        trainable=False,
    )

    def __init__(self) -> None:
        self.loaded: list[GeneratorCheckpoint] = []

    def train(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("a selected checkpoint must never be retrained")

    def load(self, checkpoint, settings, progress=None):
        self.loaded.append(checkpoint)
        return super().load(checkpoint, settings, progress)


def test_canonical_controller_loads_a_selected_checkpoint_without_training() -> None:
    generator = _FrozenToyGenerator()
    checkpoint = GeneratorCheckpoint(
        generator_id=generator.info.program_id,
        payload=ToyBaseGeneratorState(
            max_frequency=5,
            mean_amplitude=0.95,
            basis_size=8,
        ),
        location="test://already-frozen",
    )
    programs = ProgramBundle(
        problem=ToyPoissonProblem(),
        task=ToySolverTask(),
        generator=generator,
        agents={"agent": ToyResearchAgent()},
    )
    dataset = DatasetInput(
        dataset_id="preloaded-generator-dataset",
        kind=DatasetKind.NUMERICAL,
        problem={"equation": "1D Poisson"},
        geometry={"x": [0.0, 1.0]},
        settings={"development_cases": 8, "hidden_cases": 4},
    )
    application = ProductApplication.local_development()
    project = application.create_project("Preloaded Generator", dataset, programs)
    recipe = CampaignRecipe(
        name="preloaded canonical campaign",
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
            checkpoint=checkpoint,
        ),
        evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=17)),
    )

    launched = application.launch_run(project.project_id, recipe)
    session = application.wait(launched.run_id, timeout=5)

    assert session.status is RunStatus.COMPLETED, session.error
    assert generator.loaded == [checkpoint]
    assert session.result is not None
    assert any(
        event.event_type == "generator_checkpoint_selected"
        for event in session.events.snapshot()
    )
