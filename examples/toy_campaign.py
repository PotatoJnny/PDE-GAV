"""Run a three-round Poisson example on a CPU, without external data or an LLM.

The two built-in agents are deterministic demonstrations. Scientific experiments
use command agents and subprocess candidate execution instead.
"""

import json

from pde_agent_verifier_solver.app.service import ProductApplication, ProgramBundle
from pde_agent_verifier_solver.engine import (
    AgentSpec,
    CampaignRecipe,
    CaseCounts,
    DevelopmentSamplingPolicy,
    EvaluatorPolicy,
    GeneratorRecipe,
    RoundSettings,
    RoundOverride,
)
from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent, build_toy_plugin
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    DatasetInput,
    DatasetKind,
    HiddenSuiteSpec,
    OfficialEvaluationSpec,
    SampleSource,
    TerminalCohortSpec,
    TerminalEvaluationSpec,
    TerminalMetricSpec,
)


def main():
    agents = {
        "careful": ToyResearchAgent(step_size=1),
        "explorer": ToyResearchAgent(step_size=2),
    }
    programs = ProgramBundle.from_plugin(build_toy_plugin(), agents)
    app = ProductApplication.local_development()
    project = app.create_project(
        "Poisson example",
        DatasetInput(
            dataset_id="poisson-example",
            kind=DatasetKind.NUMERICAL,
            problem={"equation": "1D Poisson"},
            geometry={"x": [0.0, 1.0]},
            settings={"development_cases": 12, "hidden_cases": 4},
        ),
        programs,
    )
    recipe = CampaignRecipe(
        name="Poisson example",
        round_count=3,
        agents=tuple(AgentSpec(name, name.capitalize()) for name in agents),
        master_seed=0,
        default_round=RoundSettings(
            agent_budget_seconds=30,
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=12,
                allowed_sources=(SampleSource.ORIGINAL, SampleSource.BASE, SampleSource.GUIDED),
                starter=CaseCounts(original=2, base=2),
            ),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(original=1, base=3, guided=2),
            ),
        ),
        round_overrides={1: RoundOverride(
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(original=1, base=5),
            ),
        )},
        generator=GeneratorRecipe(perturbations_enabled=False),
        evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=65)),
        terminal_evaluation=TerminalEvaluationSpec(
            cohorts=(TerminalCohortSpec(
                cohort_id="base_model",
                suite=HiddenSuiteSpec(original=1, base=4),
            ),),
            ranking=TerminalMetricSpec(cohort_id="base_model", distribution="base"),
        ),
    )
    session = app.launch_run(project.project_id, recipe)
    session = app.wait(session.run_id, timeout=30)
    if session.status.value != "completed" or session.result is None:
        raise RuntimeError(session.error or f"Run did not complete: {session.status.value}")
    result = session.result
    summary = {
        "status": session.status.value,
        "rounds": [
            {
                "round": item.round_index,
                "selected_program": item.selected.artifact.artifact_id,
                "selection_error": item.selected.primary_error,
                "agent_case_counts": {
                    name: dict(counts) for name, counts in item.agent_case_counts.items()
                },
            }
            for item in result.rounds
        ],
        "final_program": result.selected.artifact.artifact_id,
        "terminal_error": result.selected.primary_error,
        "terminal_round_count": len(result.terminal_round_reports),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
