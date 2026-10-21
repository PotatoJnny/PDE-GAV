"""Prepared backends retain the product's explicit execution boundaries.

No scientific assets, Solver calls, subprocesses, or Campaign runs are needed
to check these composition-time guards.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pde_agent_verifier_solver.app import backend_run
from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
from pde_agent_verifier_solver.engine.agent_execution import InProcessAgentExecutionService
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    CampaignRecipe,
    EvaluatorPolicy,
    GeneratorRecipe,
    RoundSettings,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    InProcessCandidateExecutionService,
)
from pde_agent_verifier_solver.plugins.toy import build_toy_plugin
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    DatasetInput,
    DatasetKind,
    GeneratorCheckpoint,
    PreparedDataset,
)


@pytest.fixture
def prepared_binding():
    plugin = build_toy_plugin()
    checkpoint = GeneratorCheckpoint(plugin.generator.info.program_id, object())
    dataset = PreparedDataset(
        DatasetInput("execution-policy-fixture", DatasetKind.NUMERICAL), (), (),
    )
    return BackendRunBinding(plugin, checkpoint, dataset)


@pytest.fixture
def recipe():
    return CampaignRecipe(
        name="prepared execution-policy fixture",
        round_count=1,
        agents=(AgentSpec("fixture-agent", "Fixture Agent"),),
        default_round=RoundSettings(agent_budget_seconds=99),
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=False),
        evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=3)),
    )


@pytest.fixture
def composition(monkeypatch):
    from pde_agent_verifier_solver.app import service

    compiled = object()
    compiler = Mock(return_value=compiled)
    inventory = Mock()
    controller = Mock()
    monkeypatch.setattr(backend_run.CampaignCompiler, "compile", compiler)
    monkeypatch.setattr(service, "_validate_case_inventory", inventory)
    monkeypatch.setattr(backend_run, "CampaignController", controller)
    return SimpleNamespace(
        compiled=compiled, compiler=compiler, inventory=inventory, controller=controller,
    )


def _candidate_service(process="bounded_same_user_subprocess"):
    return SimpleNamespace(
        candidate_invocation_process=process,
        execute_cases=Mock(side_effect=AssertionError("Candidates must not execute")),
    )


@pytest.mark.parametrize("agent_execution", [None, InProcessAgentExecutionService()])
def test_formal_binding_rejects_missing_or_inprocess_agent(
    prepared_binding, recipe, composition, agent_execution,
):
    with pytest.raises(ValueError, match="Agent execution|in-process Agent"):
        prepared_binding.controller(
            recipe, {}, agent_execution=agent_execution,
            candidate_execution=_candidate_service(),
        )
    composition.compiler.assert_not_called()
    composition.controller.assert_not_called()


@pytest.mark.parametrize("candidate_execution", [None, InProcessCandidateExecutionService()])
def test_formal_binding_rejects_missing_or_inprocess_candidate(
    prepared_binding, recipe, composition, candidate_execution,
):
    with pytest.raises(ValueError, match="non-controller Candidate"):
        prepared_binding.controller(
            recipe, {}, agent_execution=object(), candidate_execution=candidate_execution,
        )
    composition.compiler.assert_not_called()
    composition.controller.assert_not_called()


def test_formal_binding_rejects_custom_controller_process_candidate(
    prepared_binding, recipe, composition,
):
    with pytest.raises(ValueError, match="Controller-process Candidate"):
        prepared_binding.controller(
            recipe, {}, agent_execution=object(),
            candidate_execution=_candidate_service("controller"),
        )
    composition.compiler.assert_not_called()


@pytest.mark.parametrize("candidate_execution", [
    SimpleNamespace(candidate_invocation_process="bounded_same_user_subprocess"),
    SimpleNamespace(execute_cases=lambda *args: ()),
    SimpleNamespace(execute_cases=lambda *args: (), candidate_invocation_process=""),
    SimpleNamespace(execute_cases=lambda *args: (), candidate_invocation_process=42),
])
def test_binding_rejects_malformed_candidate_service(
    prepared_binding, recipe, composition, candidate_execution,
):
    with pytest.raises(TypeError, match="candidate_execution must"):
        prepared_binding.controller(
            recipe, {}, agent_execution=object(), candidate_execution=candidate_execution,
        )
    composition.compiler.assert_not_called()


def test_formal_binding_forwards_explicit_services_and_preserves_recipe(
    prepared_binding, recipe, composition,
):
    agent_execution = object()
    candidate_execution = _candidate_service()
    agents = {"fixture-agent": object()}
    events = object()
    actual = prepared_binding.controller(
        recipe, agents, agent_execution=agent_execution,
        candidate_execution=candidate_execution, events=events,
    )
    assert actual is composition.controller.return_value
    compiled_recipe, capabilities = composition.compiler.call_args.args
    assert compiled_recipe.generator.checkpoint is prepared_binding.checkpoint
    assert recipe.generator.checkpoint is None
    assert compiled_recipe.round_count == recipe.round_count
    assert compiled_recipe.default_round is recipe.default_round
    assert compiled_recipe.evaluator is recipe.evaluator
    assert compiled_recipe.agents is recipe.agents
    assert capabilities is prepared_binding.plugin.generator.capabilities
    composition.inventory.assert_called_once_with(composition.compiled, prepared_binding.dataset)
    forwarded = composition.controller.call_args.kwargs
    assert forwarded["campaign"] is composition.compiled
    assert forwarded["dataset"] is prepared_binding.dataset
    assert forwarded["agent_execution"] is agent_execution
    assert forwarded["candidate_execution"] is candidate_execution
    assert forwarded["agents"] is agents
    assert forwarded["events"] is events
    candidate_execution.execute_cases.assert_not_called()


@pytest.mark.parametrize("candidate_execution", [
    None, InProcessCandidateExecutionService(), _candidate_service("controller"),
])
def test_local_development_retains_explicit_inprocess_compatibility(
    prepared_binding, recipe, composition, candidate_execution,
):
    prepared_binding.controller(
        recipe, {}, candidate_execution=candidate_execution, local_development=True,
    )
    forwarded = composition.controller.call_args.kwargs
    assert isinstance(forwarded["agent_execution"], InProcessAgentExecutionService)
    assert forwarded["candidate_execution"] is candidate_execution
