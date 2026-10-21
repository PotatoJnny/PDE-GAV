"""Hidden query pairing is an explicit EvaluationProgram capability."""

from types import SimpleNamespace

from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.plugins.vpfp.tasks.observable import VPFPObservableTask
from pde_agent_verifier_solver.sdk import AnswerQuery, EvaluationPhase


def _controller(evaluation):
    controller = object.__new__(CampaignController)
    controller._seeds = SeedLedger(101)
    controller.evaluation = evaluation
    return controller


def _hidden():
    def row(identity, parent=None):
        return SimpleNamespace(case=SimpleNamespace(case_id=identity, parent_case_id=parent))

    return {"base": (row("root"),), "base_perturbed": (row("child", "root"),)}


def test_ns_keeps_frozen_distribution_seeds_by_default():
    query = AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"),
        points=({"t": 0.0, "x": 0.5, "y": 0.5},),
    )
    controller = _controller(NSSolverTask().create_evaluation({"default_query": query}))
    provider = controller._hidden_plan_seed_provider(
        "official_round:1", EvaluationPhase.OFFICIAL, _hidden()
    )
    assert provider(0, "child") == controller._seeds.derive(
        "evaluation_plan", "official_round:1", "official",
        "distribution", "base_perturbed", "case_position", 0,
    )
    assert provider(0, "root") != provider(0, "child")


def test_vpfp_observable_explicitly_preserves_paired_hidden_queries():
    query = AnswerQuery.at_points(
        fields=("rho", "momentum_x", "momentum_y", "kinetic_energy"),
        points=({"t": 0.0, "x": 0.5, "y": 0.5},),
    )
    evaluation = VPFPObservableTask().create_evaluation({"default_query": query})
    assert evaluation.pair_hidden_plan_seeds is True
    controller = _controller(evaluation)
    provider = controller._hidden_plan_seed_provider(
        "official_round:1", EvaluationPhase.OFFICIAL, _hidden()
    )
    next_round = controller._hidden_plan_seed_provider(
        "official_round:2", EvaluationPhase.OFFICIAL, _hidden()
    )
    assert provider(0, "root") == provider(0, "child")
    assert provider(0, "child") != next_round(0, "child")


def test_legacy_case_records_need_no_optional_parent_attribute():
    controller = _controller(SimpleNamespace())
    hidden = {"base": (SimpleNamespace(case=SimpleNamespace(case_id="legacy")),)}
    provider = controller._hidden_plan_seed_provider(
        "official_round:1", EvaluationPhase.OFFICIAL, hidden
    )
    assert provider(0, "legacy") == controller._seeds.derive(
        "evaluation_plan", "official_round:1", "official",
        "distribution", "base", "case_position", 0,
    )
