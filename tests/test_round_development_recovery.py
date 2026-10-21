"""Small real toy-pipeline tests of the explicit first-round recovery boundary."""

import pytest

from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignCompiler, CampaignRecipe, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.memory import AgentRoundLog
from pde_agent_verifier_solver.engine.round_recovery import RoundDevelopmentRecovery
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.plugins.toy import (
    ToyBaseGeneratorState, ToyPoissonGenerator, ToyPoissonProblem,
    ToySineSolver, ToySolverTask,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, CaseSource, GenerationPurpose, GeneratorCheckpoint,
    EvaluationPhase, HiddenSuiteSpec, OfficialEvaluationSpec, PreparedDataset,
    SampleCasesRequest, SampleSource, TaskKind,
)


def _artifact(identity, *, mode=0, round_index=1, parent="candidate-s0", agent="agent-a"):
    return Artifact(
        artifact_id=identity, kind=TaskKind.SOLVER, name=identity,
        program=ToySineSolver(max_resolved_mode=mode), summary="Frozen test solver",
        usage="solve(case, query)", round_index=round_index, agent_id=agent,
        parent_artifact_id=parent, metadata={"max_resolved_mode": mode},
    )


class _Generator(ToyPoissonGenerator):
    def __init__(self):
        self.requests = []
        self.feedback = []

    def sample(self, state, request, progress=None):
        self.requests.append(request)
        return super().sample(state, request, progress)

    def update_guidance(self, state, feedback, settings, progress=None):
        self.feedback.append(feedback)
        return super().update_guidance(state, feedback, settings, progress)


class _FreshAgent:
    def __init__(self):
        self.contexts = []

    def run(self, context):
        self.contexts.append(context)
        assert context.round_plan.round_index == 2
        assert len(context.memory.rounds) == 1
        assert context.incumbent.artifact_id == "candidate-r1-0002"
        assert "OLD R1 NOTES\r\n  exact bytes" in context.memory.markdown
        assert "Recovery provenance" in context.memory.markdown
        assert len(context.candidate_history.nodes) == 3
        assert all(node.status == "eligible" for node in context.candidate_history.nodes)
        return AgentOutcome(
            _artifact("fresh-local", mode=64, round_index=2,
                      parent=context.incumbent.artifact_id),
            round_log="# Fresh R2 notes\nNo old R2 memory imported.",
        )


def _controller(*, guidance=True, rounds=2):
    generator = _Generator()
    agent = _FreshAgent()
    recipe = CampaignRecipe(
        name="Frozen first-round recovery", round_count=rounds, master_seed=101,
        agents=(AgentSpec("agent-a", "Agent A"),),
        initial_candidate=_artifact("source-s0", round_index=0, parent=None),
        generator=GeneratorRecipe(
            guidance_enabled=guidance, perturbations_enabled=False,
            checkpoint=GeneratorCheckpoint(
                generator.info.program_id, ToyBaseGeneratorState(5, 1.0, 8)
            ),
        ),
        default_round=RoundSettings(
            agent_budget_seconds=1,
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=4,
                allowed_sources=(SampleSource.BASE, SampleSource.GUIDED)
                if guidance else (SampleSource.BASE,),
            ),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=2), terminal=HiddenSuiteSpec(base=2)
            ),
        ),
        evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=17)),
    )
    campaign = CampaignCompiler().compile(recipe, generator.capabilities)
    return CampaignController(
        campaign, PreparedDataset(None, (), ()), ToyPoissonProblem(), ToySolverTask(),
        generator, {"agent-a": agent},
    ), agent


def _recovery(**overrides):
    values = {
        "round_index": 1,
        "artifacts": (_artifact("candidate-r1-0001"),
                      _artifact("candidate-r1-0002", mode=16)),
        "agent_logs": (AgentRoundLog(1, "agent-a", "Agent A", "OLD R1 NOTES\r\n  exact bytes"),),
        "agent_case_counts": {"agent-a": {"base": 2}},
        "source_run_id": "old-source-run",
        "source_manifest_sha256": "a" * 64,
        "verify_official": lambda plan, hidden, artifacts: {
            "status": "passed", "verified_cases": len(hidden["base"])
        },
        "development_case_sources": {"old-development-case": CaseSource.GENERATED},
        "development_case_id_coverage": "partial",
    }
    return RoundDevelopmentRecovery(**{**values, **overrides})


def test_recovery_rebuilds_selection_memory_guidance_and_only_runs_fresh_r2():
    controller, agent = _controller()
    verified = []

    def verify(plan, hidden, artifacts):
        assert not agent.contexts
        assert plan.round_index == 1
        assert tuple(item.artifact_id for item in artifacts) == (
            "candidate-r1-0001", "candidate-r1-0002", "candidate-s0"
        )
        assert all(node.status == "not_evaluated"
                   for node in controller._candidate_history.snapshot().nodes)
        verified.append(hidden)
        return {"status": "passed", "verified_cases": 2}

    recovery = _recovery(verify_official=verify)
    result = controller.run(recovery=recovery)

    assert len(verified) == 1
    assert len(agent.contexts) == 1
    assert result.rounds[0].selected.artifact.artifact_id == "candidate-r1-0002"
    assert result.rounds[0].agent_case_counts["agent-a"]["base"] == 2
    assert result.memory.rounds[0].agent_logs == recovery.agent_logs
    assert [feedback.selected_program_id for feedback in controller.generator.feedback] == [
        "candidate-s0", "candidate-r1-0002"
    ]
    assert [feedback.source_round_index for feedback in controller.generator.feedback] == [0, 1]
    assert all(request.round_index != 1 for request in controller.generator.requests
               if request.purpose is GenerationPurpose.DEVELOPMENT)
    assert result.rounds[1].candidate_evaluations[0].artifact.artifact_id == "candidate-r2-0003"
    assert "old-development-case" in controller._development_case_ids
    assert "a" * 64 in result.memory.rounds[0].controller_markdown


def test_verification_failure_preserves_raw_notes_but_blocks_official_and_r2():
    controller, agent = _controller()

    def reject(plan, hidden, artifacts):
        raise ValueError("saved private query differs")

    recovery = _recovery(verify_official=reject)
    with pytest.raises(ValueError, match="private query differs"):
        controller.run(recovery=recovery)
    assert not agent.contexts
    memory = controller._campaign_memory.snapshot()
    assert memory.rounds[0].status == "aborted"
    assert memory.rounds[0].agent_logs == recovery.agent_logs
    assert all(node.status == "not_evaluated"
               for node in controller._candidate_history.snapshot().nodes)
    assert not any(node.selected for node in controller._candidate_history.snapshot().nodes)


@pytest.mark.parametrize("alteration, message", [
    ({"artifacts": (_artifact("candidate-r1-0002"),)}, "contiguous Controller"),
    ({"artifacts": (_artifact("candidate-r1-0001", round_index=2),)}, "Candidate round"),
    ({"artifacts": (_artifact("candidate-r1-0001", agent="other"),)}, "Candidate Agent"),
    ({"artifacts": (_artifact("candidate-r1-0001", parent="old-r2-candidate"),)}, "Candidate parent"),
    ({"artifacts": (_artifact("candidate-r1-0001", parent="candidate-r1-0001"),)}, "cycle"),
    ({"agent_logs": ()}, "Agent order"),
    ({"agent_case_counts": {"other": {"base": 1}}}, "exactly the Campaign"),
    ({"agent_case_counts": {"agent-a": {"base": 5}}}, "development limit"),
])
def test_malformed_recovery_is_rejected_before_generator_or_agent_work(alteration, message):
    controller, agent = _controller()
    with pytest.raises(ValueError, match=message):
        controller.run(recovery=_recovery(**alteration))
    assert not controller.generator.requests
    assert not agent.contexts


def test_recovery_rejects_later_rounds_and_requires_explicit_source_and_verifier():
    with pytest.raises(ValueError, match="first-round"):
        _recovery(round_index=2)
    with pytest.raises(ValueError, match="SHA-256"):
        _recovery(source_manifest_sha256="missing")
    with pytest.raises(TypeError, match="verification"):
        _recovery(verify_official=None)
    with pytest.raises(ValueError, match="nonnegative"):
        _recovery(agent_case_counts={"agent-a": {"base": -1}})
    with pytest.raises(ValueError, match="recovered round"):
        _recovery(agent_logs=(AgentRoundLog(2, "agent-a", "Agent A", "OLD R2"),))


def test_same_agent_forward_parent_is_preserved_without_reassigning_ids():
    controller, _ = _controller(guidance=False, rounds=1)
    recovery = _recovery(artifacts=(
        _artifact("candidate-r1-0001", mode=16, parent="candidate-r1-0002"),
        _artifact("candidate-r1-0002", mode=0),
    ))
    result = controller.run(recovery=recovery)
    nodes = {node.artifact_id: node for node in result.candidate_history.nodes}
    assert nodes["candidate-r1-0001"].parent_node_id == nodes["candidate-r1-0002"].node_id
    assert nodes["candidate-r1-0001"].selected


def test_skipping_development_preserves_official_questions_and_private_plan_seeds():
    class OriginalAgent:
        def run(self, context):
            context.sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 2))
            return AgentOutcome(
                artifacts=(_artifact("local-bad"), _artifact("local-good", mode=16)),
                round_log="OLD R1 NOTES\r\n  exact bytes",
            )

    original, _ = _controller(rounds=1)
    original.agents["agent-a"] = OriginalAgent()
    saved = []
    original_official = original._official_reports

    def fingerprint(controller, hidden):
        provider = controller._hidden_plan_seed_provider(
            "official_round:1", EvaluationPhase.OFFICIAL, hidden
        )
        settings = {
            **controller.campaign.task_settings,
            "default_query": controller.campaign.evaluator.default_query,
        }
        return tuple(
            (
                distribution, dict(record.case.public_data),
                provider(index, record.case.case_id),
                controller.evaluation.make_plan(
                    record.case, EvaluationPhase.OFFICIAL, settings,
                    provider(index, record.case.case_id),
                ).query,
            )
            for distribution, records in hidden.items()
            for index, record in enumerate(records)
        )

    def capture(artifacts, hidden, **kwargs):
        if kwargs.get("seed_namespace") == "official_round:1":
            saved.append(fingerprint(original, hidden))
        return original_official(artifacts, hidden, **kwargs)

    original._official_reports = capture
    source_result = original.run()
    recovered, _ = _controller(rounds=1)

    def verify(plan, hidden, artifacts):
        assert fingerprint(recovered, hidden) == saved[0]
        return {"status": "passed", "ordered_public_problems": 2, "private_plans": 2,
                "seed_namespace": "official_round:1"}

    result = recovered.run(recovery=_recovery(verify_official=verify))
    assert result.rounds[0].hidden_distribution_metrics == source_result.rounds[0].hidden_distribution_metrics
    assert result.rounds[0].selected.primary_error == source_result.rounds[0].selected.primary_error
    assert result.memory.rounds[0].agent_logs == source_result.memory.rounds[0].agent_logs


@pytest.mark.parametrize("report", [{}, {"status": "failed"}, {"status": True}])
def test_verifier_report_requires_explicit_pass_before_official_or_fresh_agents(report):
    controller, agent = _controller()
    with pytest.raises(ValueError, match="did not explicitly pass"):
        controller.run(recovery=_recovery(verify_official=lambda *args: report))
    assert not agent.contexts
    assert all(node.status == "not_evaluated"
               for node in controller._candidate_history.snapshot().nodes)
    assert controller._campaign_memory.snapshot().rounds[0].status == "aborted"
