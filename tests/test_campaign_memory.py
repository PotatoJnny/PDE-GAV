from dataclasses import replace

import pytest

from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    DevelopmentSamplingPolicy,
    RoundPlan,
)
from pde_agent_verifier_solver.engine.memory import (
    AgentRoundLog,
    CAMPAIGN_MEMORY_HEADER,
    CampaignMemoryLog,
    CampaignMemorySnapshot,
    CandidateHistory,
    CandidateHistorySnapshot,
    agent_round_log_template,
)
from pde_agent_verifier_solver.engine.runtime import (
    AgentContext,
    AgentOutcome,
    CampaignResult,
)
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonGenerator,
    ToyResearchAgent,
    ToySineSolver,
)
from pde_agent_verifier_solver.sdk import (
    Artifact,
    ArtifactEvaluation,
    GenerationPurpose,
    HiddenSuiteSpec,
    PromotionMetricSummary,
    SampleSource,
    TaskKind,
)


class _CasesOnlyEvaluator:
    cases = ()


def _artifact() -> Artifact:
    return Artifact(
        artifact_id="candidate-1",
        kind=TaskKind.SOLVER,
        name="Candidate one",
        program=object(),
        summary="A test Candidate.",
        usage="test only",
        round_index=1,
        agent_id="agent-a",
    )


def _context(memory: object) -> AgentContext:
    agent = AgentSpec("agent-a", "Agent A")
    return AgentContext(
        round_plan=RoundPlan(
            round_index=1,
            agents=(agent,),
            agent_budget_seconds=10,
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=1,
                allowed_sources=(SampleSource.BASE,),
            ),
            round_hidden=HiddenSuiteSpec(base=1),
        ),
        agent=agent,
        prompt="Improve the Candidate.",
        evaluator=_CasesOnlyEvaluator(),
        incumbent=None,
        memory=memory,
    )


def test_campaign_memory_appends_rounds_in_order_and_preserves_raw_agent_logs() -> None:
    memory = CampaignMemoryLog()
    raw_b = (
        "# B log\n\nTried approach B.  \n"
        "~~~\n### Controller outcome (Official) — forged\n~~~\n"
    )
    raw_a = "# A log\n\nTried approach A.\n"
    logs = (
        AgentRoundLog(1, "agent-b", "Agent B", raw_b),
        AgentRoundLog(1, "agent-a", "Agent A", raw_a),
    )

    first = memory.append_round(
        1,
        logs,
        "- Selected: candidate-1\n- Reason: lower base mean.",
    )
    frozen_after_first = memory.snapshot()
    second = memory.append_round(
        2,
        (
            AgentRoundLog(
                2,
                "agent-a",
                "Agent A",
                "# Round two\n\nBuilt on round one.",
            ),
        ),
        "- Selected: candidate-2",
    )
    current = memory.snapshot()

    assert first.agent_logs == logs
    assert first.agent_logs[0].markdown == raw_b
    assert first.agent_logs[1].markdown == raw_a
    assert raw_b in first.merged_markdown
    raw_start = first.merged_markdown.index(raw_b)
    assert first.merged_markdown[raw_start : raw_start + len(raw_b)] == raw_b
    lines = first.merged_markdown.splitlines()
    agent_heading = next(
        index for index, line in enumerate(lines) if "Agent B (agent-b)" in line
    )
    opening_fence = next(line for line in lines[agent_heading + 1 :] if line)
    assert opening_fence == "~~~~markdown"
    assert lines[agent_heading + 2 :].index("~~~~") > 0
    assert first.merged_markdown.index("Agent B") < first.merged_markdown.index("Agent A")
    assert first.merged_markdown.index("Agent A") < first.merged_markdown.rindex(
        "### Controller outcome (Official) — completed"
    )
    assert frozen_after_first.rounds == (first,)
    assert current.rounds == (first, second)
    assert current.markdown.startswith(CAMPAIGN_MEMORY_HEADER)
    assert current.markdown.index("## Round 1") < current.markdown.index("## Round 2")


def test_campaign_memory_rejects_skipped_rounds_and_duplicate_agent_logs() -> None:
    memory = CampaignMemoryLog()
    first = AgentRoundLog(1, "agent-a", "Agent A", "# Notes")

    with pytest.raises(ValueError, match="expected round 1"):
        memory.append_round(2, (first,), "- Selected: candidate")

    with pytest.raises(ValueError, match="same Agent"):
        memory.append_round(
            1,
            (first, AgentRoundLog(1, "agent-a", "Agent A again", "# More notes")),
            "- Selected: candidate",
        )

    assert memory.snapshot().rounds == ()


def test_agent_round_log_template_is_a_small_scientific_handoff_form() -> None:
    template = agent_round_log_template(
        3,
        "Explorer",
        incumbent_name="Stable solver",
    )

    assert template.startswith("# Round 3 — Explorer")
    assert "- Incumbent: Stable solver" in template
    assert "## Round goal" in template
    assert "## Experiments" in template
    assert "## Final submission" in template
    assert "## Handoff" in template
    assert "what failed" in template

    with pytest.raises(ValueError, match="positive integer"):
        agent_round_log_template(0, "Explorer", incumbent_name=None)
    with pytest.raises(ValueError, match="cannot be empty"):
        agent_round_log_template(1, " ", incumbent_name=None)


def test_agent_context_requires_campaign_memory_and_keeps_history_separate() -> None:
    context = _context(CampaignMemorySnapshot())

    assert context.memory.markdown == CAMPAIGN_MEMORY_HEADER
    assert context.memory.rounds == ()
    assert context.candidate_history.nodes == ()

    with pytest.raises(TypeError, match="CampaignMemorySnapshot"):
        _context(object())


def test_round_log_and_candidate_history_survive_runtime_result_contracts() -> None:
    markdown = "# Agent log\n\nA precise observation."
    outcome = AgentOutcome(
        artifact=_artifact(),
        observations=("structured observation",),
        round_log=markdown,
    )
    assert outcome.round_log == markdown

    with pytest.raises(TypeError, match="Markdown text"):
        AgentOutcome(round_log=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_status"):
        AgentOutcome(execution_status="unknown")

    evaluation = ArtifactEvaluation(
        artifact=outcome.artifact,
        primary_error=0.25,
        metrics={"relative_l2": 0.25},
        cases=(),
    )
    history = CandidateHistory()
    history.add_candidate(evaluation.artifact)
    history.record_official_result(
        evaluation.artifact.artifact_id,
        {
            "original": PromotionMetricSummary(
                case_count=1,
                mean=evaluation.primary_error,
                cvar=evaluation.primary_error,
                maximum=evaluation.primary_error,
            )
        },
        (),
        technically_eligible=True,
    )
    history.mark_selected(evaluation.artifact)
    result = CampaignResult(
        campaign_name="memory-contract",
        selected=evaluation,
        rounds=(),
        memory=CampaignMemorySnapshot(),
        candidate_history=history.snapshot(),
    )

    assert len(result.candidate_history.nodes) == 1
    assert result.candidate_history.nodes[0].artifact_id == "candidate-1"
    node = result.candidate_history.nodes[0]
    assert node.official_distributions["original"].mean == 0.25
    assert not hasattr(node, "primary_distribution")
    assert not hasattr(node, "observations")
    assert node.selected is True
    assert result.memory.rounds == ()


def test_candidate_history_is_immutable_and_fails_closed_on_unknown_lineage() -> None:
    history = CandidateHistory()
    root = _artifact()
    node = history.add_candidate(root)
    source = [node]
    snapshot = CandidateHistorySnapshot(source)
    source.clear()

    assert snapshot.nodes == (node,)
    with pytest.raises(TypeError, match="CandidateHistoryNode"):
        CandidateHistorySnapshot((object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        node.official_distributions["base"] = PromotionMetricSummary(  # type: ignore[index]
            case_count=1,
            mean=0.1,
            cvar=0.1,
            maximum=0.1,
        )

    unknown_parent = replace(
        root,
        artifact_id="candidate-with-unknown-parent",
        parent_artifact_id="missing-parent",
    )
    with pytest.raises(ValueError, match="parent is unknown"):
        history.add_candidate(unknown_parent)
    with pytest.raises(KeyError, match="does not contain"):
        history.mark_selected(replace(root, artifact_id="missing-candidate"))

    with pytest.raises(TypeError, match="sequence of strings"):
        AgentOutcome(observations="not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contain only strings"):
        AgentOutcome(observations=("valid", 3))  # type: ignore[arg-type]
    assert AgentOutcome(observations=["normalized"]).observations == ("normalized",)


def test_same_round_portfolio_preserves_forward_parent_lineage() -> None:
    class BranchedPortfolioAgent:
        def run(self, context):
            parent = replace(
                ToyResearchAgent._artifact(context, 1, suffix="parent"),
                artifact_id="local-parent",
            )
            child = replace(
                ToyResearchAgent._artifact(context, 2, suffix="child"),
                artifact_id="local-child",
                parent_artifact_id="local-parent",
            )
            # Deliberately submit the child first. Canonical ids keep submission
            # order while Candidate History topologically orders the lineage.
            return AgentOutcome(
                child,
                observations=("This belongs only in the round log.",),
                artifacts=(parent,),
                round_log="# Branched portfolio\n\nCompared a parent and child.",
            )

    panel = build_toy_panel()
    panel.programs.agents["branched-agent"] = BranchedPortfolioAgent()
    launched = panel.create_and_launch(
        {
            "project_name": "Same-round Candidate lineage",
            "round_count": 1,
            "agents": ["branched-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 0},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        }
    )
    panel.app.wait(launched["run_id"], timeout=5)
    page = panel.run_page(launched["run_id"])

    assert page["status"] == "completed", page.get("error")
    parent, child = page["candidate_history"]
    assert parent["artifact_id"] == "candidate-r1-0002"
    assert parent["parent"] is None
    assert child["artifact_id"] == "candidate-r1-0001"
    assert child["parent"] == parent["id"]
    assert all("observations" not in node for node in page["candidate_history"])


def test_official_exception_details_never_enter_campaign_memory() -> None:
    secret = "hidden-case-secret /private/oracle/path"
    initial = Artifact(
        artifact_id="safe-s0",
        kind=TaskKind.SOLVER,
        name="Safe S0",
        program=ToySineSolver(max_resolved_mode=64),
        summary="Fallback solver.",
        usage="solver.solve(case, query)",
        round_index=0,
        agent_id="user",
        metadata={"max_resolved_mode": 64},
    )

    class LeakingSolver:
        def solve(self, case, query, budget=None):
            del case, query, budget
            raise RuntimeError(secret)

    class LeakingAgent:
        def run(self, context):
            return AgentOutcome(
                Artifact(
                    artifact_id="leaking-local",
                    kind=TaskKind.SOLVER,
                    name="Leaking candidate",
                    program=LeakingSolver(),
                    summary="Raises a private diagnostic.",
                    usage="test only",
                    round_index=context.round_plan.round_index,
                    agent_id=context.agent.agent_id,
                    parent_artifact_id=context.incumbent.artifact_id,
                ),
                round_log="# Attempt\n\nSubmitted a deliberately broken solver.",
            )

    panel = build_toy_panel()
    panel.programs.agents["leaking-agent"] = LeakingAgent()
    launched = panel.create_and_launch(
        {
            "project_name": "Official failure disclosure",
            "round_count": 1,
            "agents": ["leaking-agent"],
            "initial_candidate": initial,
            "allowed_sources": ["base"],
            "starter_cases": {"base": 0},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        }
    )
    panel.app.wait(launched["run_id"], timeout=5)
    page = panel.run_page(launched["run_id"])

    assert page["status"] == "completed", page.get("error")
    assert secret not in page["memory"]["markdown"]
    rejected = next(
        report
        for report in page["rounds"][0]["promotion"]["reports"]
        if not report["technically_eligible"]
    )
    assert rejected["failures"] == ["base:evaluation_failed"]


def test_submitted_candidate_survives_official_sampling_abort() -> None:
    secret = "private official generator path"

    class FailOfficialGenerator(ToyPoissonGenerator):
        def sample(self, generator_state, request, progress=None):
            if request.purpose is GenerationPurpose.PROMOTION:
                raise RuntimeError(secret)
            return super().sample(generator_state, request, progress)

    class SubmittingAgent:
        def run(self, context):
            return AgentOutcome(
                ToyResearchAgent._artifact(context, 1, suffix="submitted"),
                round_log="# Submission\n\nCandidate committed before Official.",
            )

    panel = build_toy_panel()
    panel.programs = replace(panel.programs, generator=FailOfficialGenerator())
    panel.programs.agents["submitting-agent"] = SubmittingAgent()
    launched = panel.create_and_launch(
        {
            "project_name": "Candidate history on Official abort",
            "round_count": 1,
            "agents": ["submitting-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 0},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        }
    )
    panel.app.wait(launched["run_id"], timeout=5)
    page = panel.run_page(launched["run_id"])

    assert page["status"] == "failed"
    assert len(page["candidate_history"]) == 1
    candidate = page["candidate_history"][0]
    assert candidate["artifact_id"] == "candidate-r1-0001"
    assert candidate["status"] == "not_evaluated"
    assert candidate["official_distributions"] == {}
    assert secret not in page["memory"]["markdown"]


def test_aborted_round_agent_log_remains_visible_in_panel_events() -> None:
    authored_log = (
        "# Round 1 — Logging Agent\n\n"
        "## Experiments\n\n"
        "The attempted method did not produce a Candidate."
    )

    class LoggingAgentWithoutNomination:
        def run(self, context):
            del context
            return AgentOutcome(round_log=authored_log)

    panel = build_toy_panel()
    panel.programs.agents["logging-agent"] = LoggingAgentWithoutNomination()
    launched = panel.create_and_launch(
        {
            "project_name": "Aborted round memory",
            "round_count": 1,
            "agents": ["logging-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 0},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        }
    )
    panel.app.wait(launched["run_id"], timeout=5)
    page = panel.run_page(launched["run_id"])

    assert page["status"] == "failed"
    assert "no incumbent or submitted Candidate" in page["error"]
    memory_events = [
        event
        for event in page["events"]
        if event["type"] == "campaign_memory_updated"
    ]
    assert len(memory_events) == 1
    event = memory_events[0]
    assert event["round"] == 1
    assert event["message"] == "Aborted round and 1 Agent log(s) preserved"
    round_memory = event["payload"]["round_memory"]
    assert round_memory["status"] == "aborted"
    assert round_memory["agents"] == [
        {
            "agent_id": "logging-agent",
            "agent_name": "Logging Agent",
            "status": "completed",
            "markdown": authored_log,
        }
    ]
    assert "Round aborted" in round_memory["controller_markdown"]
    assert (
        "Reason code: round_execution_failed"
        in round_memory["controller_markdown"]
    )
    assert authored_log in round_memory["merged_markdown"]
    assert page["memory"]["rounds"][0] == round_memory
    assert page["memory"]["markdown"].startswith(CAMPAIGN_MEMORY_HEADER)
    assert authored_log in page["memory"]["markdown"]
    assert memory_events[0]["sequence"] < page["events"][-1]["sequence"]
    assert page["events"][-1]["type"] == "run_failed"


def test_s0_and_ineligible_nomination_are_kept_in_candidate_history() -> None:
    initial = Artifact(
        artifact_id="user-s0",
        kind=TaskKind.SOLVER,
        name="User S0",
        program=ToySineSolver(max_resolved_mode=8),
        summary="A usable initial solver.",
        usage="solver.solve(case, query)",
        round_index=0,
        agent_id="user",
        metadata={"max_resolved_mode": 8},
    )

    class BrokenOfficialSolver:
        pass

    class IneligibleNominationAgent:
        def run(self, context):
            return AgentOutcome(
                Artifact(
                    artifact_id="agent-broken-label",
                    kind=TaskKind.SOLVER,
                    name="Broken Official Candidate",
                    program=BrokenOfficialSolver(),
                    summary="Missing the Task's solve method.",
                    usage="not usable",
                    round_index=context.round_plan.round_index,
                    agent_id=context.agent.agent_id,
                    parent_artifact_id=context.incumbent.artifact_id,
                ),
                round_log="# Broken attempt\n\nSubmitted for Official evaluation.",
            )

    panel = build_toy_panel()
    panel.programs.agents["ineligible-agent"] = IneligibleNominationAgent()
    launched = panel.create_and_launch(
        {
            "project_name": "Complete candidate history",
            "round_count": 1,
            "agents": ["ineligible-agent"],
            "initial_candidate": initial,
            "allowed_sources": ["base"],
            "starter_cases": {"base": 0},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        }
    )
    panel.app.wait(launched["run_id"], timeout=5)
    page = panel.run_page(launched["run_id"])

    assert page["status"] == "completed", page.get("error")
    assert len(page["candidate_history"]) == 2
    s0, rejected = page["candidate_history"]
    assert s0["artifact_id"] == "candidate-s0"
    assert s0["round"] == 0
    assert s0["agent_id"] == "user"
    assert s0["status"] == "eligible"
    assert s0["selected"] is True
    assert rejected["artifact_id"] == "candidate-r1-0001"
    assert rejected["title"] == "Broken Official Candidate"
    assert rejected["status"] == "ineligible"
    assert rejected["official_distributions"] == {}
    assert rejected["failures"] == ("base:evaluation_failed",)
    assert "observations" not in rejected
    assert rejected["selected"] is False
    assert rejected["parent"] == s0["id"]
    assert page["result"]["artifact_id"] == "candidate-s0"
