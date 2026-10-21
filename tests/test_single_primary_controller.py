"""The shared Controller evaluates one primary per Agent plus its incumbent."""

from dataclasses import replace

from pde_agent_verifier_solver.engine.agent_execution import (
    SubprocessAgentExecutionService,
    UnavailableAgentCapabilityFacade,
)
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController

from tests.test_campaign_continuation import _controller
from tests.test_single_primary_agent_execution import _program


def _run(root, *, single_primary):
    fixture = _controller(root)
    campaign = replace(fixture.campaign, rounds=fixture.campaign.rounds[:1])
    settings = {"nomination_policy": "single_primary_v1"} if single_primary else {}
    programs = {}
    for index, agent in enumerate(campaign.rounds[0].agents):
        payloads = [
            {"mode": 2 + index},
            {"mode": 3 + index},
            {"mode": 1 + index},
        ]
        source = (
            "from pde_agent_verifier_solver.engine.agent_execution import "
            "submit_agent_nomination, select_agent_primary_nomination, write_agent_worker_response\n"
        )
        for sequence, payload in enumerate(payloads, start=1):
            source += f"submit_agent_nomination({payload!r}, sequence={sequence})\n"
            if single_primary and sequence <= 2:
                source += f"select_agent_primary_nomination({sequence})\n"
        source += "write_agent_worker_response()\n"
        programs[agent.agent_id] = _program(source, settings=settings)
    service = SubprocessAgentExecutionService(
        root / "turns", hard_wall_seconds=10,
        capability_facade=UnavailableAgentCapabilityFacade(),
        termination_grace_seconds=0.1, poll_interval_seconds=0.01,
    )
    controller = CampaignController(
        campaign, fixture.dataset, fixture.problem, fixture.task,
        fixture.generator, programs, agent_execution=service,
    )
    return controller.run().rounds[0], service.audits


def test_two_agents_keep_only_explicit_primaries_and_incumbent_on_same_sized_cohort(tmp_path):
    legacy, legacy_audits = _run(tmp_path / "legacy", single_primary=False)
    primary, primary_audits = _run(tmp_path / "primary", single_primary=True)
    assert len(legacy.promotion_reports) == 7  # Three submissions each plus S0.
    assert len(primary.promotion_reports) == 3
    assert primary.hidden_suite_size == legacy.hidden_suite_size == 11
    assert primary.case_mix == legacy.case_mix
    assert primary.agent_case_counts == legacy.agent_case_counts
    expected = {"base": 5, "guided": 3, "base_perturbed": 2, "guided_perturbed": 1}
    for report in (*legacy.promotion_reports, *primary.promotion_reports):
        assert {name: report.distributions[name].case_count for name in expected} == expected
    assert any(report.candidate_digest == "candidate-s0" for report in primary.promotion_reports)
    assert [audit.nomination_count for audit in primary_audits] == [1, 1]
    assert [audit.nomination_journal_entry_count for audit in primary_audits] == [3, 3]
    assert [audit.final_nomination_sequence for audit in primary_audits] == [2, 2]
    assert [audit.nomination_count for audit in legacy_audits] == [3, 3]
