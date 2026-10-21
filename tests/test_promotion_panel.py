from pde_agent_verifier_solver.panel.campaign_view import _promotion_history_view
from pde_agent_verifier_solver.panel.staged_server import DASHBOARD_HTML


def test_panel_projects_all_promotion_distributions_without_raw_digests() -> None:
    payload = {
        "controller_policy": {"promotion_program": "test-program"},
        "rounds": [
            {
                "round_index": 0,
                "promotion_reports": [
                    {
                        "candidate_id": "incumbent",
                        "candidate_sha256": "digest-incumbent",
                        "technically_eligible": True,
                        "distributions": {
                            "base": {
                                "case_count": 64,
                                "mean": 0.2,
                                "cvar": 0.3,
                                "max": 0.4,
                            },
                            "guided": {
                                "case_count": 64,
                                "mean": 0.5,
                                "cvar": 0.6,
                                "max": 0.7,
                            },
                        },
                        "failures": [],
                    },
                    {
                        "candidate_id": "challenger",
                        "candidate_sha256": "digest-challenger",
                        "technically_eligible": True,
                        "distributions": {
                            "base": {
                                "case_count": 64,
                                "mean": 0.1,
                                "cvar": 0.2,
                                "max": 0.3,
                            },
                            "guided": {
                                "case_count": 64,
                                "mean": 0.4,
                                "cvar": 0.5,
                                "max": 0.6,
                            },
                        },
                        "failures": [],
                    },
                ],
                "promotion_decision": {
                    "selected_sha256": "digest-challenger",
                    "promoted": True,
                    "reason": "guardrails_passed",
                    "ranking": [
                        {
                            "candidate_sha256": "digest-challenger",
                            "rank": 1,
                            "eligible": True,
                            "reason": "best_base_mean",
                        },
                        {
                            "candidate_sha256": "digest-incumbent",
                            "rank": 2,
                            "eligible": True,
                            "reason": "higher_base_mean",
                        },
                    ],
                },
            }
        ],
    }

    view = _promotion_history_view(payload)

    assert view is not None
    assert view["program"] == "test-program"
    assert view["rounds"][0]["selected"] == "challenger"
    assert view["rounds"][0]["promoted"] is True
    report = view["rounds"][0]["reports"][1]
    assert [row["name"] for row in report["distributions"]] == [
        "base",
        "guided",
    ]
    assert report["distributions"][0]["mean"] == 0.1
    assert "digest-challenger" not in str(view)


def test_local_panel_renders_promotion_evidence() -> None:
    assert "Candidate selection" in DASHBOARD_HTML
    assert "promotion_history" in DASHBOARD_HTML
