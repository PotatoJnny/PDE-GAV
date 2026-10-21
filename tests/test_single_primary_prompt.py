"""Prompts use upstream explicit choice while preserving Plasma research budgets."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_canonical_codex_agent.py"
SPEC = importlib.util.spec_from_file_location("_single_primary_worker_prompt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def request(round_index=1, *, policy=True):
    return {
        "round": {"round_index": round_index, "active_budget_seconds": None,
                  "hard_wall_seconds": 14400, "time_budget_mode": "hard_wall",
                  "development_sampling": {"max_cases": 32, "allowed_sources": ["base", "guided"]}},
        "agent": {"prompt": "Improve the strong-collision solver."},
        "candidate_execution": {"case_charged_wall_seconds": 900.0},
        "worker_settings": {"nomination_policy": "single_primary_v1"} if policy else {},
    }


def test_explicit_primary_prompt_keeps_scientific_and_execution_budgets():
    prompt = WORKER._research_prompt(request())
    for expected in (
        "Explicitly choose one primary", "select-primary N",
        "Later ordinary submissions", "Only the selected", "incumbent",
        "14400 seconds of total elapsed time", "The Development Case cap is 32",
        "900-second per-Case charged-wall limit", "Missing or invalid selection never falls back",
    ):
        assert expected in prompt
    assert "zero, one, or multiple distinct Candidates" not in prompt
    assert "Only your last committed submission" not in prompt


def test_unconfigured_prompt_keeps_original_all_nomination_protocol():
    prompt = WORKER._research_prompt(request(policy=False))
    assert "zero, one, or multiple distinct Candidates" in prompt
    assert "nomination-count cap" in prompt
    assert "select-primary N" not in prompt


def test_latest_prompt_does_not_accidentally_enable_explicit_primary():
    configured = request()
    configured["worker_settings"]["nomination_policy"] = "latest"
    prompt = WORKER._research_prompt(configured)
    assert "Only your last committed submission" in prompt
    assert "select-primary N" not in prompt
