from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "experiment_recovery"
    / "single_session_competition_8h"
    / "run_competition_loop.py"
)
SPEC = importlib.util.spec_from_file_location("single_session_competition", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_neutral_prompt_removes_prescribed_methods_and_adds_competition() -> None:
    base = "prefix\n" + MODULE.METHOD_DIRECTIVE + "\nsuffix\n"
    prompt = MODULE.neutralize_prompt(base)
    assert "Fourier spectra" not in prompt
    assert "singular values" not in prompt
    assert "low-rank projections" not in prompt
    normalized = " ".join(prompt.split())
    assert "without assuming a hidden representation" in normalized
    assert "submission_a.py" in prompt
    assert "agent-invisible selection cohort" in prompt
    assert "one continuous Codex session" in prompt


@pytest.mark.skipif(
    not MODULE.DEFAULT_SOURCE_ROOT.exists(),
    reason="legacy D0 replay checkout is only available on the experiment cluster",
)
def test_real_legacy_prompt_is_neutral_and_competitive() -> None:
    legacy = MODULE._load_loop(MODULE.DEFAULT_SOURCE_ROOT)
    MODULE._install_neutral_prompt(legacy)
    prompt = legacy._solver_prompt(
        round_index=0,
        eval_density={"nx": 0, "ny": 0, "nt": 0},
        has_previous=True,
        mixed_source_agent_mode=False,
        pointwise_feedback_enabled=True,
        public_context={"smoke": True},
        previous_solver_source="def solve_case(case_dir, query_grid=None):\n    pass\n",
    )
    assert "Useful structural diagnostics include" not in prompt
    assert "Fourier spectra" not in prompt
    assert "singular values" not in prompt
    assert "low-rank projections" not in prompt
    assert "submission_a.py" in prompt
    assert "agent-invisible selection cohort" in prompt
    assert "one continuous Codex session" in prompt


def _evaluation(values: list[float]) -> dict:
    return {
        "results": [
            {"case_id": f"case_{index}", "status": "ok", "relative_l2": value}
            for index, value in enumerate(values)
        ]
    }


def test_promotion_accepts_private_improvement_without_d0_regression() -> None:
    selection = MODULE.paired_summary(
        _evaluation([1.0] * 16),
        _evaluation([0.8] * 16),
        bootstrap_samples=200,
        seed=1,
    )
    d0 = MODULE.paired_summary(
        _evaluation([1.0] * 16),
        _evaluation([1.005] * 16),
        bootstrap_samples=200,
        seed=2,
    )
    gate = MODULE.promotion_gate(
        selection,
        d0,
        min_selection_improvement=0.01,
        max_d0_regression=0.01,
    )
    assert gate["accepted"] is True


def test_promotion_rejects_d0_regression() -> None:
    selection = MODULE.paired_summary(
        _evaluation([1.0] * 16),
        _evaluation([0.8] * 16),
        bootstrap_samples=200,
        seed=3,
    )
    d0 = MODULE.paired_summary(
        _evaluation([1.0] * 16),
        _evaluation([1.02] * 16),
        bootstrap_samples=200,
        seed=4,
    )
    gate = MODULE.promotion_gate(
        selection,
        d0,
        min_selection_improvement=0.01,
        max_d0_regression=0.01,
    )
    assert gate["accepted"] is False
    assert gate["selection_pass"] is True
    assert gate["d0_guard_pass"] is False
