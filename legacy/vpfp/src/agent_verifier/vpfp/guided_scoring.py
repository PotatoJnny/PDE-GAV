from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from agent_verifier.vpfp.compute_budget import (
    SolverComputeBudget,
    coerce_budget_output,
    run_budgeted_submission,
)
from agent_verifier.vpfp.continuous_cases import (
    evaluate_continuous_macro_prediction,
)
from agent_verifier.vpfp.public_vpfp_runtime import (
    load_public_case,
    load_submission,
)
from agent_verifier.vpfp.resource_limits import wall_time_limit


GUIDANCE_SCORE_SCHEMA = "vpfp_guidance_solver_score_v1"
DEFAULT_GUIDANCE_FAILURE_REWARD = 1.0e6


class GuidanceSolverCompatibilityError(RuntimeError):
    pass


def default_guidance_solver_budget() -> SolverComputeBudget:
    return SolverComputeBudget(
        tier=2,
        wall_time_seconds=30.0,
        cpus=4,
        gpus=0,
        memory_bytes=16 * 1024**3,
    )


def normalize_guidance_solver_budget(
    value: SolverComputeBudget | Mapping[str, Any] | None,
) -> SolverComputeBudget:
    if value is None:
        return default_guidance_solver_budget()
    if isinstance(value, SolverComputeBudget):
        return value
    return SolverComputeBudget.from_dict(value)


def load_guidance_solver(path: str | Path) -> Any:
    return load_submission(path)


def run_guidance_solver(
    module: Any,
    case_dir: str | Path,
    *,
    budget: SolverComputeBudget | Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run a frozen solver at the case's actual materialized resolution.

    Compute-budget submissions are preferred because they own their spatial,
    velocity, and time discretization. Legacy entrypoints remain supported for
    archived campaigns, but receive the same wall-time guard and strict macro
    shape validation.
    """

    normalized = normalize_guidance_solver_budget(budget)
    path = Path(case_dir)
    public_case = load_public_case(path)
    query_times = np.asarray(public_case["arrays"]["t"], dtype=np.float64)
    started = time.perf_counter()
    if callable(getattr(module, "solve_with_budget", None)):
        output, elapsed = run_budgeted_submission(
            module, public_case, query_times, normalized
        )
        entrypoint = "solve_with_budget"
    else:
        with wall_time_limit(normalized.wall_time_seconds):
            if callable(getattr(module, "solve", None)):
                raw = module.solve(public_case)
                entrypoint = "solve"
            elif callable(getattr(module, "solve_at_resolution", None)):
                raw = module.solve_at_resolution(
                    public_case,
                    {
                        "nx": int(public_case["grid"].nx),
                        "ny": int(public_case["grid"].ny),
                        "nvx": int(public_case["grid"].nvx),
                        "nvy": int(public_case["grid"].nvy),
                        "nt": int(public_case["time"].nt),
                    },
                )
                entrypoint = "solve_at_resolution"
            elif callable(getattr(module, "solve_case", None)):
                raw = module.solve_case(str(path))
                entrypoint = "solve_case"
            else:
                raise GuidanceSolverCompatibilityError(
                    "target solver has no supported entrypoint"
                )
        output = coerce_budget_output(raw, public_case, query_times)
        elapsed = float(time.perf_counter() - started)
    return output, {
        "entrypoint": entrypoint,
        "wall_time_seconds": float(elapsed),
        "budget": normalized.to_dict(),
        "query_time_count": int(query_times.size),
        "materialized_resolution": {
            "nx": int(public_case["grid"].nx),
            "ny": int(public_case["grid"].ny),
            "nvx": int(public_case["grid"].nvx),
            "nvy": int(public_case["grid"].nvy),
            "nt": int(public_case["time"].nt),
        },
    }


def score_guidance_candidate(
    module: Any,
    case_dir: str | Path,
    *,
    budget: SolverComputeBudget | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output, execution = run_guidance_solver(
            module, case_dir, budget=budget
        )
        evaluation = evaluate_continuous_macro_prediction(case_dir, output)
        score = float(evaluation.get("score_smaller_is_better", np.inf))
        accepted = evaluation.get("agent_status") == "ok" and np.isfinite(score)
        if not accepted:
            return {
                "schema": GUIDANCE_SCORE_SCHEMA,
                "status": "invalid_solver_output",
                "accepted": False,
                "score_larger_is_harder": None,
                "evaluation": evaluation,
                "execution": execution,
                "wall_time_seconds": float(time.perf_counter() - started),
                "failure_signature": str(
                    evaluation.get("agent_status") or "invalid_solver_output"
                ),
            }
        return {
            "schema": GUIDANCE_SCORE_SCHEMA,
            "status": "ok",
            "accepted": True,
            "score_larger_is_harder": score,
            "evaluation": evaluation,
            "execution": execution,
            "wall_time_seconds": float(time.perf_counter() - started),
            "failure_signature": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are audited rewards.
        return {
            "schema": GUIDANCE_SCORE_SCHEMA,
            "status": "exception",
            "accepted": False,
            "score_larger_is_harder": None,
            "evaluation": {
                "agent_status": "failed",
                "exception": repr(exc),
            },
            "execution": None,
            "wall_time_seconds": float(time.perf_counter() - started),
            "failure_signature": f"{type(exc).__name__}:{exc}",
        }


def guidance_selection_rewards(
    results: list[Mapping[str, Any]],
    *,
    context: str,
) -> list[float]:
    valid = [
        float(row["score_larger_is_harder"])
        for row in results
        if row.get("accepted") is True
    ]
    if not valid:
        signatures = Counter(
            str(row.get("failure_signature") or row.get("status") or "unknown")
            for row in results
        )
        summary = ", ".join(
            f"{name} ({count})" for name, count in signatures.most_common(3)
        )
        raise GuidanceSolverCompatibilityError(
            f"all guided candidates failed for {context}; target solver is not "
            f"compatible with the materialized scoring contract: {summary}"
        )
    failure_reward = max(
        DEFAULT_GUIDANCE_FAILURE_REWARD,
        max(valid) + max(1.0, abs(max(valid))),
    )
    return [
        float(row["score_larger_is_harder"])
        if row.get("accepted") is True
        else float(failure_reward)
        for row in results
    ]
