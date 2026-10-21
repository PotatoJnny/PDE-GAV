"""Create the customer-visible result from a completed NS campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--loop-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args(argv)
    run_root = Path(args.run_root).resolve()
    loop_root = Path(args.loop_root).resolve()
    report_root = Path(args.report_root).resolve()
    final_round = int(args.rounds) - 1
    source_solver = loop_root / "rounds" / f"round_{final_round:03d}" / "submission.py"
    if not source_solver.is_file():
        raise SystemExit(f"missing final NS solver: {source_solver}")
    solver_root = run_root / "result" / "solver"
    solver_root.mkdir(parents=True, exist_ok=True)
    final_solver = solver_root / "submission.py"
    shutil.copy2(source_solver, final_solver)

    loop_summary = _read(loop_root / "final_summary.json")
    structured_report = _read(report_root / "structured_guided_5round_report.json")
    perturbation_report = _read(
        report_root / "structured_guided_perturbed_5round_report.json"
    )
    final_round_summary = loop_summary["rounds"][-1]
    result = {
        "kind": "solver",
        "problem": "Navier--Stokes",
        "name": "Selected NS solver",
        "solver": str(final_solver),
        "usage": "from submission import solve_case; solve_case(case_dir, query_grid=None)",
        "rounds_completed": int(loop_summary["rounds_completed"]),
        "primary_error": _primary_error(final_round_summary),
        "memory": str(loop_root / "agent_memory_compact.json"),
        "round_trace": str(loop_root / "round_trace.jsonl"),
        "reports": {
            "solver_evolution": str(
                report_root / "structured_guided_5round_report.json"
            ),
            "perturbations": str(
                report_root / "structured_guided_perturbed_5round_report.json"
            ),
        },
        "rounds": structured_report.get("rounds", []),
        "perturbation_summary": {
            "complete": perturbation_report.get("engineering_complete") is True,
            "rounds": len(perturbation_report.get("rounds", [])),
            "policy": perturbation_report.get("policy", {}),
        },
    }
    output = run_root / "result" / "result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def _primary_error(round_summary: dict) -> float | None:
    for key in (
        "hidden_all_summary",
        "hidden_generated_cumulative_summary",
        "hidden_holdout_summary",
        "cumulative_generated_summary",
        "generated_summary",
        "hidden_holdout_solver_metrics",
        "generated_solver_metrics",
    ):
        payload = round_summary.get(key) or {}
        summary = payload.get("summary") or payload
        for metric in ("mean_relative_l2", "median_relative_l2"):
            value = summary.get(metric)
            if value is not None:
                return float(value)
    return None


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


if __name__ == "__main__":
    main()
