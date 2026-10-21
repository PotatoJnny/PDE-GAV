"""Real CLI boundaries: numerical NS bundles -> shared CSV -> paper exports.

Fixtures are manufactured fields and local Python programs. Resource records
below are synthetic schema fixtures, not measurements of an LLM campaign.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pde_agent_verifier_solver.sdk import write_candidate_bundle_manifest


REPOSITORY = Path(__file__).resolve().parents[1]


def _cli(script: str, *arguments, temporary_root: Path, expected_returncode: int = 0):
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY / "src"),
        "MPLCONFIGDIR": str(temporary_root / "matplotlib"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / script), *map(str, arguments)],
        cwd=temporary_root, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == expected_returncode, result.stdout + "\n" + result.stderr
    return result


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _comparison(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    axis = np.array([0.0, 1.0])
    t, x, y = np.meshgrid(axis, axis, axis, indexing="ij")
    for index in range(2):
        case = data / f"case-{index}"
        case.mkdir()
        scale = index + 1
        _write(case / "public_question.json", {"time": {"t0": 0.0, "t1": 1.0}})
        _write(case / "case_spec.json", {"parameters": {"scale": scale}})
        np.savez(case / "public_arrays.npz", x=axis, y=axis,
                 fluid_mask=np.ones((2, 2), dtype=bool))
        np.savez(case / "hidden_truth_cache.npz",
                 u_true=np.stack(((1 + t + x) * scale, (2 + t + y) * scale), axis=1),
                 pressure=np.zeros((2, 2, 2)), t=axis)
        rows.append({"case_id": case.name, "case_dir": case.name,
                     "parent_case_id": f"root-{index}", "recording_id": "recording-a"})
    dataset = data / "manifest.json"
    _write(dataset, {"schema": "ns_d0_v1", "splits": {"external_test": rows}})
    source = """from pde_agent_verifier_solver.sdk import AnswerView, QueryKind
class Handle:
    def __init__(self, scale):
        self.scale = scale
    def evaluate(self, query):
        return AnswerView(QueryKind.POINTS, {
            "velocity_x": tuple((1 + p["t"] + p["x"]) * self.scale for p in query.points),
            "velocity_y": tuple((2 + p["t"] + p["y"]) * self.scale for p in query.points),
        }, points=query.points, shape=(len(query.points),))
def prepare_solution(case, solve_budget=None):
    assert not hasattr(case, "truth")
    assert "u_true" not in case.public_data["arrays"]
    return Handle(FACTOR * case.public_data["case_parameters"]["scale"])
"""
    programs = []
    for method, role, factor, seed in (("S0", "initial", 0, 0), ("final", "final", 1, 17)):
        bundle = tmp_path / method
        bundle.mkdir()
        (bundle / "submission.py").write_text(source.replace("FACTOR", str(factor)))
        write_candidate_bundle_manifest(bundle, files=("submission.py",))
        programs.append({"method": method, "role": role, "seed": seed,
                         "run_id": "shared-initial" if method == "S0" else "paper-final-17",
                         "artifact_id": method, "bundle": str(bundle)})
    spec = {
        "schema": "ns_common_evaluation_v1", "task_id": "ns-interface-test",
        "datasets": [{"dataset_id": "external", "manifest": str(dataset),
                      "split": "external_test", "cohort": "held_out"}],
        "programs": programs, "query": {"point_count": 5, "seed": 19},
        "budget": {"max_workers": 2, "case_hard_wall_seconds": 30},
    }
    manifest = tmp_path / "comparison.json"
    _write(manifest, spec)
    return manifest, spec


def _assert_exports(directory):
    assert (directory / "comparison.pdf").read_bytes().startswith(b"%PDF-")
    table = (directory / "table.tex").read_text()
    assert "\\begin{tabular}" in table and "\\end{tabular}" in table
    assert (directory / "summary.json").is_file()


def test_common_evaluation_csv_to_paper_cli_and_cost_join(tmp_path):
    manifest, _spec = _comparison(tmp_path)
    evaluation = tmp_path / "common-evaluation"
    _cli("evaluate_ns_programs.py", "--manifest", manifest, "--output", evaluation,
         temporary_root=tmp_path)
    with (evaluation / "cases.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert {
            "task_id", "method", "run_id", "seed", "artifact_id", "artifact_sha256",
            "cohort", "dataset_id", "case_id", "parent_case_id", "recording_id",
            "metric", "error", "status", "plan_sha256", "reference_sha256",
            "public_case_sha256", "solve_wall_seconds", "query_wall_seconds", "case_wall_seconds",
        } <= set(reader.fieldnames)
        rows = list(reader)
    assert len(rows) == 12 and {row["method"] for row in rows} == {"S0", "final"}
    assert all(row["status"] == "ok" for row in rows)
    for case_id in {row["case_id"] for row in rows}:
        shared = [row for row in rows if row["case_id"] == case_id]
        for identity in ("plan_sha256", "reference_sha256", "public_case_sha256"):
            assert len({row[identity] for row in shared}) == 1
            assert len(shared[0][identity]) == 64

    paper = tmp_path / "paper"
    common_arguments = ["--cases", evaluation / "cases.csv", "--methods", "final",
                        "--bootstrap-replicates", "100", "--threshold", "velocity_relative_l2=0.001"]
    _cli("summarize_comparison.py", *common_arguments, "--output", paper, "--plots",
         temporary_root=tmp_path)
    _assert_exports(paper)
    report = json.loads((paper / "summary.json").read_text())
    assert len(report["groups"]) == 3
    for group in report["groups"]:
        assert group["status"] == "ok" and group["n_independent_campaigns"] == 1
        assert group["n_test_clusters"] == 1  # The fixture declares one recording.
        assert len(group["methods"]) == 1
        method = group["methods"][0]
        assert method["method"] == "final"
        assert method["n_requested"] == 2 and method["n_failed"] == 0
        assert method["mean_error_all_cases"] == pytest.approx(0, abs=1e-14)
        assert method["execution_rate_ci95"] is None
        assert method["accuracy_pass_rate_ci95"] is None
        assert method["mean_error_ci95"] is None
    assert "S0" not in (paper / "table.tex").read_text()

    # These synthetic resource values test an explicit identity join. They do
    # not infer LLM cost from candidate timing or from the configured budget.
    costs = {
        "schema": "run_resource_report_v1", "run_id": "paper-final-17",
        "usage_coverage": "completed_turns_observed",
        "tokens": {"input_tokens": 120, "cached_input_tokens": 40, "output_tokens": 30},
        "agent_active_seconds_sum": 12.0, "agent_wall_seconds_sum": 30.0,
        "observed_activity_span_seconds": 15.0,
        "observed_llm_cost": 0.012,
    }
    _write(tmp_path / "cost-report.json", costs)
    cost_manifest = tmp_path / "cost-manifest.json"
    _write(cost_manifest, [{"method": "final", "run_id": "paper-final-17", "seed": 17,
                            "costs": "cost-report.json"}])
    cost_paper = tmp_path / "paper-with-costs"
    _cli("summarize_comparison.py", *common_arguments, "--output", cost_paper,
         "--cost-manifest", cost_manifest, "--plots", temporary_root=tmp_path)
    _assert_exports(cost_paper)
    joined = json.loads((cost_paper / "summary.json").read_text())
    assert len(joined["quality_cost_rows"]) == 3
    for row in joined["quality_cost_rows"]:
        assert row["method"] == "final" and row["campaigns"] == 1
        assert row["input_plus_output_tokens"] == 150  # Cached input is already included.
        assert row["agent_active_seconds_sum"] == 12.0
        assert row["agent_wall_seconds_sum"] == 30.0
        assert row["observed_activity_span_seconds"] == 15.0
        assert row["observed_llm_cost"] == pytest.approx(0.012)
        assert row["execution_rate"] == 1.0
        assert row["mean_error_all_cases"] == pytest.approx(0, abs=1e-14)
    with (cost_paper / "quality_cost.csv").open(newline="") as stream:
        exported_costs = list(csv.DictReader(stream))
    assert len(exported_costs) == 3
    assert all(float(row["input_plus_output_tokens"]) == 150 for row in exported_costs)

    costs["run_id"] = "another-campaign"
    _write(tmp_path / "cost-report.json", costs)
    rejected = _cli("summarize_comparison.py", *common_arguments,
                    "--output", tmp_path / "mismatched-costs", "--cost-manifest", cost_manifest,
                    temporary_root=tmp_path, expected_returncode=1)
    assert "cost report run_id differs" in rejected.stderr
    assert not (tmp_path / "mismatched-costs").exists()


def test_reference_failure_survives_both_clis_without_becoming_agent_error(tmp_path):
    manifest, _spec = _comparison(tmp_path)
    (tmp_path / "data" / "case-1" / "hidden_truth_cache.npz").unlink()
    evaluation = tmp_path / "missing-reference"
    _cli("evaluate_ns_programs.py", "--manifest", manifest, "--output", evaluation,
         temporary_root=tmp_path, expected_returncode=1)
    with (evaluation / "cases.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    failed = [row for row in rows if row["case_id"] == "case-1"]
    assert len(failed) == 6
    assert all(row["status"] == "failed" and not row["error"]
               and not row["reference_sha256"] for row in failed)
    assert all(row["public_case_sha256"] for row in failed)
    paper = tmp_path / "invalid-paper"
    _cli("summarize_comparison.py", "--cases", evaluation / "cases.csv", "--methods", "final",
         "--output", paper, "--plots", "--bootstrap-replicates", "100",
         temporary_root=tmp_path, expected_returncode=1)
    _assert_exports(paper)
    report = json.loads((paper / "summary.json").read_text())
    assert len(report["groups"]) == 3
    for group in report["groups"]:
        assert group["status"] == "invalid_evaluation"
        assert group["methods"] == [] and group["paired_comparisons"] == []
        assert group["n_requested_rows"] == 2
        assert group["evaluation_failures"][0]["case_id"] == "case-1"
    assert (paper / "invalid_evaluations.json").is_file()
