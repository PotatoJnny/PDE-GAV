from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.ns.common_evaluation import (
    load_numerical_split, run_ns_common_evaluation,
)
from pde_agent_verifier_solver.sdk import CaseVisibility, write_candidate_bundle_manifest


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _dataset(root: Path):
    root.mkdir()
    rows = []
    for index in range(2):
        case = root / f"case-{index}"
        case.mkdir()
        _write(case / "public_question.json", {"time": {"t0": 0.0, "t1": 1.0}})
        _write(case / "case_spec.json", {"parameters": {"scale": index + 1}})
        axis = np.array([0.0, 1.0])
        np.savez(case / "public_arrays.npz", x=axis, y=axis,
                 fluid_mask=np.ones((2, 2), dtype=bool))
        t, x, y = np.meshgrid(axis, axis, axis, indexing="ij")
        velocity = np.stack(((1 + t + x) * (index + 1), (2 + t + y) * (index + 1)), axis=1)
        np.savez(case / "hidden_truth_cache.npz", u_true=velocity,
                 pressure=np.ones((2, 2, 2)), t=axis)
        rows.append({"case_id": case.name, "case_dir": case.name,
                     "parent_case_id": f"parent-{index}", "recording_id": "recording-a"})
    manifest = root / "manifest.json"
    _write(manifest, {"schema": "ns_d0_v1", "splits": {"external_test": rows}})
    return manifest


def _bundle(root: Path, *, zero=False, fail_second=False, sleep=False):
    root.mkdir()
    source = '''from pde_agent_verifier_solver.sdk import AnswerView, QueryKind
import time
class Handle:
    def __init__(self, scale):
        self.scale = scale
    def evaluate(self, query):
        values = {"velocity_x": [], "velocity_y": []}
        for point in query.points:
            values["velocity_x"].append((1 + point["t"] + point["x"]) * self.scale)
            values["velocity_y"].append((2 + point["t"] + point["y"]) * self.scale)
        if "pressure" in query.fields:
            values["pressure"] = [1.0] * len(query.points)
        return AnswerView(QueryKind.POINTS, {key: tuple(value) for key, value in values.items()},
                          points=query.points, shape=(len(query.points),))
def prepare_solution(case, solve_budget=None):
    assert not hasattr(case, "truth")
    assert "case_dir" not in case.public_data
    assert "u_true" not in case.public_data["arrays"]
    assert "hidden_truth_cache" not in repr(case.public_data)
    scale = case.public_data["case_parameters"]["scale"]
    FAIL_SECOND
    SLEEP
    return Handle(SCALE)
'''.replace("FAIL_SECOND", 'if scale == 2: raise RuntimeError("intentional candidate failure")' if fail_second else "pass").replace("SLEEP", "time.sleep(10)" if sleep else "pass").replace("SCALE", "0" if zero else "scale")
    (root / "submission.py").write_text(source)
    write_candidate_bundle_manifest(root, files=("submission.py",))
    return root


def _manifest(tmp_path: Path, *, include_partial=True):
    dataset = _dataset(tmp_path / "data")
    initial = _bundle(tmp_path / "initial", zero=True)
    final = _bundle(tmp_path / "final")
    programs = [
        {"method": "S0", "run_id": "r1", "seed": 0, "artifact_id": "s0", "role": "initial", "bundle": str(initial)},
        {"method": "full", "run_id": "r1", "seed": 0, "artifact_id": "sr", "role": "final", "bundle": str(final)},
    ]
    if include_partial:
        partial = _bundle(tmp_path / "partial", fail_second=True)
        programs.append({"method": "partial", "run_id": "r2", "seed": 0,
                         "artifact_id": "sr-partial", "role": "final", "bundle": str(partial)})
    spec = {
        "schema": "ns_common_evaluation_v1", "task_id": "ns-test",
        "datasets": [{"dataset_id": "external", "manifest": str(dataset), "split": "external_test", "cohort": "held_out"}],
        "programs": programs, "query": {"point_count": 5, "seed": 19},
        "budget": {"max_workers": 2, "case_hard_wall_seconds": 30},
    }
    path = tmp_path / "comparison.json"
    _write(path, spec)
    return path, spec


def _rows(output):
    return [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]


def test_external_ns_common_test_frozen_programs_shared_plans_and_failures(tmp_path):
    manifest, spec = _manifest(tmp_path)
    output = tmp_path / "evaluation"
    summary = run_ns_common_evaluation(manifest, output)
    rows = _rows(output)
    assert len(rows) == 3 * 2 * 3
    assert summary["n_failed_program_cases"] == 1
    for case_id in ("case-0", "case-1"):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        assert len({row["plan_sha256"] for row in case_rows}) == 1
        assert len({row["reference_sha256"] for row in case_rows}) == 1
        assert all(row["parent_case_id"] and row["recording_id"] == "recording-a" for row in case_rows)
    initial = [row for row in rows if row["method"] == "S0"]
    assert all(row["error"] == pytest.approx(1.0) for row in initial)
    final = [row for row in rows if row["method"] == "full"]
    assert all(row["error"] == pytest.approx(0.0, abs=1e-14) for row in final)
    assert all(row["solve_wall_seconds"] >= 0 and row["query_wall_seconds"] >= 0 for row in final)
    # Finite-array cases perform no queued Oracle calls; zero is observed, not missing.
    assert all(row["solve_queue_wait_seconds"] == 0.0 for row in final)
    assert all(row["query_queue_wait_seconds"] == 0.0 for row in final)
    failed = [row for row in rows if row["status"] == "failed"]
    assert len(failed) == 3 and all(row["error"] is None and row["error_code"] for row in failed)
    assert all(row["solve_queue_wait_seconds"] is None for row in failed)
    assert all(row["query_queue_wait_seconds"] is None for row in failed)
    assert all(row["method"] == "partial" and row["case_id"] == "case-1" for row in failed)
    partial_summary = [group for group in summary["groups"] if group["method"] == "partial"]
    assert all(group["n_failed"] == 1 and group["success_rate"] == 0.5 for group in partial_summary)
    with (output / "cases.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows)
    for json_row, csv_row in zip(rows, csv_rows, strict=True):
        for name in ("solve_queue_wait_seconds", "query_queue_wait_seconds"):
            expected = "" if json_row[name] is None else str(json_row[name])
            assert csv_row[name] == expected
    audits = [json.loads(line) for line in (output / "execution" / "candidate_execution_audits.jsonl").read_text().splitlines()]
    assert len(audits) == 6
    assert all(audit["candidate_process_isolated"] and not audit["controller_truth_shared"] for audit in audits)
    resolved = json.loads((output / "resolved_manifest.json").read_text())
    assert all(program["artifact_sha256"] and Path(program["frozen_bundle"]).is_file() for program in resolved["programs"])
    # Editing the source after evaluation cannot alter its frozen delivery.
    Path(spec["programs"][0]["bundle"], "submission.py").write_text("raise RuntimeError('changed')")
    assert "changed" not in Path(resolved["programs"][0]["frozen_bundle"]).parent.joinpath("submission.py").read_text()
    with pytest.raises(FileExistsError):
        run_ns_common_evaluation(manifest, output)


def test_plan_replay_checks_reference_identity_before_execution(tmp_path):
    manifest, spec = _manifest(tmp_path, include_partial=False)
    first = tmp_path / "first"
    run_ns_common_evaluation(manifest, first)
    spec["query"]["plans_file"] = str(first / "plans.json")
    _write(manifest, spec)
    second = tmp_path / "second"
    run_ns_common_evaluation(manifest, second)
    assert json.loads((first / "plans.json").read_text()) == json.loads((second / "plans.json").read_text())
    for before, after in zip(_rows(first), _rows(second)):
        assert before["error"] == after["error"]
    truth = tmp_path / "data" / "case-0" / "hidden_truth_cache.npz"
    with np.load(truth) as source:
        arrays = {key: source[key].copy() for key in source.files}
    arrays["u_true"] *= 2
    np.savez(truth, **arrays)
    with pytest.raises(ValueError, match="reference_sha256"):
        run_ns_common_evaluation(manifest, tmp_path / "changed-reference")
    assert not (tmp_path / "changed-reference" / "execution" / "candidate_execution_audits.jsonl").exists()


def test_loader_subset_visibility_and_invalid_specs(tmp_path):
    manifest, spec = _manifest(tmp_path, include_partial=False)
    rows = load_numerical_split(spec["datasets"][0]["manifest"], split="external_test",
                                visibility=CaseVisibility.DEVELOPMENT, case_ids=["case-1"])
    assert len(rows) == 1 and rows[0].case.case_id == "case-1"
    assert rows[0].case.visibility is CaseVisibility.DEVELOPMENT
    assert "case_dir" not in rows[0].case.public_data
    spec["programs"] = spec["programs"][1:]
    spec["require_initial"] = True
    _write(manifest, spec)
    with pytest.raises(ValueError, match="S0"):
        run_ns_common_evaluation(manifest, tmp_path / "no-initial")
    assert not (tmp_path / "no-initial").exists()
    _write(manifest, {"schema": "pde_completed_case_dataset_v1", "problem": "ns"})
    with pytest.raises(ValueError, match="original numerical"):
        load_numerical_split(manifest, split="external_test")


def test_invalid_bundle_keeps_complete_failure_rows_and_private_diagnostic(tmp_path):
    manifest, spec = _manifest(tmp_path, include_partial=False)
    spec["programs"][0]["expected_tree_sha256"] = "0" * 64
    _write(manifest, spec)
    summary = run_ns_common_evaluation(manifest, tmp_path / "failures")
    rows = _rows(tmp_path / "failures")
    assert len(rows) == 12 and summary["n_failed_program_cases"] == 2
    assert all(row["status"] == "failed" and row["error"] is None
               for row in rows if row["method"] == "S0")
    diagnostic = json.loads((tmp_path / "failures" / "controller_private" /
                             "common_evaluation_failures.jsonl").read_text().splitlines()[0])
    assert diagnostic["stage"] == "bind_frozen_program"
    assert "frozen bundle digest differs" in diagnostic["traceback"]


def test_missing_truth_interrupts_common_evaluation_before_candidate_execution(tmp_path):
    from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError
    manifest, spec = _manifest(tmp_path, include_partial=False)
    (tmp_path / "data" / "case-1" / "hidden_truth_cache.npz").unlink()
    output = tmp_path / "missing-truth"
    with pytest.raises(CandidateExecutionInfrastructureError, match="reference"):
        run_ns_common_evaluation(manifest, output)
    plans = json.loads((output / "plans.json").read_text())["cases"]
    assert plans[1]["reference_status"] == "failed" and plans[1]["reference_sha256"] is None
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "interrupted"
    assert failure["stage"] == "prepare_reference"
    assert not (output / "summary.json").exists()
    assert not (output / "execution" / "candidate_execution_audits.jsonl").exists()
    diagnostics = (output / "controller_private" / "common_evaluation_failures.jsonl").read_text()
    assert "prepare_reference" in diagnostics and "hidden_truth_cache.npz" in diagnostics


def test_case_timeout_keeps_all_rows_and_cli_rejects_existing_output(tmp_path):
    import subprocess
    import sys

    manifest, spec = _manifest(tmp_path, include_partial=False)
    spec["budget"]["case_hard_wall_seconds"] = 0.001
    _write(manifest, spec)
    output = tmp_path / "timeout"
    summary = run_ns_common_evaluation(manifest, output)
    rows = _rows(output)
    assert summary["n_failed_program_cases"] == 4
    assert len(rows) == 12
    assert all(row["status"] == "failed" and row["error"] is None for row in rows)
    assert all(row["error_code"] == "candidate_case_timeout" for row in rows)
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_ns_programs.py"
    result = subprocess.run([sys.executable, str(script), "--manifest", str(manifest),
                             "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "FileExistsError" in result.stderr


def test_cold_start_comparison_without_initial_is_valid(tmp_path):
    manifest, spec = _manifest(tmp_path, include_partial=False)
    spec["programs"] = spec["programs"][1:]
    _write(manifest, spec)
    spec["query"]["include_pressure"] = True
    _write(manifest, spec)
    output = tmp_path / "cold-start"
    summary = run_ns_common_evaluation(manifest, output)
    assert summary["contains_initial"] is False
    assert summary["n_programs"] == 1 and summary["n_failed_program_cases"] == 0
    assert len(_rows(output)) == 8
    assert any(row["metric"] == "pressure_relative_l2" for row in _rows(output))
    assert all(row["method"] == "full" and row["status"] == "ok" for row in _rows(output))
