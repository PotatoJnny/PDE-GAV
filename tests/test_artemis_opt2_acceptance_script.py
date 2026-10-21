"""Lightweight script/summary checks using synthetic JSON, not model evidence."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/slurm_artemis_backend_opt2_acceptance.sh"
CANDIDATES = [
    {"name": "S1", "tree_sha256": "a85878d17af49f15c2de7a5c6b1f617d805bfde9247a4f4ef184f538cdc2f339"},
    {"name": "S2", "tree_sha256": "07540f9afad42fd504bf99fae71fbaafdec8510038921187ed184c9a72ad24c0"},
]


def _summary_source():
    return SCRIPT.read_text().split("<<'PY' || ARTEMIS_STATUS=1\n", 1)[1].split("\nPY\n", 1)[0]


def _junit_source():
    return SCRIPT.read_text().split("<<'JUNIT' || ARTEMIS_GATE_STATUS=1\n", 1)[1].split("\nJUNIT\n", 1)[0]


def test_script_has_valid_shell_and_bounded_offline_matrix():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True, capture_output=True)
    text = SCRIPT.read_text()
    for required in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=8",
                     "#SBATCH --time=01:00:00", "private-random native-end",
                     "baseline optimized baseline_repeat", "--parents 2 --seed 2026090703 --point-count 32",
                     "${#ARTEMIS_NEW_TESTS[@]} -eq 0"):
        assert required in text
    for forbidden in ("--auth-file", "run_canonical_codex_agent.py", "scancel ", "sbatch "):
        assert forbidden not in text
    compile(_summary_source(), str(SCRIPT), "exec")
    compile(_junit_source(), str(SCRIPT), "exec")
    assert text.index('artemis_step junit-gate') < text.index('ARTEMIS_STATUS="$ARTEMIS_GATE_STATUS"')


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _matrix(root):
    for horizon in ("private-random", "native-end"):
        for mode, elapsed in (("baseline", 10.0), ("optimized", 5.0), ("baseline_repeat", 10.1)):
            run = {"status": "completed", "evaluation_wall_seconds": elapsed,
                   "started_at_unix": 100.0, "finished_at_unix": 100.0 + elapsed,
                   "candidates": CANDIDATES, "cohort": [{"key": f"case-{i}"} for i in range(4)],
                   "cases": [{"candidate": candidate["name"], "case_key": f"case-{i}",
                              "status": "completed", "public_sessions": [{"access": "public_problem",
                              "methods": {"query_source": {"calls": 1, "points": 32, "failures": 0}}}]}
                             for candidate in CANDIDATES for i in range(4)]}
            _write(root / f"{horizon}-{mode}/result.json", run)
        for mode in ("optimized", "baseline_repeat"):
            _write(root / f"compare-{horizon}-{mode}/comparison.json", {"status": "equivalent"})


def _summarize(root, *, gate_status=0, step_status=0):
    result = subprocess.run([sys.executable, "-", str(root), str(gate_status), str(step_status)],
                            input=_summary_source(), text=True, capture_output=True)
    return result, json.loads((root / "acceptance-summary.json").read_text())


def test_summary_reports_repeat_drift_and_only_validated_speedup(tmp_path):
    _matrix(tmp_path)
    result, report = _summarize(tmp_path)
    assert result.returncode == 0, result.stderr
    assert report["status"] == "passed"
    for row in report["horizons"].values():
        assert row["evaluation_wall_speedup"] == 2.0
        assert row["baseline_repeat_eval_drift_percent"] == pytest.approx(1.0)


@pytest.mark.parametrize("failure", ["case_failed", "duplicate_case", "workload_changed", "comparison_failed"])
def test_summary_preserves_failure_and_withholds_speedup(tmp_path, failure):
    _matrix(tmp_path)
    path = tmp_path / "private-random-optimized/result.json"
    run = json.loads(path.read_text())
    if failure == "case_failed":
        run["cases"][0]["status"] = "failed"
    elif failure == "duplicate_case":
        run["cases"][1] = run["cases"][0]
    elif failure == "workload_changed":
        run["cases"][0]["public_sessions"][0]["methods"]["query_source"]["points"] = 31
    else:
        _write(tmp_path / "compare-private-random-optimized/comparison.json", {"status": "not_equivalent"})
    _write(path, run)
    result, report = _summarize(tmp_path)
    assert result.returncode == 1
    assert report["status"] == "failed" and report["problems"]
    assert all("evaluation_wall_speedup" not in row for row in report["horizons"].values())
    assert json.loads(path.read_text()) == run  # Failure artifacts were not rewritten.


def test_summary_records_gate_failure_without_reading_missing_matrix(tmp_path):
    result, report = _summarize(tmp_path, gate_status=1, step_status=1)
    assert result.returncode == 1
    assert report["status"] == "failed"
    assert report["horizons"] == {} and report["gate_status"] == 1


def _junit_reports(root, failure=None):
    for filename in ("physical-tests.xml", "deployment-tests.xml", "runner-tests.xml"):
        active = failure if filename == "physical-tests.xml" else None
        if active == "missing":
            continue
        if active == "malformed":
            (root / filename).write_text("not an XML report")
            continue
        counts = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
        child = ""
        if active in ("failure", "error", "skipped"):
            key = {"failure": "failures", "error": "errors", "skipped": "skipped"}[active]
            counts[key] = 1
            child = f'<{active} message="synthetic fixture outcome" />'
        case = f'<testcase name="synthetic_example">{child}</testcase>'
        if active == "empty":
            counts["tests"], case = 0, ""
        elif active == "miscount":
            counts["tests"] = 2
        attributes = " ".join(f'{key}="{value}"' for key, value in counts.items())
        (root / filename).write_text(f"<testsuites><testsuite {attributes}>{case}</testsuite></testsuites>")


def _check_junit(root):
    result = subprocess.run([sys.executable, "-", str(root)], input=_junit_source(),
                            text=True, capture_output=True)
    return result, json.loads((root / "junit-counts.json").read_text())


def test_junit_gate_records_nonzero_counts_when_every_selected_test_passes(tmp_path):
    _junit_reports(tmp_path)
    result, report = _check_junit(tmp_path)
    assert result.returncode == 0, result.stderr
    assert report["status"] == "passed"
    assert report["totals"] == {"tests": 3, "failures": 0, "errors": 0, "skipped": 0}
    assert all(row["passed"] == 1 for row in report["reports"].values())


@pytest.mark.parametrize("failure", ["failure", "error", "skipped", "empty", "missing", "malformed", "miscount"])
def test_junit_gate_rejects_failures_skips_empty_or_invalid_reports(tmp_path, failure):
    _junit_reports(tmp_path, failure)
    result, report = _check_junit(tmp_path)
    assert result.returncode == 1
    assert report["status"] == "failed" and report["problems"]
    assert report["reports"]["physical-tests.xml"]["status"] == "failed"
    assert report["reports"]["deployment-tests.xml"]["status"] == "passed"
    assert report["reports"]["runner-tests.xml"]["status"] == "passed"
