"""Synthetic report fixtures test acceptance failure handling, not model quality."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/slurm_artemis_backend_cold_acceptance_20260908.sh"
SPEC = importlib.util.spec_from_file_location("artemis_cold_acceptance", ROOT / "scripts/artemis_backend_cold_acceptance.py")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def test_shell_is_bounded_and_uses_frozen_offline_runner_only():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True, capture_output=True)
    text = SCRIPT.read_text()
    for required in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=64G",
                     "#SBATCH --time=02:00:00", "artemis-cold-baseline-20260908",
                     "--parents 2 --seed 2026090801 --point-count 32 --horizon native-end",
                     "--rtol 1e-10 --atol 1e-12", "trap artemis_finish EXIT", "set -euo pipefail",
                     "--validate-only", 'tee "$ARTEMIS_OUTPUT/$ARTEMIS_STEP.log"'):
        assert required in text
    for forbidden in ("--auth-file", "run_canonical_codex_agent.py", "sbatch ", "scancel ", "pip install"):
        assert forbidden not in text
    assert text.index("artemis_step junit-gate") < text.index("for ARTEMIS_BATCH in 0 1")
    assert len(HELPER.CANDIDATES) == len({item[2] for item in HELPER.CANDIDATES}) == 6
    assert all(".controller-candidates" in str(HELPER.candidate_path(item)) for item in HELPER.CANDIDATES)


def _write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _junit(root, failure=None):
    for filename in ("deployment-tests.xml", "runner-tests.xml"):
        active = failure if filename == "deployment-tests.xml" else None
        if active == "missing":
            continue
        counts = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
        child = ""
        if active in ("failure", "error", "skipped"):
            counts[{"failure": "failures", "error": "errors", "skipped": "skipped"}[active]] = 1
            child = f"<{active} />"
        case = f"<testcase classname='test_fixture' name='fixture'>{child}</testcase>"
        if active == "empty":
            counts["tests"], case = 0, ""
        elif active == "miscount":
            counts["tests"] = 2
        attributes = " ".join(f'{key}="{value}"' for key, value in counts.items())
        (root / filename).write_text(f"<testsuites><testsuite {attributes}>{case}</testsuite></testsuites>")
        _write(root, str(Path(filename).with_suffix(".events.json")), _events())


def _events(*, subtests=0):
    nodeid = "test_fixture.py::fixture"
    return {"schema": "artemis_pytest_events_v1", "exit_status": 0, "collection_finished": True,
            "collection_failures": [], "collected_count": 1,
            "collected": [{"nodeid": nodeid, "classname": "test_fixture", "name": "fixture"}],
            "reports": [{"nodeid": nodeid, "when": phase, "outcome": "passed", "subtest": False,
                         "report_type": "_pytest.reports.TestReport"} for phase in ("setup", "call", "teardown")]
                       + [{"nodeid": nodeid, "when": "call", "outcome": "passed", "subtest": True,
                           "report_type": "_pytest.subtests.SubtestReport"} for _ in range(subtests)]}


def test_junit_accepts_complete_nonempty_passing_reports(tmp_path):
    _junit(tmp_path)
    report = HELPER.junit_gate(tmp_path)
    assert report["status"] == "passed"
    assert report["totals"] == {"tests": 2, "failures": 0, "errors": 0, "skipped": 0}


def test_junit_reconciles_only_recorded_passing_subtest_events(tmp_path):
    _junit(tmp_path)
    xml = tmp_path / "deployment-tests.xml"
    xml.write_text(xml.read_text().replace('tests="1"', 'tests="14"'))
    _write(tmp_path, "deployment-tests.events.json", _events(subtests=13))
    report = HELPER.junit_gate(tmp_path)
    assert report["status"] == "passed", report["problems"]
    assert report["reports"]["deployment-tests.xml"]["passed_subtests"] == 13
    assert report["reports"]["deployment-tests.xml"]["observed"]["tests"] == 1


@pytest.mark.parametrize("failure", ["missing_sidecar", "unrecorded_subtest", "wrong_identity", "missing_testcase",
                                     "missing_phase", "duplicate_phase", "skipped_subtest", "failed_setup",
                                     "unknown_subtest_type", "collection_failed", "collection_incomplete"])
def test_junit_subtest_support_still_rejects_missing_or_failed_tests(tmp_path, failure):
    _junit(tmp_path)
    path = tmp_path / "deployment-tests.events.json"
    events = _events(subtests=1)
    xml = tmp_path / "deployment-tests.xml"
    xml.write_text(xml.read_text().replace('tests="1"', 'tests="2"'))
    if failure == "unrecorded_subtest":
        events["reports"].pop()
    elif failure == "wrong_identity":
        events["collected"][0]["name"] = "different_fixture"
    elif failure == "missing_testcase":
        xml.write_text('<testsuite tests="2" errors="0" failures="0" skipped="0" />')
    elif failure == "missing_phase":
        events["reports"].pop(0)
    elif failure == "duplicate_phase":
        events["reports"].append(dict(events["reports"][0]))
    elif failure == "skipped_subtest":
        events["reports"][-1]["outcome"] = "skipped"
    elif failure == "failed_setup":
        events["reports"][0]["outcome"] = "failed"
    elif failure == "unknown_subtest_type":
        events["reports"][-1]["report_type"] = "unsupported.SubtestReport"
    elif failure == "collection_failed":
        events["collection_failures"] = [{"nodeid": "missing.py", "outcome": "failed"}]
    elif failure == "collection_incomplete":
        events["collection_finished"] = False
    _write(tmp_path, "deployment-tests.events.json", events)
    if failure == "missing_sidecar":
        path.unlink()
    report = HELPER.junit_gate(tmp_path)
    assert report["status"] == "failed" and report["problems"]


@pytest.mark.parametrize("failure", ["failure", "error", "skipped", "empty", "miscount", "missing"])
def test_junit_rejects_invalid_or_skipped_reports(tmp_path, failure):
    _junit(tmp_path, failure)
    report = HELPER.junit_gate(tmp_path)
    assert report["status"] == "failed" and report["problems"]


def _matrix(root, monkeypatch):
    sources = {"sources": {"canonical": {"commit": "test-commit"}},
               "config_sha256": {"canonical": "test-backend-sha"}}
    monkeypatch.setattr(HELPER, "source_state", lambda: sources)
    _write(root, "source-state.json", sources)
    _write(root, "junit-counts.json", {"status": "passed"})
    steps = ["source-preflight", "deployment-tests", "runner-tests", "junit-gate", "formal-config", "split-execution"]
    _write(root, "split-execution/acceptance-summary.json", {
        "status": "passed", "completed_evaluations": 8, "canonical_commit": "test-commit",
        "backend_config_sha256": "test-backend-sha"})
    for batch in range(2):
        candidates = [{"name": name, "tree_sha256": digest}
                      for name, _, digest in HELPER.CANDIDATES[batch * 3:batch * 3 + 3]]
        cohort = [{"key": f"pair-{pair:03d}-{distribution}"}
                  for pair in range(2) for distribution in ("base", "base_perturbed")]
        for mode, elapsed in (("baseline", 20.0), ("optimized", 5.0)):
            run = {"status": "completed", "master_seed": 2026090801, "parents": 2,
                   "point_count": 32, "candidate_compute_budget": 600.0,
                   "horizon_policy": "native-end", "evaluation_phase": "development",
                   "evaluation_wall_seconds": elapsed, "candidates": candidates, "cohort": cohort,
                   "cases": [{"candidate": candidate["name"], "case_key": case["key"], "status": "completed",
                              "public_sessions": [{"access": "public_problem", "methods": {
                                  "query_source": {"calls": 1, "points": 32, "failures": 0}}}]}
                             for candidate in candidates for case in cohort]}
            _write(root, f"batch-{batch}-{mode}/result.json", run)
            steps.append(f"batch-{batch}-{mode}")
        _write(root, f"batch-{batch}-compare/comparison.json",
               {"status": "equivalent", "problems": [], "rtol": 1e-10, "atol": 1e-12})
        steps.append(f"batch-{batch}-compare")
    (root / "step-exit-codes.tsv").write_text("".join(f"{step}\t0\n" for step in steps))


def test_summary_records_exact_provenance_and_24_evaluations_per_arm(tmp_path, monkeypatch):
    _matrix(tmp_path, monkeypatch)
    report = HELPER.summarize(tmp_path, 0)
    assert report["status"] == "passed", report["problems"]
    assert report["canonical_commit"] == "test-commit"
    assert report["backend_config_sha256"] == "test-backend-sha"
    assert report["completed_evaluations_per_arm"] == {"baseline": 24, "optimized": 24}
    assert report["evaluation_wall_speedup"] == 4.0


@pytest.mark.parametrize("failure", ["duplicate", "failed", "candidate_hash", "workload", "query_failure",
                                     "budget", "comparison", "tolerance", "source_changed", "missing_step", "split", "exit"])
def test_summary_fails_closed_and_withholds_speedup(tmp_path, monkeypatch, failure):
    _matrix(tmp_path, monkeypatch)
    path = tmp_path / "batch-0-optimized/result.json"
    run = json.loads(path.read_text())
    if failure == "duplicate":
        run["cases"][1] = run["cases"][0]
    elif failure == "failed":
        run["cases"][0]["status"] = "failed"
    elif failure == "candidate_hash":
        run["candidates"][0]["tree_sha256"] = "changed"
    elif failure in ("workload", "query_failure"):
        values = run["cases"][0]["public_sessions"][0]["methods"]["query_source"]
        values["points" if failure == "workload" else "failures"] = 1
    elif failure == "budget":
        run["candidate_compute_budget"] = 599
    elif failure in ("comparison", "tolerance"):
        report = {"status": "not_equivalent" if failure == "comparison" else "equivalent",
                  "problems": [], "rtol": 1e-5 if failure == "tolerance" else 1e-10, "atol": 1e-12}
        _write(tmp_path, "batch-0-compare/comparison.json", report)
    elif failure == "source_changed":
        monkeypatch.setattr(HELPER, "source_state", lambda: {})
    elif failure == "missing_step":
        (tmp_path / "step-exit-codes.tsv").write_text("source-preflight\t0\n")
    elif failure == "split":
        _write(tmp_path, "split-execution/acceptance-summary.json", {"status": "failed"})
    _write(tmp_path, "batch-0-optimized/result.json", run)
    report = HELPER.summarize(tmp_path, int(failure == "exit"))
    assert report["status"] == "failed" and report["problems"]
    assert "evaluation_wall_speedup" not in report
    assert json.loads(path.read_text()) == run


def test_summary_preserves_missing_preflight_as_failure(tmp_path):
    report = HELPER.summarize(tmp_path, 1)
    assert report["status"] == "failed" and report["problems"]
    assert report["canonical_commit"] is None
