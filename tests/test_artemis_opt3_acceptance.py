"""Acceptance-report tests use synthetic fixtures, never performance evidence."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("opt3_acceptance", ROOT / "scripts/artemis_opt3_acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def fixtures(root, monkeypatch):
    state = {"driver": {"commit": "fixture"}}
    monkeypatch.setattr(acceptance, "sources", lambda: state)
    acceptance.write(root / "source-state.json", state)
    for name in ("physical-tests.xml", "deployment-tests.xml", "runner-tests.xml"):
        (root / name).write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="fixture"/></testsuite></testsuites>')
    for name in ("compare-native-optimized", "compare-native-baseline_repeat",
                 "compare-micro-8-optimized", "compare-micro-8-baseline_repeat", "compare-micro-1-optimized"):
        acceptance.write(root / name / "comparison.json", {"status": "equivalent", "public_workload_identical": True})
    for mode, elapsed in (("baseline", 10), ("optimized", 5), ("baseline_repeat", 11)):
        acceptance.write(root / f"native-{mode}" / "result.json", {
            "status": "completed", "evaluation_wall_seconds": elapsed,
            "cases": [{"status": "completed"} for _ in range(8)]})
        acceptance.write(root / f"native-{mode}" / "workload-summary.json", {"frozen_identity_verified": True})
    acceptance.write(root / "wire.json", {"status": "passed",
        "filesystems": {"scratch": {"status": "passed"}, "local": {"status": "passed"}}})


def test_summary_requires_all_numerical_workloads_then_reports_repeat_drift(tmp_path, monkeypatch):
    fixtures(tmp_path, monkeypatch)
    result = acceptance.summarize(tmp_path, 0)
    assert result["status"] == "passed"
    assert result["candidate_evaluations"] == 24
    assert result["candidate_evaluation_speed_ratio"] == 2
    assert result["baseline_repeat_drift_percent"] == pytest.approx(10)
    assert result["full_iteration_speedup_claimed"] is False


@pytest.mark.parametrize("failure", ["exit", "source", "numeric", "workload", "candidate", "count", "wire", "time", "skip"])
def test_failed_acceptance_withholds_speed_claims(tmp_path, monkeypatch, failure):
    fixtures(tmp_path, monkeypatch)
    code = 0
    if failure == "exit":
        code = 1
    elif failure == "source":
        monkeypatch.setattr(acceptance, "sources", lambda: {"driver": {"commit": "changed"}})
    elif failure in ("numeric", "workload"):
        acceptance.write(tmp_path / "compare-native-optimized/comparison.json",
            {"status": "not_equivalent" if failure == "numeric" else "equivalent",
             "public_workload_identical": failure != "workload"})
    elif failure == "candidate":
        acceptance.write(tmp_path / "native-optimized/workload-summary.json", {"frozen_identity_verified": False})
    elif failure in ("count", "time"):
        path = tmp_path / "native-optimized/result.json"
        data = acceptance.read(path)
        if failure == "count":
            data["cases"].pop()
        else:
            data["evaluation_wall_seconds"] = 0
        acceptance.write(path, data)
    elif failure == "wire":
        acceptance.write(tmp_path / "wire.json", {"status": "passed", "filesystems": {"local": {"status": "skipped"}}})
    elif failure == "skip":
        (tmp_path / "physical-tests.xml").write_text('<testsuite tests="1" skipped="1"><testcase><skipped/></testcase></testsuite>')
    result = acceptance.summarize(tmp_path, code)
    assert result["status"] == "failed"
    assert "candidate_evaluation_speed_ratio" not in result
    assert result["error"]


def test_slurm_script_is_offline_bounded_and_checks_numerics():
    path = ROOT / "scripts/slurm_artemis_backend_opt3_acceptance.sh"
    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
    text = path.read_text()
    for required in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --time=01:00:00",
                     "baseline optimized baseline_repeat", "--seed 2026090903",
                     "test_vp_1d1v_artemis_frozen_profile.py", "compare-micro", "junit-gate", "trap artemis_finish EXIT"):
        assert required in text
    for forbidden in ("scancel ", "sbatch ", "--auth-file", "run_canonical_codex_agent.py"):
        assert forbidden not in text
    assert text.index("junit-gate") < text.index("artemis_step wire")
