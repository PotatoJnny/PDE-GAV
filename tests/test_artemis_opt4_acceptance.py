"""Offline acceptance fixtures are not model or performance evidence."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import artemis_opt4_acceptance as a

ROOT = Path(__file__).resolve().parents[1]


def fixtures(root, monkeypatch):
    state = {"driver": {"commit": "fixture"}}
    monkeypatch.setattr(a, "sources", lambda: state)
    a.write(root / "source-state.json", state)
    a.write(root / "rpc-transport.json", {"status": "passed", "filesystems": {
        name: {"status": "passed", "variants": {"opt4_128k_trusted": {"measured_sidecar_count": 0}}}
        for name in ("scratch", "local")}})
    for name in ("physical-tests.xml", "deployment-tests.xml", "runner-tests.xml"):
        (root / name).write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="fixture"/></testsuite>')
    for name in a.COMPARISONS:
        a.write(root / name / "comparison.json", {"status": "equivalent", "public_workload_identical": True})
    for name, elapsed in (("baseline", 10), ("optimized", 5), ("baseline_repeat", 11)):
        a.write(root / f"native-{name}/result.json", {"status": "completed",
            "evaluation_wall_seconds": elapsed, "cases": [{"status": "completed"} for _ in range(8)]})
        a.write(root / f"native-{name}/workload-summary.json", {"frozen_identity_verified": True})
    for name in ("http-baseline", "http-b4096"):
        a.write(root / name / "micro.json", {"status": "completed",
            "cases": [{"calls": [{"sidecar_new_files": []} for _ in range(3)]} for _ in range(8)]})


def test_complete_summary_requires_http_no_disk_and_all_numerical_checks(tmp_path, monkeypatch):
    fixtures(tmp_path, monkeypatch)
    report = a.summarize(tmp_path, 0)
    assert report["status"] == "passed"
    assert report["candidate_evaluations"] == 24
    assert report["candidate_evaluation_speed_ratio"] == 2
    assert report["real_source_http_calls"] == 48
    assert report["real_source_http_new_sidecars"] == 0
    assert report["baseline_repeat_drift_percent"] == pytest.approx(10)


@pytest.mark.parametrize("failure", ("exit", "source", "numeric", "requests", "candidate", "count", "time", "skip", "http_missing", "http_sidecar", "http_unverified"))
def test_failure_withholds_speed_claims(tmp_path, monkeypatch, failure):
    fixtures(tmp_path, monkeypatch)
    if failure == "source":
        monkeypatch.setattr(a, "sources", lambda: {})
    elif failure in ("numeric", "requests"):
        a.write(tmp_path / "compare-native-optimized/comparison.json",
                {"status": "not_equivalent" if failure == "numeric" else "equivalent",
                 "public_workload_identical": failure != "requests"})
    elif failure == "candidate":
        a.write(tmp_path / "native-optimized/workload-summary.json", {"frozen_identity_verified": False})
    elif failure in ("count", "time"):
        path = tmp_path / "native-optimized/result.json"
        value = a.read(path)
        if failure == "count":
            value["cases"].pop()
        else:
            value["evaluation_wall_seconds"] = 0
        a.write(path, value)
    elif failure == "skip":
        (tmp_path / "physical-tests.xml").write_text('<testsuite tests="1" skipped="1"><testcase><skipped/></testcase></testsuite>')
    elif failure.startswith("http_"):
        path = tmp_path / "http-b4096/micro.json"
        value = a.read(path)
        if failure == "http_missing":
            value["cases"].pop()
        elif failure == "http_sidecar":
            value["cases"][0]["calls"][0]["sidecar_new_files"] = ["unexpected.npy"]
        else:
            del value["cases"][0]["calls"][0]["sidecar_new_files"]
        a.write(path, value)
    report = a.summarize(tmp_path, 1 if failure == "exit" else 0)
    assert report["status"] == "failed"
    assert "candidate_evaluation_speed_ratio" not in report


def test_configs_preserve_science_and_select_only_trust_and_batch_options():
    baseline = json.loads((ROOT / "configs/plasma_artemis_opt4_baseline.json").read_text())
    for batch in (1024, 2048, 4096):
        value = json.loads((ROOT / f"configs/plasma_artemis_opt4_b{batch}.json").read_text())
        settings = value["factory_settings"]
        assert settings.pop("physical_checkout").endswith("/artemis-oracle-opt4-20260909")
        assert settings.pop("expected_physical_revision")
        assert settings["oracle_settings"].pop("exact_query_execution_mode") == "trusted"
        assert settings["oracle_settings"].pop("exact_query_velocity_batch_size") == batch
        expected = json.loads(json.dumps(baseline))
        expected["factory_settings"].pop("physical_checkout")
        expected["factory_settings"].pop("expected_physical_revision")
        assert value == expected


def test_slurm_script_is_bounded_and_preserves_measurement_order():
    path = ROOT / "scripts/slurm_artemis_backend_opt4_acceptance.sh"
    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
    text = path.read_text()
    for required in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --time=01:00:00",
                     "baseline b1024 b2048 b4096 baseline_repeat", "baseline optimized baseline_repeat",
                     "--transport direct", "--transport http", "--seed 2026090903",
                     "test_artemis_opt4_trusted_source.py", "trap artemis_finish EXIT"):
        assert required in text
    for forbidden in ("scancel ", "sbatch ", "--auth-file", "run_canonical_codex_agent.py"):
        assert forbidden not in text
    assert text.index("junit-gate") < text.index("for ARTEMIS_MODE in baseline b1024")
