"""Synthetic acceptance-bookkeeping fixtures, not model/performance evidence."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from scripts import artemis_opt5_acceptance as a

ROOT = Path(__file__).resolve().parents[1]


def fixture_micro(transport, config):
    return {"status": "completed", "transport": transport, "master_seed": 2026090903,
        "parents": 2, "repeats": 2, "backend_config_sha256": config,
        "cases": [{"case_key": f"pair-{pair:03d}-{distribution}", "mode": mode,
                   "point_count": 4096, "tx_count": 16, "velocity_count": 256,
                   "calls": [{"index": index, "wall_seconds": 0.1, "sidecar_new_files": [],
                              "backend_usage": {"source_query_calls": 1, "source_queried_points": 4096}}
                             for index in range(3)]}
                  for pair in range(2) for distribution in ("base", "base_perturbed")
                  for mode in ("grid_repeated_v", "arbitrary_v")]}


def fixture_summary(root, monkeypatch):
    state = {"driver": {"commit": "fixture"}, "config_sha256": {"baseline": "d4-config", "optimized": "d5-config"}}
    monkeypatch.setattr(a, "sources", lambda: state)
    a.write(root / "source-state.json", state)
    for name in ("physical-tests.xml", "deployment-tests.xml", "runner-tests.xml"):
        (root / name).write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="fixture"/></testsuite>')
    for name in a.COMPARISONS:
        a.write(root / name / "comparison.json", {"status": "equivalent", "problems": [],
            "rtol": 1e-10, "atol": 1e-12, "public_workload_identical": True})
    for name, elapsed in (("baseline", 10), ("optimized", 5), ("baseline_repeat", 11)):
        digest = state["config_sha256"]["optimized" if name == "optimized" else "baseline"]
        a.write(root / f"native-{name}/result.json", {"status": "completed", "master_seed": 2026090903,
            "parents": 2, "point_count": 32, "candidate_compute_budget": 600,
            "horizon_policy": "native-end", "backend_config_sha256": digest,
            "evaluation_wall_seconds": elapsed, "cases": [{"status": "completed"} for _ in range(8)]})
        a.write(root / f"native-{name}/workload-summary.json", {"frozen_identity_verified": True})
        a.write(root / f"micro-{name}/micro.json", fixture_micro("direct", digest))
        if name != "baseline_repeat":
            a.write(root / f"http-{name}/micro.json", fixture_micro("http", digest))
    return state


def test_complete_acceptance_has_five_comparisons_and_no_extra_rpc_mock_requirement(tmp_path, monkeypatch):
    fixture_summary(tmp_path, monkeypatch)
    report = a.summarize(tmp_path, 0)
    assert report["status"] == "passed", report
    assert len(report["comparison_reports"]) == 5
    assert report["candidate_evaluations"] == 24 and report["candidate_evaluation_speed_ratio"] == 2
    assert report["baseline_repeat_drift_percent"] == pytest.approx(10)
    assert report["direct_source_calls"] == 72 and report["real_source_http_calls"] == 48
    assert report["real_source_http_new_sidecars"] == 0 and not report["rpc_transport_modified"]
    assert not report["full_iteration_speedup_claimed"] and not report["multi_worker_throughput_claimed"]


@pytest.mark.parametrize("failure", ["exit", "source", "numeric", "tolerance", "requests", "candidate", "count",
    "time", "skip", "config", "budget", "micro_missing", "micro_shape", "micro_counter", "micro_order",
    "micro_cohort", "micro_transport", "http_sidecar", "http_unverified"])
def test_failures_retain_summary_without_speed_claims(tmp_path, monkeypatch, failure):
    fixture_summary(tmp_path, monkeypatch)
    if failure == "source":
        monkeypatch.setattr(a, "sources", lambda: {})
    elif failure in ("numeric", "tolerance", "requests"):
        path = tmp_path / "compare-native-optimized/comparison.json"
        value = a.read(path)
        if failure == "numeric": value["status"] = "not_equivalent"
        if failure == "tolerance": value["rtol"] = 1e-3
        if failure == "requests": value["public_workload_identical"] = False
        a.write(path, value)
    elif failure == "candidate":
        a.write(tmp_path / "native-optimized/workload-summary.json", {"frozen_identity_verified": False})
    elif failure in ("count", "time", "config", "budget"):
        path = tmp_path / "native-optimized/result.json"
        value = a.read(path)
        if failure == "count": value["cases"].pop()
        if failure == "time": value["evaluation_wall_seconds"] = 0
        if failure == "config": value["backend_config_sha256"] = "wrong"
        if failure == "budget": value["candidate_compute_budget"] = 60
        a.write(path, value)
    elif failure == "skip":
        (tmp_path / "physical-tests.xml").write_text('<testsuite tests="1" skipped="1"><testcase><skipped/></testcase></testsuite>')
    elif failure.startswith(("micro_", "http_")):
        path = tmp_path / ("http-optimized/micro.json" if failure.startswith("http_") else "micro-optimized/micro.json")
        value = a.read(path)
        if failure == "micro_missing": value["cases"].pop()
        if failure == "micro_shape": value["cases"][0]["point_count"] = 1024
        if failure == "micro_counter": value["cases"][0]["calls"][0]["backend_usage"]["source_query_calls"] = 2
        if failure == "micro_order": value["cases"][0]["calls"][0]["index"] = 3
        if failure == "micro_cohort": value["cases"][0]["case_key"] = "different"
        if failure == "micro_transport": value["transport"] = "http"
        if failure == "http_sidecar": value["cases"][0]["calls"][0]["sidecar_new_files"] = ["source.npy"]
        if failure == "http_unverified": value["cases"][0]["calls"][0].pop("sidecar_new_files")
        a.write(path, value)
    report = a.summarize(tmp_path, 1 if failure == "exit" else 0)
    assert report["status"] == "failed" and "error" in report
    assert "candidate_evaluation_speed_ratio" not in report
    assert a.read(tmp_path / "acceptance-summary.json") == report


def fixture_sources(tmp_path, monkeypatch):
    root = tmp_path / "driver"
    root.mkdir()
    monkeypatch.setattr(a, "ROOT", root)
    baseline = {"physics": "artemis", "factory_settings": {
        "physical_checkout": str(tmp_path / "artemis-oracle-opt4-20260909"), "expected_physical_revision": a.P4,
        "device": "cuda", "science_config_path": "unchanged-science", "expected_science_config_sha256": "unchanged",
        "oracle_settings": {"exact_query_tx_reuse": "request", "exact_query_source_mode": "source_only",
            "factorized_jvp_mode": "forward", "exact_query_tx_mode": "demand", "exact_query_transfer_mode": "request",
            "generation_sensor_geometry": "precomputed", "exact_query_profile_mode": "frozen_basis",
            "exact_query_execution_mode": "trusted", "exact_query_velocity_batch_size": 4096}}}
    optimized = copy.deepcopy(baseline)
    optimized["factory_settings"].update(physical_checkout=str(tmp_path / "artemis-oracle-opt5-20260909"), expected_physical_revision="5" * 40)
    optimized["factory_settings"]["oracle_settings"]["exact_query_kernel_mode"] = "vp_specialized"
    a.write(root / "configs" / a.CONFIGS["baseline"], baseline)
    a.write(root / "configs" / a.CONFIGS["optimized"], optimized)
    controls = {"dirty": False, "changed": "", "p4": a.P4}
    def git(tree, *args):
        if args[0] == "status": return " M changed.py" if controls["dirty"] else ""
        if args[0] == "diff":
            assert "src/pde_agent_verifier_solver/engine" in args and a.BASE in args
            return controls["changed"]
        if args[0] == "show": return json.dumps(baseline)
        return {"driver": "d" * 40, "artemis-oracle-opt4-20260909": controls["p4"],
            "artemis-oracle-opt5-20260909": "5" * 40, "artemis-eval-benchmark-20260907": a.HARNESS}[tree.name]
    monkeypatch.setattr(a, "git", git)
    return controls


def test_sources_accepts_only_pinned_kernel_mode_delta(tmp_path, monkeypatch):
    fixture_sources(tmp_path, monkeypatch)
    report = a.sources()
    assert report["physical_d4"]["commit"] == a.P4
    assert len(report["config_sha256"]["optimized"]) == 64


@pytest.mark.parametrize("change", ["dirty", "shared", "baseline_pin", "optimized_pin", "root", "kernel", "science", "guidance", "batch", "baseline"])
def test_sources_rejects_scope_science_or_pin_changes(tmp_path, monkeypatch, change):
    controls = fixture_sources(tmp_path, monkeypatch)
    if change == "dirty": controls["dirty"] = True
    elif change == "shared": controls["changed"] = "src/pde_agent_verifier_solver/engine/evaluator.py"
    elif change == "baseline_pin": controls["p4"] = "wrong"
    else:
        path = a.ROOT / "configs" / a.CONFIGS["baseline" if change == "baseline" else "optimized"]
        value = a.read(path)
        settings = value["factory_settings"]
        if change == "optimized_pin": settings["expected_physical_revision"] = "wrong"
        if change == "root": settings["physical_checkout"] = "/wrong"
        if change == "kernel": settings["oracle_settings"]["exact_query_kernel_mode"] = "other"
        if change == "science": settings["science_config_path"] = "changed-science"
        if change == "guidance": settings["reward_guidance_settings"] = {}
        if change == "batch": settings["oracle_settings"]["exact_query_velocity_batch_size"] = 1024
        if change == "baseline": settings["oracle_settings"]["factorized_jvp_mode"] = "reverse"
        a.write(path, value)
    with pytest.raises(ValueError):
        a.sources()


def test_slurm_is_bounded_reuses_opt4_runner_and_keeps_aba_order():
    path = ROOT / "scripts/slurm_artemis_backend_opt5_acceptance.sh"
    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
    text = path.read_text()
    for required in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=64G",
        "#SBATCH --time=01:00:00", "trap artemis_finish EXIT",
        "--transport direct", "--transport http", "--seed 2026090903", "--rtol 1e-10 --atol 1e-12",
        "test_artemis_opt4_acceptance.py", "test_artemis_opt4_trusted_source.py", "test_artemis_opt5_*.py"):
        assert required in text
    assert text.count("for ARTEMIS_MODE in baseline optimized baseline_repeat") == 2
    assert text.index("junit-gate") < text.index("for ARTEMIS_MODE in baseline optimized baseline_repeat")
    assert "scripts/benchmark_artemis_opt4.py micro" in text and "scripts/benchmark_artemis_opt4.py run" in text
    for forbidden in ("sbatch ", "scancel ", "--auth-file", "run_canonical_codex_agent.py", "benchmark_artemis_opt4_rpc.py"):
        assert forbidden not in text


def test_preflight_refuses_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="reuse"):
        a.main(["preflight", "--root", str(tmp_path)])
