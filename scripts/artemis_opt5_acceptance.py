"""Offline D5 acceptance bookkeeping; no campaign or scientific implementation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scripts.artemis_opt3_acceptance import git, junit, read, write, _seconds

ROOT = Path(__file__).resolve().parents[1]
BASE = "7fceecd0d515558d74261f729508a147d5ed6ed2"
P4 = "9875c67b05a1a13d33636263dce8c648df20422e"
HARNESS = "eea9fddf88a2fe901a6ce60b507bf4e1918d7096"
CONFIGS = {"baseline": "plasma_artemis_opt5_baseline.json", "optimized": "plasma_artemis_opt5.json"}
COMPARISONS = ("compare-micro-optimized", "compare-micro-baseline_repeat", "compare-http-optimized",
               "compare-native-optimized", "compare-native-baseline_repeat")
PROTECTED = ("src/pde_agent_verifier_solver/engine", "src/pde_agent_verifier_solver/sdk",
    "src/pde_agent_verifier_solver/app", "configs/candidates", "configs/plasma_canonical_artemis.json",
    "configs/artemis_canonical_5round.json", ":(glob)scripts/*canonical*", ":(glob)configs/*opt4*",
    "scripts/benchmark_artemis_opt3.py", "scripts/benchmark_artemis_opt4.py",
    "scripts/benchmark_artemis_opt4_rpc.py", "scripts/artemis_opt4_acceptance.py",
    "scripts/slurm_artemis_backend_opt4_acceptance.sh", "tests/test_artemis_opt4_acceptance.py")


def sources():
    rows = {}
    for name, tree in (("driver", ROOT), ("physical_d4", ROOT.parent / "artemis-oracle-opt4-20260909"),
                       ("physical_d5", ROOT.parent / "artemis-oracle-opt5-20260909"),
                       ("legacy_harness", ROOT.parent / "artemis-eval-benchmark-20260907")):
        if git(tree, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"acceptance requires committed clean source: {name}")
        rows[name] = {"root": str(tree), "commit": git(tree, "rev-parse", "HEAD")}
    if rows["physical_d4"]["commit"] != P4 or rows["legacy_harness"]["commit"] != HARNESS:
        raise ValueError("frozen D4 baseline or legacy harness moved")
    if git(ROOT, "diff", "--name-only", BASE, "HEAD", "--", *PROTECTED):
        raise ValueError("protected shared/formal/D4 benchmark code changed")
    configs = {name: read(ROOT / "configs" / filename) for name, filename in CONFIGS.items()}
    baseline = configs["baseline"]
    original = json.loads(git(ROOT, "show", f"{BASE}:configs/plasma_artemis_opt4_b4096.json"))
    if baseline != original:
        raise ValueError("D5 baseline must be the unchanged D4 trusted 4096 configuration")
    normalized = []
    for name in CONFIGS:
        config = json.loads(json.dumps(configs[name]))
        settings = config["factory_settings"]
        physical = rows["physical_d4" if name == "baseline" else "physical_d5"]
        if settings.pop("physical_checkout") != physical["root"] or settings.pop("expected_physical_revision") != physical["commit"]:
            raise ValueError("physical source path or revision is not pinned")
        if any(key in settings for key in ("reward_guidance_settings", "guidance_adapter_factory")):
            raise ValueError("offline source benchmark must not train Guidance")
        options = settings["oracle_settings"]
        if options.get("exact_query_execution_mode") != "trusted" or options.get("exact_query_velocity_batch_size") != 4096:
            raise ValueError("D5 must preserve D4 trusted execution and 4096 internal velocity batch")
        if name == "optimized" and options.pop("exact_query_kernel_mode") != "vp_specialized":
            raise ValueError("D5 specialized VP kernel is not selected")
        normalized.append(config)
    if normalized[0] != normalized[1]:
        raise ValueError("D5 changed science or an unrelated execution option")
    rows["config_sha256"] = {name: sha256((ROOT / "configs" / filename).read_bytes()).hexdigest() for name, filename in CONFIGS.items()}
    return rows


def _check_micro(report, *, name, transport, state):
    variant = "optimized" if name.endswith("optimized") else "baseline"
    if (report.get("status") != "completed" or report.get("transport") != transport
            or report.get("master_seed") != 2026090903 or report.get("parents") != 2
            or report.get("repeats") != 2 or report.get("backend_config_sha256") != state["config_sha256"][variant]):
        raise ValueError(f"incomplete or misconfigured source run: {name}")
    rows = report.get("cases", [])
    expected = {(f"pair-{pair:03d}-{distribution}", mode) for pair in range(2)
                for distribution in ("base", "base_perturbed") for mode in ("grid_repeated_v", "arbitrary_v")}
    if len(rows) != 8 or {(row.get("case_key"), row.get("mode")) for row in rows} != expected:
        raise ValueError(f"source cohort/workload changed: {name}")
    for row in rows:
        if (row.get("point_count"), row.get("tx_count"), row.get("velocity_count")) != (4096, 16, 256):
            raise ValueError(f"source point workload changed: {name}")
        calls = row.get("calls", [])
        if [call.get("index") for call in calls] != [0, 1, 2]:
            raise ValueError(f"source first/repeat calls missing: {name}")
        for call in calls:
            usage = call.get("backend_usage", {})
            if usage.get("source_query_calls") != 1 or usage.get("source_queried_points") != 4096:
                raise ValueError(f"source response counters changed: {name}")
            _seconds(call["wall_seconds"])
            if transport == "http" and call.get("sidecar_new_files") != []:
                raise ValueError(f"source HTTP created sidecars or was not checked: {name}")


def summarize(root, exit_code):
    result = {"schema": "artemis_opt5_acceptance_v1", "status": "failed", "exit_code": exit_code,
        "formal_experiment_modified": False, "agent_invoked": False,
        "scope": "manufactured real-asset D4/D5 source kernels, unchanged HTTP transport, serial frozen candidates",
        "full_iteration_speedup_claimed": False, "multi_worker_throughput_claimed": False}
    try:
        state = sources()
        if read(root / "source-state.json") != state or exit_code:
            raise ValueError("source changed or an acceptance stage failed")
        tests = junit(root)
        comparisons = {name: read(root / name / "comparison.json") for name in COMPARISONS}
        for name, value in comparisons.items():
            if value.get("status") != "equivalent" or value.get("problems"):
                raise ValueError(f"numerical comparison failed: {name}")
            if value.get("rtol") != 1e-10 or value.get("atol") != 1e-12:
                raise ValueError(f"numerical tolerance changed: {name}")
            if name.startswith("compare-native-") and value.get("public_workload_identical") is not True:
                raise ValueError(f"frozen candidate requests changed: {name}")
        elapsed = {}
        for name in ("baseline", "optimized", "baseline_repeat"):
            run = read(root / f"native-{name}/result.json")
            work = read(root / f"native-{name}/workload-summary.json")
            variant = "optimized" if name == "optimized" else "baseline"
            if (run.get("status") != "completed" or len(run.get("cases", [])) != 8
                    or any(case.get("status") != "completed" for case in run["cases"])
                    or work.get("frozen_identity_verified") is not True
                    or run.get("backend_config_sha256") != state["config_sha256"][variant]
                    or run.get("master_seed") != 2026090903 or run.get("parents") != 2
                    or run.get("point_count") != 32 or run.get("candidate_compute_budget") != 600
                    or run.get("horizon_policy") != "native-end"):
                raise ValueError(f"incomplete or changed frozen candidate run: {name}")
            elapsed[name] = _seconds(run["evaluation_wall_seconds"])
        for name in ("baseline", "optimized", "baseline_repeat"):
            _check_micro(read(root / f"micro-{name}/micro.json"), name=f"micro-{name}", transport="direct", state=state)
        for name in ("baseline", "optimized"):
            _check_micro(read(root / f"http-{name}/micro.json"), name=f"http-{name}", transport="http", state=state)
        result.update(status="passed", source_state=state, tests=tests,
            candidate_evaluations=24, numerical_rtol=1e-10, numerical_atol=1e-12,
            candidate_evaluation_wall_seconds=elapsed,
            candidate_evaluation_speed_ratio=elapsed["baseline"] / elapsed["optimized"],
            baseline_repeat_drift_percent=100 * (elapsed["baseline_repeat"] / elapsed["baseline"] - 1),
            comparison_reports=list(comparisons), direct_source_calls=72,
            real_source_http_calls=48, real_source_http_new_sidecars=0,
            rpc_transport_modified=False)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    return write(root / "acceptance-summary.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "junit", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        if args.root.exists() or args.root.is_symlink():
            raise ValueError("refusing to reuse an acceptance directory")
        report = sources()
        args.root.mkdir(parents=True)
        write(args.root / "source-state.json", report)
    elif args.mode == "junit":
        report = write(args.root / "junit-summary.json", junit(args.root))
    else:
        report = summarize(args.root, args.exit_code)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("status", "passed") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
