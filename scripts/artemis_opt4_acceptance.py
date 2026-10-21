"""Offline-only D4 acceptance bookkeeping; never loaded by a campaign."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scripts.artemis_opt3_acceptance import git, junit, read, write, _seconds

ROOT = Path(__file__).resolve().parents[1]
BASE = "6b198ba6cd320a9a5db622383a6acd8dd61bbae6"
P3 = "1b317a066630b59d7274c55db9b9a01ecddc638c"
CONFIGS = ("baseline", "b1024", "b2048", "b4096")
COMPARISONS = ("compare-native-optimized", "compare-native-baseline_repeat",
              "compare-micro-b1024", "compare-micro-b2048", "compare-micro-b4096",
              "compare-micro-baseline_repeat", "compare-http-b4096")


def sources():
    rows = {}
    for name, tree in (("driver", ROOT),
                       ("physical_d3", ROOT.parent / "artemis-oracle-opt3-20260909"),
                       ("physical_d4", ROOT.parent / "artemis-oracle-opt4-20260909"),
                       ("legacy_harness", ROOT.parent / "artemis-eval-benchmark-20260907")):
        if git(tree, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"acceptance requires committed clean source: {name}")
        rows[name] = {"root": str(tree), "commit": git(tree, "rev-parse", "HEAD")}
    if rows["physical_d3"]["commit"] != P3:
        raise ValueError("D3 baseline moved")
    if rows["legacy_harness"]["commit"] != "eea9fddf88a2fe901a6ce60b507bf4e1918d7096":
        raise ValueError("legacy harness moved")
    changed = set(git(ROOT, "diff", "--name-only", BASE, "HEAD", "--",
                      "src/pde_agent_verifier_solver/engine",
                      "src/pde_agent_verifier_solver/sdk",
                      "src/pde_agent_verifier_solver/app",
                      "configs/candidates", "configs/plasma_canonical_artemis.json",
                      "configs/artemis_canonical_5round.json",
                      "scripts/run_canonical_codex_agent.py",
                      "scripts/slurm_artemis_canonical_5round.sh").splitlines())
    allowed = {"src/pde_agent_verifier_solver/engine/agent_wire.py",
               "src/pde_agent_verifier_solver/engine/candidate_execution.py"}
    if changed - allowed:
        raise ValueError(f"protected Controller/SDK/formal code changed: {sorted(changed-allowed)}")
    configs = {label: read(ROOT / f"configs/plasma_artemis_opt4_{label}.json") for label in CONFIGS}
    baseline = configs["baseline"]["factory_settings"]
    if baseline["physical_checkout"] != rows["physical_d3"]["root"] or baseline["expected_physical_revision"] != P3:
        raise ValueError("baseline must select pinned D3")
    expected = json.loads(json.dumps(baseline))
    expected.pop("physical_checkout")
    expected.pop("expected_physical_revision")
    if "reward_guidance_settings" in expected:
        raise ValueError("source acceptance cannot train Guidance")
    for batch in (1024, 2048, 4096):
        settings = json.loads(json.dumps(configs[f"b{batch}"]["factory_settings"]))
        if settings.pop("physical_checkout") != rows["physical_d4"]["root"]:
            raise ValueError("wrong D4 checkout")
        if settings.pop("expected_physical_revision") != rows["physical_d4"]["commit"]:
            raise ValueError("D4 physical commit is not pinned")
        opts = settings["oracle_settings"]
        if opts.pop("exact_query_execution_mode") != "trusted" or opts.pop("exact_query_velocity_batch_size") != batch:
            raise ValueError("wrong trusted batch variant")
        if settings != expected:
            raise ValueError("scientific configuration changed")
    rows["config_sha256"] = {key: sha256((ROOT / f"configs/plasma_artemis_opt4_{key}.json").read_bytes()).hexdigest() for key in CONFIGS}
    return rows


def summarize(root, exit_code):
    result = {"schema": "artemis_opt4_acceptance_v1", "status": "failed", "exit_code": exit_code,
              "formal_experiment_modified": False, "agent_invoked": False,
              "scope": "offline trusted source, loopback HTTP, and serial frozen candidates",
              "full_iteration_speedup_claimed": False}
    try:
        state = sources()
        if read(root / "source-state.json") != state or exit_code:
            raise ValueError("source changed or a stage failed")
        tests = junit(root)
        rpc = read(root / "rpc-transport.json")
        if rpc.get("status") != "passed" or set(rpc.get("filesystems", {})) != {"scratch", "local"}:
            raise ValueError("loopback transport acceptance incomplete")
        for filesystem in rpc["filesystems"].values():
            if filesystem.get("status") != "passed" or filesystem["variants"]["opt4_128k_trusted"]["measured_sidecar_count"] != 0:
                raise ValueError("loopback transport created source sidecars")
        comparisons = {name: read(root / name / "comparison.json") for name in COMPARISONS}
        for name, value in comparisons.items():
            if value.get("status") != "equivalent":
                raise ValueError(f"numerical acceptance failed: {name}")
            if name.startswith("compare-native-") and not value.get("public_workload_identical"):
                raise ValueError(f"candidate requests changed: {name}")
        elapsed = {}
        for name in ("baseline", "optimized", "baseline_repeat"):
            run = read(root / f"native-{name}/result.json")
            work = read(root / f"native-{name}/workload-summary.json")
            if (run.get("status") != "completed" or len(run.get("cases", [])) != 8
                    or any(c.get("status") != "completed" for c in run["cases"])
                    or not work.get("frozen_identity_verified")):
                raise ValueError(f"incomplete frozen candidate run: {name}")
            elapsed[name] = _seconds(run["evaluation_wall_seconds"])
        for name in ("http-baseline", "http-b4096"):
            report = read(root / name / "micro.json")
            calls = [call for row in report.get("cases", []) for call in row.get("calls", [])]
            if report.get("status") != "completed" or len(calls) != 24:
                raise ValueError(f"incomplete real-source HTTP run: {name}")
            if any(call.get("sidecar_new_files") != [] for call in calls):
                raise ValueError(f"source RPC created sidecars: {name}")
        result.update(status="passed", source_state=state, tests=tests,
                      candidate_evaluations=24, numerical_rtol=1e-10, numerical_atol=1e-12,
                      candidate_evaluation_wall_seconds=elapsed,
                      candidate_evaluation_speed_ratio=elapsed["baseline"]/elapsed["optimized"],
                      baseline_repeat_drift_percent=100*(elapsed["baseline_repeat"]/elapsed["baseline"]-1),
                      comparison_reports=list(comparisons),
                      real_source_http_calls=48, real_source_http_new_sidecars=0)
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
        if args.root.exists():
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
