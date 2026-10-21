"""Bookkeeping only for an isolated D3 numerical/performance comparison."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TREES = ROOT.parent
BASE = "3e46fd2b4510608e93c8321d9ed904c60d0548a2"
PHYSICAL_BASE = "91e2889dcd3f17137a13d4fd349b49539d3b5629"
CONFIGS = ("plasma_artemis_opt3_benchmark_baseline.json", "plasma_artemis_opt3_benchmark.json")


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return data


def git(tree, *arguments):
    return subprocess.check_output(["git", "-C", str(tree), *arguments], text=True).strip()


def sources():
    result = {}
    for name, tree in (("driver", ROOT), ("physical_d2", TREES / "artemis-oracle-opt2-20260907"),
                       ("physical_d3", TREES / "artemis-oracle-opt3-20260909"),
                       ("legacy_harness", TREES / "artemis-eval-benchmark-20260907")):
        dirty = git(tree, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            raise ValueError(f"offline acceptance requires committed clean source: {name}")
        result[name] = {"root": str(tree), "commit": git(tree, "rev-parse", "HEAD")}
    if result["physical_d2"]["commit"] != PHYSICAL_BASE:
        raise ValueError("D2 physical baseline moved")
    if result["legacy_harness"]["commit"] != "eea9fddf88a2fe901a6ce60b507bf4e1918d7096":
        raise ValueError("existing comparison harness moved")
    protected = ("src/pde_agent_verifier_solver/engine", "src/pde_agent_verifier_solver/sdk",
                 "src/pde_agent_verifier_solver/app", "configs/candidates",
                 "configs/plasma_canonical_artemis.json", "configs/artemis_canonical_5round.json",
                 "scripts/run_canonical_codex_agent.py", "scripts/slurm_artemis_canonical_5round.sh")
    if git(ROOT, "diff", "--name-only", BASE, "HEAD", "--", *protected):
        raise ValueError("offline optimization changed protected formal/shared code")
    configs = {name: read(ROOT / "configs" / name) for name in CONFIGS}
    a, b = (configs[name]["factory_settings"] for name in CONFIGS)
    if b.get("expected_physical_revision") != result["physical_d3"]["commit"]:
        raise ValueError("optimized configuration does not pin the reviewed physical commit")
    if a["physical_checkout"] != result["physical_d2"]["root"] or b["physical_checkout"] != result["physical_d3"]["root"]:
        raise ValueError("benchmark configuration selects unexpected physical roots")
    if b["oracle_settings"].get("exact_query_profile_mode") != "frozen_basis":
        raise ValueError("D3 profile option is not explicitly selected")
    for settings in (a, b):
        if "reward_guidance_settings" in settings or "guidance_adapter_factory" in settings:
            raise ValueError("source-only benchmark must not train Guidance")
    result["config_sha256"] = {name: sha256((ROOT / "configs" / name).read_bytes()).hexdigest() for name in CONFIGS}
    return result


def junit(root):
    reports = {}
    for name in ("physical-tests.xml", "deployment-tests.xml", "runner-tests.xml"):
        tree = ET.parse(root / name).getroot()
        suites = [node for node in tree.iter("testsuite") if not node.findall(".//testsuite")]
        counts = {key: sum(int(node.get(key, "0")) for node in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        if counts["tests"] <= 0 or any(counts[key] for key in ("failures", "errors", "skipped")):
            raise ValueError(f"unit acceptance failed, skipped or empty: {name}")
        if any(list(tree.iter(key)) for key in ("failure", "error", "skipped")):
            raise ValueError(f"unit XML contains a failure/error/skip: {name}")
        reports[name] = {**counts, "testcase_nodes": len(list(tree.iter("testcase")))}
    return {"status": "passed", "reports": reports}


def _seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("performance timing must be finite and positive")
    return value


def summarize(root, exit_code):
    report = {"schema": "artemis_opt3_acceptance_v1", "status": "failed", "exit_code": exit_code,
              "formal_experiment_modified": False, "agent_invoked": False,
              "measurement_scope": "manufactured real-asset serial candidate and source checks; codec measured separately"}
    try:
        before, after = read(root / "source-state.json"), sources()
        if before != after or exit_code:
            raise ValueError("a stage failed or source changed during the allocation")
        tests = junit(root)
        comparisons = {}
        for name in ("compare-native-optimized", "compare-native-baseline_repeat",
                     "compare-micro-8-optimized", "compare-micro-8-baseline_repeat",
                     "compare-micro-1-optimized"):
            value = read(root / name / "comparison.json")
            if value.get("status") != "equivalent":
                raise ValueError(f"numerical comparison failed: {name}")
            comparisons[name] = value
        runs = {mode: read(root / f"native-{mode}" / "result.json")
                for mode in ("baseline", "optimized", "baseline_repeat")}
        workload = {mode: read(root / f"native-{mode}" / "workload-summary.json") for mode in runs}
        for mode, run in runs.items():
            if (run.get("status") != "completed" or len(run.get("cases", [])) != 8
                    or any(case.get("status") != "completed" for case in run["cases"])
                    or not workload[mode].get("frozen_identity_verified")):
                raise ValueError(f"incomplete frozen-candidate run: {mode}")
        if any(not comparisons[name].get("public_workload_identical")
               for name in ("compare-native-optimized", "compare-native-baseline_repeat")):
            raise ValueError("candidate source workload changed")
        wire = read(root / "wire.json")
        if wire.get("status") != "passed" or any(value.get("status") != "passed" for value in wire.get("filesystems", {}).values()):
            raise ValueError("codec comparison failed or skipped")
        elapsed = {name: _seconds(value["evaluation_wall_seconds"]) for name, value in runs.items()}
        report.update(status="passed", source_state=after, tests=tests,
                      candidate_evaluations=24, numerical_rtol=1e-10, numerical_atol=1e-12,
                      candidate_evaluation_wall_seconds=elapsed,
                      candidate_evaluation_speed_ratio=elapsed["baseline"] / elapsed["optimized"],
                      baseline_repeat_drift_percent=100 * (elapsed["baseline_repeat"] / elapsed["baseline"] - 1),
                      comparison_reports=list(comparisons), workload_reports={name: f"native-{name}/workload-summary.json" for name in runs},
                      codec_report="wire.json", codec_is_not_end_to_end=True,
                      full_iteration_speedup_claimed=False)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    return write(root / "acceptance-summary.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "junit", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        if args.root.exists():
            raise ValueError("refusing to reuse offline acceptance directory")
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
