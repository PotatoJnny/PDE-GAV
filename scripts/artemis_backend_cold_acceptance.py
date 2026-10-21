"""Fail-closed bookkeeping for the frozen R4/R5 cold Backend check.

This helper never invokes an Agent or computes a scientific solution. Numerical
comparison remains the existing isolated benchmark runner's responsibility.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TREES = ROOT.parent
TURNS = Path("${PDE_GAV_RUN_ROOT}/artemis_canonical_5round_20260905/job-5243774/campaign/agent_turns")
R4 = "round-0004-artemis-critical-refiner-b67a32c690e14e0099c7ad346f9218bc"
R5E = "round-0005-artemis-broad-explorer-eb5340ee5ec54e9f90584a1d3856eb40"
R5R = "round-0005-artemis-critical-refiner-8e5a6dfdc7e64656b918b6b21d6ae896"
# R4 Official winner followed by the five retained R5 nomination bundles.
CANDIDATES = (
    ("R4_winner", R4, "57250e124d5c9fb20632f0dfb7b63c0a050ab084e0432215cf6489399f0e6573"),
    ("R5_explorer_1", R5E, "59b925364e647c8c4a3ecf464cef65b674372a82b26db880ea0cb05453d1097a"),
    ("R5_explorer_2", R5E, "9eb1d477b241d955115b08c5c71f497e3375d2d41e9501ce630bd0d03e253e72"),
    ("R5_explorer_3", R5E, "e7b559a815b094ea287171edb2c73b59351886f47303d8be04050db0064d6572"),
    ("R5_refiner_1", R5R, "99245269b000c6ed91080db34302fa2f243a402dc90ccbaa24996f0509b7a419"),
    ("R5_refiner_2", R5R, "55204db93a1340954f9ce4ee64b11536161c8cb816ceb00e7c77707575720c74"),
)
SOURCES = {
    "canonical": (ROOT, None),
    "baseline": (TREES / "artemis-cold-baseline-20260908", "941083a3f8995c101a2095f47d758b02361d0c61"),
    "benchmark": (TREES / "artemis-eval-benchmark-20260907", "eea9fddf88a2fe901a6ce60b507bf4e1918d7096"),
    "physical_baseline": (TREES / "artemis-reward-guidance", "7c22b7f1dbf46388b114ca274c0e1fec2a5b0a36"),
    "physical_optimized": (TREES / "artemis-oracle-opt2-20260907", "91e2889dcd3f17137a13d4fd349b49539d3b5629"),
}

# Explicitly loaded by the two acceptance pytest processes. Recent pytest
# records passed subtests in JUnit totals but reuses the parent testcase node;
# retain actual report events so that this difference is auditable, not guessed.
_PYTEST_EVENTS = None
_PYTEST_EVENT_PATH = None


def pytest_sessionstart(session):
    global _PYTEST_EVENTS, _PYTEST_EVENT_PATH
    xml_path = session.config.getoption("xmlpath")
    if not xml_path:
        raise ValueError("cold acceptance event recorder requires --junitxml")
    _PYTEST_EVENT_PATH = Path(xml_path).with_suffix(".events.json")
    if _PYTEST_EVENT_PATH.exists():
        raise ValueError("refusing to reuse pytest event sidecar")
    _PYTEST_EVENTS = {"schema": "artemis_pytest_events_v1", "collected": [],
                      "reports": [], "collection_failures": [], "collection_finished": False}


def pytest_collection_finish(session):
    from _pytest.junitxml import bin_xml_escape, mangle_test_address
    for item in session.items:
        names = mangle_test_address(item.nodeid)
        prefix = session.config.getoption("junitprefix")
        classnames = ([prefix] if prefix else []) + names[:-1]
        _PYTEST_EVENTS["collected"].append({"nodeid": item.nodeid,
            "classname": ".".join(classnames), "name": bin_xml_escape(names[-1])})
    _PYTEST_EVENTS["collection_finished"] = True


def pytest_collectreport(report):
    if not report.passed:
        _PYTEST_EVENTS["collection_failures"].append({"nodeid": report.nodeid, "outcome": report.outcome})


def pytest_runtest_logreport(report):
    from _pytest.subtests import SubtestReport
    _PYTEST_EVENTS["reports"].append({"nodeid": report.nodeid, "when": report.when,
        "outcome": report.outcome, "subtest": isinstance(report, SubtestReport),
        "report_type": f"{type(report).__module__}.{type(report).__name__}"})


def pytest_sessionfinish(session, exitstatus):
    _PYTEST_EVENTS.update(exit_status=int(exitstatus), collected_count=session.testscollected)
    _PYTEST_EVENT_PATH.write_text(json.dumps(_PYTEST_EVENTS, indent=2, sort_keys=True) + "\n")


def passing_subtest_count(root, filename, tree):
    events = json.loads((root / Path(filename).with_suffix(".events.json")).read_text())
    collected = events["collected"]
    ids = [item["nodeid"] for item in collected]
    if (events["schema"] != "artemis_pytest_events_v1" or events["exit_status"] != 0
            or events["collection_finished"] is not True or events["collection_failures"]
            or not ids or len(ids) != len(set(ids)) or events["collected_count"] != len(ids)):
        raise ValueError("pytest collection did not finish cleanly with unique collected tests")
    expected_xml = Counter((item["classname"], item["name"]) for item in collected)
    observed_xml = Counter((item.attrib.get("classname"), item.attrib.get("name")) for item in tree.iter("testcase"))
    if observed_xml != expected_xml:
        raise ValueError("JUnit testcase identities do not match pytest collection")
    phases = {nodeid: Counter() for nodeid in ids}
    subtests = 0
    for event in events["reports"]:
        if event["nodeid"] not in phases or event["outcome"] != "passed":
            raise ValueError("pytest report is unknown, failed or skipped")
        if event["subtest"] is True:
            if event["report_type"] != "_pytest.subtests.SubtestReport" or event["when"] != "call":
                raise ValueError("unsupported pytest subtest report")
            subtests += 1
        elif event["subtest"] is False and event["report_type"] == "_pytest.reports.TestReport":
            phases[event["nodeid"]][event["when"]] += 1
        else:
            raise ValueError("unsupported pytest ordinary report")
    if any(counts != {"setup": 1, "call": 1, "teardown": 1} for counts in phases.values()):
        raise ValueError("pytest setup, call or teardown report is missing or duplicated")
    return subtests


def candidate_path(candidate):
    _, turn, digest = candidate
    return TURNS / turn / ".controller-candidates" / digest


def write_report(root, filename, report):
    (root / filename).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def source_state():
    """Verify clean, committed source and immutable source bundles, without import execution."""
    sys.path.insert(0, str(ROOT / "src"))
    from pde_agent_verifier_solver.sdk import inspect_candidate_bundle

    sources = {}
    for name, (path, expected) in SOURCES.items():
        def git(*args):
            return subprocess.run(["git", "-C", str(path), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        commit = git("rev-parse", "HEAD")
        if git("status", "--porcelain", "--untracked-files=all") or (expected and commit != expected):
            raise ValueError(f"source must be clean and match its committed pin: {name}")
        sources[name] = {"root": str(path), "commit": commit}
    configs = {}
    for name in ("canonical", "baseline"):
        path = SOURCES[name][0] / "configs/plasma_canonical_artemis.json"
        configs[name] = sha256(path.read_bytes()).hexdigest()
    expected_configs = {
        "canonical": "39250e34c86133302b4ae0d597e1a5a171e695d3c1455feb97d6564f1f78da11",
        "baseline": "f10b8f959401b4cfcb6abc85103c344ab9c45f8421ed8ae2aa81c065a6f33fcf",
    }
    if configs != expected_configs:
        raise ValueError("cold acceptance Backend configuration pin changed")
    candidates = []
    for candidate in CANDIDATES:
        bundle = inspect_candidate_bundle(candidate_path(candidate))
        if bundle.tree_sha256 != candidate[2]:
            raise ValueError(f"frozen candidate changed: {candidate[0]}")
        candidates.append({"name": candidate[0], "tree_sha256": bundle.tree_sha256,
                           "source_path": str(candidate_path(candidate))})
    return {"sources": sources, "config_sha256": configs, "candidates": candidates}


def junit_gate(root):
    keys = ("tests", "failures", "errors", "skipped")
    report = {"status": "failed", "reports": {}, "problems": [],
              "totals": dict.fromkeys(keys, 0)}
    for filename in ("deployment-tests.xml", "runner-tests.xml"):
        row = {"status": "failed", **dict.fromkeys(keys, 0)}
        report["reports"][filename] = row
        try:
            tree = ET.parse(root / filename).getroot()
            suites = list(tree.iter("testsuite"))
            leaves = [suite for suite in suites if not suite.findall(".//testsuite")]
            if not leaves:
                raise ValueError("JUnit has no leaf suite")
            for suite in leaves:
                counts = {key: int(suite.attrib[key]) for key in keys}
                if counts["tests"] <= 0 or any(value < 0 for value in counts.values()):
                    raise ValueError("empty suite or invalid counts")
                for key in keys:
                    row[key] += counts[key]
            observed = {key: len(list(tree.iter(tag))) for key, tag in
                        zip(keys, ("testcase", "failure", "error", "skipped"))}
            row["observed"] = observed
            row["passed_subtests"] = passing_subtest_count(root, filename, tree)
            if (row["tests"] != observed["tests"] + row["passed_subtests"]
                    or any(row[key] != observed[key] for key in keys[1:])):
                raise ValueError("JUnit declared and observed counts disagree")
            for suite in (*suites, *tree.iter("testsuites")):
                if "tests" in suite.attrib and int(suite.attrib["tests"]) <= 0:
                    raise ValueError("JUnit suite declares zero tests")
                if any(int(suite.attrib.get(key, "0")) for key in keys[1:]):
                    raise ValueError("JUnit contains failure, error or skip")
            row["status"] = "passed"
        except (OSError, ET.ParseError, KeyError, TypeError, ValueError) as error:
            report["problems"].append(f"{filename}: {type(error).__name__}: {error}")
        for key in keys:
            report["totals"][key] += row[key]
    report["status"] = "failed" if report["problems"] else "passed"
    return write_report(root, "junit-counts.json", report)


def workload(run):
    rows = []
    for case in run["cases"]:
        sessions = []
        if not case["public_sessions"]:
            raise ValueError("missing public session counters")
        for session in case["public_sessions"]:
            methods = {name: {key: values[key] for key in ("calls", "points", "failures")}
                       for name, values in session["methods"].items()}
            if not methods or any(values["failures"] != 0 for values in methods.values()):
                raise ValueError("missing or failed public query workload")
            sessions.append((session["access"], methods))
        rows.append((case["candidate"], case["case_key"], sessions))
    return rows


def summarize(root, exit_code):
    report = {"schema": "artemis_backend_cold_acceptance_v1", "status": "failed",
              "canonical_commit": None, "backend_config_sha256": None,
              "agent_invoked": False, "formal_scientific_admission": False,
              "short_benchmark_only": True, "step_exit_code": exit_code,
              "expected_evaluations_per_arm": 24, "completed_evaluations_per_arm": {},
              "batches": {}, "problems": []}

    def load(relative):
        return json.loads((root / relative).read_text())

    try:
        sources = load("source-state.json")
        report.update(canonical_commit=sources["sources"]["canonical"]["commit"],
                      backend_config_sha256=sources["config_sha256"]["canonical"], source_state=sources)
        if sources != source_state():
            raise ValueError("source, configuration or candidate changed during acceptance")
        if load("junit-counts.json")["status"] != "passed":
            raise ValueError("JUnit gate did not pass")
        steps = [line.split("\t") for line in (root / "step-exit-codes.tsv").read_text().splitlines()]
        required = {"source-preflight", "deployment-tests", "runner-tests", "junit-gate", "formal-config", "split-execution"}
        required.update(f"batch-{batch}-{mode}" for batch in range(2) for mode in ("baseline", "optimized", "compare"))
        if {name for name, code in steps if code == "0"} != required or len(steps) != len(required):
            raise ValueError("missing, repeated or failed acceptance steps")
        totals = {"baseline": 0, "optimized": 0}
        times = {"baseline": 0.0, "optimized": 0.0}
        for batch in range(2):
            expected = [(name, digest) for name, _, digest in CANDIDATES[3 * batch:3 * batch + 3]]
            runs = {mode: load(f"batch-{batch}-{mode}/result.json") for mode in totals}
            row = {}
            for mode, run in runs.items():
                cases, cohort = run["cases"], run["cohort"]
                expected_ids = {(name, f"pair-{pair:03d}-{distribution}") for name, _ in expected
                                for pair in range(2) for distribution in ("base", "base_perturbed")}
                valid = (run["status"] == "completed" and len(cases) == 12
                         and {(item["candidate"], item["case_key"]) for item in cases} == expected_ids
                         and len(cohort) == len({item["key"] for item in cohort}) == 4
                         and [(item["name"], item["tree_sha256"]) for item in run["candidates"]] == expected
                         and all(item["status"] == "completed" for item in cases)
                         and run["master_seed"] == 2026090801 and run["parents"] == 2
                         and run["point_count"] == 32 and run["candidate_compute_budget"] == 600.0
                         and run["horizon_policy"] == "native-end"
                         and run["evaluation_phase"] == "development")
                elapsed = run["evaluation_wall_seconds"]
                if not valid or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError(f"batch-{batch}-{mode}: incomplete, changed or invalid frozen matrix")
                totals[mode] += len(cases)
                times[mode] += elapsed
                row[mode] = {"evaluations": len(cases), "evaluation_wall_seconds": elapsed,
                             "canonical_source": run.get("canonical_source"),
                             "physical_source": run.get("physical_source")}
            comparison = load(f"batch-{batch}-compare/comparison.json")
            if (comparison["status"] != "equivalent" or comparison.get("problems")
                    or comparison["rtol"] != 1e-10 or comparison["atol"] != 1e-12):
                raise ValueError(f"batch-{batch}: numerical comparison did not pass")
            if workload(runs["baseline"]) != workload(runs["optimized"]):
                raise ValueError(f"batch-{batch}: public calls, points or failures changed")
            report["batches"][str(batch)] = row
        report["completed_evaluations_per_arm"] = totals
        split = load("split-execution/acceptance-summary.json")
        if (split.get("status") != "passed" or split.get("completed_evaluations") != 8
                or split.get("canonical_commit") != report["canonical_commit"]
                or split.get("backend_config_sha256") != report["backend_config_sha256"]):
            raise ValueError("real-asset split Candidate execution acceptance did not pass")
        report["split_execution"] = split
        if exit_code:
            raise ValueError(f"acceptance command failed with exit {exit_code}")
        report.update(status="passed", evaluation_wall_seconds=times,
                      evaluation_wall_speedup=times["baseline"] / times["optimized"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        report["problems"].append(f"{type(error).__name__}: {error}")
    return write_report(root, "acceptance-summary.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "junit", "summary", "candidates"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--batch", type=int, choices=(0, 1))
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode == "candidates":
        if args.batch is None:
            parser.error("candidates requires --batch")
        for candidate in CANDIDATES[3 * args.batch:3 * args.batch + 3]:
            print(f"{candidate[0]}={candidate_path(candidate)}")
        return 0
    if args.root is None:
        parser.error("report mode requires --root")
    if args.mode == "preflight":
        report = write_report(args.root, "source-state.json", source_state())
    elif args.mode == "junit":
        report = junit_gate(args.root)
    else:
        report = summarize(args.root, args.exit_code)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report.get("status", "passed") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
