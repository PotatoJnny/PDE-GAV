"""Fail-closed acceptance/publication gates for the authorized R1 cold recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_canonical_artemis_campaign import (
    _RunWriter, _source_state, BACKEND_CONFIG_SHA256,
    RECOVERY_SOURCE_HEAD, RECOVERY_RESOLVED_CONFIG_SHA256,
)

BRANCH = "codex/artemis-budget-rollback-20260909"
REMOTE = "${PDE_GAV_PHYSICAL_REPOSITORY}"


def load(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("acceptance input must be a regular non-symlink file")
    return json.loads(path.read_text())


def current_source():
    state = _source_state()
    if not state["commit"] or state["status_porcelain"]:
        raise ValueError("acceptance and launch require a clean committed checkout")
    return state


def junit_report(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {name: sum(int(suite.get(name, "0")) for suite in suites)
              for name in ("tests", "failures", "errors", "skipped")}
    if totals["tests"] <= 0 or any(totals[name] for name in ("failures", "errors", "skipped")):
        raise ValueError("acceptance tests must complete with no failure/error/skip")
    return {"junit_reported_tests": totals["tests"],
            "testcase_nodes": len(list(root.iter("testcase"))), **totals}


def summarize(root, exit_code):
    report = {"schema": "artemis_budget_recovery_acceptance_v1", "status": "failed",
              "exit_code": exit_code, "backend_config_sha256": BACKEND_CONFIG_SHA256}
    try:
        before, current = load(root / "source-state.json"), current_source()
        if before != current or exit_code != 0:
            raise ValueError("acceptance failed or source changed during validation")
        tests = {name: junit_report(root / name)
                 for name in ("deployment-tests.xml", "recovery-tests.xml")}
        smoke = load(root / "real-recovery" / "acceptance-summary.json")
        guidance = load(root / "guidance" / "acceptance-summary.json")
        for result in (smoke, guidance):
            if result.get("status") != "passed" or result.get("canonical_commit") != current["commit"]:
                raise ValueError("real acceptance must pass at the exact launch revision")
            if result.get("backend_config_sha256") != BACKEND_CONFIG_SHA256:
                raise ValueError("real acceptance used different backend assets/settings")
        if (smoke.get("candidate_evaluations") != 10
                or not smoke.get("active_wait_accounting_validated")
                or not smoke.get("immutable_r1_bundles_preserved")
                or guidance.get("guidance_fit_count") != 2
                or not guidance.get("same_round_guidance_frozen")):
            raise ValueError("real acceptance lacks frozen-candidate or fresh-Guidance checks")
        report.update(status="passed", source_state=current, tests=tests,
                      source_manifest_sha256=smoke["source_manifest_sha256"],
                      replay_verification=smoke["replay_verification"],
                      real_frozen_candidate_evaluations=10, guidance_fit_count=2,
                      candidate_workers=4, model_api_called=False)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    _RunWriter(root).write_json("acceptance-summary.json", report)
    return report


def publication(root):
    state = current_source()
    remote = subprocess.check_output(("git", "-C", str(ROOT), "remote", "get-url", "origin"), text=True).strip()
    if remote.removesuffix(".git") != REMOTE:
        raise ValueError("publication remote is not the configured collaborator repository")
    rows = subprocess.check_output(("git", "-C", str(ROOT), "ls-remote", "--heads", "origin",
                                    f"refs/heads/{BRANCH}"), text=True).strip().splitlines()
    if len(rows) != 1 or rows[0].split() != [state["commit"], f"refs/heads/{BRANCH}"]:
        raise ValueError("GitHub branch does not contain the exact accepted launch HEAD")
    report = {"schema": "artemis_budget_publication_v1", "status": "verified",
              "commit": state["commit"], "remote": REMOTE, "branch": BRANCH,
              "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    _RunWriter(root).write_json("publication.json", report)
    return report


def validate_launch(acceptance, published, source):
    if (source.get("status_porcelain") != ""
            or acceptance.get("schema") != "artemis_budget_recovery_acceptance_v1"
            or acceptance.get("status") != "passed"
            or acceptance.get("source_state", {}).get("commit") != source["commit"]
            or acceptance.get("source_state", {}).get("status_porcelain") != ""
            or acceptance.get("backend_config_sha256") != BACKEND_CONFIG_SHA256):
        raise ValueError("launch requires passing exact-revision acceptance")
    if (published.get("schema") != "artemis_budget_publication_v1"
            or published.get("status") != "verified"
            or published.get("commit") != source["commit"]
            or published.get("remote") != REMOTE or published.get("branch") != BRANCH):
        raise ValueError("launch requires verified GitHub publication of the accepted revision")
    return {"status": "passed", "launch_commit": source["commit"],
            "source_manifest_sha256": acceptance["source_manifest_sha256"],
            "candidate_workers": 4, "launch_mode": "r1_reevaluation_then_fresh_r2_to_r5"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "summary", "publication", "launch"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--source-campaign", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "launch":
        if args.acceptance is None or args.publication is None or args.source_campaign is None:
            parser.error("launch requires acceptance, publication and source campaign")
        report = validate_launch(load(args.acceptance), load(args.publication), current_source())
        from scripts.artemis_round_recovery import load_frozen_round_one
        frozen = load_frozen_round_one(
            args.source_campaign, expected_source_head=RECOVERY_SOURCE_HEAD,
            expected_config_sha256=RECOVERY_RESOLVED_CONFIG_SHA256,
            expected_backend_config_sha256=BACKEND_CONFIG_SHA256,
        )
        if frozen.source_manifest_sha256 != report["source_manifest_sha256"]:
            raise ValueError("frozen recovery inputs changed after real acceptance")
    else:
        if args.root is None:
            parser.error("report mode requires --root")
        if args.mode == "preflight":
            report = current_source()
            _RunWriter(args.root).write_json("source-state.json", report)
        elif args.mode == "publication":
            report = publication(args.root)
        else:
            report = summarize(args.root, args.exit_code)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report.get("status", "passed") in ("passed", "verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
