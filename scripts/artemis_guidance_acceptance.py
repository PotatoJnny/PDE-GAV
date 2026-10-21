"""Short engineering acceptance and launch checks for Artemis B/G/P.

No new statistical model-quality threshold is imposed here. The existing D2
numerical acceptance is reused; only the added Guidance path is exercised.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COLD_COMMIT = "71c855451486c192f41dd7a427907f916d78afb6"
COLD_BACKEND_SHA256 = "39250e34c86133302b4ae0d597e1a5a171e695d3c1455feb97d6564f1f78da11"


def source_state(root=ROOT):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Guidance acceptance requires a clean committed checkout")
    return {
        "canonical_commit": git("rev-parse", "HEAD"),
        "backend_config_sha256": sha256((root / "configs/plasma_canonical_artemis.json").read_bytes()).hexdigest(),
        "campaign_config_sha256": sha256((root / "configs/artemis_canonical_5round.json").read_bytes()).hexdigest(),
    }


def write_report(root, name, report):
    (root / name).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def summarize(root, exit_code):
    report = {"schema": "artemis_guidance_engineering_acceptance_v1", "status": "failed",
              "model_api_called": False, "scientific_quality_gate": False,
              "step_exit_code": exit_code, "problems": []}
    try:
        before = json.loads((root / "source-state.json").read_text())
        report.update(before)
        if before != source_state():
            raise ValueError("Guidance code or configuration changed during acceptance")
        junit = json.loads((root / "junit-counts.json").read_text())
        if junit.get("status") != "passed":
            raise ValueError("Guidance regression tests did not pass")
        report["tests"] = junit["totals"]
        smoke = json.loads((root / "smoke/acceptance-summary.json").read_text())
        if (smoke.get("status") != "passed" or smoke.get("rounds_completed") != 2
                or smoke.get("guidance_fit_count") != 2 or smoke.get("guided_case_count", 0) <= 0
                or smoke.get("canonical_commit") != before["canonical_commit"]
                or smoke.get("backend_config_sha256") != before["backend_config_sha256"]):
            raise ValueError("Two-round real Guidance engineering smoke did not pass")
        report["smoke"] = smoke
        if exit_code:
            raise ValueError(f"Guidance acceptance command failed with exit {exit_code}")
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        report["problems"].append(f"{type(error).__name__}: {error}")
    return write_report(root, "acceptance-summary.json", report)


def validate_launch(cold, guidance, current):
    if (cold.get("status") != "passed" or cold.get("canonical_commit") != COLD_COMMIT
            or cold.get("backend_config_sha256") != COLD_BACKEND_SHA256):
        raise ValueError("The frozen D2 numerical acceptance has not passed")
    if guidance.get("status") != "passed":
        raise ValueError("Guidance engineering acceptance has not passed")
    if any(guidance.get(key) != value for key, value in current.items()):
        raise ValueError("Code or configuration changed since Guidance engineering acceptance")
    return {**current, "status": "passed", "launch_mode": "new_full_five_round_bgp_campaign",
            "scientific_quality_gate": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "summary", "launch"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--cold-report", type=Path)
    parser.add_argument("--guidance-report", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "launch":
        if args.cold_report is None or args.guidance_report is None:
            parser.error("launch requires cold and guidance reports")
        report = validate_launch(json.loads(args.cold_report.read_text()),
                                 json.loads(args.guidance_report.read_text()), source_state())
    else:
        if args.root is None:
            parser.error("report mode requires --root")
        report = (write_report(args.root, "source-state.json", source_state())
                  if args.mode == "preflight" else summarize(args.root, args.exit_code))
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report.get("status", "passed") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
