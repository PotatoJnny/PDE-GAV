#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=artemis_backend_opt2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Offline backend acceptance only. No Agent/model API, credentials, formal-run
# restart or scheduler mutation. Create the SBATCH log directory before submit.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "A separate Slurm compute allocation is required." >&2
  exit 2
fi
ARTEMIS_D1="${PDE_GAV_WORKTREES_ROOT}/artemis-backend-integrated-20260907"
ARTEMIS_D2="${PDE_GAV_WORKTREES_ROOT}/artemis-backend-opt2-20260907"
ARTEMIS_P1="${PDE_GAV_WORKTREES_ROOT}/artemis-oracle-opt-20260907"
ARTEMIS_P2="${PDE_GAV_WORKTREES_ROOT}/artemis-oracle-opt2-20260907"
ARTEMIS_RUNNER_ROOT="${PDE_GAV_WORKTREES_ROOT}/artemis-eval-benchmark-20260907"
ARTEMIS_RUNNER="$ARTEMIS_RUNNER_ROOT/scripts/benchmark_canonical_artemis_oracle.py"
ARTEMIS_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_TURNS="${PDE_GAV_RUN_ROOT}/artemis_canonical_5round_20260905/job-5243774/campaign/agent_turns"
ARTEMIS_S1="$ARTEMIS_TURNS/round-0001-artemis-critical-refiner-8327bd53f1e046ae93d9719ade25965a/.controller-candidates/a85878d17af49f15c2de7a5c6b1f617d805bfde9247a4f4ef184f538cdc2f339"
ARTEMIS_S2="$ARTEMIS_TURNS/round-0002-artemis-critical-refiner-593ae19a865e4808b90d659f75c94b0d/.controller-candidates/07540f9afad42fd504bf99fae71fbaafdec8510038921187ed184c9a72ad24c0"
ARTEMIS_OUTPUT="$ARTEMIS_D2/runs/backend-opt2-acceptance-20260907/job-$SLURM_JOB_ID"

if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing venv." >&2
fi
for ARTEMIS_TREE in "$ARTEMIS_D1" "$ARTEMIS_D2" "$ARTEMIS_P1" "$ARTEMIS_P2" "$ARTEMIS_RUNNER_ROOT"; do
  ARTEMIS_SOURCE_STATUS="$(git -C "$ARTEMIS_TREE" status --porcelain --untracked-files=all)"
  if [[ -n "$ARTEMIS_SOURCE_STATUS" ]]; then
    echo "Acceptance requires committed clean source: $ARTEMIS_TREE" >&2
    exit 2
  fi
done
if [[ -e "$ARTEMIS_OUTPUT" || -L "$ARTEMIS_OUTPUT" ]]; then
  echo "Refusing to reuse acceptance output: $ARTEMIS_OUTPUT" >&2
  exit 2
fi
if [[ ! -f "$ARTEMIS_S1/candidate_bundle.json" || ! -f "$ARTEMIS_S2/candidate_bundle.json" ]]; then
  echo "A required frozen S1/S2 candidate bundle is missing." >&2
  exit 2
fi
shopt -s nullglob
ARTEMIS_NEW_TESTS=("$ARTEMIS_P2"/tests/test_artemis_opt2_*.py)
shopt -u nullglob
if [[ ${#ARTEMIS_NEW_TESTS[@]} -eq 0 ]]; then
  echo "Missing required P2 tests/test_artemis_opt2_*.py; refusing to skip new-path tests." >&2
  exit 2
fi
mkdir -p -- "$ARTEMIS_OUTPUT"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/artemis-opt2-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd -- "$ARTEMIS_D2"

# Keep raw combined stdout/stderr plus exit code for every attempted step.
# tee preserves the runner's per-case heartbeat in the Slurm log as well.
artemis_step() {
  local ARTEMIS_STEP="$1"
  shift
  local ARTEMIS_RC=0
  echo "Artemis opt2 step started: $ARTEMIS_STEP"
  if "$@" 2>&1 | tee "$ARTEMIS_OUTPUT/$ARTEMIS_STEP.log"; then
    ARTEMIS_RC=0
  else
    ARTEMIS_RC=$?
  fi
  printf '%s\t%d\n' "$ARTEMIS_STEP" "$ARTEMIS_RC" >> "$ARTEMIS_OUTPUT/step-exit-codes.tsv"
  echo "Artemis opt2 step finished: $ARTEMIS_STEP (exit=$ARTEMIS_RC)"
  return "$ARTEMIS_RC"
}

artemis_in_tree() (
  local ARTEMIS_TEST_TREE="$1"
  shift
  cd -- "$ARTEMIS_TEST_TREE"
  "$@"
)

ARTEMIS_GATE_STATUS=0
artemis_step physical-tests artemis_in_tree "$ARTEMIS_P2" env PYTHONPATH="$ARTEMIS_P2/src" "$ARTEMIS_PYTHON" -m pytest -q \
  "$ARTEMIS_P2/tests/test_vp_1d1v_artemis_compact_factorized_field_v4.py" \
  "$ARTEMIS_P2/tests/test_vp_1d1v_artemis_compact_v5_production_runtime.py" \
  "$ARTEMIS_P2/tests/test_vp_1d1v_artemis_sensor_bin_perturbation_v4.py" \
  "$ARTEMIS_P2/tests/test_vp_1d1v_artemis_compact_sensor_bin_field_v4.py" \
  "${ARTEMIS_NEW_TESTS[@]}" \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-opt2-physical" \
  --junitxml="$ARTEMIS_OUTPUT/physical-tests.xml" || ARTEMIS_GATE_STATUS=1
artemis_step deployment-tests env PYTHONPATH="$ARTEMIS_D2/src" "$ARTEMIS_PYTHON" -m pytest -q \
  tests/test_artemis_canonical.py tests/test_artemis_formal_runner.py \
  tests/test_artemis_backend_deployment.py tests/test_artemis_opt2_acceptance_script.py \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-opt2-deployment" \
  --junitxml="$ARTEMIS_OUTPUT/deployment-tests.xml" || ARTEMIS_GATE_STATUS=1
artemis_step runner-tests artemis_in_tree "$ARTEMIS_RUNNER_ROOT" env PYTHONPATH="$ARTEMIS_RUNNER_ROOT/src" "$ARTEMIS_PYTHON" -m pytest -q \
  "$ARTEMIS_RUNNER_ROOT/tests/test_artemis_oracle_benchmark.py" \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-opt2-runner" \
  --junitxml="$ARTEMIS_OUTPUT/runner-tests.xml" || ARTEMIS_GATE_STATUS=1
# A successful pytest exit is insufficient: asset/dependency skips must not
# qualify the optimized backend. Parse all three reports before any timing run.
artemis_step junit-gate "$ARTEMIS_PYTHON" - "$ARTEMIS_OUTPUT" <<'JUNIT' || ARTEMIS_GATE_STATUS=1
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
keys = ("tests", "failures", "errors", "skipped")
report = {"schema": "artemis_opt2_junit_gate_v1", "reports": {},
          "totals": {key: 0 for key in keys}, "problems": []}
for filename in ("physical-tests.xml", "deployment-tests.xml", "runner-tests.xml"):
    row = {key: 0 for key in keys}
    row["status"] = "failed"
    report["reports"][filename] = row
    try:
        tree = ET.parse(root / filename).getroot()
        suites = list(tree.iter("testsuite"))
        leaves = [suite for suite in suites if not suite.findall(".//testsuite")]
        if not leaves:
            raise ValueError("JUnit has no leaf testsuite")
        # Leaf totals avoid double-counting a parent suite's aggregate counts.
        for suite in leaves:
            counts = {key: int(suite.attrib[key]) for key in keys}
            if counts["tests"] <= 0 or any(value < 0 for value in counts.values()):
                raise ValueError("JUnit contains an empty suite or invalid counts")
            for key in keys:
                row[key] += counts[key]
        observed = {"tests": len(list(tree.iter("testcase"))),
                    "failures": len(list(tree.iter("failure"))),
                    "errors": len(list(tree.iter("error"))),
                    "skipped": len(list(tree.iter("skipped")))}
        row["observed"] = observed
        if any(row[key] != observed[key] for key in keys):
            raise ValueError("JUnit declared and observed test counts disagree")
        # Also reject an enclosing suite that explicitly reports any failure,
        # error, skip or zero tests, even if its leaf counters are inconsistent.
        for suite in (*suites, *tree.iter("testsuites")):
            if "tests" in suite.attrib and int(suite.attrib["tests"]) <= 0:
                raise ValueError("JUnit suite declares zero tests")
            if any(int(suite.attrib.get(key, "0")) != 0 for key in keys[1:]):
                raise ValueError("JUnit contains failures, errors or skipped tests")
        row["passed"] = row["tests"]
        row["status"] = "passed"
    except (OSError, ET.ParseError, KeyError, TypeError, ValueError) as error:
        row["reason"] = f"{type(error).__name__}: {error}"
        report["problems"].append(f"{filename}: {row['reason']}")
    for key in keys:
        report["totals"][key] += row[key]
report["status"] = "passed" if not report["problems"] else "failed"
(root / "junit-counts.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True), flush=True)
raise SystemExit(0 if report["status"] == "passed" else 1)
JUNIT
artemis_step formal-config env PYTHONPATH="$ARTEMIS_D2/src" "$ARTEMIS_PYTHON" \
  scripts/run_canonical_artemis_campaign.py --config configs/artemis_canonical_5round.json \
  --validate-only || ARTEMIS_GATE_STATUS=1

ARTEMIS_STATUS="$ARTEMIS_GATE_STATUS"
if [[ "$ARTEMIS_GATE_STATUS" -eq 0 ]]; then
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
  # Six independent processes; each runs 2 frozen candidates x 2 fresh BASE
  # parents + paired PERT children = 8 evaluations, at unchanged 32 points/600s.
  # The same new seed draws the same cohort under both predeclared horizons.
  for ARTEMIS_HORIZON in private-random native-end; do
    for ARTEMIS_MODE in baseline optimized baseline_repeat; do
      ARTEMIS_CANON="$ARTEMIS_D1"
      if [[ "$ARTEMIS_MODE" == optimized ]]; then ARTEMIS_CANON="$ARTEMIS_D2"; fi
      ARTEMIS_LABEL="$ARTEMIS_HORIZON-$ARTEMIS_MODE"
      artemis_step "$ARTEMIS_LABEL" env PYTHONPATH="$ARTEMIS_CANON/src" "$ARTEMIS_PYTHON" \
        "$ARTEMIS_RUNNER" run --label "$ARTEMIS_LABEL" --canonical-root "$ARTEMIS_CANON" \
        --backend-config "$ARTEMIS_CANON/configs/plasma_canonical_artemis.json" \
        --candidate "S1=$ARTEMIS_S1" --candidate "S2=$ARTEMIS_S2" \
        --parents 2 --seed 2026090703 --point-count 32 --horizon "$ARTEMIS_HORIZON" \
        --output-root "$ARTEMIS_OUTPUT/$ARTEMIS_LABEL" || ARTEMIS_STATUS=1
    done
    for ARTEMIS_MODE in optimized baseline_repeat; do
      artemis_step "compare-$ARTEMIS_HORIZON-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" "$ARTEMIS_RUNNER" compare \
        --baseline "$ARTEMIS_OUTPUT/$ARTEMIS_HORIZON-baseline/result.json" \
        --optimized "$ARTEMIS_OUTPUT/$ARTEMIS_HORIZON-$ARTEMIS_MODE/result.json" \
        --rtol 1e-10 --atol 1e-12 \
        --output-root "$ARTEMIS_OUTPUT/compare-$ARTEMIS_HORIZON-$ARTEMIS_MODE" || ARTEMIS_STATUS=1
    done
  done
else
  echo "Matrix skipped: at least one correctness/configuration gate failed; all attempted gate artifacts retained."
fi

# Standard-library summary only: no Torch, NumPy, source mutation or Agent.
# Speed ratios are withheld if any comparison, completeness or workload check
# fails. Repeat timing drift is reported, not silently treated as a speedup.
artemis_step acceptance-summary "$ARTEMIS_PYTHON" - "$ARTEMIS_OUTPUT" "$ARTEMIS_GATE_STATUS" "$ARTEMIS_STATUS" <<'PY' || ARTEMIS_STATUS=1
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = {"schema": "artemis_backend_opt2_acceptance_v1", "short_benchmark_only": True,
          "formal_scientific_admission": False, "agent_invoked": False,
          "gate_status": int(sys.argv[2]), "step_status": int(sys.argv[3]),
          "horizons": {}, "problems": []}
expected_candidates = [
    ("S1", "a85878d17af49f15c2de7a5c6b1f617d805bfde9247a4f4ef184f538cdc2f339"),
    ("S2", "07540f9afad42fd504bf99fae71fbaafdec8510038921187ed184c9a72ad24c0"),
]

def load(relative):
    try:
        return json.loads((root / relative).read_text())
    except (OSError, ValueError) as error:
        report["problems"].append(f"{relative}: {type(error).__name__}: {error}")
        return {}

def workload(run):
    return [(case["candidate"], case["case_key"], [
        (session["access"], {name: {key: values[key] for key in ("calls", "points", "failures")}
                             for name, values in session["methods"].items()})
        for session in case["public_sessions"]]) for case in run["cases"]]

if not report["gate_status"]:
    for horizon in ("private-random", "native-end"):
        runs = {mode: load(f"{horizon}-{mode}/result.json")
                for mode in ("baseline", "optimized", "baseline_repeat")}
        row = {"runs": {}, "comparisons": {}}
        report["horizons"][horizon] = row
        before = len(report["problems"])
        for mode, run in runs.items():
            cases, cohort = run.get("cases", []), run.get("cohort", [])
            case_ids = [(case.get("candidate"), case.get("case_key")) for case in cases]
            candidates = [(item.get("name"), item.get("tree_sha256")) for item in run.get("candidates", [])]
            valid = (run.get("status") == "completed" and len(cases) == len(set(case_ids)) == 8
                     and len(cohort) == len({item.get("key") for item in cohort}) == 4
                     and candidates == expected_candidates
                     and all(case.get("status") == "completed" for case in cases))
            elapsed = run.get("evaluation_wall_seconds", 0)
            valid = valid and isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0
            if not valid:
                report["problems"].append(f"{horizon}-{mode}: incomplete/duplicate/failed results or candidate mismatch")
            row["runs"][mode] = {"status": run.get("status", "missing"), "evaluation_wall_seconds": elapsed,
                "generation_wall_seconds": run.get("generation_wall_seconds"),
                "experiment_wall_seconds": run.get("experiment_wall_seconds"),
                "runner_elapsed_seconds": run.get("finished_at_unix", 0) - run.get("started_at_unix", 0)}
        for mode in ("optimized", "baseline_repeat"):
            comparison = load(f"compare-{horizon}-{mode}/comparison.json")
            row["comparisons"][mode] = comparison.get("status", "missing")
            if comparison.get("status") != "equivalent":
                report["problems"].append(f"{horizon}-{mode}: numeric comparison did not pass")
            try:
                assert workload(runs["baseline"]) == workload(runs[mode])
            except (AssertionError, KeyError, TypeError) as error:
                report["problems"].append(f"{horizon}-{mode}: public workload changed or missing ({type(error).__name__})")
        if len(report["problems"]) == before:
            baseline = runs["baseline"]["evaluation_wall_seconds"]
            row["evaluation_wall_speedup"] = baseline / runs["optimized"]["evaluation_wall_seconds"]
            row["baseline_repeat_eval_drift_percent"] = 100 * (runs["baseline_repeat"]["evaluation_wall_seconds"] / baseline - 1)
report["status"] = "passed" if not (report["gate_status"] or report["step_status"] or report["problems"]) else "failed"
if report["status"] != "passed":
    for row in report["horizons"].values():
        row.pop("evaluation_wall_speedup", None)
(root / "acceptance-summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True), flush=True)
raise SystemExit(0 if report["status"] == "passed" else 1)
PY
echo "Artemis opt2 acceptance finished (status=$ARTEMIS_STATUS): $ARTEMIS_OUTPUT"
exit "$ARTEMIS_STATUS"
