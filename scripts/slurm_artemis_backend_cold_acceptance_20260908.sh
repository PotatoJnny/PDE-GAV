#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=artemis_cold_acceptance
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Frozen-candidate engineering check only; no Agent/model API or formal run.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "A separate Slurm compute allocation is required." >&2
  exit 2
fi
ARTEMIS_DEPLOY="${PDE_GAV_WORKTREES_ROOT}/plasma-canonical-integration"
ARTEMIS_BASELINE="${PDE_GAV_WORKTREES_ROOT}/artemis-cold-baseline-20260908"
ARTEMIS_BENCHMARK="${PDE_GAV_WORKTREES_ROOT}/artemis-eval-benchmark-20260907"
ARTEMIS_RUNNER="$ARTEMIS_BENCHMARK/scripts/benchmark_canonical_artemis_oracle.py"
ARTEMIS_HELPER="$ARTEMIS_DEPLOY/scripts/artemis_backend_cold_acceptance.py"
ARTEMIS_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_OUTPUT="$ARTEMIS_DEPLOY/runs/cold-backend-acceptance-20260908/job-$SLURM_JOB_ID"
if [[ -e "$ARTEMIS_OUTPUT" || -L "$ARTEMIS_OUTPUT" ]]; then
  echo "Refusing to reuse acceptance output: $ARTEMIS_OUTPUT" >&2
  exit 2
fi
mkdir -p -- "$ARTEMIS_OUTPUT"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/artemis-cold-acceptance-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd -- "$ARTEMIS_DEPLOY"

artemis_step() {
  local ARTEMIS_STEP="$1"
  shift
  local ARTEMIS_RC=0
  echo "Artemis cold acceptance step started: $ARTEMIS_STEP"
  if "$@" 2>&1 | tee "$ARTEMIS_OUTPUT/$ARTEMIS_STEP.log"; then
    ARTEMIS_RC=0
  else
    ARTEMIS_RC=$?
  fi
  printf '%s\t%d\n' "$ARTEMIS_STEP" "$ARTEMIS_RC" >> "$ARTEMIS_OUTPUT/step-exit-codes.tsv"
  echo "Artemis cold acceptance step finished: $ARTEMIS_STEP (exit=$ARTEMIS_RC)"
  return "$ARTEMIS_RC"
}

artemis_finish() {
  local ARTEMIS_FINAL_STATUS=$?
  trap - EXIT
  # Always retain a failed summary on an ordinary error; never silently continue
  # a matrix after a test, numerical-equivalence or public-workload failure.
  if "$ARTEMIS_PYTHON" "$ARTEMIS_HELPER" summary --root "$ARTEMIS_OUTPUT" \
      --exit-code "$ARTEMIS_FINAL_STATUS" 2>&1 | tee "$ARTEMIS_OUTPUT/acceptance-summary.log"; then
    ARTEMIS_FINAL_STATUS=0
  else
    ARTEMIS_FINAL_STATUS=1
  fi
  echo "Artemis cold acceptance finished (status=$ARTEMIS_FINAL_STATUS): $ARTEMIS_OUTPUT"
  exit "$ARTEMIS_FINAL_STATUS"
}
trap artemis_finish EXIT
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing venv without installation." >&2
fi
artemis_step source-preflight "$ARTEMIS_PYTHON" "$ARTEMIS_HELPER" preflight --root "$ARTEMIS_OUTPUT"
artemis_step deployment-tests env PYTHONPATH="$ARTEMIS_DEPLOY/src" "$ARTEMIS_PYTHON" -m pytest -q \
  -p scripts.artemis_backend_cold_acceptance \
  tests/test_artemis_canonical.py tests/test_artemis_formal_runner.py \
  tests/test_artemis_backend_deployment.py tests/test_canonical_codex_agent.py \
  tests/test_candidate_execution.py tests/test_canonical_candidate_boundary.py \
  tests/test_unified_runtime_boundaries.py tests/test_agent_parallel_execution.py \
  tests/test_artemis_candidate_execution.py tests/test_artemis_split_execution_check.py \
  tests/test_artemis_cold_acceptance_script.py \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-cold-deployment" \
  --junitxml="$ARTEMIS_OUTPUT/deployment-tests.xml"
artemis_step runner-tests env PYTHONPATH="$ARTEMIS_BENCHMARK/src" "$ARTEMIS_PYTHON" -m pytest -q \
  -p scripts.artemis_backend_cold_acceptance \
  "$ARTEMIS_BENCHMARK/tests/test_artemis_oracle_benchmark.py" \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-cold-runner" \
  --junitxml="$ARTEMIS_OUTPUT/runner-tests.xml"
artemis_step junit-gate "$ARTEMIS_PYTHON" "$ARTEMIS_HELPER" junit --root "$ARTEMIS_OUTPUT"
artemis_step formal-config env PYTHONPATH="$ARTEMIS_DEPLOY/src" "$ARTEMIS_PYTHON" \
  scripts/run_canonical_artemis_campaign.py --config configs/artemis_canonical_5round.json --validate-only

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
# The unchanged harness supports at most four candidates and at least two
# BASE parents per process. Two batches of three cover R4 winner + all five R5
# nominations on the same two fresh BASE/PERT pairs: 24 evaluations per arm.
for ARTEMIS_BATCH in 0 1; do
  ARTEMIS_CANDIDATE_LINES="$("$ARTEMIS_PYTHON" "$ARTEMIS_HELPER" candidates --batch "$ARTEMIS_BATCH")"
  mapfile -t ARTEMIS_CANDIDATES <<< "$ARTEMIS_CANDIDATE_LINES"
  if [[ "${#ARTEMIS_CANDIDATES[@]}" -ne 3 ]]; then
    echo "Cold acceptance requires exactly three frozen candidates per batch." >&2
    exit 2
  fi
  ARTEMIS_ARGS=()
  for ARTEMIS_CANDIDATE in "${ARTEMIS_CANDIDATES[@]}"; do
    ARTEMIS_ARGS+=(--candidate "$ARTEMIS_CANDIDATE")
  done
  for ARTEMIS_MODE in baseline optimized; do
    ARTEMIS_CANON="$ARTEMIS_BASELINE"
    if [[ "$ARTEMIS_MODE" == optimized ]]; then ARTEMIS_CANON="$ARTEMIS_DEPLOY"; fi
    ARTEMIS_LABEL="batch-$ARTEMIS_BATCH-$ARTEMIS_MODE"
    artemis_step "$ARTEMIS_LABEL" env PYTHONPATH="$ARTEMIS_CANON/src" "$ARTEMIS_PYTHON" \
      "$ARTEMIS_RUNNER" run --label "$ARTEMIS_LABEL" --canonical-root "$ARTEMIS_CANON" \
      --backend-config "$ARTEMIS_CANON/configs/plasma_canonical_artemis.json" \
      "${ARTEMIS_ARGS[@]}" --parents 2 --seed 2026090801 --point-count 32 --horizon native-end \
      --output-root "$ARTEMIS_OUTPUT/$ARTEMIS_LABEL"
  done
  artemis_step "batch-$ARTEMIS_BATCH-compare" "$ARTEMIS_PYTHON" "$ARTEMIS_RUNNER" compare \
    --baseline "$ARTEMIS_OUTPUT/batch-$ARTEMIS_BATCH-baseline/result.json" \
    --optimized "$ARTEMIS_OUTPUT/batch-$ARTEMIS_BATCH-optimized/result.json" \
    --rtol 1e-10 --atol 1e-12 --output-root "$ARTEMIS_OUTPUT/batch-$ARTEMIS_BATCH-compare"
done
artemis_step split-execution env PYTHONPATH="$ARTEMIS_DEPLOY/src" "$ARTEMIS_PYTHON" \
  scripts/check_artemis_split_execution.py \
  --direct-result "$ARTEMIS_OUTPUT/batch-0-optimized/result.json" \
  --output-root "$ARTEMIS_OUTPUT/split-execution"
