#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=artemis_backend_opt5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "D5 acceptance requires its own Slurm compute allocation." >&2
  exit 2
fi
ARTEMIS_D5="${PDE_GAV_WORKTREES_ROOT}/artemis-backend-opt5-20260909"
ARTEMIS_P5="${PDE_GAV_WORKTREES_ROOT}/artemis-oracle-opt5-20260909"
ARTEMIS_HARNESS="${PDE_GAV_WORKTREES_ROOT}/artemis-eval-benchmark-20260907"
ARTEMIS_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_OUTPUT="$ARTEMIS_D5/runs/backend-opt5-acceptance-20260909/job-$SLURM_JOB_ID"
cd -- "$ARTEMIS_D5"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing Python venv." >&2
fi
export PYTHONPATH="$ARTEMIS_D5:$ARTEMIS_D5/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/artemis-opt5-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$ARTEMIS_PYTHON" scripts/artemis_opt5_acceptance.py preflight --root "$ARTEMIS_OUTPUT"
artemis_finish() {
  local ARTEMIS_EXIT=$?
  trap - EXIT
  "$ARTEMIS_PYTHON" scripts/artemis_opt5_acceptance.py summary --root "$ARTEMIS_OUTPUT" --exit-code "$ARTEMIS_EXIT" || ARTEMIS_EXIT=1
  exit "$ARTEMIS_EXIT"
}
trap artemis_finish EXIT
artemis_step() {
  local ARTEMIS_LABEL="$1"
  shift
  echo "Starting offline stage: $ARTEMIS_LABEL"
  "$@" 2>&1 | tee "$ARTEMIS_OUTPUT/$ARTEMIS_LABEL.log"
}
artemis_in_tree() (
  local ARTEMIS_TEST_ROOT="$1"
  shift
  cd -- "$ARTEMIS_TEST_ROOT"
  "$@"
)
artemis_step physical-tests artemis_in_tree "$ARTEMIS_P5" env PYTHONPATH="$ARTEMIS_P5/src" "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  "$ARTEMIS_P5/tests/test_vp_1d1v_artemis_compact_factorized_field_v4.py" \
  "$ARTEMIS_P5/tests/test_vp_1d1v_artemis_compact_v5_production_runtime.py" \
  "$ARTEMIS_P5/tests/test_vp_1d1v_artemis_sensor_bin_perturbation_v4.py" \
  "$ARTEMIS_P5/tests/test_vp_1d1v_artemis_compact_sensor_bin_field_v4.py" \
  "$ARTEMIS_P5"/tests/test_artemis_opt2_*.py \
  "$ARTEMIS_P5/tests/test_vp_1d1v_artemis_frozen_profile.py" \
  "$ARTEMIS_P5/tests/test_artemis_opt4_trusted_source.py" \
  "$ARTEMIS_P5"/tests/test_artemis_opt5_*.py \
  --junitxml="$ARTEMIS_OUTPUT/physical-tests.xml"
artemis_step deployment-tests "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  tests/test_artemis_canonical.py tests/test_artemis_backend_deployment.py \
  tests/test_artemis_opt3_deployment.py tests/test_artemis_opt3_benchmark.py \
  tests/test_artemis_opt4_acceptance.py tests/test_artemis_opt4_benchmark.py \
  tests/test_artemis_opt4_rpc.py tests/test_candidate_execution.py \
  tests/test_candidate_solve_budget.py tests/test_artemis_solve_budget.py \
  tests/test_canonical_candidate_boundary.py tests/test_agent_capability.py \
  tests/test_artemis_opt5_acceptance.py --junitxml="$ARTEMIS_OUTPUT/deployment-tests.xml"
artemis_step runner-tests artemis_in_tree "$ARTEMIS_HARNESS" env PYTHONPATH="$ARTEMIS_HARNESS/src" "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  "$ARTEMIS_HARNESS/tests/test_artemis_oracle_benchmark.py" --junitxml="$ARTEMIS_OUTPUT/runner-tests.xml"
artemis_step junit-gate "$ARTEMIS_PYTHON" scripts/artemis_opt5_acceptance.py junit --root "$ARTEMIS_OUTPUT"

artemis_config() {
  if [[ "$1" == optimized ]]; then
    echo "$ARTEMIS_D5/configs/plasma_artemis_opt5.json"
  else
    echo "$ARTEMIS_D5/configs/plasma_artemis_opt5_baseline.json"
  fi
}
for ARTEMIS_MODE in baseline optimized baseline_repeat; do
  ARTEMIS_CONFIG="$(artemis_config "$ARTEMIS_MODE")"
  artemis_step "micro-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py micro \
    --label "micro-$ARTEMIS_MODE" --canonical-root "$ARTEMIS_D5" --backend-config "$ARTEMIS_CONFIG" \
    --threads 8 --seed 2026090903 --repeats 2 --transport direct --output-root "$ARTEMIS_OUTPUT/micro-$ARTEMIS_MODE"
done
for ARTEMIS_MODE in optimized baseline_repeat; do
  artemis_step "compare-micro-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py compare-micro \
    --baseline "$ARTEMIS_OUTPUT/micro-baseline/micro.json" --optimized "$ARTEMIS_OUTPUT/micro-$ARTEMIS_MODE/micro.json" \
    --rtol 1e-10 --atol 1e-12 --output-root "$ARTEMIS_OUTPUT/compare-micro-$ARTEMIS_MODE"
done
for ARTEMIS_MODE in baseline optimized; do
  ARTEMIS_CONFIG="$(artemis_config "$ARTEMIS_MODE")"
  artemis_step "http-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py micro \
    --label "http-$ARTEMIS_MODE" --canonical-root "$ARTEMIS_D5" --backend-config "$ARTEMIS_CONFIG" \
    --threads 8 --seed 2026090903 --repeats 2 --transport http --output-root "$ARTEMIS_OUTPUT/http-$ARTEMIS_MODE"
done
artemis_step compare-http-optimized "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py compare-micro \
  --baseline "$ARTEMIS_OUTPUT/http-baseline/micro.json" --optimized "$ARTEMIS_OUTPUT/http-optimized/micro.json" \
  --rtol 1e-10 --atol 1e-12 --output-root "$ARTEMIS_OUTPUT/compare-http-optimized"
for ARTEMIS_MODE in baseline optimized baseline_repeat; do
  ARTEMIS_CONFIG="$(artemis_config "$ARTEMIS_MODE")"
  artemis_step "native-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py run \
    --label "native-$ARTEMIS_MODE" --canonical-root "$ARTEMIS_D5" --backend-config "$ARTEMIS_CONFIG" \
    --threads 8 --seed 2026090903 --horizon native-end --output-root "$ARTEMIS_OUTPUT/native-$ARTEMIS_MODE"
done
for ARTEMIS_MODE in optimized baseline_repeat; do
  artemis_step "compare-native-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py compare \
    --baseline "$ARTEMIS_OUTPUT/native-baseline/result.json" --optimized "$ARTEMIS_OUTPUT/native-$ARTEMIS_MODE/result.json" \
    --rtol 1e-10 --atol 1e-12 --output-root "$ARTEMIS_OUTPUT/compare-native-$ARTEMIS_MODE"
done
