#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=artemis_backend_opt4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "D4 acceptance requires its own Slurm compute allocation." >&2
  exit 2
fi
ARTEMIS_D4="${PDE_GAV_WORKTREES_ROOT}/artemis-backend-opt4-20260909"
ARTEMIS_P4="${PDE_GAV_WORKTREES_ROOT}/artemis-oracle-opt4-20260909"
ARTEMIS_HARNESS="${PDE_GAV_WORKTREES_ROOT}/artemis-eval-benchmark-20260907"
ARTEMIS_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_OUTPUT="$ARTEMIS_D4/runs/backend-opt4-acceptance-20260909/job-$SLURM_JOB_ID"
cd -- "$ARTEMIS_D4"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing Python venv." >&2
fi
export PYTHONPATH="$ARTEMIS_D4:$ARTEMIS_D4/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/artemis-opt4-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$ARTEMIS_PYTHON" scripts/artemis_opt4_acceptance.py preflight --root "$ARTEMIS_OUTPUT"
artemis_finish() {
  local ARTEMIS_EXIT=$?
  trap - EXIT
  "$ARTEMIS_PYTHON" scripts/artemis_opt4_acceptance.py summary --root "$ARTEMIS_OUTPUT" --exit-code "$ARTEMIS_EXIT" || ARTEMIS_EXIT=1
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
artemis_step physical-tests artemis_in_tree "$ARTEMIS_P4" env PYTHONPATH="$ARTEMIS_P4/src" "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  "$ARTEMIS_P4/tests/test_vp_1d1v_artemis_compact_factorized_field_v4.py" \
  "$ARTEMIS_P4/tests/test_vp_1d1v_artemis_compact_v5_production_runtime.py" \
  "$ARTEMIS_P4/tests/test_vp_1d1v_artemis_sensor_bin_perturbation_v4.py" \
  "$ARTEMIS_P4/tests/test_vp_1d1v_artemis_compact_sensor_bin_field_v4.py" \
  "$ARTEMIS_P4"/tests/test_artemis_opt2_*.py \
  "$ARTEMIS_P4/tests/test_vp_1d1v_artemis_frozen_profile.py" \
  "$ARTEMIS_P4/tests/test_artemis_opt4_trusted_source.py" \
  --junitxml="$ARTEMIS_OUTPUT/physical-tests.xml"
artemis_step deployment-tests "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  tests/test_artemis_canonical.py tests/test_artemis_backend_deployment.py \
  tests/test_artemis_opt3_deployment.py tests/test_artemis_opt3_benchmark.py \
  tests/test_artemis_opt4_acceptance.py tests/test_artemis_opt4_benchmark.py \
  tests/test_artemis_opt4_rpc.py tests/test_candidate_execution.py \
  tests/test_candidate_solve_budget.py tests/test_artemis_solve_budget.py \
  tests/test_canonical_candidate_boundary.py tests/test_agent_capability.py \
  --junitxml="$ARTEMIS_OUTPUT/deployment-tests.xml"
artemis_step runner-tests artemis_in_tree "$ARTEMIS_HARNESS" env PYTHONPATH="$ARTEMIS_HARNESS/src" "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  "$ARTEMIS_HARNESS/tests/test_artemis_oracle_benchmark.py" --junitxml="$ARTEMIS_OUTPUT/runner-tests.xml"
artemis_step junit-gate "$ARTEMIS_PYTHON" scripts/artemis_opt4_acceptance.py junit --root "$ARTEMIS_OUTPUT"
artemis_step rpc-transport "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4_rpc.py \
  --scratch-parent ${PDE_GAV_RUN_ROOT} --local-parent "$SLURM_TMPDIR" \
  --output "$ARTEMIS_OUTPUT/rpc-transport.json" --iterations 32

for ARTEMIS_MODE in baseline b1024 b2048 b4096 baseline_repeat; do
  ARTEMIS_CONFIG="$ARTEMIS_D4/configs/plasma_artemis_opt4_${ARTEMIS_MODE}.json"
  if [[ "$ARTEMIS_MODE" == baseline_repeat ]]; then ARTEMIS_CONFIG="$ARTEMIS_D4/configs/plasma_artemis_opt4_baseline.json"; fi
  artemis_step "micro-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py micro \
    --label "micro-$ARTEMIS_MODE" --canonical-root "$ARTEMIS_D4" --backend-config "$ARTEMIS_CONFIG" \
    --threads 8 --seed 2026090903 --transport direct --output-root "$ARTEMIS_OUTPUT/micro-$ARTEMIS_MODE"
done
for ARTEMIS_MODE in b1024 b2048 b4096 baseline_repeat; do
  artemis_step "compare-micro-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py compare-micro \
    --baseline "$ARTEMIS_OUTPUT/micro-baseline/micro.json" --optimized "$ARTEMIS_OUTPUT/micro-$ARTEMIS_MODE/micro.json" \
    --output-root "$ARTEMIS_OUTPUT/compare-micro-$ARTEMIS_MODE"
done
for ARTEMIS_MODE in baseline b4096; do
  artemis_step "http-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py micro \
    --label "http-$ARTEMIS_MODE" --canonical-root "$ARTEMIS_D4" \
    --backend-config "$ARTEMIS_D4/configs/plasma_artemis_opt4_${ARTEMIS_MODE}.json" \
    --threads 8 --seed 2026090903 --transport http --output-root "$ARTEMIS_OUTPUT/http-$ARTEMIS_MODE"
done
artemis_step compare-http-b4096 "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py compare-micro \
  --baseline "$ARTEMIS_OUTPUT/http-baseline/micro.json" --optimized "$ARTEMIS_OUTPUT/http-b4096/micro.json" \
  --output-root "$ARTEMIS_OUTPUT/compare-http-b4096"
for ARTEMIS_MODE in baseline optimized baseline_repeat; do
  ARTEMIS_CONFIG="$ARTEMIS_D4/configs/plasma_artemis_opt4_baseline.json"
  if [[ "$ARTEMIS_MODE" == optimized ]]; then ARTEMIS_CONFIG="$ARTEMIS_D4/configs/plasma_artemis_opt4_b4096.json"; fi
  artemis_step "native-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py run \
    --label "native-$ARTEMIS_MODE" --canonical-root "$ARTEMIS_D4" --backend-config "$ARTEMIS_CONFIG" \
    --threads 8 --seed 2026090903 --horizon native-end --output-root "$ARTEMIS_OUTPUT/native-$ARTEMIS_MODE"
done
for ARTEMIS_MODE in optimized baseline_repeat; do
  artemis_step "compare-native-$ARTEMIS_MODE" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt4.py compare \
    --baseline "$ARTEMIS_OUTPUT/native-baseline/result.json" --optimized "$ARTEMIS_OUTPUT/native-$ARTEMIS_MODE/result.json" \
    --output-root "$ARTEMIS_OUTPUT/compare-native-$ARTEMIS_MODE"
done
