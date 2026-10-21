#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=artemis_backend_opt3
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Offline source acceptance requires its own Slurm compute allocation." >&2
  exit 2
fi
ARTEMIS_D3="${PDE_GAV_WORKTREES_ROOT}/artemis-backend-opt3-20260909"
ARTEMIS_P3="${PDE_GAV_WORKTREES_ROOT}/artemis-oracle-opt3-20260909"
ARTEMIS_HARNESS="${PDE_GAV_WORKTREES_ROOT}/artemis-eval-benchmark-20260907"
ARTEMIS_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_OUTPUT="$ARTEMIS_D3/runs/backend-opt3-acceptance-20260909/job-$SLURM_JOB_ID"
ARTEMIS_SCRATCH="${PDE_GAV_RUN_ROOT}/artemis_opt3_offline_20260909/job-$SLURM_JOB_ID"
cd -- "$ARTEMIS_D3"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing Python venv." >&2
fi
export PYTHONPATH="$ARTEMIS_D3:$ARTEMIS_D3/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/artemis-opt3-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$ARTEMIS_PYTHON" scripts/artemis_opt3_acceptance.py preflight --root "$ARTEMIS_OUTPUT"
artemis_finish() {
  local ARTEMIS_EXIT=$?
  trap - EXIT
  "$ARTEMIS_PYTHON" scripts/artemis_opt3_acceptance.py summary --root "$ARTEMIS_OUTPUT" --exit-code "$ARTEMIS_EXIT" || ARTEMIS_EXIT=1
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
artemis_step physical-tests artemis_in_tree "$ARTEMIS_P3" env PYTHONPATH="$ARTEMIS_P3/src" "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  "$ARTEMIS_P3/tests/test_vp_1d1v_artemis_compact_factorized_field_v4.py" \
  "$ARTEMIS_P3/tests/test_vp_1d1v_artemis_compact_v5_production_runtime.py" \
  "$ARTEMIS_P3/tests/test_vp_1d1v_artemis_sensor_bin_perturbation_v4.py" \
  "$ARTEMIS_P3/tests/test_vp_1d1v_artemis_compact_sensor_bin_field_v4.py" \
  "$ARTEMIS_P3"/tests/test_artemis_opt2_*.py "$ARTEMIS_P3/tests/test_vp_1d1v_artemis_frozen_profile.py" \
  --junitxml="$ARTEMIS_OUTPUT/physical-tests.xml"
artemis_step deployment-tests "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  tests/test_artemis_canonical.py tests/test_artemis_backend_deployment.py \
  tests/test_artemis_opt3_deployment.py tests/test_artemis_opt3_acceptance.py \
  tests/test_artemis_opt3_benchmark.py tests/test_artemis_oracle_wire_benchmark.py \
  --junitxml="$ARTEMIS_OUTPUT/deployment-tests.xml"
artemis_step runner-tests artemis_in_tree "$ARTEMIS_HARNESS" env PYTHONPATH="$ARTEMIS_HARNESS/src" "$ARTEMIS_PYTHON" -m pytest -o addopts='' -q \
  "$ARTEMIS_HARNESS/tests/test_artemis_oracle_benchmark.py" --junitxml="$ARTEMIS_OUTPUT/runner-tests.xml"
artemis_step junit-gate "$ARTEMIS_PYTHON" scripts/artemis_opt3_acceptance.py junit --root "$ARTEMIS_OUTPUT"
artemis_step wire "$ARTEMIS_PYTHON" scripts/benchmark_artemis_oracle_wire.py \
  --scratch-parent "$ARTEMIS_SCRATCH/wire-scratch" --local-parent "$SLURM_TMPDIR" \
  --output "$ARTEMIS_OUTPUT/wire.json" --iterations 32 --warmup 2

for ARTEMIS_THREADS in 8 1; do
  ARTEMIS_MODES=(baseline optimized)
  if [[ "$ARTEMIS_THREADS" == 8 ]]; then ARTEMIS_MODES+=(baseline_repeat); fi
  for ARTEMIS_MODE in "${ARTEMIS_MODES[@]}"; do
    ARTEMIS_CONFIG="$ARTEMIS_D3/configs/plasma_artemis_opt3_benchmark_baseline.json"
    if [[ "$ARTEMIS_MODE" == optimized ]]; then ARTEMIS_CONFIG="$ARTEMIS_D3/configs/plasma_artemis_opt3_benchmark.json"; fi
    ARTEMIS_LABEL="micro-$ARTEMIS_THREADS-$ARTEMIS_MODE"
    artemis_step "$ARTEMIS_LABEL" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt3.py micro \
      --label "$ARTEMIS_LABEL" --canonical-root "$ARTEMIS_D3" --backend-config "$ARTEMIS_CONFIG" \
      --threads "$ARTEMIS_THREADS" --seed 2026090903 --output-root "$ARTEMIS_OUTPUT/$ARTEMIS_LABEL"
  done
  for ARTEMIS_MODE in "${ARTEMIS_MODES[@]:1}"; do
    ARTEMIS_LABEL="compare-micro-$ARTEMIS_THREADS-$ARTEMIS_MODE"
    artemis_step "$ARTEMIS_LABEL" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt3.py compare-micro \
      --baseline "$ARTEMIS_OUTPUT/micro-$ARTEMIS_THREADS-baseline/micro.json" \
      --optimized "$ARTEMIS_OUTPUT/micro-$ARTEMIS_THREADS-$ARTEMIS_MODE/micro.json" \
      --output-root "$ARTEMIS_OUTPUT/$ARTEMIS_LABEL"
  done
done
for ARTEMIS_MODE in baseline optimized baseline_repeat; do
  ARTEMIS_CONFIG="$ARTEMIS_D3/configs/plasma_artemis_opt3_benchmark_baseline.json"
  if [[ "$ARTEMIS_MODE" == optimized ]]; then ARTEMIS_CONFIG="$ARTEMIS_D3/configs/plasma_artemis_opt3_benchmark.json"; fi
  ARTEMIS_LABEL="native-$ARTEMIS_MODE"
  artemis_step "$ARTEMIS_LABEL" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt3.py run \
    --label "$ARTEMIS_LABEL" --canonical-root "$ARTEMIS_D3" --backend-config "$ARTEMIS_CONFIG" \
    --threads 8 --seed 2026090903 --horizon native-end --output-root "$ARTEMIS_OUTPUT/$ARTEMIS_LABEL"
done
for ARTEMIS_MODE in optimized baseline_repeat; do
  ARTEMIS_LABEL="compare-native-$ARTEMIS_MODE"
  artemis_step "$ARTEMIS_LABEL" "$ARTEMIS_PYTHON" scripts/benchmark_artemis_opt3.py compare \
    --baseline "$ARTEMIS_OUTPUT/native-baseline/result.json" \
    --optimized "$ARTEMIS_OUTPUT/native-$ARTEMIS_MODE/result.json" \
    --output-root "$ARTEMIS_OUTPUT/$ARTEMIS_LABEL"
done
