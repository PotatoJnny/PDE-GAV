#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=vpfp_guide_debug
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
VPFP_DEBUG_ROOT=${PDE_GAV_WORKTREES_ROOT}/vpfp-demand-batched-oracle-20260908
VPFP_DEBUG_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-guidance-training-debug-$SLURM_JOB_ID"
VPFP_DEBUG_PYTHON=${PDE_GAV_PYTHON}
VPFP_DEBUG_STARTED=$SECONDS
VPFP_DEBUG_TEST_EXIT=not_run
VPFP_DEBUG_RUN_EXIT=not_run
VPFP_DEBUG_REVISION=unverified
VPFP_DEBUG_FINAL=false
test -n "$SLURM_JOB_ID"
test -n "$SLURM_TMPDIR"
test ! -e "$VPFP_DEBUG_OUTPUT"
test ! -L "$VPFP_DEBUG_OUTPUT"
mkdir -- "$VPFP_DEBUG_OUTPUT"
vpfp_debug_finish() {
  local code=$?
  trap - EXIT
  if [[ "$(git -C "$VPFP_DEBUG_ROOT" rev-parse HEAD)" == "$VPFP_DEBUG_REVISION" && -z "$(git -C "$VPFP_DEBUG_ROOT" status --porcelain)" ]]; then
    VPFP_DEBUG_FINAL=true
  elif [[ "$code" == 0 ]]; then code=1; fi
  printf 'field\tvalue\nsource_revision\t%s\nsource_final_verified\t%s\ntests\t%s\ndiagnosis\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t1\nproduction_training_updates\t0\ndiagnostic_replay_requested_updates\t500\n' \
    "$VPFP_DEBUG_REVISION" "$VPFP_DEBUG_FINAL" "$VPFP_DEBUG_TEST_EXIT" \
    "$VPFP_DEBUG_RUN_EXIT" "$code" "$((SECONDS-VPFP_DEBUG_STARTED))" \
    > "$VPFP_DEBUG_OUTPUT/launch.tsv"
  exit "$code"
}
trap vpfp_debug_finish EXIT
cd "$VPFP_DEBUG_ROOT"
VPFP_DEBUG_REVISION="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
set +eu
set +o pipefail
source /etc/profile
set -euo pipefail
if ! module load StdEnv/2023 python/3.12 scipy-stack pytest; then
  echo 'Cluster modules unavailable; using existing unchanged venv.' >&2
fi
test -x "$VPFP_DEBUG_PYTHON"
export PYTHONPATH="$VPFP_DEBUG_ROOT/src:$VPFP_DEBUG_ROOT/scripts"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/vpfp-guidance-debug-pycache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
VPFP_DEBUG_TEST_EXIT=0
CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=10s 180 \
  "$VPFP_DEBUG_PYTHON" -m pytest -o addopts= -q -p no:cacheprovider \
  --basetemp "$SLURM_TMPDIR/vpfp-guidance-debug-tests" \
  tests/test_vpfp_guidance_training_diagnostics.py tests/test_vpfp_guidance_training_debug_runner.py \
  tests/test_vpfp_solver_error_guidance.py \
  > "$VPFP_DEBUG_OUTPUT/tests.log" 2>&1 || VPFP_DEBUG_TEST_EXIT=$?
if [[ "$VPFP_DEBUG_TEST_EXIT" != 0 ]]; then exit "$VPFP_DEBUG_TEST_EXIT"; fi
test "$(git rev-parse HEAD)" == "$VPFP_DEBUG_REVISION"
test -z "$(git status --porcelain)"
VPFP_DEBUG_RUN_EXIT=0
timeout --signal=TERM --kill-after=10s 1530 "$VPFP_DEBUG_PYTHON" \
  scripts/run_vpfp_guidance_training_debug.py --output "$VPFP_DEBUG_OUTPUT/diagnosis" \
  --wall-seconds 1500 > "$VPFP_DEBUG_OUTPUT/diagnosis.log" 2>&1 || VPFP_DEBUG_RUN_EXIT=$?
exit "$VPFP_DEBUG_RUN_EXIT"
