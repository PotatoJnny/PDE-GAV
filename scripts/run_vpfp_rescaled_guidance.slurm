#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=vpfp_rescaled_guide
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
VPFP_RESCALE_ROOT=${PDE_GAV_WORKTREES_ROOT}/vpfp-demand-batched-oracle-20260908
VPFP_RESCALE_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-rescaled-guidance-${SLURM_JOB_ID:-}"
VPFP_RESCALE_PYTHON=${PDE_GAV_PYTHON}
VPFP_RESCALE_STARTED=$SECONDS
VPFP_RESCALE_TEST_EXIT=not_run
VPFP_RESCALE_TRAIN_EXIT=not_run
VPFP_RESCALE_EFFECT_EXIT=not_run
VPFP_RESCALE_REASON=setup_failed
VPFP_RESCALE_REVISION=unverified
VPFP_RESCALE_SCRIPT_SHA256=unverified
VPFP_RESCALE_FINAL=false
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
test -n "${SLURM_TMPDIR:-}"
test ! -e "$VPFP_RESCALE_OUTPUT"
test ! -L "$VPFP_RESCALE_OUTPUT"
mkdir -- "$VPFP_RESCALE_OUTPUT"
vpfp_rescale_finish() {
  local code=$?
  trap - EXIT
  if [[ "$(git -C "$VPFP_RESCALE_ROOT" rev-parse HEAD)" == "$VPFP_RESCALE_REVISION" && -z "$(git -C "$VPFP_RESCALE_ROOT" status --porcelain)" ]]; then
    VPFP_RESCALE_FINAL=true
  else
    VPFP_RESCALE_REASON=source_changed
    if [[ "$code" == 0 ]]; then code=1; fi
  fi
  printf 'field\tvalue\nsource_revision\t%s\nsource_script_sha256\t%s\nsource_final_verified\t%s\ntests\t%s\ntraining\t%s\neffect\t%s\nreason\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t1\nproduction_assets_replaced\tfalse\n' \
    "$VPFP_RESCALE_REVISION" "$VPFP_RESCALE_SCRIPT_SHA256" "$VPFP_RESCALE_FINAL" \
    "$VPFP_RESCALE_TEST_EXIT" "$VPFP_RESCALE_TRAIN_EXIT" "$VPFP_RESCALE_EFFECT_EXIT" \
    "$VPFP_RESCALE_REASON" "$code" "$((SECONDS-VPFP_RESCALE_STARTED))" \
    > "$VPFP_RESCALE_OUTPUT/launch.tsv"
  exit "$code"
}
trap vpfp_rescale_finish EXIT
trap 'VPFP_RESCALE_REASON=terminated; exit 143' TERM
trap 'VPFP_RESCALE_REASON=interrupted; exit 130' INT
cd "$VPFP_RESCALE_ROOT"
VPFP_RESCALE_REVISION="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
read -r VPFP_RESCALE_SCRIPT_SHA256 VPFP_RESCALE_SCRIPT_NAME < <(sha256sum -- "$VPFP_RESCALE_ROOT/scripts/run_vpfp_rescaled_guidance.slurm")
vpfp_rescale_verify_source() {
  if [[ "$(git rev-parse HEAD)" != "$VPFP_RESCALE_REVISION" || -n "$(git status --porcelain)" ]]; then
    VPFP_RESCALE_REASON=source_changed
    return 1
  fi
}
set +eu
set +o pipefail
source /etc/profile
set -euo pipefail
if ! module load StdEnv/2023 python/3.12 scipy-stack pytest; then
  echo 'Cluster modules unavailable; using existing unchanged venv.' >&2
fi
test -x "$VPFP_RESCALE_PYTHON"
export PYTHONPATH="$VPFP_RESCALE_ROOT/src:$VPFP_RESCALE_ROOT/scripts"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/vpfp-rescaled-guidance-pycache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
vpfp_rescale_verify_source
VPFP_RESCALE_REASON=tests_failed
VPFP_RESCALE_TEST_EXIT=0
# Keep the allocated GPU visible so device-copy/gradient tests actually run.
timeout --signal=TERM --kill-after=10s 300 \
  "$VPFP_RESCALE_PYTHON" -m pytest -o addopts= -q -p no:cacheprovider \
  --basetemp "$SLURM_TMPDIR/vpfp-rescaled-guidance-tests" \
  tests/test_vpfp_solver_error_guidance.py \
  tests/test_vpfp_calibrated_solver_error_guidance.py \
  tests/test_vpfp_exponential_solver_error_guidance.py \
  tests/test_vpfp_standardized_solver_error_guidance.py \
  tests/test_vpfp_rescaled_solver_error_guidance.py \
  tests/test_vpfp_guidance_effect_runner.py \
  tests/test_vpfp_guidance_effect_summary.py \
  tests/test_vpfp_exponential_guidance_runner.py \
  tests/test_vpfp_rescaled_guidance_runner.py \
  tests/test_vpfp_rescaling_physics.py \
  > "$VPFP_RESCALE_OUTPUT/tests.log" 2>&1 || VPFP_RESCALE_TEST_EXIT=$?
if [[ "$VPFP_RESCALE_TEST_EXIT" != 0 ]]; then exit "$VPFP_RESCALE_TEST_EXIT"; fi
vpfp_rescale_verify_source
VPFP_RESCALE_REASON=training_failed
VPFP_RESCALE_TRAIN_EXIT=0
timeout --signal=TERM --kill-after=10s 1230 "$VPFP_RESCALE_PYTHON" \
  scripts/run_vpfp_rescaled_guidance_training.py \
  --config configs/vpfp_rescaled_guidance_20260908.json \
  --output "$VPFP_RESCALE_OUTPUT/training" --wall-seconds 1200 \
  > "$VPFP_RESCALE_OUTPUT/training.log" 2>&1 || VPFP_RESCALE_TRAIN_EXIT=$?
if [[ "$VPFP_RESCALE_TRAIN_EXIT" != 0 ]]; then exit "$VPFP_RESCALE_TRAIN_EXIT"; fi
vpfp_rescale_verify_source
# No development MSE/ranking gate: successful training must reach real solver tests.
VPFP_RESCALE_REASON=effect_failed
VPFP_RESCALE_EFFECT_EXIT=0
timeout --signal=TERM --kill-after=10s 9030 "$VPFP_RESCALE_PYTHON" \
  scripts/run_vpfp_rescaled_guidance_effect.py \
  --config configs/vpfp_rescaled_guidance_20260908.json \
  --training-result "$VPFP_RESCALE_OUTPUT/training/result.json" \
  --output "$VPFP_RESCALE_OUTPUT/effect" --wall-seconds 9000 \
  > "$VPFP_RESCALE_OUTPUT/effect.log" 2>&1 || VPFP_RESCALE_EFFECT_EXIT=$?
if [[ "$VPFP_RESCALE_EFFECT_EXIT" == 0 ]]; then VPFP_RESCALE_REASON=completed; fi
exit "$VPFP_RESCALE_EFFECT_EXIT"
