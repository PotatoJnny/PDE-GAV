#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --job-name=vpfp_guide_calibration
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
VPFP_CALIBRATION_ROOT=${PDE_GAV_WORKTREES_ROOT}/vpfp-demand-batched-oracle-20260908
VPFP_CALIBRATION_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-guidance-calibration-${SLURM_JOB_ID:-}"
VPFP_CALIBRATION_PYTHON=${PDE_GAV_PYTHON}
VPFP_CALIBRATION_STARTED=$SECONDS
VPFP_CALIBRATION_TEST_EXIT=not_run
VPFP_CALIBRATION_TRAIN_EXIT=not_run
VPFP_CALIBRATION_EFFECT_EXIT=not_run
VPFP_CALIBRATION_REASON=setup_failed
VPFP_CALIBRATION_REVISION=unverified
VPFP_CALIBRATION_SCRIPT_SHA256=unverified
VPFP_CALIBRATION_FINAL=false
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
test -n "${SLURM_TMPDIR:-}"
test ! -e "$VPFP_CALIBRATION_OUTPUT"
test ! -L "$VPFP_CALIBRATION_OUTPUT"
mkdir -- "$VPFP_CALIBRATION_OUTPUT"
vpfp_calibration_finish() {
  local code=$?
  trap - EXIT
  if [[ "$(git -C "$VPFP_CALIBRATION_ROOT" rev-parse HEAD)" == "$VPFP_CALIBRATION_REVISION" && -z "$(git -C "$VPFP_CALIBRATION_ROOT" status --porcelain)" ]]; then
    VPFP_CALIBRATION_FINAL=true
  else
    VPFP_CALIBRATION_REASON=source_changed
    if [[ "$code" == 0 ]]; then code=1; fi
  fi
  printf 'field\tvalue\nsource_revision\t%s\nsource_script_sha256\t%s\nsource_final_verified\t%s\ntests\t%s\ntraining\t%s\neffect\t%s\nreason\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t1\nproduction_assets_replaced\tfalse\n' \
    "$VPFP_CALIBRATION_REVISION" "$VPFP_CALIBRATION_SCRIPT_SHA256" "$VPFP_CALIBRATION_FINAL" \
    "$VPFP_CALIBRATION_TEST_EXIT" "$VPFP_CALIBRATION_TRAIN_EXIT" "$VPFP_CALIBRATION_EFFECT_EXIT" \
    "$VPFP_CALIBRATION_REASON" "$code" "$((SECONDS-VPFP_CALIBRATION_STARTED))" \
    > "$VPFP_CALIBRATION_OUTPUT/launch.tsv"
  exit "$code"
}
trap vpfp_calibration_finish EXIT
trap 'VPFP_CALIBRATION_REASON=terminated; exit 143' TERM
trap 'VPFP_CALIBRATION_REASON=interrupted; exit 130' INT
cd "$VPFP_CALIBRATION_ROOT"
VPFP_CALIBRATION_REVISION="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
read -r VPFP_CALIBRATION_SCRIPT_SHA256 VPFP_CALIBRATION_SCRIPT_NAME < <(sha256sum -- "$VPFP_CALIBRATION_ROOT/scripts/run_vpfp_guidance_calibration.slurm")
vpfp_calibration_verify_source() {
  if [[ "$(git rev-parse HEAD)" != "$VPFP_CALIBRATION_REVISION" || -n "$(git status --porcelain)" ]]; then
    VPFP_CALIBRATION_REASON=source_changed
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
test -x "$VPFP_CALIBRATION_PYTHON"
export PYTHONPATH="$VPFP_CALIBRATION_ROOT/src:$VPFP_CALIBRATION_ROOT/scripts"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/vpfp-guidance-calibration-pycache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
vpfp_calibration_verify_source
VPFP_CALIBRATION_REASON=tests_failed
VPFP_CALIBRATION_TEST_EXIT=0
CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=10s 180 \
  "$VPFP_CALIBRATION_PYTHON" -m pytest -o addopts= -q -p no:cacheprovider \
  --basetemp "$SLURM_TMPDIR/vpfp-guidance-calibration-tests" \
  tests/test_vpfp_calibrated_solver_error_guidance.py \
  tests/test_vpfp_guidance_calibration_metrics.py \
  tests/test_vpfp_solver_error_guidance.py \
  tests/test_vpfp_guidance_effect_runner.py \
  tests/test_vpfp_guidance_calibration_runner.py \
  > "$VPFP_CALIBRATION_OUTPUT/tests.log" 2>&1 || VPFP_CALIBRATION_TEST_EXIT=$?
if [[ "$VPFP_CALIBRATION_TEST_EXIT" != 0 ]]; then exit "$VPFP_CALIBRATION_TEST_EXIT"; fi
vpfp_calibration_verify_source
VPFP_CALIBRATION_REASON=training_failed
VPFP_CALIBRATION_TRAIN_EXIT=0
timeout --signal=TERM --kill-after=10s 930 "$VPFP_CALIBRATION_PYTHON" \
  scripts/run_vpfp_guidance_calibration.py --output "$VPFP_CALIBRATION_OUTPUT/training" \
  --wall-seconds 900 > "$VPFP_CALIBRATION_OUTPUT/training.log" 2>&1 || VPFP_CALIBRATION_TRAIN_EXIT=$?
if [[ "$VPFP_CALIBRATION_TRAIN_EXIT" != 0 ]]; then exit "$VPFP_CALIBRATION_TRAIN_EXIT"; fi
vpfp_calibration_verify_source
VPFP_CALIBRATION_SELECTED="$VPFP_CALIBRATION_OUTPUT/training/selected-effect-config.json"
if [[ ! -e "$VPFP_CALIBRATION_SELECTED" && ! -L "$VPFP_CALIBRATION_SELECTED" ]]; then
  VPFP_CALIBRATION_EFFECT_EXIT=skipped_no_eligible_candidate
  VPFP_CALIBRATION_REASON=skipped_no_eligible_candidate
  printf '%s\n' 'skipped_no_eligible_candidate' > "$VPFP_CALIBRATION_OUTPUT/effect.log"
  exit 0
fi
VPFP_CALIBRATION_REASON=invalid_selected_effect_config
test -f "$VPFP_CALIBRATION_SELECTED"
test ! -L "$VPFP_CALIBRATION_SELECTED"
vpfp_calibration_verify_source
VPFP_CALIBRATION_REASON=effect_failed
VPFP_CALIBRATION_EFFECT_EXIT=0
timeout --signal=TERM --kill-after=10s 2130 "$VPFP_CALIBRATION_PYTHON" \
  scripts/run_vpfp_strong_guidance_effect.py --config "$VPFP_CALIBRATION_SELECTED" \
  --output "$VPFP_CALIBRATION_OUTPUT/effect" --wall-seconds 2100 \
  > "$VPFP_CALIBRATION_OUTPUT/effect.log" 2>&1 || VPFP_CALIBRATION_EFFECT_EXIT=$?
if [[ "$VPFP_CALIBRATION_EFFECT_EXIT" == 0 ]]; then VPFP_CALIBRATION_REASON=completed; fi
# Completion is not a claim of increased difficulty or physical qualification.
exit "$VPFP_CALIBRATION_EFFECT_EXIT"
