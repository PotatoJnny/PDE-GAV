#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --job-name=vpfp_exponential_guide
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
VPFP_EXP_ROOT=${PDE_GAV_WORKTREES_ROOT}/vpfp-demand-batched-oracle-20260908
VPFP_EXP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-exponential-guidance-${SLURM_JOB_ID:-}"
VPFP_EXP_PYTHON=${PDE_GAV_PYTHON}
VPFP_EXP_STARTED=$SECONDS
VPFP_EXP_TEST_EXIT=not_run
VPFP_EXP_TRAIN_EXIT=not_run
VPFP_EXP_EFFECT_EXIT=not_run
VPFP_EXP_REASON=setup_failed
VPFP_EXP_REVISION=unverified
VPFP_EXP_SCRIPT_SHA256=unverified
VPFP_EXP_FINAL=false
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
test -n "${SLURM_TMPDIR:-}"
test ! -e "$VPFP_EXP_OUTPUT"
test ! -L "$VPFP_EXP_OUTPUT"
mkdir -- "$VPFP_EXP_OUTPUT"
vpfp_exp_finish() {
  local code=$?
  trap - EXIT
  if [[ "$(git -C "$VPFP_EXP_ROOT" rev-parse HEAD)" == "$VPFP_EXP_REVISION" && -z "$(git -C "$VPFP_EXP_ROOT" status --porcelain)" ]]; then
    VPFP_EXP_FINAL=true
  else
    VPFP_EXP_REASON=source_changed
    if [[ "$code" == 0 ]]; then code=1; fi
  fi
  printf 'field\tvalue\nsource_revision\t%s\nsource_script_sha256\t%s\nsource_final_verified\t%s\ntests\t%s\ntraining\t%s\neffect\t%s\nreason\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t1\nproduction_assets_replaced\tfalse\n' \
    "$VPFP_EXP_REVISION" "$VPFP_EXP_SCRIPT_SHA256" "$VPFP_EXP_FINAL" \
    "$VPFP_EXP_TEST_EXIT" "$VPFP_EXP_TRAIN_EXIT" "$VPFP_EXP_EFFECT_EXIT" \
    "$VPFP_EXP_REASON" "$code" "$((SECONDS-VPFP_EXP_STARTED))" \
    > "$VPFP_EXP_OUTPUT/launch.tsv"
  exit "$code"
}
trap vpfp_exp_finish EXIT
trap 'VPFP_EXP_REASON=terminated; exit 143' TERM
trap 'VPFP_EXP_REASON=interrupted; exit 130' INT
cd "$VPFP_EXP_ROOT"
VPFP_EXP_REVISION="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
read -r VPFP_EXP_SCRIPT_SHA256 VPFP_EXP_SCRIPT_NAME < <(sha256sum -- "$VPFP_EXP_ROOT/scripts/run_vpfp_exponential_guidance.slurm")
vpfp_exp_verify_source() {
  if [[ "$(git rev-parse HEAD)" != "$VPFP_EXP_REVISION" || -n "$(git status --porcelain)" ]]; then
    VPFP_EXP_REASON=source_changed
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
test -x "$VPFP_EXP_PYTHON"
export PYTHONPATH="$VPFP_EXP_ROOT/src:$VPFP_EXP_ROOT/scripts"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/vpfp-exponential-guidance-pycache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
vpfp_exp_verify_source
VPFP_EXP_REASON=tests_failed
VPFP_EXP_TEST_EXIT=0
CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=10s 180 \
  "$VPFP_EXP_PYTHON" -m pytest -o addopts= -q -p no:cacheprovider \
  --basetemp "$SLURM_TMPDIR/vpfp-exponential-guidance-tests" \
  tests/test_vpfp_solver_error_guidance.py \
  tests/test_vpfp_calibrated_solver_error_guidance.py \
  tests/test_vpfp_exponential_solver_error_guidance.py \
  tests/test_vpfp_guidance_effect_runner.py \
  tests/test_vpfp_guidance_effect_summary.py \
  tests/test_vpfp_exponential_guidance_runner.py \
  > "$VPFP_EXP_OUTPUT/tests.log" 2>&1 || VPFP_EXP_TEST_EXIT=$?
if [[ "$VPFP_EXP_TEST_EXIT" != 0 ]]; then exit "$VPFP_EXP_TEST_EXIT"; fi
vpfp_exp_verify_source
VPFP_EXP_REASON=training_failed
VPFP_EXP_TRAIN_EXIT=0
timeout --signal=TERM --kill-after=10s 630 "$VPFP_EXP_PYTHON" \
  scripts/run_vpfp_exponential_guidance.py --output "$VPFP_EXP_OUTPUT/training" \
  --wall-seconds 600 > "$VPFP_EXP_OUTPUT/training.log" 2>&1 || VPFP_EXP_TRAIN_EXIT=$?
if [[ "$VPFP_EXP_TRAIN_EXIT" != 0 ]]; then exit "$VPFP_EXP_TRAIN_EXIT"; fi
vpfp_exp_verify_source
VPFP_EXP_EFFECT_CONFIG="$VPFP_EXP_OUTPUT/training/effect-config.json"
VPFP_EXP_REASON=missing_effect_config
test -e "$VPFP_EXP_EFFECT_CONFIG"
VPFP_EXP_REASON=invalid_effect_config
test -f "$VPFP_EXP_EFFECT_CONFIG"
test ! -L "$VPFP_EXP_EFFECT_CONFIG"
vpfp_exp_verify_source
# Real paired solver comparison is mandatory after successful training.
# Development MSE/ranking/eligibility do not gate this stage.
VPFP_EXP_REASON=effect_failed
VPFP_EXP_EFFECT_EXIT=0
timeout --signal=TERM --kill-after=10s 2430 "$VPFP_EXP_PYTHON" \
  scripts/run_vpfp_strong_guidance_effect.py --config "$VPFP_EXP_EFFECT_CONFIG" \
  --output "$VPFP_EXP_OUTPUT/effect" --wall-seconds 2400 \
  > "$VPFP_EXP_OUTPUT/effect.log" 2>&1 || VPFP_EXP_EFFECT_EXIT=$?
if [[ "$VPFP_EXP_EFFECT_EXIT" == 0 ]]; then VPFP_EXP_REASON=completed; fi
# Completion records execution, not demonstrated increased solver difficulty.
exit "$VPFP_EXP_EFFECT_EXIT"
