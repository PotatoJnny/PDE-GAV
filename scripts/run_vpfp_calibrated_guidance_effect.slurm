#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --job-name=vpfp_calibrated_effect
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
VPFP_GUIDANCE_EFFECT_ROOT=${PDE_GAV_WORKTREES_ROOT}/vpfp-demand-batched-oracle-20260908
VPFP_GUIDANCE_EFFECT_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-calibrated-guidance-effect-${SLURM_JOB_ID}"
VPFP_GUIDANCE_EFFECT_PYTHON=${PDE_GAV_PYTHON}
VPFP_GUIDANCE_EFFECT_STARTED=$SECONDS
VPFP_GUIDANCE_EFFECT_TEST_EXIT=not_run
VPFP_GUIDANCE_EFFECT_EVALUATION_EXIT=not_run
VPFP_GUIDANCE_EFFECT_REVISION=unverified
VPFP_GUIDANCE_EFFECT_FINAL=false
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_TMPDIR:-}"
test ! -e "${VPFP_GUIDANCE_EFFECT_OUTPUT}"
test ! -L "${VPFP_GUIDANCE_EFFECT_OUTPUT}"
mkdir -- "${VPFP_GUIDANCE_EFFECT_OUTPUT}"
vpfp_guidance_effect_finish() {
  local code=$?
  trap - EXIT
  if [[ "$(git -C "${VPFP_GUIDANCE_EFFECT_ROOT}" rev-parse HEAD)" == "${VPFP_GUIDANCE_EFFECT_REVISION}" && -z "$(git -C "${VPFP_GUIDANCE_EFFECT_ROOT}" status --porcelain)" ]]; then
    VPFP_GUIDANCE_EFFECT_FINAL=true
  elif [[ "${code}" == 0 ]]; then
    code=1
  fi
  printf 'field\tvalue\nsource_revision\t%s\nsource_final_verified\t%s\ntests\t%s\nevaluation\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t1\nnew_training_updates\t0\n' \
    "${VPFP_GUIDANCE_EFFECT_REVISION}" "${VPFP_GUIDANCE_EFFECT_FINAL}" \
    "${VPFP_GUIDANCE_EFFECT_TEST_EXIT}" "${VPFP_GUIDANCE_EFFECT_EVALUATION_EXIT}" \
    "${code}" "$((SECONDS-VPFP_GUIDANCE_EFFECT_STARTED))" \
    > "${VPFP_GUIDANCE_EFFECT_OUTPUT}/launch.tsv"
  exit "${code}"
}
trap vpfp_guidance_effect_finish EXIT
cd "${VPFP_GUIDANCE_EFFECT_ROOT}"
VPFP_GUIDANCE_EFFECT_REVISION="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
set +eu
set +o pipefail
source /etc/profile
set -euo pipefail
if ! module load StdEnv/2023 python/3.12 scipy-stack pytest; then
  echo 'Cluster modules unavailable; using the existing unchanged venv.' >&2
fi
test -x "${VPFP_GUIDANCE_EFFECT_PYTHON}"
export PYTHONPATH="${VPFP_GUIDANCE_EFFECT_ROOT}/src:${VPFP_GUIDANCE_EFFECT_ROOT}/scripts"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-guidance-effect-pycache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
VPFP_GUIDANCE_EFFECT_TEST_EXIT=0
CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=10s 180 \
  "${VPFP_GUIDANCE_EFFECT_PYTHON}" -m pytest -o addopts= -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-guidance-effect-tests" \
  tests/test_vpfp_guidance_effect_summary.py tests/test_vpfp_guidance_effect_runner.py \
  tests/test_vpfp_strong_solver_guidance_round.py tests/test_vpfp_demand_solver_reward.py \
  tests/test_vpfp_solver_error_guidance.py tests/test_vpfp_calibrated_solver_error_guidance.py \
  > "${VPFP_GUIDANCE_EFFECT_OUTPUT}/tests.log" 2>&1 || VPFP_GUIDANCE_EFFECT_TEST_EXIT=$?
if [[ "${VPFP_GUIDANCE_EFFECT_TEST_EXIT}" != 0 ]]; then exit "${VPFP_GUIDANCE_EFFECT_TEST_EXIT}"; fi
test "$(git rev-parse HEAD)" == "${VPFP_GUIDANCE_EFFECT_REVISION}"
test -z "$(git status --porcelain)"
# Fixed strength 2 is the original registered maximum nonzero value, not the
# old selected winner (which was 0). This job evaluates; it never trains.
# Exit 0 means the comparison completed, NOT that Guidance made cases harder.
VPFP_GUIDANCE_EFFECT_EVALUATION_EXIT=0
timeout --signal=TERM --kill-after=10s 3030 "${VPFP_GUIDANCE_EFFECT_PYTHON}" \
  scripts/run_vpfp_strong_guidance_effect.py \
  --config configs/vpfp_calibrated_guidance_effect_20260908.json \
  --output "${VPFP_GUIDANCE_EFFECT_OUTPUT}/evaluation" --wall-seconds 3000 \
  > "${VPFP_GUIDANCE_EFFECT_OUTPUT}/evaluation.log" 2>&1 || VPFP_GUIDANCE_EFFECT_EVALUATION_EXIT=$?
exit "${VPFP_GUIDANCE_EFFECT_EVALUATION_EXIT}"
