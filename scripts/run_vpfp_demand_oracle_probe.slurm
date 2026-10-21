#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=vpfp_demand_batch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
VPFP_DEMAND_ROOT=${PDE_GAV_WORKTREES_ROOT}/vpfp-demand-batched-oracle-20260908
VPFP_DEMAND_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-demand-batched-oracle-${SLURM_JOB_ID}"
VPFP_DEMAND_REQUESTED_GPUS=1
if [[ "${VPFP_DEMAND_CPU_ONLY:-0}" == 1 ]]; then
  VPFP_DEMAND_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-demand-batched-oracle-cpu-${SLURM_JOB_ID}"
  VPFP_DEMAND_REQUESTED_GPUS=0
fi
VPFP_DEMAND_PYTHON=${PDE_GAV_PYTHON}
VPFP_DEMAND_STARTED=$SECONDS
VPFP_DEMAND_TEST_EXIT=not_run VPFP_DEMAND_PROBE_EXIT=not_run
VPFP_DEMAND_REVISION=unverified VPFP_DEMAND_FINAL=false
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_TMPDIR:-}"
test ! -e "${VPFP_DEMAND_OUTPUT}"
mkdir -- "${VPFP_DEMAND_OUTPUT}"
vpfp_demand_finish() {
  local code=$?
  trap - EXIT
  if [[ "$(git rev-parse HEAD)" == "${VPFP_DEMAND_REVISION}" && -z "$(git status --porcelain)" ]]; then
    VPFP_DEMAND_FINAL=true
  fi
  printf 'field\tvalue\nsource_revision\t%s\nsource_final_verified\t%s\ntests\t%s\nprobe\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t%s\nnew_training_updates\t0\nnew_diffusion_draws\t0\n' \
    "${VPFP_DEMAND_REVISION}" "${VPFP_DEMAND_FINAL}" "${VPFP_DEMAND_TEST_EXIT}" \
    "${VPFP_DEMAND_PROBE_EXIT}" "${code}" "$((SECONDS-VPFP_DEMAND_STARTED))" "${VPFP_DEMAND_REQUESTED_GPUS}" \
    > "${VPFP_DEMAND_OUTPUT}/launch.tsv"
  exit "${code}"
}
trap vpfp_demand_finish EXIT
cd "${VPFP_DEMAND_ROOT}"
VPFP_DEMAND_REVISION="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
set +eu
set +o pipefail
source /etc/profile
set -euo pipefail
if ! module load StdEnv/2023 python/3.12 scipy-stack pytest; then
  echo 'Cluster modules unavailable; using the existing unchanged venv.' >&2
fi
test -x "${VPFP_DEMAND_PYTHON}"
export PYTHONPATH="${VPFP_DEMAND_ROOT}/src:${VPFP_DEMAND_ROOT}/scripts"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-demand-pycache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
VPFP_DEMAND_TEST_EXIT=0
CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=10s 480 \
  "${VPFP_DEMAND_PYTHON}" -m pytest -o addopts= -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-demand-tests" \
  tests/test_vpfp_demand_solver_reward.py tests/test_vpfp_demand_rollouts.py \
  tests/test_vpfp_batched_source.py tests/test_vpfp_canonical_resident_runtime.py \
  tests/test_vpfp_canonical_binding.py \
  tests/test_vpfp_public_demand_contract.py tests/test_vpfp_canonical_demand_oracle.py \
  tests/test_vpfp_observable_demand_contract.py tests/test_macro_matching.py \
  tests/test_reference_pde_programs.py tests/test_vpfp_moving_frame_backend.py \
  tests/test_canonical_public_problem_capability.py tests/test_canonical_session_lifecycle.py \
  tests/test_public_problem_bundle.py tests/test_public_sessions.py \
  tests/test_vpfp_frozen_solver_reward.py tests/test_vpfp_strong_guidance_rollouts.py \
  tests/test_vpfp_strong_solver_guidance_round.py \
  > "${VPFP_DEMAND_OUTPUT}/tests.log" 2>&1 || VPFP_DEMAND_TEST_EXIT=$?
if [[ "${VPFP_DEMAND_TEST_EXIT}" != 0 ]]; then exit "${VPFP_DEMAND_TEST_EXIT}"; fi
if [[ "${VPFP_DEMAND_CPU_ONLY:-0}" == 1 ]]; then exit 0; fi
VPFP_DEMAND_PROBE_EXIT=0
timeout --signal=TERM --kill-after=10s 360 "${VPFP_DEMAND_PYTHON}" \
  scripts/probe_vpfp_demand_oracle.py --output "${VPFP_DEMAND_OUTPUT}/probe" --include-legacy \
  > "${VPFP_DEMAND_OUTPUT}/probe.log" 2>&1 || VPFP_DEMAND_PROBE_EXIT=$?
exit "${VPFP_DEMAND_PROBE_EXIT}"
