#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=vpfp_noise_geometry
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "A Slurm compute allocation with local scratch is required." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-noise-geometry
VPFP_PYTHON=${PDE_GAV_PYTHON}
cd "${VPFP_CHECKOUT}"
VPFP_SOURCE_STATUS="$(git status --porcelain --untracked-files=all -- src scripts configs legacy/vpfp)"
if [[ -n "${VPFP_SOURCE_STATUS}" ]]; then
  echo "Freeze the relevant source before diagnostics." >&2
  echo "${VPFP_SOURCE_STATUS}" >&2
  exit 2
fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing explicit venv." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-noise-geometry-pycache"
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
VPFP_EXIT=0
for VPFP_REGIME in weak strong; do
  if [[ "${VPFP_REGIME}" == weak ]]; then
    VPFP_CACHE=${PDE_GAV_RUN_ROOT}/vpfp-sobolev-matching-weak-5232589
    VPFP_PILOT=${PDE_GAV_RUN_ROOT}/vpfp-value-control-pilot-weak-5236275
  else
    VPFP_CACHE=${PDE_GAV_RUN_ROOT}/vpfp-sobolev-matching-strong-5232590
    VPFP_PILOT=${PDE_GAV_RUN_ROOT}/vpfp-value-control-pilot-strong-5236280
  fi
  VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-noise-geometry-${VPFP_REGIME}-${SLURM_JOB_ID}"
  if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then
    echo "Refusing existing output: ${VPFP_OUTPUT}" >&2
    exit 2
  fi
  VPFP_REGIME_EXIT=0
  "${VPFP_PYTHON}" scripts/diagnose_vpfp_noise_geometry.py \
    --cache "${VPFP_CACHE}" --pilot "${VPFP_PILOT}" --output "${VPFP_OUTPUT}" \
    --regime "${VPFP_REGIME}" --answers 16 --seed 2026090507 --device cuda \
    || VPFP_REGIME_EXIT=$?
  if [[ "${VPFP_REGIME_EXIT}" != 0 ]]; then VPFP_EXIT="${VPFP_REGIME_EXIT}"; fi
  echo "VPFP ${VPFP_REGIME} diagnosis exit=${VPFP_REGIME_EXIT}; output=${VPFP_OUTPUT}"
done
exit "${VPFP_EXIT}"
