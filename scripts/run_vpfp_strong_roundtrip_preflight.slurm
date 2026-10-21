#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=vpfp_strong_roundtrip_preflight
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" -ne 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Strong roundtrip preflight requires Slurm compute allocation/local scratch and no positional arguments." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-roundtrip-gate
VPFP_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-strong-main-5244223
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-strong-roundtrip-preflight-${SLURM_JOB_ID}"
VPFP_CONFIG="${VPFP_CHECKOUT}/configs/vpfp_unified_live_strong_collision.json"
VPFP_RECEIPT="${VPFP_OUTPUT}.launch-exit-codes.tsv"
VPFP_GPU_LOG="${VPFP_OUTPUT}.gpu-preflight.log"
cd "${VPFP_CHECKOUT}"
for VPFP_PATH in "${VPFP_CONFIG}" "${VPFP_SOURCE}/corpus/manifest.json" "${VPFP_SOURCE}/result.json" \
    "${VPFP_SOURCE}/frozen-plan.json" scripts/run_vpfp_strong_roundtrip_preflight.py scripts/run_vpfp_strong_initial_pilot.py; do
  if [[ ! -f "${VPFP_PATH}" ]]; then
    echo "Missing original strong source or preflight driver: ${VPFP_PATH}" >&2
    exit 2
  fi
done
if [[ ! -x "${VPFP_PYTHON}" ]]; then
  echo "Explicit Python environment is unavailable." >&2
  exit 2
fi
for VPFP_PATH in "${VPFP_OUTPUT}" "${VPFP_RECEIPT}" "${VPFP_GPU_LOG}"; do
  if [[ -e "${VPFP_PATH}" || -L "${VPFP_PATH}" ]]; then
    echo "Refusing to overwrite: ${VPFP_PATH}" >&2
    exit 2
  fi
done
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" ]]; then
  echo "Commit/freeze relevant strong-preflight source before launching." >&2
  echo "${VPFP_DIRTY}" >&2
  exit 2
fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the explicit existing venv." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-strong-roundtrip-preflight-pycache"
VPFP_GPU_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "No usable CUDA device"; print(torch.cuda.get_device_name(0))' \
  > "${VPFP_GPU_LOG}" 2>&1 || VPFP_GPU_EXIT=$?
VPFP_PREFLIGHT_EXIT=not_run
VPFP_EXIT="${VPFP_GPU_EXIT}"
if [[ "${VPFP_GPU_EXIT}" == 0 ]]; then
  VPFP_PREFLIGHT_EXIT=0
  "${VPFP_PYTHON}" scripts/run_vpfp_strong_roundtrip_preflight.py --config "${VPFP_CONFIG}" \
    --source "${VPFP_SOURCE}" --output "${VPFP_OUTPUT}" || VPFP_PREFLIGHT_EXIT=$?
  VPFP_EXIT="${VPFP_PREFLIGHT_EXIT}"
fi
{
  printf 'stage\texit_code\ngpu_preflight\t%s\npreflight\t%s\nlaunch\t%s\n' \
    "${VPFP_GPU_EXIT}" "${VPFP_PREFLIGHT_EXIT}" "${VPFP_EXIT}"
} > "${VPFP_RECEIPT}"
echo "Strong preflight artifacts: ${VPFP_OUTPUT}; exit=${VPFP_EXIT}. Zero formal training, zero PDE solves; no automatic followup."
exit "${VPFP_EXIT}"
