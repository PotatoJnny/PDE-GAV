#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=vpfp_regime_coherent_preflight
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires one Slurm compute allocation, local scratch and no positional arguments." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-regime-coherent-v2
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-regime-coherent-preflight-${SLURM_JOB_ID}"
cd "${VPFP_CHECKOUT}"
if [[ -n "$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)" ]]; then
  echo "Freeze relevant source before launching the registered preflight." >&2
  exit 2
fi
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then
  echo "Refusing to overwrite preflight artifacts." >&2
  exit 2
fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the explicit existing venv without installing packages." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-regime-coherent-pycache" CUBLAS_WORKSPACE_CONFIG=:4096:8
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
"${VPFP_PYTHON}" scripts/run_vpfp_regime_coherent_preflight.py --output "${VPFP_OUTPUT}"
