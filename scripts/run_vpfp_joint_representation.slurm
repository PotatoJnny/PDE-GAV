#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=vpfp_joint_representation
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Representation probes require a Slurm compute allocation and local scratch." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-regime-joint-modeling
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-joint-representation-${SLURM_JOB_ID}"
VPFP_RECEIPT="${VPFP_OUTPUT}.launch-exit-codes.tsv"
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" || -e "${VPFP_RECEIPT}" || -L "${VPFP_RECEIPT}" ]]; then
  echo "Refusing to overwrite representation probe artifacts." >&2
  exit 2
fi
cd "${VPFP_CHECKOUT}"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using existing explicit venv." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-joint-representation-pycache"
echo "Stage-zero representation probes only: 0 training steps, 0 new PDE solves, no asset changes."
VPFP_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_joint_representation_probe.py --output "${VPFP_OUTPUT}" || VPFP_EXIT=$?
printf 'stage\texit_code\nrepresentation_probes\t%s\n' "${VPFP_EXIT}" > "${VPFP_RECEIPT}"
echo "Artifacts: ${VPFP_OUTPUT}; receipt: ${VPFP_RECEIPT}; exit: ${VPFP_EXIT}"
exit "${VPFP_EXIT}"
