#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=vpfp_weak_dynamics_diagnostic
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# User-approved development diagnosis, no training/selection/deployment.
# Charges original weak20 GPUh; root coordinates <=2 campaign GPUs.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then exit 2; fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-weak-dynamics-diagnostics
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-weak-dynamics-diagnostic-${SLURM_JOB_ID}"
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
VPFP_REVISION=not_verified VPFP_RUN_EXIT=not_run VPFP_STARTED_SECONDS=$SECONDS
vpfp_finish() {
  local status=$?
  trap - EXIT
  printf 'stage\texit_code\nsource_revision\t%s\npipeline\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t1\nweak_stage_total_cap_gpu_seconds\t72000\nscientific_admission\tfalse\n' \
    "${VPFP_REVISION}" "${VPFP_RUN_EXIT}" "${status}" "$((SECONDS-VPFP_STARTED_SECONDS))" > "${VPFP_OUTPUT}/launch-receipt.tsv.partial"
  mv -- "${VPFP_OUTPUT}/launch-receipt.tsv.partial" "${VPFP_OUTPUT}/launch-receipt.tsv"
  exit "${status}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "${VPFP_WEAK_CPU_GATE_RECEIPT:-}" != /* || ! "${VPFP_WEAK_CPU_GATE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Explicit successful CPU gate receipt and SHA required." >&2; exit 2
fi
cd "${VPFP_CHECKOUT}"
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" || ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then exit 2; fi
git ls-files --error-unmatch -- scripts/run_vpfp_weak_dynamics_launch.py \
  scripts/run_vpfp_weak_dynamics_diagnostic.py scripts/run_vpfp_velocity_dynamics_preflight.py >/dev/null
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using existing venv without installing packages." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-weak-dynamics-pycache"
VPFP_RUN_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_weak_dynamics_launch.py \
  --output "${VPFP_OUTPUT}/pipeline" --cpu-gate "${VPFP_WEAK_CPU_GATE_RECEIPT}" \
  --cpu-gate-sha256 "${VPFP_WEAK_CPU_GATE_SHA256}" > "${VPFP_OUTPUT}/pipeline.log" 2>&1 || VPFP_RUN_EXIT=$?
exit "${VPFP_RUN_EXIT}"
