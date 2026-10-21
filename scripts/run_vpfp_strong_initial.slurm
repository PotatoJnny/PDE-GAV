#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=vpfp_strong_initial_gate
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" -ne 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Strong initial pilot requires Slurm compute allocation/local scratch and no positional arguments." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-roundtrip-gate
VPFP_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-strong-main-5244223
VPFP_VERIFIED_PREFLIGHT=${PDE_GAV_RUN_ROOT}/vpfp-strong-roundtrip-preflight-5259322/result.json
VPFP_VERIFIED_PREFLIGHT_SHA=1c37c6dc2509e6fdaf1bd6bd28bec7ad00bef871274f598f6fdca952b86e3c80
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-strong-initial-${SLURM_JOB_ID}"
VPFP_CONFIG="${VPFP_CHECKOUT}/configs/vpfp_unified_live_strong_collision.json"
VPFP_RECEIPT="${VPFP_OUTPUT}.launch-exit-codes.tsv"
VPFP_GPU_LOG="${VPFP_OUTPUT}.gpu-preflight.log"
cd "${VPFP_CHECKOUT}"
for VPFP_PATH in "${VPFP_CONFIG}" "${VPFP_SOURCE}/corpus/manifest.json" "${VPFP_SOURCE}/result.json" \
    "${VPFP_SOURCE}/frozen-reference-features.json" scripts/run_vpfp_strong_initial_pilot.py scripts/run_plasma_canonical_smoke.py; do
  if [[ ! -f "${VPFP_PATH}" ]]; then
    echo "Missing frozen strong source or driver: ${VPFP_PATH}" >&2
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
  echo "Commit/freeze relevant strong-pilot source before launching." >&2
  echo "${VPFP_DIRTY}" >&2
  exit 2
fi
# The recorded successful real-data preflight is a prerequisite, not a
# replacement for this allocation's own per-arm preflight. No weight resume.
if [[ ! -f "${VPFP_VERIFIED_PREFLIGHT}" || -L "${VPFP_VERIFIED_PREFLIGHT}" ]] \
    || ! VPFP_PREFLIGHT_SHA_LINE="$(sha256sum -- "${VPFP_VERIFIED_PREFLIGHT}")" \
    || [[ "${VPFP_PREFLIGHT_SHA_LINE%% *}" != "${VPFP_VERIFIED_PREFLIGHT_SHA}" ]]; then
  echo "Verified strong preflight result is missing, unreadable, or has changed SHA." >&2
  printf 'stage\texit_code\npreflight_binding\tfailed\ngpu_preflight\tnot_run\nexperiment\tnot_run\nlaunch\t2\n' > "${VPFP_RECEIPT}"
  exit 2
fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the explicit existing venv." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-strong-initial-pycache"
VPFP_GPU_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "No usable CUDA device"; print(torch.cuda.get_device_name(0))' \
  > "${VPFP_GPU_LOG}" 2>&1 || VPFP_GPU_EXIT=$?
VPFP_EXPERIMENT_EXIT=not_run
VPFP_EXIT="${VPFP_GPU_EXIT}"
if [[ "${VPFP_GPU_EXIT}" == 0 ]]; then
  VPFP_EXPERIMENT_EXIT=0
  "${VPFP_PYTHON}" scripts/run_vpfp_strong_initial_pilot.py --config "${VPFP_CONFIG}" \
    --source "${VPFP_SOURCE}" --output "${VPFP_OUTPUT}" --seed 1620260906 --device cuda || VPFP_EXPERIMENT_EXIT=$?
  VPFP_EXIT="${VPFP_EXPERIMENT_EXIT}"
fi
declare -A VPFP_SMOKE_EXITS
for VPFP_ARM in cov_base initial_conditioned; do
  VPFP_SMOKE_EXIT=gpu_preflight_failed
  # Independently smoke an existing best asset even if a later audit failed.
  if [[ "${VPFP_GPU_EXIT}" == 0 ]]; then
    VPFP_ARM_ROOT="${VPFP_OUTPUT}/seed-0/${VPFP_ARM}"
    VPFP_CHECKPOINT="${VPFP_ARM_ROOT}/checkpoints/best.pt"
    VPFP_SMOKE_EXIT=checkpoint_missing
    if [[ -f "${VPFP_CHECKPOINT}" ]]; then
      if VPFP_SHA_LINE="$(sha256sum -- "${VPFP_CHECKPOINT}")"; then
        VPFP_SHA="${VPFP_SHA_LINE%% *}"
        VPFP_SMOKE_EXIT=0
        "${VPFP_PYTHON}" scripts/run_plasma_canonical_smoke.py --physics vpfp --config "${VPFP_CONFIG}" \
          --diffusion-checkpoint "${VPFP_CHECKPOINT}" --diffusion-checkpoint-sha256 "${VPFP_SHA}" \
          --output "${VPFP_ARM_ROOT}/canonical-campaign-smoke" --device cuda \
          > "${VPFP_ARM_ROOT}/canonical-campaign-smoke.log" 2>&1 || VPFP_SMOKE_EXIT=$?
      else
        VPFP_SMOKE_EXIT="checkpoint_sha_failed:$?"
      fi
    fi
  fi
  VPFP_SMOKE_EXITS["${VPFP_ARM}"]="${VPFP_SMOKE_EXIT}"
  if [[ "${VPFP_SMOKE_EXIT}" != 0 && "${VPFP_EXIT}" == 0 ]]; then
    VPFP_EXIT=1
  fi
done
{
  printf 'stage\texit_code\npreflight_binding\t0\ngpu_preflight\t%s\nexperiment\t%s\n' "${VPFP_GPU_EXIT}" "${VPFP_EXPERIMENT_EXIT}"
  for VPFP_ARM in cov_base initial_conditioned; do
    printf 'smoke_%s\t%s\n' "${VPFP_ARM}" "${VPFP_SMOKE_EXITS[${VPFP_ARM}]}"
  done
  printf 'launch\t%s\n' "${VPFP_EXIT}"
} > "${VPFP_RECEIPT}"
echo "Strong initial pilot artifacts: ${VPFP_OUTPUT}; exit=${VPFP_EXIT}. No automatic followup or model admission."
exit "${VPFP_EXIT}"
