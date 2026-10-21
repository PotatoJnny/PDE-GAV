#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=gkw_parallel_agent_recovery
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

echo "Retired: finite-condition GeneratedCodes v1 and its lookup champion cannot seed the continuous-condition GKW campaign; cold-start a new campaign." >&2
exit 2

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Parallel GKW Agent iteration requires a Slurm compute node." >&2
  exit 2
fi
UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${UNIFIED_ROOT}" ]]; then
  echo "UNIFIED_PLASMA_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT}")"

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU binding is absent}"
if (( ${#ALLOCATED_GPUS[@]} != 3 )); then
  echo "Parallel GKW main iteration requires exactly three GPU tokens." >&2
  exit 2
fi
if [[ -z "${ALLOCATED_GPUS[0]}" || -z "${ALLOCATED_GPUS[1]}" || -z "${ALLOCATED_GPUS[2]}" || \
      "${ALLOCATED_GPUS[0]}" == "${ALLOCATED_GPUS[1]}" || \
      "${ALLOCATED_GPUS[0]}" == "${ALLOCATED_GPUS[2]}" || \
      "${ALLOCATED_GPUS[1]}" == "${ALLOCATED_GPUS[2]}" ]]; then
  echo "Parallel GKW main GPU tokens must be nonempty and distinct." >&2
  exit 2
fi

export PDE_AGENT_WORKER_GPU_DEVICES="${ALLOCATED_GPUS[0]},${ALLOCATED_GPUS[1]}"
export PDE_CONTROLLER_GPU_DEVICE="${ALLOCATED_GPUS[2]}"
export CUDA_VISIBLE_DEVICES="${PDE_CONTROLLER_GPU_DEVICE}"

export GKW_UNIFIED_CONFIG="${UNIFIED_ROOT}/configs/gkw_unified_competitive_recovery.json"
export GKW_PHYSICAL_ROOT="${GKW_PHYSICAL_ROOT:-${PDE_GAV_WORKTREES_ROOT}/gkw-fresh-physical-5ca50ae}"
export PLASMA_STORAGE_ROOT="${PLASMA_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
export PLASMA_RUN_ROOT="${PLASMA_RUN_ROOT:-${PLASMA_STORAGE_ROOT}/unified-gkw-parallel-recovery-20260829-v1/recovery-${SLURM_JOB_ID}}"
export GKW_RECOVERY_COLD_START="${GKW_RECOVERY_COLD_START:-${PDE_GAV_RUN_ROOT}/unified-gkw-parallel-main-20260829-v1/main-5070278/result/model/submission.py}"
EXPECTED_RECOVERY_SHA256="ed7673cf3df144df10bc68ddf2aa17ade57624e8558bd536d14f35c83b1ea8a7"
if [[ ! -f "${GKW_RECOVERY_COLD_START}" || -L "${GKW_RECOVERY_COLD_START}" ]]; then
  echo "Recovery cold start is absent or a symlink: ${GKW_RECOVERY_COLD_START}" >&2
  exit 2
fi
ACTUAL_RECOVERY_SHA256="$(sha256sum -- "${GKW_RECOVERY_COLD_START}")"
ACTUAL_RECOVERY_SHA256="${ACTUAL_RECOVERY_SHA256%% *}"
if [[ "${ACTUAL_RECOVERY_SHA256}" != "${EXPECTED_RECOVERY_SHA256}" ]]; then
  echo "Recovery cold start does not match the promoted R1 winner." >&2
  exit 2
fi

exec "${UNIFIED_ROOT}/scripts/slurm_gkw_unified_live_agent_iteration.sh"
