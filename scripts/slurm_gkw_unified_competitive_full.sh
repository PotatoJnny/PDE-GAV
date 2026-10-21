#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=gkw_open_search_fresh
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

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

export GKW_UNIFIED_CONFIG="${UNIFIED_ROOT}/configs/gkw_unified_competitive_full.json"
export GKW_PHYSICAL_ROOT="${GKW_PHYSICAL_ROOT:-${PDE_GAV_WORKTREES_ROOT}/gkw-fresh-physical-5ca50ae}"
export PLASMA_STORAGE_ROOT="${PLASMA_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
export PLASMA_RUN_ROOT="${PLASMA_RUN_ROOT:-${PLASMA_STORAGE_ROOT}/unified-gkw-open-search-fresh-20260830-v1/main-${SLURM_JOB_ID}}"

exec "${UNIFIED_ROOT}/scripts/slurm_gkw_unified_live_agent_iteration.sh"
