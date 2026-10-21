#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=gkw_unified_strong
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${UNIFIED_ROOT}" ]]; then
  echo "UNIFIED_PLASMA_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT}")"

GKW_STRONG_PROFILE="${GKW_STRONG_PROFILE:-rehearsal}"
case "${GKW_STRONG_PROFILE}" in
  rehearsal)
    CONFIG_NAME="gkw_unified_strong_rehearsal.json"
    ;;
  full)
    echo "gkw_unified_strong_full.json is a four-round configuration benchmark; it requires a segmented or parallel launcher and cannot run on this one-L40S rehearsal allocation." >&2
    exit 2
    ;;
  *)
    echo "GKW_STRONG_PROFILE must be rehearsal or full." >&2
    exit 2
    ;;
esac

export GKW_UNIFIED_CONFIG="${UNIFIED_ROOT}/configs/${CONFIG_NAME}"
export PLASMA_STORAGE_ROOT="${PLASMA_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
export PLASMA_RUN_ROOT="${PLASMA_RUN_ROOT:-${PLASMA_STORAGE_ROOT}/unified-gkw-strong-20260825-v1/${GKW_STRONG_PROFILE}-${SLURM_JOB_ID:-unknown}}"
exec "${UNIFIED_ROOT}/scripts/slurm_gkw_unified_live_agent_iteration.sh"
