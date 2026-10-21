#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=vpfp_weak_unified_agent_r1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Weak-collision VPFP Agent iteration requires a Slurm compute node." >&2
  exit 2
fi
UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${UNIFIED_ROOT}" ]]; then
  echo "UNIFIED_PLASMA_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT}")"
CONFIG="${UNIFIED_ROOT}/configs/vpfp_unified_live_weak_collision.json"
PLASMA_RUN_ROOT="${PDE_GAV_RUN_ROOT}/unified-plasma-agent-iterations-20260824-v1/vpfp-weak-r1"
if [[ ! -e "${UNIFIED_ROOT}/.git" || ! -f "${CONFIG}" ]]; then
  echo "Unified plasma checkout/config is incomplete." >&2
  exit 2
fi
if [[ -n "$(git -C "${UNIFIED_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Unified plasma checkout must be clean." >&2
  exit 2
fi
if [[ -e "${PLASMA_RUN_ROOT}" ]]; then
  echo "Refusing to reuse run root: ${PLASMA_RUN_ROOT}" >&2
  exit 2
fi

VENV="${PLASMA_VENV:-${HOME}/environment/venv}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Required cluster venv is absent: ${VENV}" >&2
  exit 2
fi
source "${VENV}/bin/activate"
CODEX_BINARY="${CODEX_BINARY:-codex}"
CODEX_AUTH_FILE="${CODEX_AUTH_FILE:-${HOME}/.codex/auth.json}"
if [[ ! -x "${CODEX_BINARY}" || ! -f "${CODEX_AUTH_FILE}" || -L "${CODEX_AUTH_FILE}" ]]; then
  echo "Pinned Codex binary/auth are unavailable." >&2
  exit 2
fi

export UNIFIED_ROOT PLASMA_RUN_ROOT CODEX_BINARY CODEX_AUTH_FILE
export PYTHONPATH="${UNIFIED_ROOT}/src:${UNIFIED_ROOT}/legacy/vpfp/src"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${PYTHONPYCACHEPREFIX}"
cd "${UNIFIED_ROOT}"

python scripts/run_unified_plasma_agent_iteration.py \
  --legacy-macro-reference \
  --config "${CONFIG}"
