#!/usr/bin/env bash
# Requires the site variables PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
unset PLASMA_COHORT_STORAGE_ROOT

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "NS Agent iteration requires a Slurm compute node." >&2
  exit 2
fi
UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${UNIFIED_ROOT}" ]]; then
  echo "UNIFIED_PLASMA_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT}")"
CONFIG="${NS_UNIFIED_CONFIG:?Set the explicit NS unified config}"
CONFIG="$(realpath -- "${CONFIG}")"
PLASMA_STORAGE_ROOT="${PLASMA_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
PLASMA_RUN_ROOT="${PLASMA_RUN_ROOT:?Set a fresh NS run root below PLASMA_STORAGE_ROOT}"

if [[ ! -e "${UNIFIED_ROOT}/.git" || ! -f "${CONFIG}" ]]; then
  echo "Unified checkout or selected NS config is incomplete." >&2
  exit 2
fi
if [[ -n "$(git -C "${UNIFIED_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Unified checkout must be clean." >&2
  exit 2
fi
if [[ -e "${PLASMA_RUN_ROOT}" ]]; then
  echo "Refusing to reuse NS run root: ${PLASMA_RUN_ROOT}" >&2
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

mkdir -p "${SLURM_TMPDIR}/pycache"
export HF_HOME="${SLURM_TMPDIR}/huggingface"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UNIFIED_ROOT PLASMA_RUN_ROOT PLASMA_STORAGE_ROOT
export CODEX_BINARY CODEX_AUTH_FILE
export PYTHONPATH="${UNIFIED_ROOT}/src:${UNIFIED_ROOT}/legacy/ns/src"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${UNIFIED_ROOT}"

# The launcher hashes and cross-binds every formal transferred artifact once
# before constructing a runtime. Online case generation performs no report
# audit and no transfer hashing.
python scripts/run_unified_plasma_agent_iteration.py \
  --legacy-macro-reference \
  --config "${CONFIG}"
