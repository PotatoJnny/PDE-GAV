#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=gkw_unified_agent_r1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_RUN_ROOT, PDE_GAV_DEPS_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
unset PLASMA_COHORT_STORAGE_ROOT

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "GKW Agent iteration requires a Slurm compute node." >&2
  exit 2
fi
UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${UNIFIED_ROOT}" ]]; then
  echo "UNIFIED_PLASMA_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT}")"
CONFIG="${GKW_UNIFIED_CONFIG:-${UNIFIED_ROOT}/configs/gkw_unified_live_agent_iteration.json}"
PLASMA_STORAGE_ROOT="${PLASMA_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
PLASMA_RUN_ROOT="${PLASMA_RUN_ROOT:-${PLASMA_STORAGE_ROOT}/unified-plasma-agent-iterations-20260824-v1/gkw-r1}"
GKW_PHYSICAL_COMMIT="${GKW_PHYSICAL_COMMIT:-80632f11a39b8577124ffa652556af1521d9b992}"
GKW_PHYSICAL_ROOT="${GKW_PHYSICAL_ROOT:?Set the clean pinned physical checkout}"
GKW_PHYSICAL_ROOT="$(realpath -- "${GKW_PHYSICAL_ROOT}")"
GKW_EXTERNAL_ROOT="${GKW_EXTERNAL_ROOT:-${PDE_GAV_DEPS_ROOT}/neural-gyrokinetics-568411b}"
GKW_EXTERNAL_COMMIT="568411b33be486262b8c5cfe1dfb3e0fe97f6419"
GYARADAX_ROOT="${GYARADAX_ROOT:-${PDE_GAV_DEPS_ROOT}/gyaradax-318e3d11}"
GYARADAX_COMMIT="318e3d11d7326d99e96f2bd5a8998a462261dcd4"
DEPS_ROOT="${GKW_QUALIFICATION_DEPS_ROOT:-${PDE_GAV_DEPS_ROOT}/gkw-perturbation-qualification-python-v1}"
GKW_PERTURBATION_CERTIFICATE="${GKW_PERTURBATION_CERTIFICATE:?Set the continuous-condition qualification certificate}"
GKW_QUALIFICATION_REPORT="${GKW_QUALIFICATION_REPORT:?Set the continuous-condition qualification report}"
GKW_QUALIFICATION_GENERATOR_MANIFEST="${GKW_QUALIFICATION_GENERATOR_MANIFEST:?Set the continuous-condition generator manifest}"
GKW_QUALIFICATION_OPERATOR_MANIFEST="${GKW_QUALIFICATION_OPERATOR_MANIFEST:?Set the continuous-condition operator manifest}"
GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST="${GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST:?Set the continuous-condition physical-source manifest}"
GKW_QUALIFICATION_SEED_PLAN="${GKW_QUALIFICATION_SEED_PLAN:?Set the continuous-condition seed plan}"
GKW_QUALIFICATION_REPORT_SHA256="${GKW_QUALIFICATION_REPORT_SHA256:?Set the continuous-condition qualification report SHA-256}"
GKW_QUALIFICATION_GENERATOR_MANIFEST_SHA256="${GKW_QUALIFICATION_GENERATOR_MANIFEST_SHA256:?Set the continuous-condition generator manifest SHA-256}"
GKW_QUALIFICATION_OPERATOR_MANIFEST_SHA256="${GKW_QUALIFICATION_OPERATOR_MANIFEST_SHA256:?Set the continuous-condition operator manifest SHA-256}"
GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST_SHA256="${GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST_SHA256:?Set the continuous-condition physical-source manifest SHA-256}"
GKW_QUALIFICATION_SEED_PLAN_MANIFEST_SHA256="${GKW_QUALIFICATION_SEED_PLAN_MANIFEST_SHA256:?Set the continuous-condition seed-plan manifest SHA-256}"
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
if [[ ! -e "${GKW_PHYSICAL_ROOT}/.git" || "$(git -C "${GKW_PHYSICAL_ROOT}" rev-parse HEAD)" != "${GKW_PHYSICAL_COMMIT}" || -n "$(git -C "${GKW_PHYSICAL_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "GKW physical checkout is absent, dirty, or unpinned." >&2
  exit 2
fi
if [[ "${GKW_EXTERNAL_ROOT}" != /scratch/* || ! -e "${GKW_EXTERNAL_ROOT}/.git" || "$(git -C "${GKW_EXTERNAL_ROOT}" rev-parse HEAD)" != "${GKW_EXTERNAL_COMMIT}" || -n "$(git -C "${GKW_EXTERNAL_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "neural-gyrokinetics checkout is absent, dirty, or unpinned." >&2
  exit 2
fi
if [[ "${GYARADAX_ROOT}" != /scratch/* || ! -e "${GYARADAX_ROOT}/.git" || "$(git -C "${GYARADAX_ROOT}" rev-parse HEAD)" != "${GYARADAX_COMMIT}" || -n "$(git -C "${GYARADAX_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Gyaradax checkout is absent, dirty, or unpinned." >&2
  exit 2
fi
if [[ "${DEPS_ROOT}" != /scratch/* || ! -d "${DEPS_ROOT}" ]]; then
  echo "Frozen GKW Python dependency bundle is absent: ${DEPS_ROOT}" >&2
  exit 2
fi
for path in \
  "${GKW_PERTURBATION_CERTIFICATE}" \
  "${GKW_QUALIFICATION_REPORT}" \
  "${GKW_QUALIFICATION_GENERATOR_MANIFEST}" \
  "${GKW_QUALIFICATION_OPERATOR_MANIFEST}" \
  "${GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST}" \
  "${GKW_QUALIFICATION_SEED_PLAN}"; do
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "Pinned GKW qualification artifact is absent or a symlink: ${path}" >&2
    exit 2
  fi
done

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

export UNIFIED_ROOT PLASMA_RUN_ROOT CODEX_BINARY CODEX_AUTH_FILE
export PLASMA_STORAGE_ROOT
export GKW_PHYSICAL_ROOT GKW_PHYSICAL_COMMIT GKW_EXTERNAL_ROOT GYARADAX_ROOT
export GKW_PERTURBATION_CERTIFICATE GKW_QUALIFICATION_REPORT
export GKW_QUALIFICATION_GENERATOR_MANIFEST GKW_QUALIFICATION_OPERATOR_MANIFEST
export GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST GKW_QUALIFICATION_SEED_PLAN
export GKW_QUALIFICATION_REPORT_SHA256
export GKW_QUALIFICATION_GENERATOR_MANIFEST_SHA256
export GKW_QUALIFICATION_OPERATOR_MANIFEST_SHA256
export GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST_SHA256
export GKW_QUALIFICATION_SEED_PLAN_MANIFEST_SHA256
export GKW_QUALIFICATION_DEPS_ROOT="${DEPS_ROOT}"
export PYTHONPATH="${UNIFIED_ROOT}/src:${GKW_PHYSICAL_ROOT}/src:${GKW_EXTERNAL_ROOT}:${GYARADAX_ROOT}:${DEPS_ROOT}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${UNIFIED_ROOT}"

python -c '
import importlib.metadata as metadata

expected = {
    "kappamodules": "0.1.112",
    "einops": "0.8.1",
    "omegaconf": "2.3.0",
    "antlr4-python3-runtime": "4.9.3",
    "PyYAML": "6.0.2",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"frozen GKW Agent dependencies changed: {actual!r}")
'

RECOVERY_ARGUMENTS=()
if [[ -n "${PLASMA_RECOVERY_MANIFEST:-}" || \
      -n "${PLASMA_RECOVERY_MANIFEST_SHA256:-}" ]]; then
  if [[ -z "${PLASMA_RECOVERY_MANIFEST:-}" || \
        -z "${PLASMA_RECOVERY_MANIFEST_SHA256:-}" ]]; then
    echo "Recovery manifest path and SHA-256 must be supplied together." >&2
    exit 2
  fi
  if [[ ! -f "${PLASMA_RECOVERY_MANIFEST}" || \
        -L "${PLASMA_RECOVERY_MANIFEST}" ]]; then
    echo "Recovery manifest is absent or is a symlink." >&2
    exit 2
  fi
  RECOVERY_ARGUMENTS=(
    --recover-development-manifest "${PLASMA_RECOVERY_MANIFEST}"
    --recover-development-manifest-sha256 "${PLASMA_RECOVERY_MANIFEST_SHA256}"
  )
fi

# The controller's normal preflight independently revalidates the physical,
# neural-gyrokinetics, and Gyaradax roots against the config's exact commits.
python scripts/run_unified_plasma_agent_iteration.py \
  --legacy-macro-reference \
  --config "${CONFIG}" \
  "${RECOVERY_ARGUMENTS[@]}"
