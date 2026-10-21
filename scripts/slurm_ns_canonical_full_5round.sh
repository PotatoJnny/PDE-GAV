#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1
#SBATCH --job-name=ns_canonical_5r_2a

set -euo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "The formal canonical NS campaign requires a Slurm array task." >&2
  exit 2
fi

REPOSITORY_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
REPOSITORY_ROOT="$(realpath -- "${REPOSITORY_ROOT:?Submit from the repository or set UNIFIED_PLASMA_ROOT}")"
PYTHON_BIN="${NS_CANONICAL_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"
CODEX_BINARY="${CODEX_BINARY:-codex}"
CODEX_AUTH_FILE="${CODEX_AUTH_FILE:-${HOME}/.codex/auth.json}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Canonical NS Python is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -x "${CODEX_BINARY}" || ! -f "${CODEX_AUTH_FILE}" || -L "${CODEX_AUTH_FILE}" ]]; then
  echo "Canonical Codex binary/auth are unavailable or unsafe." >&2
  exit 2
fi
GIT_COMMAND=(git -C "${REPOSITORY_ROOT}")
if ! "${GIT_COMMAND[@]}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ -d "${REPOSITORY_ROOT}/.git-real" ]]; then
    GIT_COMMAND=(
      git
      --git-dir="${REPOSITORY_ROOT}/.git-real"
      --work-tree="${REPOSITORY_ROOT}"
    )
  else
    echo "The repository has no usable Git metadata: ${REPOSITORY_ROOT}" >&2
    exit 2
  fi
fi
if [[ -n "$("${GIT_COMMAND[@]}" status --porcelain --untracked-files=all)" ]]; then
  echo "Formal canonical NS execution requires a clean, committed checkout." >&2
  exit 2
fi

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU binding is absent}"
if (( ${#ALLOCATED_GPUS[@]} != 4 )); then
  echo "Formal canonical NS execution requires exactly four GPU tokens." >&2
  exit 2
fi
for ((i=0; i<4; i++)); do
  for ((j=i+1; j<4; j++)); do
    if [[ -z "${ALLOCATED_GPUS[i]}" || "${ALLOCATED_GPUS[i]}" == "${ALLOCATED_GPUS[j]}" ]]; then
      echo "Formal canonical NS GPU tokens must be nonempty and distinct." >&2
      exit 2
    fi
  done
done
[[ -n "${ALLOCATED_GPUS[3]}" ]] || exit 2

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    PROFILE="structured_numerical"
    CONFIG="${REPOSITORY_ROOT}/configs/ns_canonical_structured_5round.json"
    ;;
  1)
    PROFILE="experimental_piv"
    CONFIG="${REPOSITORY_ROOT}/configs/ns_canonical_piv_5round.json"
    ;;
  *)
    echo "SLURM_ARRAY_TASK_ID must be 0 or 1." >&2
    exit 2
    ;;
esac

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
RUN_FAMILY_ROOT="${NS_CANONICAL_RUN_FAMILY_ROOT:?Set a fresh absolute NS canonical run-family root}"
if [[ "${RUN_FAMILY_ROOT}" != /* ]]; then
  echo "NS_CANONICAL_RUN_FAMILY_ROOT must be absolute: ${RUN_FAMILY_ROOT}" >&2
  exit 2
fi
mkdir -p -- "${RUN_FAMILY_ROOT}"
RUN_FAMILY_ROOT="$(realpath -- "${RUN_FAMILY_ROOT}")"
RUN_ROOT="${RUN_FAMILY_ROOT}/${PROFILE}/array-${ARRAY_JOB_ID}-task-${SLURM_ARRAY_TASK_ID}"
if [[ -e "${RUN_ROOT}" || -L "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse canonical NS run root: ${RUN_ROOT}" >&2
  exit 2
fi

export PDE_AGENT_WORKER_GPU_DEVICES="${ALLOCATED_GPUS[0]},${ALLOCATED_GPUS[1]}"
export PDE_CONTROLLER_GPU_DEVICE="${ALLOCATED_GPUS[2]}"
export PDE_CANDIDATE_GPU_DEVICE="${ALLOCATED_GPUS[3]}"
# Keep the dedicated Controller GPU at cuda:0; Agent GPUs are leased only during hidden evaluation.
export CUDA_VISIBLE_DEVICES="${PDE_CONTROLLER_GPU_DEVICE},${PDE_AGENT_WORKER_GPU_DEVICES},${PDE_CANDIDATE_GPU_DEVICE}"
export PYTHONPATH="${REPOSITORY_ROOT}/src:${REPOSITORY_ROOT}/legacy/ns/src:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p -- "${PYTHONPYCACHEPREFIX}"
cd -- "${REPOSITORY_ROOT}"

exec "${PYTHON_BIN}" scripts/run_canonical_ns_campaign.py \
  --config "${CONFIG}" \
  --run-root "${RUN_ROOT}" \
  --codex-binary "${CODEX_BINARY}" \
  --auth-file "${CODEX_AUTH_FILE}" \
  --worker-gpus "${PDE_AGENT_WORKER_GPU_DEVICES}" \
  --controller-gpu "${PDE_CONTROLLER_GPU_DEVICE}" \
  --candidate-gpus "${PDE_CANDIDATE_GPU_DEVICE}"
