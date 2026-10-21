#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=artemis_canonical_5r_2a
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "The formal Artemis campaign requires a Slurm compute allocation." >&2
  exit 2
fi

ARTEMIS_REPOSITORY="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
ARTEMIS_REPOSITORY="$(realpath -- "${ARTEMIS_REPOSITORY:?Submit from the integration checkout}")"
ARTEMIS_PYTHON="${ARTEMIS_CANONICAL_PYTHON:-python}"
ARTEMIS_CODEX="${CODEX_BINARY:-codex}"
ARTEMIS_AUTH="${CODEX_AUTH_FILE:-${HOME}/.codex/auth.json}"

# Follow cluster instructions when the modules exist; retain the explicit
# existing-venv fallback used by the admitted Plasma backend on the cluster.
if ! module load scipy-stack pytest; then
  echo "scipy-stack/pytest modules unavailable; using existing venv: ${ARTEMIS_PYTHON}" >&2
fi
if [[ ! -x "${ARTEMIS_PYTHON}" || ! -x "${ARTEMIS_CODEX}" || \
      ! -f "${ARTEMIS_AUTH}" || -L "${ARTEMIS_AUTH}" ]]; then
  echo "Formal Python/Codex executable or regular non-symlink authentication source unavailable." >&2
  exit 2
fi
git -C "${ARTEMIS_REPOSITORY}" rev-parse --is-inside-work-tree >/dev/null
if [[ -n "$(git -C "${ARTEMIS_REPOSITORY}" status --porcelain --untracked-files=all)" ]]; then
  echo "Formal Artemis execution requires a clean, committed checkout." >&2
  exit 2
fi

IFS=',' read -r -a ARTEMIS_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU binding is absent}"
if (( ${#ARTEMIS_GPUS[@]} != 3 )); then
  echo "Formal Artemis requires exactly three GPU tokens." >&2
  exit 2
fi
if [[ -z "${ARTEMIS_GPUS[0]}" || -z "${ARTEMIS_GPUS[1]}" || -z "${ARTEMIS_GPUS[2]}" || \
      "${ARTEMIS_GPUS[0]}" == "${ARTEMIS_GPUS[1]}" || \
      "${ARTEMIS_GPUS[0]}" == "${ARTEMIS_GPUS[2]}" || \
      "${ARTEMIS_GPUS[1]}" == "${ARTEMIS_GPUS[2]}" ]]; then
  echo "Formal Artemis GPU tokens must be nonempty and distinct." >&2
  exit 2
fi

ARTEMIS_RUN_FAMILY="${ARTEMIS_CANONICAL_RUN_FAMILY_ROOT:-${PDE_GAV_RUN_ROOT}/artemis_canonical_5round_20260905}"
ARTEMIS_JOB_ROOT="${ARTEMIS_RUN_FAMILY}/job-${SLURM_JOB_ID}"
if [[ -e "${ARTEMIS_JOB_ROOT}" || -L "${ARTEMIS_JOB_ROOT}" ]]; then
  echo "Refusing to reuse formal Artemis job root: ${ARTEMIS_JOB_ROOT}" >&2
  exit 2
fi
mkdir -p -- "${ARTEMIS_RUN_FAMILY}"
mkdir -- "${ARTEMIS_JOB_ROOT}"

export PDE_AGENT_WORKER_GPU_DEVICES="${ARTEMIS_GPUS[0]},${ARTEMIS_GPUS[1]}"
export PDE_CONTROLLER_GPU_DEVICE="${ARTEMIS_GPUS[2]}"
export CUDA_VISIBLE_DEVICES="${PDE_CONTROLLER_GPU_DEVICE}"
export PYTHONPATH="${ARTEMIS_REPOSITORY}/src"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/artemis-canonical-pycache"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p -- "${PYTHONPYCACHEPREFIX}"
cd -- "${ARTEMIS_REPOSITORY}"

# Check the actual compute host, without model calls, auth reads, policy
# changes or any scientific mini-run. A failure retains its report and stops.
"${ARTEMIS_PYTHON}" scripts/check_canonical_codex_environment.py \
  --codex-binary "${ARTEMIS_CODEX}" \
  --config scripts/canonical_codex.config.toml \
  --output "${ARTEMIS_JOB_ROOT}/environment-check"

ARTEMIS_RECOVERY_ARGS=()
ARTEMIS_EXECUTION_ARGS=()
if [[ "${ARTEMIS_REUSE_AGENT_GPUS_FOR_EVALUATION:-0}" == 1 ]]; then
  ARTEMIS_EXECUTION_ARGS=(--reuse-agent-gpus-for-evaluation
    --backend-execution-config "$ARTEMIS_REPOSITORY/configs/plasma_artemis_d5_guided.json")
fi
if [[ -n "${ARTEMIS_ROUND_ONE_RECOVERY_SOURCE:-}" ]]; then
  ARTEMIS_RECOVERY_ARGS=(--recover-round-one-from "$ARTEMIS_ROUND_ONE_RECOVERY_SOURCE"
    --candidate-workers "${ARTEMIS_RECOVERY_CANDIDATE_WORKERS:-4}")
fi
exec "${ARTEMIS_PYTHON}" scripts/run_canonical_artemis_campaign.py \
  --config configs/artemis_canonical_5round.json \
  --run-root "${ARTEMIS_JOB_ROOT}/campaign" \
  --codex-binary "${ARTEMIS_CODEX}" --auth-file "${ARTEMIS_AUTH}" \
  --worker-gpus "${PDE_AGENT_WORKER_GPU_DEVICES}" \
  --controller-gpu "${PDE_CONTROLLER_GPU_DEVICE}" "${ARTEMIS_RECOVERY_ARGS[@]}" "${ARTEMIS_EXECUTION_ARGS[@]}"
