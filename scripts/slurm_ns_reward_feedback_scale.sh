#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=ns_reward_scale
#SBATCH --output=ns_reward_scale-%j.out
#SBATCH --error=ns_reward_scale-%j.err

set -euo pipefail

UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT:?Submit from the repository or set UNIFIED_PLASMA_ROOT}")"
PYTHON_BIN="${NS_REWARD_SCALE_PYTHON:-${UNIFIED_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Reward-scale Python is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
SOURCE_RUN="${NS_REWARD_SCALE_SOURCE_RUN:-${UNIFIED_ROOT}/project_runs/ns_structured_guided_perturbed_5round}"
OUTPUT_ROOT="${NS_REWARD_SCALE_OUT:-${UNIFIED_ROOT}/project_runs/ns_reward_feedback_scale_${SLURM_JOB_ID}}"

export PYTHONPATH="${UNIFIED_ROOT}/src:${UNIFIED_ROOT}/legacy/ns/src:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${UNIFIED_ROOT}"
"${PYTHON_BIN}" scripts/benchmark_ns_reward_feedback_scale.py \
  --reward-table "${SOURCE_RUN}/campaign/agent_adversarial_loop/rounds/round_004/guidance_filter/filtered_reward_table.jsonl" \
  --loop-root "${SOURCE_RUN}/campaign/agent_adversarial_loop" \
  --pca-checkpoint "${SOURCE_RUN}/artifacts/pca/structured_coefficient_pca.pt" \
  --out "${OUTPUT_ROOT}" \
  --train-sizes 64 128 256 512 1024 \
  --repeat-count 3 \
  --heldout-cases 1024 \
  --steps 5000 \
  --batch-size 256 \
  --width 512 \
  --depth 6 \
  --time-width 128 \
  --device cuda \
  --amp bfloat16

echo "Reward-model scale report: ${OUTPUT_ROOT}/report.json"
