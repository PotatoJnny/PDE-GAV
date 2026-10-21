#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ns_structured_flow_fresh
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
  echo "Formal NS iteration requires a Slurm compute node." >&2
  exit 2
fi
UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT:?Set UNIFIED_PLASMA_ROOT or submit from the checkout}")"
IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU binding is absent}"
if (( ${#ALLOCATED_GPUS[@]} != 3 )); then
  echo "Formal NS iteration requires exactly three GPU tokens." >&2
  exit 2
fi
if [[ -z "${ALLOCATED_GPUS[0]}" || -z "${ALLOCATED_GPUS[1]}" || -z "${ALLOCATED_GPUS[2]}" || \
      "${ALLOCATED_GPUS[0]}" == "${ALLOCATED_GPUS[1]}" || \
      "${ALLOCATED_GPUS[0]}" == "${ALLOCATED_GPUS[2]}" || \
      "${ALLOCATED_GPUS[1]}" == "${ALLOCATED_GPUS[2]}" ]]; then
  echo "Formal NS GPU tokens must be nonempty and distinct." >&2
  exit 2
fi

REQUIRED_ARTIFACT_ENV=(
  NS_PCA_CHECKPOINT NS_PCA_SELECTION NS_PCA_AUDIT
  NS_FLOW_CHECKPOINT NS_FLOW_SELECTION NS_FLOW_TRAIN_REPORT
  NS_FLOW_SAMPLE_AUDIT NS_ANSWER_FIRST_REPORT
  NS_PCA_CHECKPOINT_SHA256 NS_PCA_SELECTION_SHA256 NS_PCA_AUDIT_SHA256
  NS_FLOW_CHECKPOINT_SHA256 NS_FLOW_SELECTION_SHA256 NS_FLOW_TRAIN_REPORT_SHA256
  NS_FLOW_SAMPLE_AUDIT_SHA256 NS_ANSWER_FIRST_REPORT_SHA256
  NS_PERTURBATION_AMPLITUDE
)
for name in "${REQUIRED_ARTIFACT_ENV[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Formal NS input is unset: ${name}" >&2
    exit 2
  fi
done
export "${REQUIRED_ARTIFACT_ENV[@]}"

export PDE_AGENT_WORKER_GPU_DEVICES="${ALLOCATED_GPUS[0]},${ALLOCATED_GPUS[1]}"
export PDE_CONTROLLER_GPU_DEVICE="${ALLOCATED_GPUS[2]}"
export CUDA_VISIBLE_DEVICES="${PDE_CONTROLLER_GPU_DEVICE}"
export NS_UNIFIED_CONFIG="${UNIFIED_ROOT}/configs/ns_unified_formal_template.json"
export PLASMA_STORAGE_ROOT="${PLASMA_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
export PLASMA_RUN_ROOT="${PLASMA_RUN_ROOT:-${PLASMA_STORAGE_ROOT}/unified-ns-structured-flow-fresh-v1/main-${SLURM_JOB_ID}}"

# No collaborator path or digest is defaulted here. The controller preflight
# verifies all 8 regular files, all 8 supplied SHA-256 values, selected rank /
# step bindings, and the 100% continuous answer-first contract before launch.
"${UNIFIED_ROOT}/scripts/slurm_ns_unified_live_agent_iteration.sh"
