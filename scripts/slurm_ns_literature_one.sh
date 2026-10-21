#!/bin/bash
#SBATCH --job-name=ns-pde-agent
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@600
# Choose --account and --partition explicitly at sbatch time on the target cluster.
set -euo pipefail
if (( $# < 4 || $# > 5 )); then
    echo "usage: sbatch [cluster flags] $0 SOURCE PYTHON COMPARISONS ARM [--resume]" >&2
    exit 2
fi
SOURCE_ROOT=$1
PYTHON_BIN=$2
COMPARISONS=$3
ARM=$4
RESUME_ARGUMENT=${5:-}
if [[ -n "$RESUME_ARGUMENT" && "$RESUME_ARGUMENT" != --resume ]]; then
    echo "The fifth argument may only be --resume" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?requires an allocation}"
: "${CUDA_VISIBLE_DEVICES:?requires one allocated GPU}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == -1 ]]; then
    echo "Exactly one allocated GPU is required" >&2
    exit 2
fi
export PYTHONPATH="$SOURCE_ROOT/src:$SOURCE_ROOT/legacy/ns/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$SOURCE_ROOT"
EXTRA_ARGS=()
if [[ -n "$RESUME_ARGUMENT" ]]; then EXTRA_ARGS+=("$RESUME_ARGUMENT"); fi
# Exit 75 means safely paused. Submit the same arm with --resume for its next segment.
# exec delivers Slurm USR1 directly to the existing allocation pause handler.
exec "$PYTHON_BIN" scripts/run_ns_literature_arm.py \
    --plan "$COMPARISONS" --arm "$ARM" --gpu-token "$CUDA_VISIBLE_DEVICES" \
    --segment-seconds 10800 --flush-seconds 600 "${EXTRA_ARGS[@]}"
