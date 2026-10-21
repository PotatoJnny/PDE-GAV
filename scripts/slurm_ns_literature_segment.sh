#!/bin/bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --job-name=ns-literature
#SBATCH --qos=normal
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
if (( $# < 4 || $# > 5 )); then
    echo "usage: sbatch $0 SOURCE PYTHON CONFIG OUTPUT [MAX_SEGMENTS]" >&2
    exit 2
fi
SOURCE_ROOT=$1
PYTHON_BIN=$2
CONFIG=$3
OUTPUT=$4
MAX_SEGMENTS=${5:-12}
: "${SLURM_JOB_ID:?requires a Slurm allocation}"
: "${SLURM_TMPDIR:?requires node-local temporary storage}"
export NS_LITERATURE_SEGMENT_STARTED_UNIX
NS_LITERATURE_SEGMENT_STARTED_UNIX=$(date +%s)
export PYTHONPATH="$SOURCE_ROOT/src:$SOURCE_ROOT/legacy/ns/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/ns-literature-pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$SOURCE_ROOT"
# exec lets the baseline's allocation handler receive Slurm's early USR1.
exec "$PYTHON_BIN" "$SOURCE_ROOT/scripts/run_ns_literature_slurm_segment.py" \
    --config "$CONFIG" --output "$OUTPUT" --max-segments "$MAX_SEGMENTS"
