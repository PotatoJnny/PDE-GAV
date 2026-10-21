#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --job-name=ns_draft

set -euo pipefail
umask 077
MODE="${1:?Specify preflight or a configured NS experiment mode}"
BATCH_ROOT="${2:?Specify frozen batch root}"
PYTHON_BIN="${3:?Specify Python executable}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_TMPDIR:?Slurm local temporary directory required}"
export NS_JOB_GPU_TOKENS="${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a JOB_GPU_IDS <<< "${NS_JOB_GPU_TOKENS}"
if [[ "${MODE}" == fixed ]]; then
    # Fixed selection borrows both idle Agent GPUs through the existing lease.
    export CUDA_VISIBLE_DEVICES="${NS_JOB_GPU_TOKENS}"
elif [[ "${MODE}" == bon || "${MODE}" == aide ]]; then
    export CUDA_VISIBLE_DEVICES=""
else
    [[ "${#JOB_GPU_IDS[@]}" == 3 ]]
    # Keep the dedicated Controller GPU at cuda:0; Agent GPUs are leased only during hidden evaluation.
    export CUDA_VISIBLE_DEVICES="${JOB_GPU_IDS[2]},${JOB_GPU_IDS[0]},${JOB_GPU_IDS[1]}"
fi
SNAPSHOT_ROOT="${BATCH_ROOT}/source"
export PYTHONPATH="${SNAPSHOT_ROOT}/src:${SNAPSHOT_ROOT}/legacy/ns/src"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# Oracle requests execute concurrently. Each numerical request gets one CPU
# thread; Agent commands keep their explicit eight-thread environment and
# Candidate subprocesses independently retain their managed one-thread limits.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
mkdir -p -- "${PYTHONPYCACHEPREFIX}"
cd -- "${SNAPSHOT_ROOT}"
# Check this allocated node before spending a model call or starting an Agent.
"${PYTHON_BIN}" scripts/check_canonical_codex_environment.py \
    --codex-binary ${CODEX_BINARY} \
    --config "${SNAPSHOT_ROOT}/scripts/canonical_codex.config.toml" \
    --output "${BATCH_ROOT}/job_status/environment-${SLURM_JOB_ID}"
exec "${PYTHON_BIN}" scripts/run_ns_draft_job.py --mode "${MODE}" --batch-root "${BATCH_ROOT}"
