#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --job-name=ns-eval-short

set -euo pipefail
umask 077
PLAN="${1:?Specify prepared plan.json}"
LANE="${2:?Specify lane index}"
PYTHON_BIN="${3:?Specify Python executable}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_TMPDIR:?Slurm local temporary directory required}"
: "${CUDA_VISIBLE_DEVICES:?One allocated GPU required}"
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]]
# Slurm executes a spool copy of this script. The plan owns the frozen source
# path; BASH_SOURCE and the submission working directory need not point there.
SHORT_SOURCE_ROOT="$("${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["source_root"])' "${PLAN}")"
export PYTHONPATH="${SHORT_SOURCE_ROOT}/src:${SHORT_SOURCE_ROOT}/legacy/ns/src"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/ns-eval-pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p -- "${PYTHONPYCACHEPREFIX}"
cd -- "${SHORT_SOURCE_ROOT}"
# exec makes the batch-shell USR1 reach the Python budget handler directly.
# No Codex process, Agent service or Generator training is started here.
exec "${PYTHON_BIN}" scripts/ns_evaluation_shards.py run \
    --plan "${PLAN}" --lane "${LANE}" --auto-continue
