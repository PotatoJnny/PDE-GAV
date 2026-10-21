#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --job-name=ns-eval-tail
set -euo pipefail
umask 077
PLAN="${1:?prepared plan.json required}"
PYTHON_BIN="${2:?Python executable required}"
: "${SLURM_JOB_ID:?Slurm required}"
: "${SLURM_TMPDIR:?Slurm scratch required}"
: "${CUDA_VISIBLE_DEVICES:?GPU required}"
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]]
EVAL_SOURCE="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_root"])' "${PLAN}")"
export PYTHONPATH="${EVAL_SOURCE}/src:${EVAL_SOURCE}/legacy/ns/src"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/ns-eval-pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p -- "${PYTHONPYCACHEPREFIX}"
cd -- "${EVAL_SOURCE}"
exec "${PYTHON_BIN}" scripts/ns_campaign_evaluation_only.py --plan "${PLAN}"
