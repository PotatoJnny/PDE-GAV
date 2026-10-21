#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=gkw_canonical_smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "GKW canonical smoke requires a Slurm allocation." >&2
  exit 2
fi
GKW_CHECKOUT="${PLASMA_INTEGRATION_ROOT:-${SLURM_SUBMIT_DIR:-}}"
GKW_PYTHON="${PLASMA_PYTHON:-python}"
if [[ ! -f "${GKW_CHECKOUT}/scripts/run_plasma_canonical_smoke.py" || ! -x "${GKW_PYTHON}" ]]; then
  echo "The integration checkout or Python environment is unavailable." >&2
  exit 2
fi
GKW_CHECKOUT="$(realpath -- "${GKW_CHECKOUT}")"
GKW_SMOKE_ROOT="${GKW_CHECKOUT}/artifacts/plasma-canonical-smoke-${SLURM_JOB_ID}"
if [[ -e "${GKW_SMOKE_ROOT}" ]]; then
  echo "Refusing to reuse GKW smoke output: ${GKW_SMOKE_ROOT}" >&2
  exit 2
fi
mkdir -p "${GKW_SMOKE_ROOT}"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing explicit venv." >&2
fi
export PYTHONPATH="${GKW_CHECKOUT}/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Match the qualified deployment: JAX geometry on CPU, Torch models on CUDA.
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/gkw-canonical-pycache"
cd "${GKW_CHECKOUT}"
exec "${GKW_PYTHON}" scripts/run_plasma_canonical_smoke.py \
  --physics gkw --config configs/plasma_canonical_gkw.json \
  --output "${GKW_SMOKE_ROOT}/gkw" --device cuda
