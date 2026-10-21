#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --job-name=artemis_parallel_smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Parallel smoke requires its own three-GPU Slurm allocation." >&2
  exit 2
fi
ARTEMIS_PARALLEL_ROOT="${PDE_GAV_WORKTREES_ROOT}/artemis-multigpu-driver-20260910"
ARTEMIS_PARALLEL_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_PARALLEL_OUTPUT="${PDE_GAV_RUN_ROOT}/artemis_parallel_smoke_20260910/job-$SLURM_JOB_ID"
cd -- "$ARTEMIS_PARALLEL_ROOT"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing Python venv." >&2
fi
# Keep all three Slurm tokens visible. The parent uses cuda:0; each isolated
# worker selects its assigned raw token before importing/loading CUDA.
export PYTHONPATH="$ARTEMIS_PARALLEL_ROOT:$ARTEMIS_PARALLEL_ROOT/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/artemis-parallel-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ -e "$ARTEMIS_PARALLEL_OUTPUT" || -L "$ARTEMIS_PARALLEL_OUTPUT" ]]; then
  echo "Refusing to reuse parallel smoke output." >&2
  exit 2
fi
mkdir -p -- "$ARTEMIS_PARALLEL_OUTPUT"
"$ARTEMIS_PARALLEL_PYTHON" -m pytest -o addopts='' -q \
  tests/test_artemis_parallel_smoke.py tests/test_artemis_parallel_oracle.py \
  tests/test_evaluation_resources.py tests/test_artemis_gpu_reuse_scope.py \
  tests/test_unified_runtime_boundaries.py tests/test_public_session_parallelism.py \
  tests/test_prepared_backend_execution_policy.py \
  tests/test_candidate_oracle_lanes.py \
  tests/test_candidate_execution.py tests/test_candidate_solve_budget.py \
  tests/test_artemis_solve_budget.py tests/test_agent_parallel_execution.py \
  --junitxml="$ARTEMIS_PARALLEL_OUTPUT/software-tests.xml" \
  2>&1 | tee "$ARTEMIS_PARALLEL_OUTPUT/software-tests.log"
exec "$ARTEMIS_PARALLEL_PYTHON" scripts/check_artemis_parallel_evaluation.py \
  --backend-config configs/plasma_artemis_opt5.json \
  --output-root "$ARTEMIS_PARALLEL_OUTPUT/real"
