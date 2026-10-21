#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=vpfp_strong_canonical_2r
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "VPFP iteration requires a Slurm compute allocation." >&2
  exit 2
fi
VPFP_ITERATION_REPO="${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-canonical-iteration-20260909"
VPFP_ITERATION_PYTHON="${PDE_GAV_PYTHON}"
VPFP_ITERATION_CODEX="${CODEX_BINARY}"
VPFP_ITERATION_AUTH="${CODEX_AUTH_FILE}"
cd -- "${VPFP_ITERATION_REPO}"
if ! module load scipy-stack pytest; then
  echo "Cluster scipy-stack/pytest modules unavailable; using existing venv." >&2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing mutable source; commit the isolated iteration checkout first." >&2
  exit 2
fi
IFS=',' read -r -a VPFP_ITERATION_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU binding required}"
if (( ${#VPFP_ITERATION_GPUS[@]} != 2 )); then
  echo "Need one Controller GPU and one Agent GPU." >&2
  exit 2
fi
export PDE_CONTROLLER_GPU_DEVICE="${VPFP_ITERATION_GPUS[0]}"
export PDE_AGENT_WORKER_GPU_DEVICES="${VPFP_ITERATION_GPUS[1]}"
export CUDA_VISIBLE_DEVICES="${PDE_CONTROLLER_GPU_DEVICE}"
export PYTHONPATH="${VPFP_ITERATION_REPO}/src"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-iteration-pycache"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
VPFP_ITERATION_JOB_ROOT="${PDE_GAV_RUN_ROOT}/vpfp-strong-canonical-iteration-${SLURM_JOB_ID}"
if [[ -e "${VPFP_ITERATION_JOB_ROOT}" || -L "${VPFP_ITERATION_JOB_ROOT}" ]]; then
  echo "Refusing to reuse an existing run directory." >&2
  exit 2
fi
mkdir -- "${VPFP_ITERATION_JOB_ROOT}"
echo "VPFP job root: ${VPFP_ITERATION_JOB_ROOT}"
git rev-parse HEAD
"${VPFP_ITERATION_PYTHON}" -m pytest -q -p no:cacheprovider \
  tests/test_vpfp_round_solver_guidance.py \
  tests/test_vpfp_iteration_backend.py \
  tests/test_vpfp_split_candidate_execution.py \
  tests/test_vpfp_strong_initial_bundle.py \
  tests/test_vpfp_iteration_execution_check.py \
  tests/test_canonical_vpfp_runner.py \
  tests/test_vpfp_observable_demand_contract.py \
  tests/test_vpfp_canonical_binding.py \
  tests/test_vpfp_canonical_demand_oracle.py \
  tests/test_vpfp_canonical_resident_runtime.py \
  tests/test_vpfp_perturbation_backend.py \
  tests/test_vpfp_rescaled_solver_error_guidance.py \
  tests/test_vpfp_standardized_solver_error_guidance.py \
  tests/test_candidate_execution.py \
  tests/test_unified_runtime_boundaries.py \
  tests/test_canonical_public_problem_capability.py \
  --junitxml="${VPFP_ITERATION_JOB_ROOT}/tests.xml"

"${VPFP_ITERATION_PYTHON}" scripts/run_canonical_vpfp_campaign.py --validate-only
"${VPFP_ITERATION_PYTHON}" scripts/check_canonical_codex_environment.py \
  --codex-binary "${VPFP_ITERATION_CODEX}" --config scripts/canonical_codex.config.toml \
  --output "${VPFP_ITERATION_JOB_ROOT}/environment-check"
# Real two-round PDE/Guidance/private-scoring wiring check, NOT an LLM run.
# Failure stops here and is retained; no implicit resubmission or weaker gate.
"${VPFP_ITERATION_PYTHON}" scripts/check_vpfp_iteration_execution.py \
  --config configs/vpfp_strong_canonical_2round.json \
  --output "${VPFP_ITERATION_JOB_ROOT}/preflight"
# Only the shared formal Agent worker follows a successful preflight.
exec "${VPFP_ITERATION_PYTHON}" scripts/run_canonical_vpfp_campaign.py \
  --config configs/vpfp_strong_canonical_2round.json \
  --run-root "${VPFP_ITERATION_JOB_ROOT}/campaign" \
  --codex-binary "${VPFP_ITERATION_CODEX}" --auth-file "${VPFP_ITERATION_AUTH}" \
  --worker-gpu "${PDE_AGENT_WORKER_GPU_DEVICES}" --controller-gpu "${PDE_CONTROLLER_GPU_DEVICE}"
