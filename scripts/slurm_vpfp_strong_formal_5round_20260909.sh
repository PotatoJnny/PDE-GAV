#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
# User-approved Full strong VPFP: no scientific overrides at submission time.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=vpfp_strong_full_5r_s101
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

VPFP_FORMAL_REPO=${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-formal-5round-20260909
VPFP_FORMAL_PYTHON=${PDE_GAV_PYTHON}
VPFP_FORMAL_CODEX=${CODEX_BINARY}
VPFP_FORMAL_AUTH=${CODEX_AUTH_FILE}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
test -n "${SLURM_TMPDIR:-}"
cd -- "$VPFP_FORMAL_REPO"
if ! module load scipy-stack pytest; then
  echo 'Cluster modules unavailable; using existing unchanged venv.' >&2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo 'Refusing mutable formal source.' >&2
  exit 2
fi
IFS=',' read -r -a VPFP_FORMAL_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU binding is required}"
if (( ${#VPFP_FORMAL_GPUS[@]} != 3 )); then
  echo 'Formal profile requires one Controller and two distinct Agent GPUs.' >&2
  exit 2
fi
export PDE_CONTROLLER_GPU_DEVICE="${VPFP_FORMAL_GPUS[0]}"
export PDE_AGENT_WORKER_GPU_DEVICES="${VPFP_FORMAL_GPUS[1]},${VPFP_FORMAL_GPUS[2]}"
# Keep the dedicated Controller GPU at cuda:0; Agent GPUs are leased only during hidden evaluation.
export CUDA_VISIBLE_DEVICES="$PDE_CONTROLLER_GPU_DEVICE,$PDE_AGENT_WORKER_GPU_DEVICES"
export PYTHONPATH="$VPFP_FORMAL_REPO/src:$VPFP_FORMAL_REPO/scripts:$VPFP_FORMAL_REPO/legacy/vpfp/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/vpfp-formal-pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
VPFP_FORMAL_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-strong-formal-5round-${SLURM_JOB_ID}"
test ! -e "$VPFP_FORMAL_OUTPUT"
test ! -L "$VPFP_FORMAL_OUTPUT"
mkdir -- "$VPFP_FORMAL_OUTPUT"
echo "Formal VPFP run root: $VPFP_FORMAL_OUTPUT"
git rev-parse HEAD

"$VPFP_FORMAL_PYTHON" -m pytest -o addopts= -q -p no:cacheprovider \
  tests/test_vpfp_formal_runner.py \
  tests/test_vpfp_formal_feedback.py \
  tests/test_vpfp_formal_evaluation_bank.py \
  tests/test_vpfp_formal_preflight.py \
  tests/test_vpfp_round_solver_guidance.py \
  tests/test_vpfp_iteration_backend.py \
  tests/test_vpfp_split_candidate_execution.py \
  tests/test_vpfp_strong_initial_bundle.py \
  tests/test_vpfp_observable_demand_contract.py \
  tests/test_vpfp_canonical_binding.py \
  tests/test_vpfp_canonical_demand_oracle.py \
  tests/test_vpfp_canonical_resident_runtime.py \
  tests/test_vpfp_perturbation_backend.py \
  tests/test_vpfp_rescaled_solver_error_guidance.py \
  tests/test_vpfp_standardized_solver_error_guidance.py \
  tests/test_candidate_execution.py \
  tests/test_agent_execution_service.py \
  tests/test_unified_runtime_boundaries.py \
  tests/test_prepared_backend_execution_policy.py \
  tests/test_promotion_program.py \
  tests/test_runtime_multi_candidate_terminal.py \
  --junitxml="$VPFP_FORMAL_OUTPUT/tests.xml"

"$VPFP_FORMAL_PYTHON" scripts/run_vpfp_strong_formal_campaign.py --validate-only
"$VPFP_FORMAL_PYTHON" scripts/check_canonical_codex_environment.py \
  --codex-binary "$VPFP_FORMAL_CODEX" --config scripts/canonical_codex.config.toml \
  --output "$VPFP_FORMAL_OUTPUT/environment-check"
"$VPFP_FORMAL_PYTHON" scripts/check_vpfp_formal_preflight.py \
  --config configs/vpfp_strong_formal_5round_s101.json \
  --output "$VPFP_FORMAL_OUTPUT/preflight" --worker-counts 4,8,16

echo 'All engineering gates passed; starting the unchanged formal Campaign.'
exec "$VPFP_FORMAL_PYTHON" scripts/run_vpfp_strong_formal_campaign.py \
  --config configs/vpfp_strong_formal_5round_s101.json \
  --run-root "$VPFP_FORMAL_OUTPUT/campaign" \
  --codex-binary "$VPFP_FORMAL_CODEX" --auth-file "$VPFP_FORMAL_AUTH" \
  --worker-gpus "$PDE_AGENT_WORKER_GPU_DEVICES" --controller-gpu "$PDE_CONTROLLER_GPU_DEVICE" \
  --preflight-report "$VPFP_FORMAL_OUTPUT/preflight/summary.json"
