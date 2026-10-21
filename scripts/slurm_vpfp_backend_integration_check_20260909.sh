#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
# Engineering compatibility check only: no LLM, no Guide training, no formal run.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=vpfp_backend_merge_check
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

VPFP_MERGE_REPO=${PDE_GAV_WORKTREES_ROOT}/vpfp-backend-publish-20260909
VPFP_MERGE_PYTHON=${PDE_GAV_PYTHON}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
test -n "${SLURM_TMPDIR:-}"
cd -- "$VPFP_MERGE_REPO"
if ! module load scipy-stack pytest; then
  echo 'Cluster modules unavailable; using existing unchanged venv.' >&2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo 'Integration check requires clean committed source.' >&2
  exit 2
fi
VPFP_MERGE_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-backend-merge-check-${SLURM_JOB_ID}"
test ! -e "$VPFP_MERGE_OUTPUT"
test ! -L "$VPFP_MERGE_OUTPUT"
mkdir -- "$VPFP_MERGE_OUTPUT"
export PYTHONPATH="$VPFP_MERGE_REPO/src:$VPFP_MERGE_REPO/scripts:$VPFP_MERGE_REPO/legacy/vpfp/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/vpfp-merge-pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
echo "VPFP integration check output: $VPFP_MERGE_OUTPUT"
git rev-parse HEAD
"$VPFP_MERGE_PYTHON" -m pytest -o addopts= -q -p no:cacheprovider \
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
  tests/test_vpfp_campaign_guidance_paircheck.py \
  tests/test_vpfp_answer_first_perturbation.py \
  tests/test_vpfp_structured_diffusion.py \
  tests/test_candidate_execution.py \
  tests/test_agent_execution_service.py \
  tests/test_unified_runtime_boundaries.py \
  tests/test_canonical_public_problem_capability.py \
  tests/test_prepared_backend_execution_policy.py \
  tests/test_promotion_program.py \
  tests/test_runtime_multi_candidate_terminal.py \
  tests/test_ns_common_evaluation.py \
  tests/test_toy_product_flow.py \
  tests/test_product_skeleton.py \
  tests/test_promotion_panel.py \
  tests/test_unified_campaign_seed_replay.py \
  --junitxml="$VPFP_MERGE_OUTPUT/tests.xml"
"$VPFP_MERGE_PYTHON" scripts/run_canonical_vpfp_campaign.py --validate-only
exec "$VPFP_MERGE_PYTHON" scripts/check_vpfp_iteration_execution.py \
  --config configs/vpfp_strong_canonical_2round.json \
  --output "$VPFP_MERGE_OUTPUT/quick" --quick-only
