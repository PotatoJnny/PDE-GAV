#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=vpfp_g2_solver_paircheck
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

VPFP_PAIRCHECK_REPO=${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-canonical-iteration-20260909
VPFP_PAIRCHECK_PYTHON=${PDE_GAV_PYTHON}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
test -n "${SLURM_TMPDIR:-}"
cd -- "$VPFP_PAIRCHECK_REPO"
if ! module load scipy-stack pytest; then
  echo 'Cluster modules unavailable; using existing unchanged venv.' >&2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo 'Paircheck requires the committed isolated source checkout.' >&2
  exit 2
fi
VPFP_PAIRCHECK_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-campaign-guidance-paircheck-${SLURM_JOB_ID}"
test ! -e "$VPFP_PAIRCHECK_OUTPUT"
test ! -L "$VPFP_PAIRCHECK_OUTPUT"
mkdir -- "$VPFP_PAIRCHECK_OUTPUT"
export PYTHONPATH="$VPFP_PAIRCHECK_REPO/src:$VPFP_PAIRCHECK_REPO/scripts"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/vpfp-paircheck-pycache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONOPTIMIZE
echo "VPFP paired test output: $VPFP_PAIRCHECK_OUTPUT"
git rev-parse HEAD
"$VPFP_PAIRCHECK_PYTHON" -m pytest -o addopts= -q -p no:cacheprovider \
  tests/test_vpfp_campaign_guidance_paircheck.py \
  tests/test_vpfp_rescaled_solver_error_guidance.py \
  tests/test_vpfp_guidance_effect_summary.py \
  tests/test_vpfp_split_candidate_execution.py \
  --junitxml="$VPFP_PAIRCHECK_OUTPUT/tests.xml"
exec "$VPFP_PAIRCHECK_PYTHON" scripts/run_vpfp_campaign_guidance_paircheck.py \
  --config configs/vpfp_campaign_guidance_paircheck_20260909.json \
  --output "$VPFP_PAIRCHECK_OUTPUT/evaluation" --wall-seconds 9900
