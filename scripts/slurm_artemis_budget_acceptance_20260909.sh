#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=artemis_budget_accept
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires a Slurm compute allocation." >&2
  exit 2
fi
ARTEMIS_TEST_ROOT="${PDE_GAV_WORKTREES_ROOT}/artemis-budget-rollback-20260909"
ARTEMIS_TEST_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_TEST_SOURCE="${PDE_GAV_RUN_ROOT}/artemis_canonical_d2_guided_5round_20260908/job-5309385/campaign"
ARTEMIS_TEST_OUTPUT="$ARTEMIS_TEST_ROOT/runs/budget-acceptance-20260909/job-$SLURM_JOB_ID"
if [[ -e "$ARTEMIS_TEST_OUTPUT" || -L "$ARTEMIS_TEST_OUTPUT" ]]; then
  echo "Refusing to reuse acceptance output." >&2
  exit 2
fi
mkdir -p -- "$ARTEMIS_TEST_OUTPUT"
cd -- "$ARTEMIS_TEST_ROOT"
export PYTHONPATH="$ARTEMIS_TEST_ROOT:$ARTEMIS_TEST_ROOT/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/artemis-budget-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing venv without installation." >&2
fi
artemis_test_finish() {
  local ARTEMIS_TEST_RC=$?
  trap - EXIT
  "$ARTEMIS_TEST_PYTHON" scripts/artemis_budget_acceptance.py summary \
    --root "$ARTEMIS_TEST_OUTPUT" --exit-code "$ARTEMIS_TEST_RC"
}
trap artemis_test_finish EXIT
"$ARTEMIS_TEST_PYTHON" scripts/artemis_budget_acceptance.py preflight --root "$ARTEMIS_TEST_OUTPUT"
"$ARTEMIS_TEST_PYTHON" -m pytest -q -p scripts.artemis_backend_cold_acceptance \
  tests/test_candidate_solve_budget.py tests/test_candidate_execution.py \
  tests/test_evaluator_contract.py tests/test_continuous_solution.py \
  tests/test_artemis_canonical.py tests/test_artemis_candidate_execution.py \
  tests/test_artemis_solve_budget.py tests/test_artemis_formal_runner.py \
  tests/test_artemis_formal_protocol.py tests/test_artemis_backend_deployment.py \
  tests/test_campaign_memory.py tests/test_unified_runtime_boundaries.py \
  --junitxml="$ARTEMIS_TEST_OUTPUT/deployment-tests.xml" \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-budget-deployment"
"$ARTEMIS_TEST_PYTHON" -m pytest -q -p scripts.artemis_backend_cold_acceptance \
  tests/test_round_development_recovery.py tests/test_artemis_round_recovery.py \
  tests/test_artemis_budget_acceptance.py tests/test_artemis_guidance_execution.py \
  --junitxml="$ARTEMIS_TEST_OUTPUT/recovery-tests.xml" \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-budget-recovery"
"$ARTEMIS_TEST_PYTHON" scripts/run_canonical_artemis_campaign.py --validate-only \
  --recover-round-one-from "$ARTEMIS_TEST_SOURCE" --candidate-workers 4
"$ARTEMIS_TEST_PYTHON" scripts/check_artemis_budget_recovery.py \
  --source-campaign "$ARTEMIS_TEST_SOURCE" --output-root "$ARTEMIS_TEST_OUTPUT/real-recovery"
"$ARTEMIS_TEST_PYTHON" scripts/check_artemis_guidance_execution.py \
  --backend-config configs/plasma_canonical_artemis.json --official-count 96 \
  --output-root "$ARTEMIS_TEST_OUTPUT/guidance"
