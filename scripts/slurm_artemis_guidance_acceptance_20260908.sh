#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=artemis_guidance_smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "A Slurm compute allocation is required." >&2
  exit 2
fi
ARTEMIS_GUIDANCE_ROOT="${PDE_GAV_WORKTREES_ROOT}/artemis-guidance-canonical-20260908"
ARTEMIS_GUIDANCE_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_GUIDANCE_OUTPUT="$ARTEMIS_GUIDANCE_ROOT/runs/guidance-acceptance-20260908/job-$SLURM_JOB_ID"
if [[ -e "$ARTEMIS_GUIDANCE_OUTPUT" || -L "$ARTEMIS_GUIDANCE_OUTPUT" ]]; then
  echo "Refusing to reuse Guidance acceptance output." >&2
  exit 2
fi
mkdir -p -- "$ARTEMIS_GUIDANCE_OUTPUT"
export PYTHONPATH="$ARTEMIS_GUIDANCE_ROOT/src"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/artemis-guidance-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd -- "$ARTEMIS_GUIDANCE_ROOT"
artemis_guidance_finish() {
  local ARTEMIS_GUIDANCE_RC=$?
  trap - EXIT
  if "$ARTEMIS_GUIDANCE_PYTHON" scripts/artemis_guidance_acceptance.py summary \
      --root "$ARTEMIS_GUIDANCE_OUTPUT" --exit-code "$ARTEMIS_GUIDANCE_RC"; then
    exit 0
  else
    exit 1
  fi
}
trap artemis_guidance_finish EXIT
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing venv without installation." >&2
fi
"$ARTEMIS_GUIDANCE_PYTHON" scripts/artemis_guidance_acceptance.py preflight --root "$ARTEMIS_GUIDANCE_OUTPUT"
"$ARTEMIS_GUIDANCE_PYTHON" -m pytest -q -p scripts.artemis_backend_cold_acceptance \
  tests/test_artemis_canonical.py tests/test_artemis_formal_runner.py \
  tests/test_artemis_backend_deployment.py tests/test_artemis_formal_protocol.py \
  tests/test_artemis_candidate_execution.py tests/test_artemis_split_execution_check.py \
  tests/test_candidate_execution.py tests/test_canonical_candidate_boundary.py \
  tests/test_unified_runtime_boundaries.py tests/test_agent_parallel_execution.py \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-guidance-deployment" \
  --junitxml="$ARTEMIS_GUIDANCE_OUTPUT/deployment-tests.xml"
"$ARTEMIS_GUIDANCE_PYTHON" -m pytest -q -p scripts.artemis_backend_cold_acceptance \
  tests/test_artemis_guidance_execution.py tests/test_artemis_guidance_acceptance.py \
  -o cache_dir="$SLURM_TMPDIR/pytest-artemis-guidance-runner" \
  --junitxml="$ARTEMIS_GUIDANCE_OUTPUT/runner-tests.xml"
"$ARTEMIS_GUIDANCE_PYTHON" scripts/artemis_backend_cold_acceptance.py junit --root "$ARTEMIS_GUIDANCE_OUTPUT"
"$ARTEMIS_GUIDANCE_PYTHON" scripts/run_canonical_artemis_campaign.py --validate-only
"$ARTEMIS_GUIDANCE_PYTHON" scripts/check_artemis_guidance_execution.py \
  --backend-config configs/plasma_canonical_artemis.json \
  --official-count 96 --output-root "$ARTEMIS_GUIDANCE_OUTPUT/smoke"
