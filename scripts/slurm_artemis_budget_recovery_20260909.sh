#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=artemis_r1_budget_replay
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ $# -ne 3 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires source campaign, passing acceptance, verified publication and Slurm allocation." >&2
  exit 2
fi
ARTEMIS_RECOVERY_ROOT="${PDE_GAV_WORKTREES_ROOT}/artemis-budget-rollback-20260909"
cd -- "$ARTEMIS_RECOVERY_ROOT"
${PDE_GAV_PYTHON} scripts/artemis_budget_acceptance.py launch \
  --acceptance "$2" --publication "$3" --source-campaign "$1"
export UNIFIED_PLASMA_ROOT="$ARTEMIS_RECOVERY_ROOT"
export ARTEMIS_CANONICAL_RUN_FAMILY_ROOT="${PDE_GAV_RUN_ROOT}/artemis_r1_budget_recovery_20260909"
export ARTEMIS_ROUND_ONE_RECOVERY_SOURCE="$1"
export ARTEMIS_RECOVERY_CANDIDATE_WORKERS=4
exec bash scripts/slurm_artemis_canonical_5round.sh
