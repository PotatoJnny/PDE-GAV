#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=artemis_d2_guided_5r
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ $# -ne 2 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires D2 and Guidance acceptance reports and a Slurm allocation." >&2
  exit 2
fi
ARTEMIS_GUIDANCE_ROOT="${PDE_GAV_WORKTREES_ROOT}/artemis-guidance-canonical-20260908"
cd -- "$ARTEMIS_GUIDANCE_ROOT"
${PDE_GAV_PYTHON} scripts/artemis_guidance_acceptance.py launch \
  --cold-report "$1" --guidance-report "$2"
export UNIFIED_PLASMA_ROOT="$ARTEMIS_GUIDANCE_ROOT"
export ARTEMIS_CANONICAL_RUN_FAMILY_ROOT="${PDE_GAV_RUN_ROOT}/artemis_canonical_d2_guided_5round_20260908"
exec bash scripts/slurm_artemis_canonical_5round.sh
