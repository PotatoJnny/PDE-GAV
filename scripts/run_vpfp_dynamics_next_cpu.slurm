#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=vpfp_next_cpu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Research-only; no child submissions or online asset replacement.
# Requires the site variables PDE_GAV_WORKTREES_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
exec bash ${PDE_GAV_WORKTREES_ROOT}/vpfp-dynamics-next-stage-20260907/scripts/run_vpfp_dynamics_next_job.sh cpu

