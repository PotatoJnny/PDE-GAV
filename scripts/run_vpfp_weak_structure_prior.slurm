#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --job-name=vpfp_weak_structure_prior
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
exec bash ${PDE_GAV_WORKTREES_ROOT}/vpfp-weak-noise-device-fix/scripts/run_vpfp_weak_prior_job.sh structure
