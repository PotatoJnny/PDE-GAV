#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --job-name=vpfp_demand_batch_cpu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
export VPFP_DEMAND_CPU_ONLY=1
exec bash ${PDE_GAV_WORKTREES_ROOT}/vpfp-demand-batched-oracle-20260908/scripts/run_vpfp_demand_oracle_probe.slurm
