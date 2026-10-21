#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=artemis_d5_gpu_reuse
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ $# -ne 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires a completed parallel software/GPU smoke report and Slurm allocation." >&2
  exit 2
fi
ARTEMIS_GPU_REUSE_ROOT="${PDE_GAV_WORKTREES_ROOT}/artemis-multigpu-driver-20260910"
cd -- "$ARTEMIS_GPU_REUSE_ROOT"
${PDE_GAV_PYTHON} -c '
import json, subprocess, sys
with open(sys.argv[1]) as handle:
    report = json.load(handle)
if report.get("schema") != "artemis_parallel_evaluation_smoke_v1" or report.get("status") != "passed":
    raise SystemExit("A passing short parallel integration smoke is required")
tested = report.get("driver_source", {}).get("commit")
if not tested:
    raise SystemExit("Smoke report is missing its tested driver revision")
# Check source identity once at launch; documentation-only commits do not
# require another numerical run. This is not an online Oracle check.
subprocess.run(["git", "diff", "--quiet", tested, "HEAD", "--", "src", "scripts", "configs"], check=True)
with open("configs/plasma_artemis_d5_guided.json") as handle:
    execution_config = json.load(handle)
if report.get("physical_source", {}).get("commit") != execution_config["factory_settings"]["expected_physical_revision"]:
    raise SystemExit("Smoke report did not exercise the selected physical backend")
' "$1"
export UNIFIED_PLASMA_ROOT="$ARTEMIS_GPU_REUSE_ROOT"
export ARTEMIS_CANONICAL_RUN_FAMILY_ROOT="${PDE_GAV_RUN_ROOT}/artemis_d5_gpu_reuse_20260910"
export ARTEMIS_ROUND_ONE_RECOVERY_SOURCE="${PDE_GAV_RUN_ROOT}/artemis_canonical_d2_guided_5round_20260908/job-5309385/campaign"
export ARTEMIS_RECOVERY_CANDIDATE_WORKERS=6
export ARTEMIS_REUSE_AGENT_GPUS_FOR_EVALUATION=1
exec bash scripts/slurm_artemis_canonical_5round.sh
