#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=artemis_d2_cold_5r
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# This is a new full campaign from the original S0, not a mid-round resume.
# The canonical launcher supplies the updated b2cd7b8 execution contract.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

ARTEMIS_COLD_ROOT="${PDE_GAV_WORKTREES_ROOT}/plasma-canonical-integration"
ARTEMIS_COLD_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_COLD_REPORT="${1:?Supply the completed cold acceptance-summary.json path}"
if [[ $# -ne 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires one acceptance report and a Slurm compute allocation." >&2
  exit 2
fi

"${ARTEMIS_COLD_PYTHON}" - "${ARTEMIS_COLD_ROOT}" "${ARTEMIS_COLD_REPORT}" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
if report.get("status") != "passed":
    raise SystemExit("Cold Backend acceptance did not pass")
commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if report.get("canonical_commit") != commit:
    raise SystemExit("Canonical code changed since cold Backend acceptance")
digest = hashlib.sha256((root / "configs/plasma_canonical_artemis.json").read_bytes()).hexdigest()
if report.get("backend_config_sha256") != digest:
    raise SystemExit("Backend configuration changed since cold acceptance")
print(json.dumps({"cold_acceptance": str(report_path.resolve()), "canonical_commit": commit,
                  "backend_config_sha256": digest, "launch_mode": "new_full_five_round_campaign"}), flush=True)
PY

export UNIFIED_PLASMA_ROOT="${ARTEMIS_COLD_ROOT}"
export ARTEMIS_CANONICAL_RUN_FAMILY_ROOT="${PDE_GAV_RUN_ROOT}/artemis_canonical_d2_cold_5round_20260908"
cd -- "${ARTEMIS_COLD_ROOT}"
exec bash scripts/slurm_artemis_canonical_5round.sh
