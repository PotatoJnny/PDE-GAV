#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=gkw_lifecycle_mem_smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "GKW lifecycle memory smoke requires a Slurm compute node." >&2
  exit 2
fi

UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${UNIFIED_ROOT}" ]]; then
  echo "UNIFIED_PLASMA_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT}")"

export UNIFIED_PLASMA_ROOT="${UNIFIED_ROOT}"
export GKW_UNIFIED_CONFIG="${UNIFIED_ROOT}/configs/gkw_unified_lifecycle_memory_smoke.json"
export PLASMA_STORAGE_ROOT="${PLASMA_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
GKW_LIFECYCLE_SMOKE_ROOT="${GKW_LIFECYCLE_SMOKE_ROOT:-${PLASMA_STORAGE_ROOT}/unified-gkw-lifecycle-memory-smoke-20260827-v1}"
case "${GKW_LIFECYCLE_SMOKE_ROOT}/" in
  "${PLASMA_STORAGE_ROOT}/"*) ;;
  *) echo "GKW_LIFECYCLE_SMOKE_ROOT must be below PLASMA_STORAGE_ROOT." >&2; exit 2 ;;
esac
export PLASMA_RUN_ROOT="${GKW_LIFECYCLE_SMOKE_ROOT}/job-${SLURM_JOB_ID}"

TIME_REPORT="${SLURM_TMPDIR}/gkw-lifecycle-memory-smoke-${SLURM_JOB_ID}.time"
if [[ -e "${TIME_REPORT}" || -L "${TIME_REPORT}" ]]; then
  echo "Refusing to overwrite time report: ${TIME_REPORT}" >&2
  exit 2
fi
if [[ ! -x /usr/bin/time ]]; then
  echo "GNU time is required at /usr/bin/time." >&2
  exit 2
fi

set +e
/usr/bin/time -v -o "${TIME_REPORT}" \
  "${UNIFIED_ROOT}/scripts/slurm_gkw_unified_live_agent_iteration.sh"
CONTROLLER_STATUS=$?
set -e
cat -- "${TIME_REPORT}" >&2
if [[ "${CONTROLLER_STATUS}" -ne 0 ]]; then
  echo "GKW lifecycle memory smoke controller failed with status ${CONTROLLER_STATUS}." >&2
  exit "${CONTROLLER_STATUS}"
fi

RESULT="${PLASMA_RUN_ROOT}/result/result.json"
LEDGER="${PLASMA_RUN_ROOT}/_controller/sampling_ledger.json"
for required in "${RESULT}" "${LEDGER}"; do
  if [[ ! -s "${required}" || -L "${required}" ]]; then
    echo "GKW lifecycle memory smoke lacks required output: ${required}" >&2
    exit 3
  fi
done
if ! grep -Fq '"status": "complete"' "${LEDGER}"; then
  echo "GKW lifecycle memory smoke sampling ledger is incomplete." >&2
  exit 3
fi

RESULT_PYTHON="${PLASMA_VENV:-${HOME}/environment/venv}/bin/python"
if [[ ! -x "${RESULT_PYTHON}" ]]; then
  echo "GKW lifecycle memory smoke lacks the result validator Python." >&2
  exit 3
fi
if ! "${RESULT_PYTHON}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
terminal = payload.get("terminal")
cold_start = terminal.get("cold_start") if isinstance(terminal, dict) else None
case_count = terminal.get("case_count") if isinstance(terminal, dict) else None
valid_count = cold_start.get("valid_member_count") if isinstance(cold_start, dict) else None
invalid_count = cold_start.get("invalid_member_count") if isinstance(cold_start, dict) else None
valid = (
    isinstance(terminal, dict)
    and terminal.get("evaluation_status") == "valid"
    and isinstance(cold_start, dict)
    and type(case_count) is int
    and type(valid_count) is int
    and type(invalid_count) is int
    and invalid_count == 0
    and valid_count == case_count
)
raise SystemExit(0 if valid else 1)
' "${RESULT}"; then
  echo "GKW lifecycle memory smoke terminal evaluation is not fully valid." >&2
  exit 3
fi

MAX_RSS_KB="$(awk -F: '/Maximum resident set size \(kbytes\)/ {gsub(/[[:space:]]/, "", $2); print $2}' "${TIME_REPORT}")"
if [[ ! "${MAX_RSS_KB}" =~ ^[0-9]+$ ]]; then
  echo "GKW lifecycle memory smoke could not parse maximum RSS." >&2
  exit 3
fi
MAX_RSS_LIMIT_KB=$((28 * 1024 * 1024))
if (( MAX_RSS_KB > MAX_RSS_LIMIT_KB )); then
  echo "GKW lifecycle memory smoke exceeded the 28 GiB RSS gate: ${MAX_RSS_KB} KiB." >&2
  exit 3
fi

printf 'GKW lifecycle memory smoke complete: result=%s max_rss_kib=%s\n' \
  "${RESULT}" "${MAX_RSS_KB}"
