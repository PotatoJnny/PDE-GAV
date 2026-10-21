#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3
#SBATCH --job-name=gkw_fresh_objective_ab
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Requires the site variables PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "GKW fresh objective A/B requires a Slurm array task." >&2
  exit 2
fi

UNIFIED_ROOT="${UNIFIED_PLASMA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${UNIFIED_ROOT}" ]]; then
  echo "UNIFIED_PLASMA_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
UNIFIED_ROOT="$(realpath -- "${UNIFIED_ROOT}")"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
case "${TASK_ID}" in
  0)
    ARM="equal"
    REPLICATE="1"
    CONFIG_NAME="gkw_unified_fresh_equal_pilot.json"
    ;;
  1)
    ARM="equal"
    REPLICATE="2"
    CONFIG_NAME="gkw_unified_fresh_equal_pilot.json"
    ;;
  2)
    ARM="primary_guardrail"
    REPLICATE="1"
    CONFIG_NAME="gkw_unified_fresh_primary_guardrail_pilot.json"
    ;;
  3)
    ARM="primary_guardrail"
    REPLICATE="2"
    CONFIG_NAME="gkw_unified_fresh_primary_guardrail_pilot.json"
    ;;
  *)
    echo "SLURM_ARRAY_TASK_ID must be in 0..3." >&2
    exit 2
    ;;
esac

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
export UNIFIED_PLASMA_ROOT="${UNIFIED_ROOT}"
export GKW_UNIFIED_CONFIG="${UNIFIED_ROOT}/configs/${CONFIG_NAME}"
export PLASMA_STORAGE_ROOT="${PLASMA_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
GKW_FRESH_AB_ROOT="${GKW_FRESH_AB_ROOT:-${PLASMA_STORAGE_ROOT}/unified-gkw-fresh-objective-ab-20260827-v2}"
case "${GKW_FRESH_AB_ROOT}/" in
  "${PLASMA_STORAGE_ROOT}/"*) ;;
  *) echo "GKW_FRESH_AB_ROOT must be below PLASMA_STORAGE_ROOT." >&2; exit 2 ;;
esac
export PLASMA_RUN_ROOT="${GKW_FRESH_AB_ROOT}/${ARM}/replicate-${REPLICATE}-array-${ARRAY_JOB_ID}-task-${TASK_ID}"

WATCHDOG_SECONDS=255600
WATCHDOG_GRACE_SECONDS=60
TIMEOUT_DIAGNOSTIC="${SLURM_TMPDIR}/gkw-controller-timeout-${SLURM_JOB_ID}-${TASK_ID}.stderr"
if [[ -e "${TIMEOUT_DIAGNOSTIC}" || -L "${TIMEOUT_DIAGNOSTIC}" ]]; then
  echo "Refusing to overwrite timeout diagnostic: ${TIMEOUT_DIAGNOSTIC}" >&2
  exit 2
fi
touch -- "${TIMEOUT_DIAGNOSTIC}"
chmod 600 "${TIMEOUT_DIAGNOSTIC}"
trap 'rm -f -- "${TIMEOUT_DIAGNOSTIC}"' EXIT

# Keep controller/Agent stderr live on the original descriptor while reserving
# timeout's own stderr for an unspoofable deadline diagnostic.
exec 3>&2
set +e
LC_ALL=C timeout --verbose --signal=TERM \
  --kill-after="${WATCHDOG_GRACE_SECONDS}s" \
  "${WATCHDOG_SECONDS}s" \
  bash -c 'exec 2>&3 3>&-; exec "$@"' \
  gkw-controller \
  "${UNIFIED_ROOT}/scripts/slurm_gkw_unified_live_agent_iteration.sh" \
  2> "${TIMEOUT_DIAGNOSTIC}"
CONTROLLER_STATUS=$?
set -e
exec 3>&-

if [[ -s "${TIMEOUT_DIAGNOSTIC}" ]]; then
  cat -- "${TIMEOUT_DIAGNOSTIC}" >&2
fi

WATCHDOG_TRIGGERED=false
WATCHDOG_FORCED_KILL=false
if grep -Eq '^timeout: sending signal (TERM|KILL) to command ' "${TIMEOUT_DIAGNOSTIC}"; then
  WATCHDOG_TRIGGERED=true
fi
if grep -Eq '^timeout: sending signal KILL to command ' "${TIMEOUT_DIAGNOSTIC}"; then
  WATCHDOG_FORCED_KILL=true
fi

if [[ "${WATCHDOG_TRIGGERED}" == true ]]; then
  WATCHDOG_FINISHED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  mkdir -p -- "${PLASMA_RUN_ROOT}"
  FAILURE_MARKER="${PLASMA_RUN_ROOT}/controller_watchdog_failure.json"
  TEMPORARY_MARKER="${FAILURE_MARKER}.tmp-${SLURM_JOB_ID}-${TASK_ID}"
  {
    printf '{\n'
    printf '  "schema": "unified-controller-watchdog-failure-v1",\n'
    printf '  "status": "failed",\n'
    printf '  "reason": "controller_watchdog_timeout",\n'
    printf '  "timeout_signal_observed": true,\n'
    printf '  "watchdog_seconds": %s,\n' "${WATCHDOG_SECONDS}"
    printf '  "kill_grace_seconds": %s,\n' "${WATCHDOG_GRACE_SECONDS}"
    printf '  "forced_kill": %s,\n' "${WATCHDOG_FORCED_KILL}"
    printf '  "exit_status": %s,\n' "${CONTROLLER_STATUS}"
    printf '  "slurm_job_id": "%s",\n' "${SLURM_JOB_ID}"
    printf '  "slurm_array_job_id": "%s",\n' "${ARRAY_JOB_ID}"
    printf '  "slurm_array_task_id": "%s",\n' "${TASK_ID}"
    printf '  "finished_at": "%s"\n' "${WATCHDOG_FINISHED_AT}"
    printf '}\n'
  } > "${TEMPORARY_MARKER}"
  chmod 600 "${TEMPORARY_MARKER}"
  mv -- "${TEMPORARY_MARKER}" "${FAILURE_MARKER}"
  echo "Controller watchdog failure recorded at ${FAILURE_MARKER}" >&2
elif [[ "${CONTROLLER_STATUS}" -ne 0 ]]; then
  FAILED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  mkdir -p -- "${PLASMA_RUN_ROOT}"
  FAILURE_MARKER="${PLASMA_RUN_ROOT}/controller_process_failure.json"
  TEMPORARY_MARKER="${FAILURE_MARKER}.tmp-${SLURM_JOB_ID}-${TASK_ID}"
  {
    printf '{\n'
    printf '  "schema": "unified-controller-process-failure-v1",\n'
    printf '  "status": "failed",\n'
    printf '  "reason": "controller_process_failure",\n'
    printf '  "timeout_signal_observed": false,\n'
    printf '  "exit_status": %s,\n' "${CONTROLLER_STATUS}"
    printf '  "slurm_job_id": "%s",\n' "${SLURM_JOB_ID}"
    printf '  "slurm_array_job_id": "%s",\n' "${ARRAY_JOB_ID}"
    printf '  "slurm_array_task_id": "%s",\n' "${TASK_ID}"
    printf '  "finished_at": "%s"\n' "${FAILED_AT}"
    printf '}\n'
  } > "${TEMPORARY_MARKER}"
  chmod 600 "${TEMPORARY_MARKER}"
  mv -- "${TEMPORARY_MARKER}" "${FAILURE_MARKER}"
  echo "Controller process failure recorded at ${FAILURE_MARKER}" >&2
fi

exit "${CONTROLLER_STATUS}"
