#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=""
OUTPUT_FILE=""
SUBMISSION=""
MODEL=""
REASONING_EFFORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --output-file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --submission)
      SUBMISSION="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --reasoning-effort)
      REASONING_EFFORT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

for value_name in WORKSPACE OUTPUT_FILE SUBMISSION MODEL REASONING_EFFORT; do
  if [[ -z "${!value_name}" ]]; then
    echo "missing required argument: ${value_name}" >&2
    exit 2
  fi
done
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "instruction-only solver must run inside a Slurm compute allocation" >&2
  exit 2
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "activate the cluster venv before launching the solver agent" >&2
  exit 2
fi
case "$(command -v python)" in
  "${VIRTUAL_ENV}/"*) ;;
  *)
    echo "python is not provided by the active VIRTUAL_ENV" >&2
    exit 2
    ;;
esac

path_in_run_storage() {
  local candidate="$1"
  case "${candidate}/" in
    /scratch/*) return 0 ;;
  esac
  if [[ -n "${VPFP_RUN_STORAGE_ROOT:-}" ]]; then
    local configured_root
    configured_root="$(realpath -m "${VPFP_RUN_STORAGE_ROOT}")"
    case "${candidate}/" in
      "${configured_root}/"*) return 0 ;;
    esac
  fi
  return 1
}

WORKSPACE="$(realpath "${WORKSPACE}")"
OUTPUT_FILE="$(realpath -m "${OUTPUT_FILE}")"
SUBMISSION="$(realpath -m "${SUBMISSION}")"
if ! path_in_run_storage "${WORKSPACE}"; then
  echo "instruction-only solver workspace needs configured run storage" >&2
  exit 2
fi
case "${OUTPUT_FILE}" in
  "${WORKSPACE}/"*) ;;
  *) echo "output file must stay inside the solver workspace" >&2; exit 2 ;;
esac
case "${SUBMISSION}" in
  "${WORKSPACE}/"*) ;;
  *) echo "submission must stay inside the solver workspace" >&2; exit 2 ;;
esac

SCRIPT_DIR="${VPFP_REPO_SCRIPT_DIR:-$(dirname "$(realpath "$0")")}"
SCRIPT_DIR="$(realpath "${SCRIPT_DIR}")"
CODEX_BINARY="${VPFP_CODEX_BINARY:-}"
if [[ -z "${CODEX_BINARY}" ]]; then
  for candidate in "${HOME}"/.codex/packages/standalone/releases/*/bin/codex; do
    if [[ -x "${candidate}" ]]; then
      CODEX_BINARY="${candidate}"
    fi
  done
fi
if [[ -z "${CODEX_BINARY}" ]]; then
  CODEX_BINARY="${HOME}/.codex/packages/standalone/current/bin/codex"
fi
if [[ ! -x "${CODEX_BINARY}" ]]; then
  CODEX_BINARY="$(command -v codex || true)"
fi
if [[ -z "${CODEX_BINARY}" || ! -x "${CODEX_BINARY}" ]]; then
  echo "Codex CLI is unavailable on the compute node" >&2
  exit 2
fi
CODEX_BINARY="$(realpath "${CODEX_BINARY}")"
CODEX_AUTH="${VPFP_CODEX_AUTH:-${HOME}/.codex/auth.json}"
CODEX_CONFIG_TEMPLATE="${SCRIPT_DIR}/vpfp_instruction_only_codex.config.toml"
AUTH_SUPERVISOR="${SCRIPT_DIR}/run_codex_with_ephemeral_auth.py"
for required in \
  "${CODEX_BINARY}" \
  "${CODEX_AUTH}" \
  "${CODEX_CONFIG_TEMPLATE}" \
  "${AUTH_SUPERVISOR}" \
  "${WORKSPACE}/prompt.md" \
  "${WORKSPACE}/public_eval.py" \
  "${WORKSPACE}/solver_agent_instructions.md" \
  "${WORKSPACE}/solver_instruction_preflight.sh"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing instruction-only solver runtime input: ${required}" >&2
    exit 2
  fi
done

RUNTIME_PARENT="$(realpath -m "${VPFP_CODEX_RUNTIME_PARENT:-${SLURM_TMPDIR}}")"
case "${RUNTIME_PARENT}/" in
  "${WORKSPACE}/"*)
    echo "Codex runtime must stay outside the solver workspace" >&2
    exit 2
    ;;
esac
case "${RUNTIME_PARENT}/" in
  "${SLURM_TMPDIR}/"*) ;;
  *)
    if ! path_in_run_storage "${RUNTIME_PARENT}"; then
      echo "Codex runtime parent needs run storage or SLURM_TMPDIR" >&2
      exit 2
    fi
    ;;
esac
mkdir -p "${RUNTIME_PARENT}"
RUNTIME_ROOT="$(mktemp -d "${RUNTIME_PARENT}/vpfp-instruction-agent.XXXXXX")"
RUNTIME_CODEX_HOME="${RUNTIME_ROOT}/codex-home"
AGENT_HOME="${RUNTIME_ROOT}/agent-home"
mkdir -p "${RUNTIME_CODEX_HOME}" "${AGENT_HOME}" "${WORKSPACE}/.cache"
chmod 700 "${RUNTIME_CODEX_HOME}" "${AGENT_HOME}"
cp "${CODEX_AUTH}" "${RUNTIME_CODEX_HOME}/auth.json"
cp "${CODEX_CONFIG_TEMPLATE}" "${RUNTIME_CODEX_HOME}/config.toml"
chmod 600 "${RUNTIME_CODEX_HOME}/auth.json"
chmod 644 "${RUNTIME_CODEX_HOME}/config.toml"

cleanup() {
  chmod 700 "${RUNTIME_CODEX_HOME}" 2>/dev/null || true
  rm -f "${RUNTIME_CODEX_HOME}/auth.json" 2>/dev/null || true
  rm -rf "${RUNTIME_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT

RUN_ENV=(
  "HOME=${AGENT_HOME}"
  "CODEX_HOME=${RUNTIME_CODEX_HOME}"
  "PATH=${PATH}"
  "VIRTUAL_ENV=${VIRTUAL_ENV}"
  "LANG=${LANG:-C.UTF-8}"
  "TERM=${TERM:-dumb}"
  "TMPDIR=${SLURM_TMPDIR}"
  "TMP=${SLURM_TMPDIR}"
  "TEMP=${SLURM_TMPDIR}"
  "USER=${USER}"
  "PYTHONDONTWRITEBYTECODE=1"
  "PYTHONUNBUFFERED=1"
  "PYTHONNOUSERSITE=1"
  "SLURM_JOB_ID=${SLURM_JOB_ID}"
  "SLURM_JOB_NAME=${SLURM_JOB_NAME:-vpfp_instruction_solver}"
  "SLURM_TMPDIR=${SLURM_TMPDIR}"
  "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
  "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
  "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  "SLURM_GPUS=${SLURM_GPUS:-}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
  "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  "LIBRARY_PATH=${LIBRARY_PATH:-}"
  "CPATH=${CPATH:-}"
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
  "MKL_NUM_THREADS=${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
  "XDG_CACHE_HOME=${WORKSPACE}/.cache"
  "TORCH_HOME=${WORKSPACE}/.cache/torch"
  "HF_HOME=${WORKSPACE}/.cache/huggingface"
  "PIP_CACHE_DIR=${WORKSPACE}/.cache/pip"
)
for name in \
  HTTP_PROXY HTTPS_PROXY NO_PROXY \
  SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
  if [[ -n "${!name:-}" ]]; then
    RUN_ENV+=("${name}=${!name}")
  fi
done
if [[ -n "${VPFP_ONLINE_GENERATOR_SOCKET:-}" ]]; then
  if ! grep -q '"VPFP_ONLINE_GENERATOR_SOCKET"' "${RUNTIME_CODEX_HOME}/config.toml"; then
    echo "Codex shell environment policy omits the online generator socket" >&2
    exit 2
  fi
  RUN_ENV+=("VPFP_ONLINE_GENERATOR_SOCKET=${VPFP_ONLINE_GENERATOR_SOCKET}")
fi

(
  cd "${WORKSPACE}"
  env -i "${RUN_ENV[@]}" \
    /bin/bash ./solver_instruction_preflight.sh
) > "${WORKSPACE}/solver_instruction_preflight.runner.log" 2>&1
test -f "${WORKSPACE}/solver_instruction_preflight.json"

set +e
env -i "${RUN_ENV[@]}" \
  "${VIRTUAL_ENV}/bin/python" "${AUTH_SUPERVISOR}" \
  --auth "${RUNTIME_CODEX_HOME}/auth.json" \
  --audit "${RUNTIME_CODEX_HOME}/auth_lifecycle.json" \
  -- \
  "${CODEX_BINARY}" exec \
  --model "${MODEL}" \
  --config "model_reasoning_effort=\"${REASONING_EFFORT}\"" \
  --ephemeral \
  --ignore-rules \
  --strict-config \
  --enable shell_tool \
  --disable apps \
  --disable plugins \
  --disable browser_use \
  --disable browser_use_external \
  --disable browser_use_full_cdp_access \
  --disable computer_use \
  --disable image_generation \
  --disable in_app_browser \
  --disable memories \
  --disable goals \
  --disable workspace_dependencies \
  --disable skill_mcp_dependency_install \
  --config "model_instructions_file=\"${WORKSPACE}/solver_agent_instructions.md\"" \
  --config 'history.persistence="none"' \
  --json \
  --color never \
  --output-last-message "${OUTPUT_FILE}" \
  --skip-git-repo-check \
  --cd "${WORKSPACE}" \
  - < "${WORKSPACE}/prompt.md"
CODEX_RETURN_CODE=$?
set -e

if [[ -e "${RUNTIME_CODEX_HOME}/auth.json" ]]; then
  echo "Codex auth file still exists after the supervised invocation" >&2
  exit 97
fi
if [[ ! -f "${RUNTIME_CODEX_HOME}/auth_lifecycle.json" ]]; then
  echo "missing Codex auth lifecycle audit" >&2
  exit 97
fi
cp "${RUNTIME_CODEX_HOME}/auth_lifecycle.json" "${WORKSPACE}/codex_auth_lifecycle.json"
if ! grep -q '"auth_removed_before_first_event_forwarded": true' "${WORKSPACE}/codex_auth_lifecycle.json"; then
  echo "Codex auth was not removed before solver tools became available" >&2
  exit 97
fi
if [[ "${CODEX_RETURN_CODE}" != "0" ]]; then
  echo "Codex instruction-only solver exited with status ${CODEX_RETURN_CODE}" >&2
  exit "${CODEX_RETURN_CODE}"
fi
if [[ ! -f "${WORKSPACE}/solver_instruction_preflight.json" ]]; then
  echo "solver agent did not run the required Slurm/venv/GPU preflight" >&2
  exit 97
fi
if ! grep -q '"approvals_reviewer": "auto_review"' "${WORKSPACE}/solver_instruction_preflight.json"; then
  echo "solver preflight did not record auto-review configuration" >&2
  exit 97
fi
if [[ ! -f "${SUBMISSION}" || -L "${SUBMISSION}" ]]; then
  echo "instruction-only solver agent did not write a regular submission.py" >&2
  exit 3
fi
if [[ ! -f "${OUTPUT_FILE}" || -L "${OUTPUT_FILE}" ]]; then
  echo "instruction-only solver agent did not write its final message" >&2
  exit 3
fi
