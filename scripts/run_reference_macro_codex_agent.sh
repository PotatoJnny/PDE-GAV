#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  echo "usage: $0 --model MODEL --reasoning-effort EFFORT --codex-binary PATH --auth-file PATH" >&2
}

MODEL=""
REASONING_EFFORT=""
CODEX_BINARY=""
CODEX_AUTH_SOURCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MODEL="$2"
      shift 2
      ;;
    --reasoning-effort)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      REASONING_EFFORT="$2"
      shift 2
      ;;
    --codex-binary)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      CODEX_BINARY="$2"
      shift 2
      ;;
    --auth-file)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      CODEX_AUTH_SOURCE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

for required_argument in MODEL REASONING_EFFORT CODEX_BINARY CODEX_AUTH_SOURCE; do
  if [[ -z "${!required_argument}" ]]; then
    echo "missing required argument: ${required_argument}" >&2
    usage
    exit 2
  fi
done
if [[ ! "${MODEL}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "model contains unsupported characters" >&2
  exit 2
fi
case "${REASONING_EFFORT}" in
  minimal|low|medium|high|xhigh|max|ultra) ;;
  *) echo "unsupported reasoning effort: ${REASONING_EFFORT}" >&2; exit 2 ;;
esac

for required_environment in \
  PDE_MACRO_WORKSPACE PDE_MACRO_CONTEXT PDE_MACRO_PROMPT PDE_MACRO_SUBMISSION \
  SLURM_JOB_ID SLURM_TMPDIR VIRTUAL_ENV CUDA_VISIBLE_DEVICES; do
  if [[ -z "${!required_environment:-}" ]]; then
    echo "missing required environment: ${required_environment}" >&2
    exit 2
  fi
done

# A process worker owns exactly one accelerator. Preserve Slurm's opaque device
# token without interpreting it as a local ordinal, but reject values that could
# expose multiple GPUs (or deliberately hide every GPU) before launching Codex.
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^([0-9]+|GPU-[0-9A-Fa-f-]+|MIG-[A-Za-z0-9._:/-]+)$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one GPU or MIG device" >&2
  exit 2
fi
if [[ -n "${CUDA_DEVICE_ORDER:-}" ]]; then
  case "${CUDA_DEVICE_ORDER}" in
    FASTEST_FIRST|PCI_BUS_ID) ;;
    *)
      echo "CUDA_DEVICE_ORDER must be FASTEST_FIRST or PCI_BUS_ID" >&2
      exit 2
      ;;
  esac
fi

if [[ ! -d "${VIRTUAL_ENV}" ]]; then
  echo "VIRTUAL_ENV is not a directory" >&2
  exit 2
fi
VIRTUAL_ENV="$(realpath -- "${VIRTUAL_ENV}")"
if [[ ! -x "${VIRTUAL_ENV}/bin/python" ]]; then
  echo "active VIRTUAL_ENV has no executable bin/python" >&2
  exit 2
fi
ACTIVE_PYTHON="$(command -v python || true)"
if [[ -z "${ACTIVE_PYTHON}" ]] || \
   [[ "$(realpath -- "${ACTIVE_PYTHON}")" != "$(realpath -- "${VIRTUAL_ENV}/bin/python")" ]]; then
  echo "python is not provided by the active VIRTUAL_ENV" >&2
  exit 2
fi

path_is_in_workspace() {
  local candidate="$1"
  case "${candidate}/" in
    "${WORKSPACE}/"*) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -d "${PDE_MACRO_WORKSPACE}" || -L "${PDE_MACRO_WORKSPACE}" ]]; then
  echo "PDE_MACRO_WORKSPACE must be a real directory" >&2
  exit 2
fi
WORKSPACE="$(realpath -- "${PDE_MACRO_WORKSPACE}")"

resolve_workspace_input() {
  local raw_path="$1"
  local label="$2"
  local resolved
  if [[ ! -f "${raw_path}" || -L "${raw_path}" ]]; then
    echo "${label} must be a regular non-symlink file" >&2
    return 2
  fi
  resolved="$(realpath -- "${raw_path}")"
  if ! path_is_in_workspace "${resolved}"; then
    echo "${label} must stay inside PDE_MACRO_WORKSPACE" >&2
    return 2
  fi
  printf '%s\n' "${resolved}"
}

CONTEXT="$(resolve_workspace_input "${PDE_MACRO_CONTEXT}" PDE_MACRO_CONTEXT)"
PROMPT="$(resolve_workspace_input "${PDE_MACRO_PROMPT}" PDE_MACRO_PROMPT)"
SUBMISSION="$(realpath -m -- "${PDE_MACRO_SUBMISSION}")"
if ! path_is_in_workspace "${SUBMISSION}"; then
  echo "PDE_MACRO_SUBMISSION must stay inside PDE_MACRO_WORKSPACE" >&2
  exit 2
fi
if [[ ! -d "$(dirname -- "${SUBMISSION}")" ]]; then
  echo "PDE_MACRO_SUBMISSION parent directory does not exist" >&2
  exit 2
fi
if [[ -e "${SUBMISSION}" && ( ! -f "${SUBMISSION}" || -L "${SUBMISSION}" ) ]]; then
  echo "existing PDE_MACRO_SUBMISSION must be a regular non-symlink file" >&2
  exit 2
fi
if [[ "${SUBMISSION}" == "${CONTEXT}" || "${SUBMISSION}" == "${PROMPT}" ]]; then
  echo "PDE_MACRO_SUBMISSION must not overwrite a runtime input" >&2
  exit 2
fi

SCRIPT_DIR="$(dirname -- "$(realpath -- "$0")")"
REPOSITORY_ROOT="$(dirname -- "${SCRIPT_DIR}")"
AUTH_SUPERVISOR="${SCRIPT_DIR}/run_codex_with_ephemeral_auth.py"
CODEX_CONFIG_TEMPLATE="${SCRIPT_DIR}/reference_macro_codex.config.toml"
for control_file in "${AUTH_SUPERVISOR}" "${CODEX_CONFIG_TEMPLATE}"; do
  if [[ ! -f "${control_file}" || -L "${control_file}" ]]; then
    echo "missing trusted launcher input: ${control_file}" >&2
    exit 2
  fi
done

if [[ ! -x "${CODEX_BINARY}" ]]; then
  echo "Codex binary is not executable" >&2
  exit 2
fi
CODEX_BINARY="$(realpath -- "${CODEX_BINARY}")"
if [[ ! -f "${CODEX_BINARY}" || ! -x "${CODEX_BINARY}" ]]; then
  echo "resolved Codex binary is not an executable regular file" >&2
  exit 2
fi
if [[ ! -f "${CODEX_AUTH_SOURCE}" || -L "${CODEX_AUTH_SOURCE}" ]]; then
  echo "Codex auth source must be a regular non-symlink file" >&2
  exit 2
fi
CODEX_AUTH_SOURCE="$(realpath -- "${CODEX_AUTH_SOURCE}")"
if path_is_in_workspace "${CODEX_AUTH_SOURCE}"; then
  echo "Codex auth source must remain outside the Agent workspace" >&2
  exit 2
fi

if [[ ! -d "${SLURM_TMPDIR}" || -L "${SLURM_TMPDIR}" ]]; then
  echo "SLURM_TMPDIR must be a real directory" >&2
  exit 2
fi
SLURM_TMPDIR="$(realpath -- "${SLURM_TMPDIR}")"
if path_is_in_workspace "${SLURM_TMPDIR}"; then
  echo "ephemeral Codex runtime must remain outside the Agent workspace" >&2
  exit 2
fi

FINAL_MESSAGE="${WORKSPACE}/codex-last-message.txt"
AUTH_AUDIT_COPY="${WORKSPACE}/codex-auth-lifecycle.json"
for agent_output in "${FINAL_MESSAGE}" "${AUTH_AUDIT_COPY}"; do
  if [[ -e "${agent_output}" || -L "${agent_output}" ]]; then
    echo "refusing to overwrite pre-existing launcher output: ${agent_output}" >&2
    exit 2
  fi
done

RUNTIME_ROOT=""
cleanup() {
  if [[ -n "${RUNTIME_ROOT}" && -d "${RUNTIME_ROOT}" ]]; then
    chmod -R u+rwX "${RUNTIME_ROOT}" 2>/dev/null || true
    rm -rf -- "${RUNTIME_ROOT}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

RUNTIME_ROOT="$(mktemp -d "${SLURM_TMPDIR}/reference-macro-codex.XXXXXX")"
RUNTIME_CODEX_HOME="${RUNTIME_ROOT}/codex-home"
AGENT_HOME="${RUNTIME_ROOT}/agent-home"
mkdir -p "${RUNTIME_CODEX_HOME}" "${AGENT_HOME}"
chmod 700 "${RUNTIME_ROOT}" "${RUNTIME_CODEX_HOME}" "${AGENT_HOME}"
install -m 600 "${CODEX_AUTH_SOURCE}" "${RUNTIME_CODEX_HOME}/auth.json"
install -m 600 "${CODEX_CONFIG_TEMPLATE}" "${RUNTIME_CODEX_HOME}/config.toml"

AGENT_PYTHONPATH="${REPOSITORY_ROOT}/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
  if [[ "${PYTHONPATH}" == *$'\n'* || "${PYTHONPATH}" == *$'\r'* ]]; then
    echo "PYTHONPATH must not contain newline characters" >&2
    exit 2
  fi
  AGENT_PYTHONPATH="${AGENT_PYTHONPATH}:${PYTHONPATH}"
fi
LOOPBACK_NO_PROXY="127.0.0.1,localhost"
if [[ -n "${NO_PROXY:-}" ]]; then
  LOOPBACK_NO_PROXY="${LOOPBACK_NO_PROXY},${NO_PROXY}"
fi

RUN_ENV=(
  "HOME=${AGENT_HOME}"
  "CODEX_HOME=${RUNTIME_CODEX_HOME}"
  "PATH=${VIRTUAL_ENV}/bin:${PATH}"
  "VIRTUAL_ENV=${VIRTUAL_ENV}"
  "PYTHONPATH=${AGENT_PYTHONPATH}"
  "LANG=${LANG:-C.UTF-8}"
  "TERM=${TERM:-dumb}"
  "TMPDIR=${SLURM_TMPDIR}"
  "TMP=${SLURM_TMPDIR}"
  "TEMP=${SLURM_TMPDIR}"
  "USER=${USER:-reference-macro-agent}"
  "PYTHONDONTWRITEBYTECODE=1"
  "PYTHONUNBUFFERED=1"
  "PYTHONNOUSERSITE=1"
  "SLURM_JOB_ID=${SLURM_JOB_ID}"
  "SLURM_JOB_NAME=${SLURM_JOB_NAME:-reference_macro_agent}"
  "SLURM_TMPDIR=${SLURM_TMPDIR}"
  "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
  "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
  "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  "SLURM_GPUS=${SLURM_GPUS:-}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
  "MKL_NUM_THREADS=${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
  "PDE_MACRO_WORKSPACE=${WORKSPACE}"
  "PDE_MACRO_CONTEXT=${CONTEXT}"
  "PDE_MACRO_PROMPT=${PROMPT}"
  "PDE_MACRO_SUBMISSION=${SUBMISSION}"
  "NO_PROXY=${LOOPBACK_NO_PROXY}"
)
if [[ -n "${CUDA_DEVICE_ORDER:-}" ]]; then
  RUN_ENV+=("CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}")
fi
for inherited_name in \
  HTTP_PROXY HTTPS_PROXY \
  SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
  if [[ -n "${!inherited_name:-}" ]]; then
    RUN_ENV+=("${inherited_name}=${!inherited_name}")
  fi
done

set +e
env -i "${RUN_ENV[@]}" \
  "${VIRTUAL_ENV}/bin/python" "${AUTH_SUPERVISOR}" \
  --auth "${RUNTIME_CODEX_HOME}/auth.json" \
  --audit "${RUNTIME_CODEX_HOME}/auth-lifecycle.json" \
  --audit-copy "${AUTH_AUDIT_COPY}" \
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
  --config 'history.persistence="none"' \
  --json \
  --color never \
  --output-last-message "${FINAL_MESSAGE}" \
  --skip-git-repo-check \
  --cd "${WORKSPACE}" \
  - < "${PROMPT}"
CODEX_RETURN_CODE=$?
set -e

if [[ -e "${RUNTIME_CODEX_HOME}/auth.json" ]]; then
  echo "ephemeral Codex auth still exists after the supervised invocation" >&2
  exit 97
fi
if [[ ! -f "${RUNTIME_CODEX_HOME}/auth-lifecycle.json" ]]; then
  echo "missing internal Codex auth lifecycle audit" >&2
  exit 97
fi
"${VIRTUAL_ENV}/bin/python" -c \
  'import json, pathlib, sys; p=json.loads(pathlib.Path(sys.argv[1]).read_text()); raise SystemExit(0 if p.get("auth_removed_before_first_event_forwarded") is True else 97)' \
  "${RUNTIME_CODEX_HOME}/auth-lifecycle.json"
if [[ "${CODEX_RETURN_CODE}" -ne 0 ]]; then
  echo "Codex reference macro Agent exited with status ${CODEX_RETURN_CODE}" >&2
  exit "${CODEX_RETURN_CODE}"
fi
if [[ ! -f "${SUBMISSION}" || -L "${SUBMISSION}" ]]; then
  echo "Codex reference macro Agent did not write a regular submission" >&2
  exit 3
fi
if [[ ! -f "${FINAL_MESSAGE}" || -L "${FINAL_MESSAGE}" ]]; then
  echo "Codex reference macro Agent did not write a regular final message" >&2
  exit 3
fi
