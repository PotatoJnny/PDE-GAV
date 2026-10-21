#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "instruction-only solver tools must run on a Slurm compute node" >&2
  exit 2
fi
case "$(realpath .)/" in
  /scratch/*) ;;
  *)
    echo "instruction-only solver workspace must be under /scratch" >&2
    exit 2
    ;;
esac
for required in \
  prompt.md \
  public_eval.py \
  solver_agent_instructions.md \
  case_manifest.json \
  cases; do
  if [[ ! -e "${required}" ]]; then
    echo "missing public solver workspace input: ${required}" >&2
    exit 2
  fi
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "the cluster Python venv is not active" >&2
  exit 2
fi
PYTHON_BIN="$(command -v python)"
case "${PYTHON_BIN}" in
  "${VIRTUAL_ENV}/"*) ;;
  *)
    echo "python is not provided by VIRTUAL_ENV: ${PYTHON_BIN}" >&2
    exit 2
    ;;
esac

python - <<'PY'
import json
import os
import socket
from pathlib import Path

import torch

assert torch.cuda.is_available()
value = torch.ones(1, device="cuda").sum()
assert float(value) == 1.0

payload = {
    "approval_policy": "on-request",
    "approvals_reviewer": "auto_review",
    "context_isolation_enforcement": (
        "public_workspace_redaction_and_model_instructions"
    ),
    "credential_access_prohibited_by_instruction": True,
    "gpu_available_to_runner": True,
    "gpu_tool_access_requires_auto_reviewed_escalation": True,
    "host": socket.gethostname(),
    "host_filesystem_os_isolated": False,
    "network_access_allowed": True,
    "network_policy": "bounded_literature_and_dependency_access",
    "passed": True,
    "probe_layer": "instruction_only_runner_environment",
    "python": str(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"),
    "sandbox_bypass_enabled": False,
    "sandbox_mode": "workspace-write",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_tmpdir": os.environ.get("SLURM_TMPDIR"),
    "venv": os.environ["VIRTUAL_ENV"],
    "workspace": str(Path.cwd()),
    "workspace_public_redaction_required": True,
}
text = json.dumps(payload, sort_keys=True)
Path("solver_instruction_preflight.json").write_text(text + "\n")
print(text)
PY
