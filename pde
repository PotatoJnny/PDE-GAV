#!/usr/bin/env bash
set -euo pipefail

PRODUCT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PRODUCT_ROOT}/src:${PYTHONPATH:-}"
exec python -m pde_agent_verifier_solver "$@"
