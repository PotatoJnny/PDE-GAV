#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=plasma_canonical_smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Canonical Plasma smoke requires a Slurm allocation." >&2
  exit 2
fi
PLASMA_CHECKOUT="${PLASMA_INTEGRATION_ROOT:-${SLURM_SUBMIT_DIR:-}}"
PLASMA_PYTHON="${PLASMA_PYTHON:-python}"
if [[ ! -f "${PLASMA_CHECKOUT}/scripts/run_plasma_canonical_smoke.py" || ! -x "${PLASMA_PYTHON}" ]]; then
  echo "The integration checkout or requested Python environment is unavailable." >&2
  exit 2
fi
PLASMA_CHECKOUT="$(realpath -- "${PLASMA_CHECKOUT}")"
PLASMA_SMOKE_ROOT="${PLASMA_CHECKOUT}/artifacts/plasma-canonical-smoke-${SLURM_JOB_ID}"
if [[ -e "${PLASMA_SMOKE_ROOT}" ]]; then
  echo "Refusing to reuse Plasma smoke output: ${PLASMA_SMOKE_ROOT}" >&2
  exit 2
fi
mkdir -p "${PLASMA_SMOKE_ROOT}"
if ! module load scipy-stack pytest; then
  echo "Requested module names unavailable; using the existing explicit cluster venv." >&2
fi
export PYTHONPATH="${PLASMA_CHECKOUT}/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# GKW's qualified launchers use CPU JAX geometry and GPU Torch inference.
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/plasma-canonical-pycache"
cd "${PLASMA_CHECKOUT}"
PLASMA_FAILURES=0

run_plasma_case() {
  local plasma_label="$1" plasma_physics="$2" plasma_config="$3"
  if "${PLASMA_PYTHON}" scripts/run_plasma_canonical_smoke.py \
      --physics "${plasma_physics}" --config "${plasma_config}" \
      --output "${PLASMA_SMOKE_ROOT}/${plasma_label}" --device cuda \
      > "${PLASMA_SMOKE_ROOT}/${plasma_label}.log" 2>&1; then
    echo "Canonical smoke completed: ${plasma_label}"
  else
    PLASMA_FAILURES=$((PLASMA_FAILURES + 1))
    echo "Canonical smoke failed: ${plasma_label}; detailed report/log retained." >&2
  fi
}

# Fresh interpreter per physics keeps the separately pinned physical packages isolated.
run_plasma_case artemis artemis configs/plasma_canonical_artemis.json
run_plasma_case gkw gkw configs/plasma_canonical_gkw.json
run_plasma_case vpfp-weak vpfp configs/vpfp_unified_live_weak_collision.json
run_plasma_case vpfp-strong vpfp configs/vpfp_unified_live_strong_collision.json

# The separate learned-model experiment does train new weights; the four
# canonical smoke reports above are explicitly no-training diagnostics.
export VPFP_INTEGRATION_ROOT="${PLASMA_CHECKOUT}" VPFP_PYTHON="${PLASMA_PYTHON}"
if ! bash scripts/slurm_vpfp_learned_answer_qualification.sh \
    > "${PLASMA_SMOKE_ROOT}/vpfp-learned-qualification.log" 2>&1; then
  PLASMA_FAILURES=$((PLASMA_FAILURES + 1))
  echo "Learned VPFP qualification had failures; its scratch reports/logs were retained." >&2
fi
echo "Plasma smoke artifacts: ${PLASMA_SMOKE_ROOT}; failed stages: ${PLASMA_FAILURES}"
exit "${PLASMA_FAILURES}"
