#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=vpfp_learned_canonical_smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Learned VPFP Campaign smoke requires a Slurm allocation." >&2
  exit 2
fi
VPFP_CHECKOUT="${PLASMA_INTEGRATION_ROOT:-${SLURM_SUBMIT_DIR:-}}"
VPFP_PYTHON="${PLASMA_PYTHON:-python}"
VPFP_ASSET_ROOT=${PDE_GAV_RUN_ROOT}/vpfp-learned-answer-smoke-5229231
if [[ ! -f "${VPFP_CHECKOUT}/scripts/run_plasma_canonical_smoke.py" || ! -x "${VPFP_PYTHON}" ]]; then
  echo "The integration checkout or Python environment is unavailable." >&2
  exit 2
fi
VPFP_CHECKOUT="$(realpath -- "${VPFP_CHECKOUT}")"
VPFP_SMOKE_ROOT="${VPFP_CHECKOUT}/artifacts/plasma-canonical-smoke-${SLURM_JOB_ID}"
if [[ -e "${VPFP_SMOKE_ROOT}" ]]; then
  echo "Refusing to reuse learned VPFP output: ${VPFP_SMOKE_ROOT}" >&2
  exit 2
fi
mkdir -p "${VPFP_SMOKE_ROOT}"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing explicit venv." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-learned-canonical-pycache"
cd "${VPFP_CHECKOUT}"
VPFP_FAILURES=0
for VPFP_REGIME in weak strong; do
  case "${VPFP_REGIME}" in
    weak) VPFP_SHA=39d2c60f2f44637148a07624b4c3b76fd7bbeb26478477f71430ebd6c9bf0579 ;;
    strong) VPFP_SHA=79ae78b9fec9a9f842c4026e640b57ca3036e73d5be9bc1e98862d70d7009460 ;;
  esac
  if ! "${VPFP_PYTHON}" scripts/run_plasma_canonical_smoke.py \
      --physics vpfp --config "configs/vpfp_unified_live_${VPFP_REGIME}_collision.json" \
      --learned-checkpoint "${VPFP_ASSET_ROOT}/${VPFP_REGIME}-learned.npz" \
      --learned-checkpoint-sha256 "${VPFP_SHA}" \
      --output "${VPFP_SMOKE_ROOT}/vpfp-learned-${VPFP_REGIME}" --device cuda \
      > "${VPFP_SMOKE_ROOT}/vpfp-learned-${VPFP_REGIME}.log" 2>&1; then
    VPFP_FAILURES=$((VPFP_FAILURES + 1))
  fi
done
echo "Learned VPFP Campaign reports: ${VPFP_SMOKE_ROOT}; failed stages: ${VPFP_FAILURES}"
exit "${VPFP_FAILURES}"
