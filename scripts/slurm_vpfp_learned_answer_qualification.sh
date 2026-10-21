#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=vpfp_learned_answer_smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_RUN_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "This bounded VPFP qualification requires a Slurm allocation." >&2
  exit 2
fi
VPFP_CHECKOUT="${VPFP_INTEGRATION_ROOT:-${SLURM_SUBMIT_DIR:-}}"
VPFP_PYTHON="${VPFP_PYTHON:-python}"
VPFP_RUN_ROOT="${PDE_GAV_RUN_ROOT}/vpfp-learned-answer-smoke-${SLURM_JOB_ID}"
if [[ ! -f "${VPFP_CHECKOUT}/scripts/train_vpfp_answer_distribution.py" || ! -x "${VPFP_PYTHON}" ]]; then
  echo "The VPFP integration checkout or Python environment is unavailable." >&2
  exit 2
fi
if [[ -e "${VPFP_RUN_ROOT}" ]]; then
  echo "Refusing to reuse VPFP qualification run root: ${VPFP_RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${VPFP_RUN_ROOT}"
export PYTHONPATH="${VPFP_CHECKOUT}/src:${VPFP_CHECKOUT}/legacy/vpfp/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-pycache"
cd "${VPFP_CHECKOUT}"
VPFP_FAILURES=0
for VPFP_REGIME in weak strong; do
  VPFP_CONFIG="${VPFP_CHECKOUT}/configs/vpfp_unified_live_${VPFP_REGIME}_collision.json"
  VPFP_CORPUS="${VPFP_RUN_ROOT}/${VPFP_REGIME}-corpus.npz"
  VPFP_CHECKPOINT="${VPFP_RUN_ROOT}/${VPFP_REGIME}-learned.npz"
  if "${VPFP_PYTHON}" scripts/train_vpfp_answer_distribution.py collect \
      --config "${VPFP_CONFIG}" --output "${VPFP_CORPUS}" --samples 32 --batch-size 4 \
      > "${VPFP_RUN_ROOT}/${VPFP_REGIME}-collect.log" 2>&1; then
    if "${VPFP_PYTHON}" scripts/train_vpfp_answer_distribution.py train \
        --corpus "${VPFP_CORPUS}" --output "${VPFP_CHECKPOINT}" --rank 8 \
        > "${VPFP_RUN_ROOT}/${VPFP_REGIME}-train.log" 2>&1; then
      read -r VPFP_CHECKPOINT_SHA VPFP_UNUSED < <(sha256sum "${VPFP_CHECKPOINT}")
      if ! "${VPFP_PYTHON}" scripts/qualify_vpfp_answer_distribution.py \
          --config "${VPFP_CONFIG}" --corpus "${VPFP_CORPUS}" \
          --checkpoint "${VPFP_CHECKPOINT}" --checkpoint-sha256 "${VPFP_CHECKPOINT_SHA}" \
          --report "${VPFP_RUN_ROOT}/${VPFP_REGIME}-qualification.json" --samples 8 \
          > "${VPFP_RUN_ROOT}/${VPFP_REGIME}-qualification.log" 2>&1; then
        VPFP_FAILURES=$((VPFP_FAILURES + 1))
      fi
    else
      VPFP_FAILURES=$((VPFP_FAILURES + 1))
    fi
  else
    VPFP_FAILURES=$((VPFP_FAILURES + 1))
  fi
done
echo "VPFP bounded learned-answer qualification artifacts: ${VPFP_RUN_ROOT}; failed regimes: ${VPFP_FAILURES}"
exit "${VPFP_FAILURES}"
