#!/usr/bin/env bash
# Run only inside a compute allocation. This script never submits/modifies jobs.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

VPFP_REGIME="${1:?expected weak or strong}"
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/run_vpfp_sobolev_ablation_job.sh weak|strong" >&2
  exit 2
fi
case "${VPFP_REGIME}" in
  weak)
    VPFP_SEED=320260904
    VPFP_SOURCE_CORPUS=${PDE_GAV_RUN_ROOT}/vpfp-full-coefficient-edm-weak-5230989/corpus
    ;;
  strong)
    VPFP_SEED=420260904
    VPFP_SOURCE_CORPUS=${PDE_GAV_RUN_ROOT}/vpfp-full-coefficient-edm-strong-5230988/corpus
    ;;
  *) echo "Unknown VPFP regime: ${VPFP_REGIME}" >&2; exit 2 ;;
esac
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "VPFP Sobolev ablation requires a Slurm compute allocation and local scratch." >&2
  exit 2
fi

VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-sobolev-matching
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_RUN_ROOT="${PDE_GAV_RUN_ROOT}/vpfp-sobolev-matching-${VPFP_REGIME}-${SLURM_JOB_ID}"
VPFP_CONFIG="${VPFP_CHECKOUT}/configs/vpfp_unified_live_${VPFP_REGIME}_collision.json"
VPFP_LAUNCH_STATUS="${VPFP_RUN_ROOT}.launch-exit-codes.tsv"
cd "${VPFP_CHECKOUT}"
if [[ ! -x "${VPFP_PYTHON}" || ! -r "${VPFP_CONFIG}" || ! -d "${VPFP_SOURCE_CORPUS}" \
      || ! -f scripts/run_vpfp_sobolev_ablation.py || ! -f scripts/run_plasma_canonical_smoke.py ]]; then
  echo "Missing frozen source corpus, configuration, runtime, or launch program." >&2
  exit 2
fi
if [[ -e "${VPFP_RUN_ROOT}" || -L "${VPFP_RUN_ROOT}" \
      || -e "${VPFP_LAUNCH_STATUS}" || -L "${VPFP_LAUNCH_STATUS}" ]]; then
  echo "Refusing to overwrite an existing Sobolev run or launch-status artifact." >&2
  exit 2
fi
# Include staged and untracked executable source; git diff alone misses the
# latter. Root-level Slurm logs/docs are not imported by this pipeline.
VPFP_SOURCE_STATUS="$(git status --porcelain --untracked-files=all -- src scripts configs legacy/vpfp)"
if [[ -n "${VPFP_SOURCE_STATUS}" ]]; then
  echo "Commit/freeze relevant source before this reproducible ablation:" >&2
  echo "${VPFP_SOURCE_STATUS}" >&2
  exit 2
fi
VPFP_REVISION="$(git rev-parse HEAD)"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing explicit venv." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-sobolev-pycache"

echo "VPFP Sobolev ${VPFP_REGIME}: revision=${VPFP_REVISION}; source=${VPFP_SOURCE_CORPUS}; output=${VPFP_RUN_ROOT}"
echo "Two arms, each 1200 training seconds with a 100000-step cap; source corpus is read-only."
VPFP_EXIT=0
VPFP_ABLATION_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_sobolev_ablation.py \
  --config "${VPFP_CONFIG}" --source-corpus "${VPFP_SOURCE_CORPUS}" \
  --output "${VPFP_RUN_ROOT}" --seed "${VPFP_SEED}" \
  --train-seconds 1200 --steps 100000 --batch-size 4 --collect-batch-size 8 \
  --spatial-weight .25 --temporal-weight .25 --sample-steps 64 --device cuda \
  || VPFP_ABLATION_EXIT=$?
VPFP_EXIT="${VPFP_ABLATION_EXIT}"
VPFP_BASELINE_SMOKE_EXIT=not_run
VPFP_SOBOLEV_SMOKE_EXIT=not_run

# A failed audit/training stage must not hide existing checkpoints or prevent
# an independent arm's ABI smoke. This does not certify physical fidelity,
# guidance, solver performance, or isolated formal LLM-agent validation.
for VPFP_ARM in baseline sobolev; do
  VPFP_CHECKPOINT="${VPFP_RUN_ROOT}/${VPFP_ARM}/checkpoints/best.pt"
  VPFP_SMOKE_EXIT=checkpoint_missing
  if [[ -f "${VPFP_CHECKPOINT}" ]]; then
    if VPFP_SHA_LINE="$(sha256sum -- "${VPFP_CHECKPOINT}")"; then
      VPFP_SHA="${VPFP_SHA_LINE%% *}"
      VPFP_SMOKE_EXIT=0
      "${VPFP_PYTHON}" scripts/run_plasma_canonical_smoke.py \
        --physics vpfp --config "${VPFP_CONFIG}" \
        --diffusion-checkpoint "${VPFP_CHECKPOINT}" --diffusion-checkpoint-sha256 "${VPFP_SHA}" \
        --output "${VPFP_RUN_ROOT}/${VPFP_ARM}/canonical-campaign-smoke" --device cuda \
        > "${VPFP_RUN_ROOT}/${VPFP_ARM}/canonical-campaign-smoke.log" 2>&1 \
        || VPFP_SMOKE_EXIT=$?
    else
      VPFP_SMOKE_EXIT=$?
      echo "Cannot hash ${VPFP_ARM} checkpoint; smoke skipped." >&2
    fi
  else
    echo "No ${VPFP_ARM} best checkpoint; independent smoke skipped." >&2
  fi
  if [[ "${VPFP_ARM}" == baseline ]]; then
    VPFP_BASELINE_SMOKE_EXIT="${VPFP_SMOKE_EXIT}"
  else
    VPFP_SOBOLEV_SMOKE_EXIT="${VPFP_SMOKE_EXIT}"
  fi
  if [[ "${VPFP_SMOKE_EXIT}" != 0 && "${VPFP_EXIT}" == 0 ]]; then
    if [[ "${VPFP_SMOKE_EXIT}" =~ ^[0-9]+$ ]]; then
      VPFP_EXIT="${VPFP_SMOKE_EXIT}"
    else
      VPFP_EXIT=1
    fi
  fi
  echo "VPFP ${VPFP_REGIME} ${VPFP_ARM} canonical smoke exit: ${VPFP_SMOKE_EXIT}"
done

# The CLI itself requires a nonexistent output, so this small sibling receipt
# also survives failures that happen before it creates the run directory.
printf 'stage\texit_code\nablation\t%s\nbaseline_smoke\t%s\nsobolev_smoke\t%s\nlaunch\t%s\n' \
  "${VPFP_ABLATION_EXIT}" "${VPFP_BASELINE_SMOKE_EXIT}" "${VPFP_SOBOLEV_SMOKE_EXIT}" "${VPFP_EXIT}" \
  > "${VPFP_LAUNCH_STATUS}"
echo "VPFP Sobolev artifacts: ${VPFP_RUN_ROOT}; launch status: ${VPFP_LAUNCH_STATUS}; exit: ${VPFP_EXIT}."
echo "No automatic scientific admission. Previous corpus and jobs were not modified."
exit "${VPFP_EXIT}"
