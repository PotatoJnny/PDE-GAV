#!/usr/bin/env bash
# Research-only allocation; no scheduler mutation or deployed-asset replacement.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/run_vpfp_regime_job.sh strong|weak" >&2
  exit 2
fi
VPFP_REGIME="$1"
case "${VPFP_REGIME}" in
  strong)
    VPFP_SEED=1120260905
    VPFP_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-scaleup-strong-5241247
    VPFP_DRIVER=scripts/run_vpfp_strong_main_experiment.py
    VPFP_RUN_LABEL=vpfp-strong-main
    VPFP_ARMS=(value_base cov_base)
    ;;
  weak)
    VPFP_SEED=1320260905
    VPFP_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-scaleup-weak-5241246
    VPFP_DRIVER=scripts/run_vpfp_weak_regime_debug.py
    VPFP_RUN_LABEL=vpfp-weak-regime-debug
    VPFP_ARMS=(value_base cov_centered_iso cov_shrink_half cov_base)
    ;;
  *) echo "Unknown VPFP regime: ${VPFP_REGIME}" >&2; exit 2 ;;
esac
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "VPFP regime experiments require a Slurm compute allocation and local scratch." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-regime-main-debug
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_CONFIG="${VPFP_CHECKOUT}/configs/vpfp_unified_live_${VPFP_REGIME}_collision.json"
VPFP_RUN_ROOT="${PDE_GAV_RUN_ROOT}/${VPFP_RUN_LABEL}-${SLURM_JOB_ID}"
VPFP_RECEIPT="${VPFP_RUN_ROOT}.launch-exit-codes.tsv"
VPFP_GPU_LOG="${VPFP_RUN_ROOT}.gpu-preflight.log"
cd "${VPFP_CHECKOUT}"
for VPFP_REQUIRED in "${VPFP_CONFIG}" "${VPFP_SOURCE}/corpus/manifest.json" \
    "${VPFP_SOURCE}/result.json" "${VPFP_DRIVER}" scripts/run_plasma_canonical_smoke.py; do
  if [[ ! -f "${VPFP_REQUIRED}" ]]; then
    echo "Missing frozen configuration, source experiment or entry point: ${VPFP_REQUIRED}" >&2
    exit 2
  fi
done
if [[ ! -x "${VPFP_PYTHON}" ]]; then
  echo "Existing explicit Python runtime unavailable." >&2
  exit 2
fi
for VPFP_ARTIFACT in "${VPFP_RUN_ROOT}" "${VPFP_RECEIPT}" "${VPFP_GPU_LOG}"; do
  if [[ -e "${VPFP_ARTIFACT}" || -L "${VPFP_ARTIFACT}" ]]; then
    echo "Refusing to overwrite experiment artifact: ${VPFP_ARTIFACT}" >&2
    exit 2
  fi
done
VPFP_SOURCE_STATUS="$(git status --porcelain --untracked-files=all -- src scripts configs legacy/vpfp)"
if [[ -n "${VPFP_SOURCE_STATUS}" ]]; then
  echo "Commit/freeze relevant source before launching." >&2
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
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-regime-pycache"
VPFP_ARGS=(--config "${VPFP_CONFIG}" --output "${VPFP_RUN_ROOT}" --regime "${VPFP_REGIME}"
  --seed "${VPFP_SEED}" --replicas 2 --batch-size 4 --generated-answers 128 --sample-steps 64 --device cuda)
if [[ "${VPFP_REGIME}" == strong ]]; then
  VPFP_ARGS+=(--source-corpus "${VPFP_SOURCE}/corpus" --train-answers 2048 --heldout-answers 128
    --train-seconds 7200 --checkpoint-seconds 1800 3600 --steps 1000000 --collect-batch-size 8)
else
  VPFP_ARGS+=(--source "${VPFP_SOURCE}" --train-seconds 1800 --checkpoint-seconds 360 900 --steps 100000)
fi
echo "VPFP ${VPFP_REGIME}: revision=${VPFP_REVISION}; output=${VPFP_RUN_ROOT}"
if [[ "${VPFP_REGIME}" == strong ]]; then
  echo "Independent 2048 train + four 128 heldout splits; two seeds x two arms x 7200 seconds."
  echo "All five splits newly solved; 1800/3600-second snapshots; physical audit never selects best."
else
  echo "Same 512 train/128 validation; two seeds x four arms x 1800 seconds."
  echo "Uncentered anchor and three fixed-center covariance strengths; opened development references."
fi
VPFP_GPU_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "No usable CUDA device"; print({"cuda_available": True, "device": torch.cuda.get_device_name(0)})' \
  > "${VPFP_GPU_LOG}" 2>&1 || VPFP_GPU_EXIT=$?
VPFP_EXPERIMENT_EXIT=not_run
VPFP_EXIT="${VPFP_GPU_EXIT}"
if [[ "${VPFP_GPU_EXIT}" == 0 ]]; then
  VPFP_EXPERIMENT_EXIT=0
  "${VPFP_PYTHON}" "${VPFP_DRIVER}" "${VPFP_ARGS[@]}" || VPFP_EXPERIMENT_EXIT=$?
  VPFP_EXIT="${VPFP_EXPERIMENT_EXIT}"
fi
declare -A VPFP_SMOKE_EXITS
for VPFP_ARM in "${VPFP_ARMS[@]}"; do
  VPFP_ARM_ROOT="${VPFP_RUN_ROOT}/seed-0/${VPFP_ARM}"
  VPFP_CHECKPOINT="${VPFP_ARM_ROOT}/checkpoints/best.pt"
  VPFP_SMOKE_EXIT=checkpoint_missing
  if [[ "${VPFP_GPU_EXIT}" != 0 ]]; then
    VPFP_SMOKE_EXIT=gpu_preflight_failed
  elif [[ -f "${VPFP_CHECKPOINT}" ]]; then
    if VPFP_SHA_LINE="$(sha256sum -- "${VPFP_CHECKPOINT}")"; then
      VPFP_SHA="${VPFP_SHA_LINE%% *}"
      VPFP_SMOKE_EXIT=0
      "${VPFP_PYTHON}" scripts/run_plasma_canonical_smoke.py \
        --physics vpfp --config "${VPFP_CONFIG}" --diffusion-checkpoint "${VPFP_CHECKPOINT}" \
        --diffusion-checkpoint-sha256 "${VPFP_SHA}" --output "${VPFP_ARM_ROOT}/canonical-campaign-smoke" --device cuda \
        > "${VPFP_ARM_ROOT}/canonical-campaign-smoke.log" 2>&1 || VPFP_SMOKE_EXIT=$?
    else
      VPFP_SMOKE_EXIT=$?
      echo "Cannot hash seed-0/${VPFP_ARM}; smoke skipped with failure preserved." >&2
    fi
  fi
  VPFP_SMOKE_EXITS["${VPFP_ARM}"]="${VPFP_SMOKE_EXIT}"
  if [[ "${VPFP_SMOKE_EXIT}" != 0 && "${VPFP_EXIT}" == 0 ]]; then
    VPFP_EXIT=1
  fi
done
{
  printf 'stage\texit_code\ngpu_preflight\t%s\nexperiment\t%s\n' "${VPFP_GPU_EXIT}" "${VPFP_EXPERIMENT_EXIT}"
  for VPFP_ARM in "${VPFP_ARMS[@]}"; do
    printf 'smoke_%s\t%s\n' "${VPFP_ARM}" "${VPFP_SMOKE_EXITS[${VPFP_ARM}]}"
  done
  printf 'launch\t%s\n' "${VPFP_EXIT}"
} > "${VPFP_RECEIPT}"
echo "VPFP artifacts: ${VPFP_RUN_ROOT}; launch receipt: ${VPFP_RECEIPT}; exit=${VPFP_EXIT}."
echo "Research experiment only; no automatic followup job, model replacement or scientific admission."
exit "${VPFP_EXIT}"
