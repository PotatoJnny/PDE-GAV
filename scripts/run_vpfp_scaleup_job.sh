#!/usr/bin/env bash
# One bounded research allocation; never submits, cancels or replaces jobs.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/run_vpfp_scaleup_job.sh weak|strong" >&2
  exit 2
fi
VPFP_REGIME="$1"
case "${VPFP_REGIME}" in
  weak)
    VPFP_SEED=920260905
    VPFP_CACHE=${PDE_GAV_RUN_ROOT}/vpfp-sobolev-matching-weak-5232589
    VPFP_PILOT=${PDE_GAV_RUN_ROOT}/vpfp-temporal-training-weak-5240924
    ;;
  strong)
    VPFP_SEED=1020260905
    VPFP_CACHE=${PDE_GAV_RUN_ROOT}/vpfp-sobolev-matching-strong-5232590
    VPFP_PILOT=${PDE_GAV_RUN_ROOT}/vpfp-temporal-training-strong-5240925
    ;;
  *) echo "Unknown VPFP regime: ${VPFP_REGIME}" >&2; exit 2 ;;
esac
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "VPFP scale-up requires a Slurm compute allocation and local scratch." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-scaleup-experiment
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_CONFIG="${VPFP_CHECKOUT}/configs/vpfp_unified_live_${VPFP_REGIME}_collision.json"
VPFP_SELECTION="${VPFP_CHECKOUT}/configs/vpfp_scaleup_selection_${VPFP_REGIME}.json"
VPFP_RUN_ROOT="${PDE_GAV_RUN_ROOT}/vpfp-scaleup-${VPFP_REGIME}-${SLURM_JOB_ID}"
VPFP_RECEIPT="${VPFP_RUN_ROOT}.launch-exit-codes.tsv"
VPFP_GPU_LOG="${VPFP_RUN_ROOT}.gpu-preflight.log"
cd "${VPFP_CHECKOUT}"
for VPFP_REQUIRED in "${VPFP_CONFIG}" "${VPFP_SELECTION}" "${VPFP_CACHE}/corpus/manifest.json" \
    "${VPFP_PILOT}/result.json" scripts/run_vpfp_scaleup_experiment.py scripts/run_plasma_canonical_smoke.py; do
  if [[ ! -f "${VPFP_REQUIRED}" ]]; then
    echo "Missing frozen configuration, completed pilot or reviewed selection: ${VPFP_REQUIRED}" >&2
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
  echo "Commit/freeze relevant source and reviewed candidate selection before launching." >&2
  echo "${VPFP_SOURCE_STATUS}" >&2
  exit 2
fi
VPFP_CANDIDATE="$("${VPFP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"])' "${VPFP_SELECTION}")"
case "${VPFP_CANDIDATE}" in
  value_low|cov_base|cov_low) ;;
  *) echo "Invalid reviewed VPFP candidate: ${VPFP_CANDIDATE}" >&2; exit 2 ;;
esac
VPFP_REVISION="$(git rev-parse HEAD)"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing explicit venv." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-scaleup-pycache"
echo "VPFP ${VPFP_REGIME}: revision=${VPFP_REVISION}; candidate=${VPFP_CANDIDATE}; output=${VPFP_RUN_ROOT}"
echo "Same 512 train/128 validation; fresh 384 evaluation answers; two seeds x two arms x 1800 seconds."
echo "EMA learning-curve snapshots at 360/900 seconds; no checkpoint selection by physical audit."
VPFP_GPU_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "No usable CUDA device"; print({"cuda_available": True, "device": torch.cuda.get_device_name(0)})' \
  > "${VPFP_GPU_LOG}" 2>&1 || VPFP_GPU_EXIT=$?
VPFP_EXPERIMENT_EXIT=not_run
VPFP_EXIT="${VPFP_GPU_EXIT}"
if [[ "${VPFP_GPU_EXIT}" == 0 ]]; then
  VPFP_EXPERIMENT_EXIT=0
  "${VPFP_PYTHON}" scripts/run_vpfp_scaleup_experiment.py \
    --config "${VPFP_CONFIG}" --cache "${VPFP_CACHE}" --pilot "${VPFP_PILOT}" --selection "${VPFP_SELECTION}" \
    --output "${VPFP_RUN_ROOT}" --regime "${VPFP_REGIME}" --candidate "${VPFP_CANDIDATE}" --seed "${VPFP_SEED}" \
    --replicas 2 --train-seconds 1800 --steps 100000 --batch-size 4 --collect-batch-size 8 \
    --checkpoint-seconds 360 900 --generated-answers 128 --sample-steps 64 --device cuda \
    || VPFP_EXPERIMENT_EXIT=$?
  VPFP_EXIT="${VPFP_EXPERIMENT_EXIT}"
fi
declare -A VPFP_SMOKE_EXITS
for VPFP_ARM in value_base "${VPFP_CANDIDATE}"; do
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
printf 'stage\texit_code\ngpu_preflight\t%s\nexperiment\t%s\nbaseline_smoke\t%s\ncandidate_smoke\t%s\nlaunch\t%s\n' \
  "${VPFP_GPU_EXIT}" "${VPFP_EXPERIMENT_EXIT}" "${VPFP_SMOKE_EXITS[value_base]}" \
  "${VPFP_SMOKE_EXITS[${VPFP_CANDIDATE}]}" "${VPFP_EXIT}" > "${VPFP_RECEIPT}"
echo "VPFP artifacts: ${VPFP_RUN_ROOT}; launch receipt: ${VPFP_RECEIPT}; exit=${VPFP_EXIT}."
echo "Research learning curve only; no deployed model replacement or scientific admission."
exit "${VPFP_EXIT}"
