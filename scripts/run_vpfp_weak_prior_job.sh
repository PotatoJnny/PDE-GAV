#!/usr/bin/env bash
# Research-only allocation; no scheduler mutation or deployed-asset replacement.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/run_vpfp_weak_prior_job.sh noise|structure" >&2
  exit 2
fi
VPFP_EXPERIMENT="$1"
VPFP_SEED=1320260905
VPFP_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-scaleup-weak-5241246
VPFP_ANCHORS=${PDE_GAV_RUN_ROOT}/vpfp-weak-regime-debug-5244224
case "${VPFP_EXPERIMENT}" in
  noise)
    VPFP_DRIVER=scripts/run_vpfp_weak_noise_factorial.py
    VPFP_RUN_LABEL=vpfp-weak-noise-factorial
    VPFP_ARMS=(cov_energy_iso cov_unit_shape)
    ;;
  structure)
    VPFP_DRIVER=scripts/run_vpfp_weak_structure_prior.py
    VPFP_RUN_LABEL=vpfp-weak-structure-prior
    VPFP_ARMS=(cov_shrink_half branch_identity_control lowmode_coherent)
    ;;
  *) echo "Unknown weak pilot: ${VPFP_EXPERIMENT}" >&2; exit 2 ;;
esac
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "VPFP regime experiments require a Slurm compute allocation and local scratch." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-weak-noise-device-fix
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_CONFIG="${VPFP_CHECKOUT}/configs/vpfp_unified_live_weak_collision.json"
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
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-weak-prior-pycache"
VPFP_ARGS=(--config "${VPFP_CONFIG}" --output "${VPFP_RUN_ROOT}" --regime weak
  --seed "${VPFP_SEED}" --replicas 2 --batch-size 4 --generated-answers 128 --sample-steps 64 --device cuda
  --source "${VPFP_SOURCE}" --train-seconds 1800 --checkpoint-seconds 360 900 --steps 100000)
if [[ "${VPFP_EXPERIMENT}" == noise ]]; then
  if [[ ! -f "${VPFP_ANCHORS}/result.json" || ! -f "${VPFP_ANCHORS}/frozen-checkpoints.json" ]]; then
    echo "Missing completed immutable weak noise anchors." >&2
    exit 2
  fi
  VPFP_ARGS+=(--anchors "${VPFP_ANCHORS}")
fi
echo "VPFP weak ${VPFP_EXPERIMENT}: revision=${VPFP_REVISION}; output=${VPFP_RUN_ROOT}"
echo "Same 512 train/128 validation; two seeds x ${#VPFP_ARMS[@]} new arms x 1800 seconds."
echo "Opened development references; no new PDE solves, blind test, or automatic admission."
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
