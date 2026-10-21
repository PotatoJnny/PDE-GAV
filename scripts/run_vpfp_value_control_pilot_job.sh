#!/usr/bin/env bash
# A bounded compute-allocation pilot; never submits or modifies Slurm jobs.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/run_vpfp_value_control_pilot_job.sh weak|strong" >&2
  exit 2
fi
VPFP_REGIME="$1"
case "${VPFP_REGIME}" in
  weak)
    VPFP_SEED=520260905
    VPFP_CACHE=${PDE_GAV_RUN_ROOT}/vpfp-sobolev-matching-weak-5232589
    ;;
  strong)
    VPFP_SEED=620260905
    VPFP_CACHE=${PDE_GAV_RUN_ROOT}/vpfp-sobolev-matching-strong-5232590
    ;;
  *) echo "Unknown VPFP regime: ${VPFP_REGIME}" >&2; exit 2 ;;
esac
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "VPFP value-control pilot requires a Slurm compute allocation and local scratch." >&2
  exit 2
fi

VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-value-controls-pilot
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_RUN_ROOT="${PDE_GAV_RUN_ROOT}/vpfp-value-control-pilot-${VPFP_REGIME}-${SLURM_JOB_ID}"
VPFP_CONFIG="${VPFP_CHECKOUT}/configs/vpfp_unified_live_${VPFP_REGIME}_collision.json"
VPFP_LAUNCH_STATUS="${VPFP_RUN_ROOT}.launch-exit-codes.tsv"
VPFP_PREFLIGHT_LOG="${VPFP_RUN_ROOT}.gpu-preflight.log"
cd "${VPFP_CHECKOUT}"
if [[ ! -x "${VPFP_PYTHON}" || ! -r "${VPFP_CONFIG}" || ! -f "${VPFP_CACHE}/corpus/manifest.json" \
      || ! -f scripts/run_vpfp_value_control_pilot.py || ! -f scripts/run_plasma_canonical_smoke.py ]]; then
  echo "Missing frozen historical cache, configuration, runtime, or launch program." >&2
  exit 2
fi
for VPFP_ARTIFACT in "${VPFP_RUN_ROOT}" "${VPFP_LAUNCH_STATUS}" "${VPFP_PREFLIGHT_LOG}"; do
  if [[ -e "${VPFP_ARTIFACT}" || -L "${VPFP_ARTIFACT}" ]]; then
    echo "Refusing to overwrite an existing pilot artifact: ${VPFP_ARTIFACT}" >&2
    exit 2
  fi
done
# Git diff alone misses untracked executable source. Historical caches remain
# read-only and are additionally hash-bound by the Python pilot.
VPFP_SOURCE_STATUS="$(git status --porcelain --untracked-files=all -- src scripts configs legacy/vpfp)"
if [[ -n "${VPFP_SOURCE_STATUS}" ]]; then
  echo "Commit/freeze relevant source before this reproducible direction pilot:" >&2
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
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-value-control-pycache"

echo "VPFP value-control ${VPFP_REGIME}: revision=${VPFP_REVISION}; cache=${VPFP_CACHE}; output=${VPFP_RUN_ROOT}"
echo "Two seeds x four arms x 240 training seconds; 64 generated answers per arm; no online PDE collection."
VPFP_EXIT=0
VPFP_PREFLIGHT_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "Slurm allocation has no usable CUDA device"; print({"cuda_available": True, "device": torch.cuda.get_device_name(0)})' \
  > "${VPFP_PREFLIGHT_LOG}" 2>&1 || VPFP_PREFLIGHT_EXIT=$?
VPFP_PILOT_EXIT=not_run
if [[ "${VPFP_PREFLIGHT_EXIT}" == 0 ]]; then
  VPFP_PILOT_EXIT=0
  "${VPFP_PYTHON}" scripts/run_vpfp_value_control_pilot.py \
    --config "${VPFP_CONFIG}" --cache "${VPFP_CACHE}" --output "${VPFP_RUN_ROOT}" \
    --regime "${VPFP_REGIME}" --seed "${VPFP_SEED}" --replicas 2 \
    --train-seconds 240 --steps 100000 --batch-size 4 --generated-answers 64 \
    --temporal-weight .25 --sample-steps 64 --device cuda \
    || VPFP_PILOT_EXIT=$?
  VPFP_EXIT="${VPFP_PILOT_EXIT}"
else
  VPFP_EXIT="${VPFP_PREFLIGHT_EXIT}"
  echo "CUDA preflight failed; no training or smoke attempted. See ${VPFP_PREFLIGHT_LOG}." >&2
fi

declare -A VPFP_SMOKE_EXITS
# Freeze/training/audit failures never hide an existing arm checkpoint. Only
# seed 0 needs the four independent canonical macro-only ABI smoke checks.
# These checks do not qualify physics, guidance or distributional fidelity.
for VPFP_ARM in full_base full_time value_base value_time; do
  VPFP_ARM_ROOT="${VPFP_RUN_ROOT}/seed-0/${VPFP_ARM}"
  VPFP_CHECKPOINT="${VPFP_ARM_ROOT}/checkpoints/best.pt"
  VPFP_SMOKE_EXIT=checkpoint_missing
  if [[ "${VPFP_PREFLIGHT_EXIT}" != 0 ]]; then
    VPFP_SMOKE_EXIT=gpu_preflight_failed
  elif [[ -f "${VPFP_CHECKPOINT}" ]]; then
    if VPFP_SHA_LINE="$(sha256sum -- "${VPFP_CHECKPOINT}")"; then
      VPFP_SHA="${VPFP_SHA_LINE%% *}"
      VPFP_SMOKE_EXIT=0
      "${VPFP_PYTHON}" scripts/run_plasma_canonical_smoke.py \
        --physics vpfp --config "${VPFP_CONFIG}" \
        --diffusion-checkpoint "${VPFP_CHECKPOINT}" --diffusion-checkpoint-sha256 "${VPFP_SHA}" \
        --output "${VPFP_ARM_ROOT}/canonical-campaign-smoke" --device cuda \
        > "${VPFP_ARM_ROOT}/canonical-campaign-smoke.log" 2>&1 \
        || VPFP_SMOKE_EXIT=$?
    else
      VPFP_SMOKE_EXIT=$?
      echo "Cannot hash seed-0/${VPFP_ARM} checkpoint; smoke skipped." >&2
    fi
  else
    echo "No seed-0/${VPFP_ARM} best checkpoint; independent smoke skipped." >&2
  fi
  VPFP_SMOKE_EXITS["${VPFP_ARM}"]="${VPFP_SMOKE_EXIT}"
  if [[ "${VPFP_SMOKE_EXIT}" != 0 && "${VPFP_EXIT}" == 0 ]]; then
    if [[ "${VPFP_SMOKE_EXIT}" =~ ^[0-9]+$ ]]; then
      VPFP_EXIT="${VPFP_SMOKE_EXIT}"
    else
      VPFP_EXIT=1
    fi
  fi
  echo "VPFP ${VPFP_REGIME} seed-0/${VPFP_ARM} canonical smoke exit: ${VPFP_SMOKE_EXIT}"
done

# Keep a sibling receipt even if the pilot failed before making its fresh
# output. No failed or skipped stage is deleted or silently marked successful.
printf 'stage\texit_code\ngpu_preflight\t%s\npilot\t%s\nfull_base_smoke\t%s\nfull_time_smoke\t%s\nvalue_base_smoke\t%s\nvalue_time_smoke\t%s\nlaunch\t%s\n' \
  "${VPFP_PREFLIGHT_EXIT}" "${VPFP_PILOT_EXIT}" \
  "${VPFP_SMOKE_EXITS[full_base]}" "${VPFP_SMOKE_EXITS[full_time]}" \
  "${VPFP_SMOKE_EXITS[value_base]}" "${VPFP_SMOKE_EXITS[value_time]}" "${VPFP_EXIT}" \
  > "${VPFP_LAUNCH_STATUS}"
echo "VPFP pilot artifacts: ${VPFP_RUN_ROOT}; launch status: ${VPFP_LAUNCH_STATUS}; exit: ${VPFP_EXIT}."
echo "Development screen only: no automatic large run or scientific admission. Historical runs were not modified."
exit "${VPFP_EXIT}"
