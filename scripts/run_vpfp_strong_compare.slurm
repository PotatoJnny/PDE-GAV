#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=vpfp_strong_compare
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Explicit manual submission only. The parent campaign reserves eight of its
# 48 GPU hours and controls the global two-GPU concurrency limit.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires a Slurm compute allocation, local scratch and no positional arguments." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-moving-frame-campaign
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-strong-main-5244223
VPFP_PILOT=${PDE_GAV_RUN_ROOT}/vpfp-strong-initial-5259887
VPFP_PREFLIGHT=${PDE_GAV_RUN_ROOT}/vpfp-regime-coherent-preflight-5267643/strong-preflight/result.json
VPFP_PREFLIGHT_SHA=5f76d5781691aa40fdf8c6b1a43b3b5c221821cab09fa459ec7caef9be09110e
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-strong-regime-coherent-${SLURM_JOB_ID}"
VPFP_RECEIPT="${VPFP_OUTPUT}.launch-exit-codes.tsv"
VPFP_GPU_LOG="${VPFP_OUTPUT}.gpu-preflight.log"
VPFP_RUN_LOG="${VPFP_OUTPUT}.experiment.log"
VPFP_TEST_LOG="${VPFP_OUTPUT}.mechanics-tests.log"
for VPFP_PATH in "${VPFP_OUTPUT}" "${VPFP_RECEIPT}" "${VPFP_GPU_LOG}" "${VPFP_RUN_LOG}" "${VPFP_TEST_LOG}"; do
  if [[ -e "${VPFP_PATH}" || -L "${VPFP_PATH}" ]]; then
    echo "Refusing to overwrite ${VPFP_PATH}" >&2
    exit 2
  fi
done
VPFP_SOURCE_EXIT=not_run VPFP_BINDING_EXIT=not_run VPFP_GPU_EXIT=not_run VPFP_RUN_EXIT=not_run
VPFP_TEST_EXIT=not_run
VPFP_REVISION=not_verified
vpfp_finish() {
  local status=$?
  trap - EXIT
  printf 'stage\texit_code\nsource_freeze\t%s\nstrong_preflight_binding\t%s\ngpu_preflight\t%s\nmechanics_tests\t%s\nstrong_comparisons\t%s\nlaunch\t%s\nsource_revision\t%s\n' \
    "${VPFP_SOURCE_EXIT}" "${VPFP_BINDING_EXIT}" "${VPFP_GPU_EXIT}" "${VPFP_TEST_EXIT}" "${VPFP_RUN_EXIT}" "${status}" "${VPFP_REVISION}" > "${VPFP_RECEIPT}"
  echo "Strong comparison exit=${status}; retained artifacts: ${VPFP_OUTPUT}. No automatic followup/admission."
  exit "${status}"
}
trap vpfp_finish EXIT
cd "${VPFP_CHECKOUT}"
VPFP_SOURCE_EXIT=2
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then echo "A formal Git commit is required." >&2; exit 2; fi
git ls-files --error-unmatch -- scripts/run_vpfp_strong_compare.slurm scripts/run_vpfp_strong_regime_coherent.py >/dev/null
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" ]]; then
  echo "Commit/freeze all relevant source and this wrapper before manual launch." >&2
  exit 2
fi
for VPFP_PATH in "${VPFP_CHECKOUT}/configs/vpfp_unified_live_strong_collision.json" \
    "${VPFP_SOURCE}/corpus/manifest.json" "${VPFP_SOURCE}/result.json" "${VPFP_SOURCE}/frozen-reference-features.json" \
    "${VPFP_PILOT}/frozen-plan.json" "${VPFP_PILOT}/frozen-checkpoints.json" "${VPFP_PILOT}/result.json"; do
  if [[ ! -f "${VPFP_PATH}" || -L "${VPFP_PATH}" ]]; then
    echo "Missing/nonregular frozen input: ${VPFP_PATH}" >&2
    exit 2
  fi
done
if [[ ! -x "${VPFP_PYTHON}" ]]; then echo "Explicit venv is unavailable." >&2; exit 2; fi
VPFP_SOURCE_EXIT=0
VPFP_BINDING_EXIT=2
if [[ ! -f "${VPFP_PREFLIGHT}" || -L "${VPFP_PREFLIGHT}" ]]; then
  echo "Successful strong-only preflight receipt is missing/nonregular." >&2
  exit 2
fi
if VPFP_SHA_LINE="$(sha256sum -- "${VPFP_PREFLIGHT}")"; then
  if [[ "${VPFP_SHA_LINE%% *}" != "${VPFP_PREFLIGHT_SHA}" ]]; then
    echo "Strong-only preflight SHA changed." >&2
    exit 2
  fi
else
  VPFP_BINDING_EXIT=$?
  exit "${VPFP_BINDING_EXIT}"
fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the explicit existing venv without installing packages." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-strong-compare-pycache"
# Bind only the completed strong substage; the independent moving-frame stage
# is deliberately neither required nor reported as passed by this wrapper.
"${VPFP_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); r=p["preflight"]["initial_conditioned_rho_cov"]; assert p["execution_passed"] is True and p["preflight_only"] is True and p["training"] == {} and p["audits"] == {}; assert r["execution_passed"] is True and r["normalization_gate"]["passed"] is True' "${VPFP_PREFLIGHT}" || { VPFP_BINDING_EXIT=$?; exit "${VPFP_BINDING_EXIT}"; }
VPFP_BINDING_EXIT=0
VPFP_GPU_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))' \
  > "${VPFP_GPU_LOG}" 2>&1 || VPFP_GPU_EXIT=$?
if [[ "${VPFP_GPU_EXIT}" != 0 ]]; then exit "${VPFP_GPU_EXIT}"; fi
VPFP_TEST_EXIT=0
timeout --signal=TERM --kill-after=30s 900 "${VPFP_PYTHON}" -m pytest -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-strong-compare-tests" \
  tests/test_vpfp_training_resume.py tests/test_vpfp_conditional_rho_covariance_gate.py \
  tests/test_vpfp_numerical_roundtrip.py tests/test_vpfp_strong_regime_coherent.py \
  > "${VPFP_TEST_LOG}" 2>&1 || VPFP_TEST_EXIT=$?
if [[ "${VPFP_TEST_EXIT}" != 0 ]]; then exit "${VPFP_TEST_EXIT}"; fi
VPFP_RUN_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_strong_regime_coherent.py \
  --config "${VPFP_CHECKOUT}/configs/vpfp_unified_live_strong_collision.json" \
  --source "${VPFP_SOURCE}" --pilot "${VPFP_PILOT}" --output "${VPFP_OUTPUT}" --device cuda \
  > "${VPFP_RUN_LOG}" 2>&1 || VPFP_RUN_EXIT=$?
# The driver fixes six training runs,14 native states and128 attempts/state;
# the eight-hour allocation never silently changes counts or follows up.
exit "${VPFP_RUN_EXIT}"
