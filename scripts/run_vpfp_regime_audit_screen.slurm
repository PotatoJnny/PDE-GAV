#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --job-name=vpfp_regime_audit_screen
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Manual-only stage. Default to the exact reviewed, completed mechanics receipt
# issued by the parent. Explicit frozen path/SHA overrides are permitted; never
# discover a latest result or bind an actively changing overall preflight file.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires a Slurm compute allocation, local scratch and no positional arguments." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-moving-frame-campaign
VPFP_AUDIT_MECHANICS_RESULT="${VPFP_AUDIT_MECHANICS_RESULT:-${VPFP_CHECKOUT}/configs/vpfp_regime_mechanics_receipt_5267643.json}"
VPFP_AUDIT_MECHANICS_SHA256="${VPFP_AUDIT_MECHANICS_SHA256:-0800081e703fef70bbb5eb4078854a39dca9b390e9b4c6591f43703fe8a31399}"
if [[ ! "${VPFP_AUDIT_MECHANICS_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Mechanics binding requires a reviewed receipt path and valid SHA256; no automatic selection." >&2
  exit 2
fi
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_STRONG_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-strong-main-5244223
VPFP_WEAK_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-scaleup-weak-5241246
VPFP_STRONG_RUN=${PDE_GAV_RUN_ROOT}/vpfp-strong-initial-5259887
VPFP_WEAK_RUN=${PDE_GAV_RUN_ROOT}/vpfp-hermite-joint-recovery-5258664
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-regime-audit-screen-${SLURM_JOB_ID}"
VPFP_RECEIPT="${VPFP_OUTPUT}.launch-exit-codes.tsv"
VPFP_GPU_LOG="${VPFP_OUTPUT}.gpu-preflight.log"
VPFP_STRONG_LOG="${VPFP_OUTPUT}.strong.log"
VPFP_WEAK_LOG="${VPFP_OUTPUT}.weak.log"
VPFP_TEST_LOG="${VPFP_OUTPUT}.mechanics-tests.log"
for VPFP_PATH in "${VPFP_OUTPUT}" "${VPFP_RECEIPT}" "${VPFP_GPU_LOG}" "${VPFP_STRONG_LOG}" "${VPFP_WEAK_LOG}" "${VPFP_TEST_LOG}"; do
  if [[ -e "${VPFP_PATH}" || -L "${VPFP_PATH}" ]]; then
    echo "Refusing to overwrite ${VPFP_PATH}" >&2
    exit 2
  fi
done
VPFP_SOURCE_EXIT=not_run VPFP_BINDING_EXIT=not_run VPFP_GPU_EXIT=not_run
VPFP_TEST_EXIT=not_run
VPFP_STRONG_EXIT=not_run VPFP_WEAK_EXIT=not_run VPFP_REVISION=not_verified
vpfp_finish() {
  local status=$?
  trap - EXIT
  printf 'stage\texit_code\nsource_freeze\t%s\nmechanics_binding\t%s\ngpu_preflight\t%s\nmechanics_tests\t%s\nstrong_audit\t%s\nweak_audit\t%s\nlaunch\t%s\nsource_revision\t%s\nmechanics_result_sha256\t%s\n' \
    "${VPFP_SOURCE_EXIT}" "${VPFP_BINDING_EXIT}" "${VPFP_GPU_EXIT}" "${VPFP_TEST_EXIT}" "${VPFP_STRONG_EXIT}" "${VPFP_WEAK_EXIT}" "${status}" "${VPFP_REVISION}" "${VPFP_AUDIT_MECHANICS_SHA256}" > "${VPFP_RECEIPT}"
  echo "Audit screen exit=${status}; retained artifacts: ${VPFP_OUTPUT}. Descriptive development audit, not admission."
  exit "${status}"
}
trap vpfp_finish EXIT
cd "${VPFP_CHECKOUT}"
VPFP_SOURCE_EXIT=2
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then echo "A formal Git commit is required." >&2; exit 2; fi
git ls-files --error-unmatch -- scripts/run_vpfp_regime_audit_screen.slurm scripts/run_vpfp_regime_fidelity_coverage.py >/dev/null
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" ]]; then
  echo "Commit/freeze all relevant source and this wrapper before manual launch." >&2
  exit 2
fi
if [[ ! -x "${VPFP_PYTHON}" ]]; then echo "Explicit venv is unavailable." >&2; exit 2; fi
# Fixed target names, verified against the completed generation receipts.
# No checkpoint search, feature-based selection or fallback to another model.
VPFP_STRONG_DIRS=(
  "${VPFP_STRONG_RUN}/seed-0/cov_base/evaluation"
  "${VPFP_STRONG_RUN}/seed-0/initial_conditioned/evaluation"
  "${VPFP_STRONG_RUN}/seed-1/cov_base/evaluation"
  "${VPFP_STRONG_RUN}/seed-1/initial_conditioned/evaluation"
)
VPFP_WEAK_DIRS=(
  "${VPFP_WEAK_RUN}/seed-0/hermite_permuted_control/evaluation/best"
  "${VPFP_WEAK_RUN}/seed-0/hermite_joint/evaluation/best"
  "${VPFP_WEAK_RUN}/seed-1/hermite_permuted_control/evaluation/best"
  "${VPFP_WEAK_RUN}/seed-1/hermite_joint/evaluation/best"
)
for VPFP_PATH in "${VPFP_STRONG_SOURCE}/corpus/manifest.json" "${VPFP_WEAK_SOURCE}/corpus/manifest.json" \
    "${VPFP_STRONG_RUN}/result.json" "${VPFP_WEAK_RUN}/result.json"; do
  if [[ ! -f "${VPFP_PATH}" || -L "${VPFP_PATH}" ]]; then echo "Missing/nonregular source ${VPFP_PATH}" >&2; exit 2; fi
done
for VPFP_DIR in "${VPFP_STRONG_DIRS[@]}" "${VPFP_WEAK_DIRS[@]}"; do
  if [[ ! -d "${VPFP_DIR}" || -L "${VPFP_DIR}" ]]; then echo "Missing/nonregular target ${VPFP_DIR}" >&2; exit 2; fi
  for VPFP_FILE in generated-coefficients.npy generated-conditions.npy generated-provenance.json; do
    if [[ ! -f "${VPFP_DIR}/${VPFP_FILE}" || -L "${VPFP_DIR}/${VPFP_FILE}" ]]; then
      echo "Missing/nonregular frozen target input ${VPFP_DIR}/${VPFP_FILE}" >&2
      exit 2
    fi
  done
done
VPFP_SOURCE_EXIT=0
VPFP_BINDING_EXIT=2
if [[ ! -f "${VPFP_AUDIT_MECHANICS_RESULT}" || -L "${VPFP_AUDIT_MECHANICS_RESULT}" ]]; then
  echo "Explicit mechanics result is missing/nonregular." >&2
  exit 2
fi
if VPFP_SHA_LINE="$(sha256sum -- "${VPFP_AUDIT_MECHANICS_RESULT}")"; then
  if [[ "${VPFP_SHA_LINE%% *}" != "${VPFP_AUDIT_MECHANICS_SHA256}" ]]; then
    echo "Explicit mechanics result SHA changed." >&2
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
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-regime-audit-pycache"
# Only successful completed mechanics are a prerequisite for this old-native
# audit. This does NOT assert the whole moving-frame preflight passed.
"${VPFP_PYTHON}" -c 'import hashlib,json,sys; from pathlib import Path; p=json.load(open(sys.argv[1])); assert p["schema"] == "vpfp_completed_mechanics_receipt_v1" and p["stage"] == "mechanics_tests"; assert type(p.get("exit_code")) is int and p["exit_code"] == 0; names={Path(x).name for x in p["command"]}; assert {"test_vpfp_regime_fidelity_coverage.py","test_vpfp_regime_fidelity_coverage_cli.py"} <= names; log=Path(p["log"]); assert log.is_file() and not log.is_symlink(); print("Frozen mechanics substage passed; overall preflight status at capture =", p.get("overall_preflight_execution_passed_at_capture"))' "${VPFP_AUDIT_MECHANICS_RESULT}" || { VPFP_BINDING_EXIT=$?; exit "${VPFP_BINDING_EXIT}"; }
VPFP_BINDING_EXIT=0
VPFP_GPU_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))' \
  > "${VPFP_GPU_LOG}" 2>&1 || VPFP_GPU_EXIT=$?
if [[ "${VPFP_GPU_EXIT}" != 0 ]]; then exit "${VPFP_GPU_EXIT}"; fi
VPFP_TESTS=(
  tests/test_vpfp_coherent_diffusion.py tests/test_vpfp_training_resume.py tests/test_vpfp_training_snapshots.py
  tests/test_vpfp_conditional_rho_covariance_gate.py tests/test_vpfp_numerical_roundtrip.py
  tests/test_vpfp_regime_fidelity_coverage.py tests/test_vpfp_regime_fidelity_coverage_cli.py
  tests/test_vpfp_weak_coherent_driver.py tests/test_vpfp_strong_regime_coherent.py
  tests/test_vpfp_campaign_launchers.py
)
# Discover test filenames only, never model/answer targets. This includes the
# new direct audit/backend/binding/smoke tests whose names contain moving_frame.
VPFP_DISCOVERED_TESTS="$(rg --files tests -g 'test_vpfp_*moving_frame*.py' | sort)"
while IFS= read -r VPFP_TEST; do
  if [[ -n "${VPFP_TEST}" ]]; then VPFP_TESTS+=("${VPFP_TEST}"); fi
done <<< "${VPFP_DISCOVERED_TESTS}"
for VPFP_TEST in tests/test_vpfp_moving_frame.py tests/test_vpfp_moving_frame_audit.py "${VPFP_TESTS[@]}"; do
  if [[ ! -f "${VPFP_TEST}" ]]; then echo "Missing registered mechanics test ${VPFP_TEST}" >&2; exit 2; fi
done
VPFP_TEST_EXIT=0
timeout --signal=TERM --kill-after=30s 900 "${VPFP_PYTHON}" -m pytest -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-regime-audit-tests" "${VPFP_TESTS[@]}" \
  > "${VPFP_TEST_LOG}" 2>&1 || VPFP_TEST_EXIT=$?
if [[ "${VPFP_TEST_EXIT}" != 0 ]]; then exit "${VPFP_TEST_EXIT}"; fi
VPFP_STRONG_ARGS=()
for VPFP_DIR in "${VPFP_STRONG_DIRS[@]}"; do VPFP_STRONG_ARGS+=(--generated-dir "${VPFP_DIR}"); done
VPFP_WEAK_ARGS=()
for VPFP_DIR in "${VPFP_WEAK_DIRS[@]}"; do VPFP_WEAK_ARGS+=(--generated-dir "${VPFP_DIR}"); done
# One invocation per regime caches calibration/reference features once for
# all four targets. An independent regime still runs after the other fails.
VPFP_STRONG_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_regime_fidelity_coverage.py --source "${VPFP_STRONG_SOURCE}" \
  "${VPFP_STRONG_ARGS[@]}" --output "${VPFP_OUTPUT}/strong" --device cuda --sample-count 128 --bootstrap-repeats 49 \
  > "${VPFP_STRONG_LOG}" 2>&1 || VPFP_STRONG_EXIT=$?
VPFP_WEAK_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_regime_fidelity_coverage.py --source "${VPFP_WEAK_SOURCE}" \
  "${VPFP_WEAK_ARGS[@]}" --output "${VPFP_OUTPUT}/weak" --device cuda --sample-count 128 --bootstrap-repeats 49 \
  > "${VPFP_WEAK_LOG}" 2>&1 || VPFP_WEAK_EXIT=$?
if [[ "${VPFP_STRONG_EXIT}" != 0 ]]; then exit "${VPFP_STRONG_EXIT}"; fi
exit "${VPFP_WEAK_EXIT}"
