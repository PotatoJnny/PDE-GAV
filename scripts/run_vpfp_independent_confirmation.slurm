#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=vpfp_independent_confirmation
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Root manually checks prior strong/weak safety dependencies before submission.
# No child submission. Missing BOTH selection variables means PREFLIGHT ONLY.
# To request the reviewed data confirmation, explicitly export:
# VPFP_CONFIRMATION_SELECTION, VPFP_CONFIRMATION_SELECTION_SHA256 and
# VPFP_CONFIRMATION_SEED. This script never discovers or chooses candidates.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
VPFP_STARTED_SECONDS=$SECONDS
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires a Slurm compute allocation and no positional arguments." >&2
  exit 2
fi
VPFP_MODE=preflight_only_not_confirmation
if [[ -n "${VPFP_CONFIRMATION_SELECTION:-}" || -n "${VPFP_CONFIRMATION_SELECTION_SHA256:-}" ]]; then
  if [[ "${VPFP_CONFIRMATION_SELECTION:-}" != /* || ! "${VPFP_CONFIRMATION_SELECTION_SHA256:-}" =~ ^[0-9a-f]{64}$ \
      || ! "${VPFP_CONFIRMATION_SEED:-}" =~ ^[0-9]+$ ]]; then
    echo "Confirmation requires an absolute reviewed selection, exact SHA256 and explicit nonnegative seed." >&2
    exit 2
  fi
  VPFP_MODE=explicit_selected_coarse_confirmation_handoff
elif [[ -n "${VPFP_CONFIRMATION_SEED:-}" ]]; then
  echo "A seed without an explicit selection is ambiguous; remove it for preflight-only." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-independent-regime-confirmation
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-independent-confirmation-${SLURM_JOB_ID}"
# mkdir is the exclusive claim; do not reuse a partial/complete output directory.
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then echo "Output already exists; preserving it." >&2; exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
VPFP_SOURCE_EXIT=not_run VPFP_BINDING_EXIT=not_requested VPFP_GPU_EXIT=not_run
VPFP_ENTRY_EXIT=not_run VPFP_TEST_EXIT=not_run VPFP_CONFIRM_EXIT=not_requested VPFP_REVISION=not_verified
VPFP_CONFIRM_SECONDS=0
vpfp_receipt() {
  local status=$1
  printf 'stage\texit_code\nmode\t%s\nsource_freeze\t%s\nselection_binding\t%s\ngpu_preflight\t%s\nentrypoint_help\t%s\nmechanics_tests\t%s\nconfirmation_handoff\t%s\nlaunch\t%s\nsource_revision\t%s\nselection_sha256\t%s\nconfirmation_budget_seconds\t%s\nactual_elapsed_seconds\t%s\nscientific_admission\tfalse\nrefined_reference_confirmation\tfalse\n' \
    "${VPFP_MODE}" "${VPFP_SOURCE_EXIT}" "${VPFP_BINDING_EXIT}" "${VPFP_GPU_EXIT}" "${VPFP_ENTRY_EXIT}" \
    "${VPFP_TEST_EXIT}" "${VPFP_CONFIRM_EXIT}" "${status}" "${VPFP_REVISION}" "${VPFP_CONFIRMATION_SELECTION_SHA256:-not_requested}" \
    "${VPFP_CONFIRM_SECONDS}" "$((SECONDS-VPFP_STARTED_SECONDS))" > "${VPFP_OUTPUT}/launch-receipt.tsv.partial"
  mv -- "${VPFP_OUTPUT}/launch-receipt.tsv.partial" "${VPFP_OUTPUT}/launch-receipt.tsv"
}
vpfp_finish() {
  local status=$?
  trap - EXIT
  vpfp_receipt "${status}"
  echo "${VPFP_MODE}: exit=${status}; artifacts retained at ${VPFP_OUTPUT}. No admission or refined-reference claim."
  exit "${status}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
vpfp_receipt running
cd "${VPFP_CHECKOUT}"
VPFP_SOURCE_EXIT=2
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then echo "Formal source commit required." >&2; exit 2; fi
git ls-files --error-unmatch -- scripts/run_vpfp_independent_confirmation.slurm scripts/run_vpfp_independent_confirmation.py \
  scripts/run_vpfp_independent_physics.py src/pde_agent_verifier_solver/plugins/vpfp/independent_physics.py \
  src/pde_agent_verifier_solver/plugins/vpfp/physical_controls.py src/pde_agent_verifier_solver/plugins/vpfp/reference_refinement.py >/dev/null
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" ]]; then echo "Commit/freeze all relevant source before manual submission." >&2; exit 2; fi
if [[ ! -x "${VPFP_PYTHON}" ]]; then echo "Explicit existing venv missing." >&2; exit 2; fi
VPFP_SOURCE_EXIT=0
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using explicit existing venv; no package installation." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-independent-confirmation-pycache"
if [[ "${VPFP_MODE}" != preflight_only_not_confirmation ]]; then
  VPFP_BINDING_EXIT=2
  if [[ ! -f "${VPFP_CONFIRMATION_SELECTION}" || -L "${VPFP_CONFIRMATION_SELECTION}" ]]; then echo "Selection missing/nonregular." >&2; exit 2; fi
  if VPFP_SHA_LINE="$(sha256sum -- "${VPFP_CONFIRMATION_SELECTION}")"; then
    if [[ "${VPFP_SHA_LINE%% *}" != "${VPFP_CONFIRMATION_SELECTION_SHA256}" ]]; then echo "Selection SHA changed." >&2; exit 2; fi
  else
    VPFP_BINDING_EXIT=$?
    exit "${VPFP_BINDING_EXIT}"
  fi
  "${VPFP_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["schema"]=="vpfp_independent_confirmation_selection_v1" and p["scientific_admission"] is False and p["review"].strip() and p["candidates"]' \
    "${VPFP_CONFIRMATION_SELECTION}" || { VPFP_BINDING_EXIT=$?; exit "${VPFP_BINDING_EXIT}"; }
  VPFP_BINDING_EXIT=0
fi
VPFP_GPU_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))' \
  > "${VPFP_OUTPUT}/gpu-preflight.log" 2>&1 || VPFP_GPU_EXIT=$?
if [[ "${VPFP_GPU_EXIT}" != 0 ]]; then exit "${VPFP_GPU_EXIT}"; fi
VPFP_ENTRY_EXIT=0
timeout --signal=TERM --kill-after=15s 120 "${VPFP_PYTHON}" scripts/run_vpfp_independent_confirmation.py --help \
  > "${VPFP_OUTPUT}/entrypoint-help.log" 2>&1 || VPFP_ENTRY_EXIT=$?
if [[ "${VPFP_ENTRY_EXIT}" != 0 ]]; then exit "${VPFP_ENTRY_EXIT}"; fi
VPFP_TESTS=(tests/test_vpfp_independent_physics.py tests/test_vpfp_physical_controls.py
  tests/test_vpfp_reference_refinement.py tests/test_vpfp_independent_physics_cli.py
  tests/test_vpfp_diffusion_data.py tests/test_vpfp_independent_confirmation.py
  tests/test_vpfp_regime_fidelity_coverage.py tests/test_vpfp_regime_fidelity_coverage_cli.py
  tests/test_vpfp_independent_confirmation_launch.py)
VPFP_DISCOVERED_TESTS="$(rg --files tests -g 'test_vpfp_*moving_frame*.py' | sort)"
while IFS= read -r VPFP_TEST; do
  if [[ -n "${VPFP_TEST}" ]]; then VPFP_TESTS+=("${VPFP_TEST}"); fi
done <<< "${VPFP_DISCOVERED_TESTS}"
for VPFP_TEST in tests/test_vpfp_moving_frame.py tests/test_vpfp_moving_frame_audit.py tests/test_vpfp_moving_frame_backend.py "${VPFP_TESTS[@]}"; do
  if [[ ! -f "${VPFP_TEST}" ]]; then echo "Missing required test ${VPFP_TEST}" >&2; exit 2; fi
done
# Up to one hour TOTAL for binding/GPU/help/tests, not one extra hour per gate.
VPFP_GATE_SECONDS=$((3600-(SECONDS-VPFP_STARTED_SECONDS)))
if (( VPFP_GATE_SECONDS <= 0 )); then VPFP_TEST_EXIT=124; exit 124; fi
VPFP_TEST_EXIT=0
vpfp_receipt running
timeout --signal=TERM --kill-after=15s "${VPFP_GATE_SECONDS}" "${VPFP_PYTHON}" -m pytest -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-independent-confirmation-tests" "${VPFP_TESTS[@]}" \
  > "${VPFP_OUTPUT}/mechanics-tests.log" 2>&1 || VPFP_TEST_EXIT=$?
if [[ "${VPFP_TEST_EXIT}" != 0 ]]; then exit "${VPFP_TEST_EXIT}"; fi
if [[ "${VPFP_MODE}" == preflight_only_not_confirmation ]]; then
  echo "PREFLIGHT ONLY completed. No candidate selected, references collected or512-answer confirmation run."
  exit 0
fi
# Never borrow unused preflight time to exceed the registered11h confirmation.
# The Slurm12h hard cap includes everything; reserve60s for termination/receipt.
VPFP_CONFIRM_SECONDS=$((43200-(SECONDS-VPFP_STARTED_SECONDS)-60))
if (( VPFP_CONFIRM_SECONDS > 39600 )); then VPFP_CONFIRM_SECONDS=39600; fi
if (( VPFP_CONFIRM_SECONDS <= 30 )); then VPFP_CONFIRM_EXIT=124; exit 124; fi
VPFP_CONFIRM_EXIT=0
vpfp_receipt running
timeout --signal=TERM --kill-after=15s "${VPFP_CONFIRM_SECONDS}" "${VPFP_PYTHON}" scripts/run_vpfp_independent_confirmation.py \
  --selection "${VPFP_CONFIRMATION_SELECTION}" --selection-sha256 "${VPFP_CONFIRMATION_SELECTION_SHA256}" \
  --seed "${VPFP_CONFIRMATION_SEED}" --output "${VPFP_OUTPUT}/confirmation" \
  --wall-seconds "${VPFP_CONFIRM_SECONDS}" --device cuda > "${VPFP_OUTPUT}/confirmation.log" 2>&1 || VPFP_CONFIRM_EXIT=$?
exit "${VPFP_CONFIRM_EXIT}"
