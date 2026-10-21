#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=vpfp_strong_audit_analysis_cpu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# CPU mechanics on a scheduler allocation, never a login node. No training,
# real corpus extraction, generation or scientific confirmation. CUDA-specific
# tests skip here and still have to pass the later one-GPU confirmation gate.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires a Slurm compute allocation and no positional arguments." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-audit-analysis
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-strong-audit-analysis-cpu-gate-${SLURM_JOB_ID}"
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then echo "Preserving existing output." >&2; exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
VPFP_STARTED_SECONDS=$SECONDS
VPFP_REVISION=not_verified VPFP_TEST_EXIT=not_run VPFP_FIGURE_EXIT=not_requested
vpfp_finish() {
  local status=$?
  trap - EXIT
  printf 'stage\texit_code\nsource_revision\t%s\nmechanics_tests\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t0\nrequested_cpus\t8\nscientific_admission\tfalse\nCUDA_mechanics_completed\tfalse\n' \
    "${VPFP_REVISION}" "${VPFP_TEST_EXIT}" "${status}" "$((SECONDS-VPFP_STARTED_SECONDS))" > "${VPFP_OUTPUT}/receipt.tsv.partial"
  printf 'completed_audit_figures\t%s\nfigure_source_sha256\t%s\n' "${VPFP_FIGURE_EXIT}" "${VPFP_FIGURE_SHA256:-not_requested}" >> "${VPFP_OUTPUT}/receipt.tsv.partial"
  mv -- "${VPFP_OUTPUT}/receipt.tsv.partial" "${VPFP_OUTPUT}/receipt.tsv"
  echo "CPU mechanics exit=${status}; ${VPFP_OUTPUT}; no GPU/physics qualification."
  exit "${status}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
cd "${VPFP_CHECKOUT}"
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then echo "Formal commit required." >&2; exit 2; fi
git ls-files --error-unmatch -- scripts/run_vpfp_strong_audit_analysis_cpu.slurm >/dev/null
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" ]]; then echo "Freeze all relevant source before submitting this gate." >&2; exit 2; fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using explicit existing venv without installing packages." >&2
fi
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-strong-audit-analysis-cpu-pycache"
VPFP_TESTS=(tests/test_vpfp_strong_initial_layer_audit.py tests/test_vpfp_strong_audit_summary.py
  tests/test_vpfp_strong_initial_layer_launch.py tests/test_vpfp_independent_physics.py tests/test_vpfp_physical_controls.py
  tests/test_vpfp_reference_refinement.py tests/test_vpfp_independent_confirmation.py
  tests/test_vpfp_independent_confirmation_launch.py tests/test_vpfp_independent_physics_cli.py
  tests/test_vpfp_strong_regime_followup.py
  tests/test_vpfp_independent_cpu_gate.py tests/test_vpfp_diffusion_data.py tests/test_vpfp_regime_coverage_plots.py
  tests/test_vpfp_regime_fidelity_coverage.py tests/test_vpfp_regime_fidelity_coverage_cli.py)
VPFP_DISCOVERED_TESTS="$(rg --files tests -g 'test_vpfp_*moving_frame*.py' | sort)"
while IFS= read -r VPFP_TEST; do
  if [[ -n "${VPFP_TEST}" ]]; then VPFP_TESTS+=("${VPFP_TEST}"); fi
done <<< "${VPFP_DISCOVERED_TESTS}"
for VPFP_TEST in "${VPFP_TESTS[@]}"; do
  if [[ ! -f "${VPFP_TEST}" ]]; then echo "Required mechanics test missing: ${VPFP_TEST}" >&2; exit 2; fi
done
VPFP_TEST_EXIT=0
timeout --signal=TERM --kill-after=15s 900 "${VPFP_PYTHON}" -m pytest -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-strong-audit-analysis-cpu-tests" "${VPFP_TESTS[@]}" \
  > "${VPFP_OUTPUT}/mechanics-tests.log" 2>&1 || VPFP_TEST_EXIT=$?
if [[ "${VPFP_TEST_EXIT}" != 0 ]]; then exit "${VPFP_TEST_EXIT}"; fi
# Fixed completed aggregate report: drawing these curves evaluates
# no fields, generates no answers and makes no physical acceptance decision.
VPFP_FIGURE_RESULT=${PDE_GAV_RUN_ROOT}/vpfp-strong-regime-followup-5286256/followup/regime-audit/result.json
VPFP_FIGURE_SHA256=6af6ea4418a9ec5eadd6dbc52209e1e5380c4505013bceabe04135551074b5f4
if [[ -n "${VPFP_FIGURE_RESULT}" ]]; then
  VPFP_FIGURE_EXIT=2
  if [[ "${VPFP_FIGURE_RESULT:-}" != /* || ! "${VPFP_FIGURE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Figure source requires an explicit absolute result and SHA." >&2; exit 2
  fi
  VPFP_FIGURE_SECONDS=$((1200-(SECONDS-VPFP_STARTED_SECONDS)-30))
  if (( VPFP_FIGURE_SECONDS <= 0 )); then VPFP_FIGURE_EXIT=124; exit 124; fi
  VPFP_FIGURE_EXIT=0
  timeout --signal=TERM --kill-after=15s "${VPFP_FIGURE_SECONDS}" "${VPFP_PYTHON}" scripts/summarize_vpfp_strong_audit.py \
    --audit-result "${VPFP_FIGURE_RESULT}" --expected-sha256 "${VPFP_FIGURE_SHA256}" \
    --output "${VPFP_OUTPUT}/completed-audit-figures" > "${VPFP_OUTPUT}/figures.log" 2>&1 || VPFP_FIGURE_EXIT=$?
  if [[ "$(git rev-parse HEAD)" != "${VPFP_REVISION}" || -n "$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)" ]]; then
    echo "Source changed during tests/summary; preserving output but invalidating gate." >&2
    VPFP_FIGURE_EXIT=2
  fi
  exit "${VPFP_FIGURE_EXIT}"
fi
exit "${VPFP_TEST_EXIT}"
