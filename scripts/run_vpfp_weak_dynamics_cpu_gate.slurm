#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=vpfp_weak_dynamics_cpu_gate
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Mechanics only; no corpus extraction, training or admission.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then exit 2; fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-weak-dynamics-diagnostics
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-weak-dynamics-cpu-gate-${SLURM_JOB_ID}"
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
VPFP_REVISION=not_verified VPFP_TEST_EXIT=not_run VPFP_STARTED_SECONDS=$SECONDS
vpfp_finish() {
  local status=$?
  trap - EXIT
  printf 'stage\texit_code\nsource_revision\t%s\nmechanics_tests\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t0\nscientific_admission\tfalse\n' \
    "${VPFP_REVISION}" "${VPFP_TEST_EXIT}" "${status}" "$((SECONDS-VPFP_STARTED_SECONDS))" > "${VPFP_OUTPUT}/receipt.tsv.partial"
  mv -- "${VPFP_OUTPUT}/receipt.tsv.partial" "${VPFP_OUTPUT}/receipt.tsv"
  exit "${status}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
cd "${VPFP_CHECKOUT}"
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" || ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then exit 2; fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using existing venv without installing packages." >&2
fi
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-weak-dynamics-cpu-pycache"
VPFP_TESTS=(tests/test_vpfp_weak_dynamics_diagnostic.py tests/test_vpfp_weak_dynamics_driver.py
  tests/test_vpfp_coherent_velocity_dynamics.py tests/test_vpfp_velocity_dynamics_preflight.py
  tests/test_vpfp_weak_dynamics_launch.py tests/test_vpfp_independent_physics.py
  tests/test_vpfp_physical_controls.py tests/test_vpfp_coherent_diffusion.py
  tests/test_vpfp_weak_coherent_driver.py tests/test_vpfp_regime_fidelity_coverage.py
  tests/test_vpfp_diffusion_data.py)
VPFP_DISCOVERED_TESTS="$(rg --files tests -g 'test_vpfp_*moving_frame*.py' | sort)"
while IFS= read -r VPFP_TEST; do
  if [[ -n "${VPFP_TEST}" ]]; then VPFP_TESTS+=("${VPFP_TEST}"); fi
done <<< "${VPFP_DISCOVERED_TESTS}"
VPFP_TEST_EXIT=0
timeout --signal=TERM --kill-after=15s 1100 "${VPFP_PYTHON}" -m pytest -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-weak-dynamics-cpu-tests" "${VPFP_TESTS[@]}" \
  > "${VPFP_OUTPUT}/mechanics-tests.log" 2>&1 || VPFP_TEST_EXIT=$?
if [[ "${VPFP_TEST_EXIT}" == 0 ]]; then
  VPFP_FINAL_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
  VPFP_FINAL_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
  if [[ "${VPFP_FINAL_REVISION}" != "${VPFP_REVISION}" || -n "${VPFP_FINAL_DIRTY}" ]]; then
    echo "Source changed during CPU tests; do not issue a successful same-source gate." >&2
    VPFP_TEST_EXIT=2
  fi
fi
exit "${VPFP_TEST_EXIT}"
