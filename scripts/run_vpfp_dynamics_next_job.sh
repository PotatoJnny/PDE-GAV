#!/usr/bin/env bash
# Common environment/receipt for this specific VPFP development round.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_VENV, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 1 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then exit 2; fi
VPFP_PHASE=$1
case "${VPFP_PHASE}" in cpu|strong|weak|velocity_pilot|rho_continuation) ;; *) exit 2 ;; esac
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-dynamics-next-stage-20260907
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-dynamics-next-${VPFP_PHASE}-${SLURM_JOB_ID}"
VPFP_STARTED=$SECONDS VPFP_STAGE_EXIT=not_run VPFP_REVISION=unverified VPFP_MODULES=not_run VPFP_GPU=1
if [[ "${VPFP_PHASE}" == cpu ]]; then VPFP_GPU=0; fi
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
vpfp_finish() {
  local code=$?
  trap - EXIT
  printf 'stage\tvalue\nsource_revision\t%s\nphase\t%s\nexperiment\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t%s\nmodules\t%s\nscientific_admission\tfalse\n' \
    "${VPFP_REVISION}" "${VPFP_PHASE}" "${VPFP_STAGE_EXIT}" "${code}" "$((SECONDS-VPFP_STARTED))" "${VPFP_GPU}" "${VPFP_MODULES}" \
    > "${VPFP_OUTPUT}/launch-receipt.tsv.partial"
  mv -- "${VPFP_OUTPUT}/launch-receipt.tsv.partial" "${VPFP_OUTPUT}/launch-receipt.tsv"
  exit "${code}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "${VPFP_CHECKOUT}"
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ || -n "$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)" ]]; then
  echo "Freeze relevant source before tests or numerical work." >&2; exit 2
fi
# System profile scripts assume an ordinary shell (LC_BYOBU may be unset).
# Its module initialization also invokes an unsupported -q; do not propagate
# task-local strict options into unmodified host profile function internals.
set +eu
set +o pipefail
source /etc/profile
VPFP_PROFILE_STATUS=$?
set -euo pipefail
printf 'system_profile_return=%s\n' "${VPFP_PROFILE_STATUS}"
if module load StdEnv/2023 python/3.12 arrow/23.0.1 scipy-stack pytest apptainer/1.4.5; then
  VPFP_MODULES=standard
else
  # This cluster exposes only its Slurm module. Previous frozen VPFP
  # jobs used this exact existing CVMFS-backed venv; do not install or replace it.
  VPFP_MODULES=unavailable_existing_venv
  echo "Requested cluster modules unavailable; using existing ${PDE_GAV_VENV} unchanged." >&2
fi
source "${PDE_GAV_VENV}/bin/activate"
test -n "${VIRTUAL_ENV:-}"
case "$(command -v python)" in "${VIRTUAL_ENV}/"*) ;; *) exit 2 ;; esac
VPFP_PYTHON=${PDE_GAV_PYTHON}
export PYTHONPATH="${VPFP_CHECKOUT}/src:${VPFP_CHECKOUT}/scripts" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-next-pycache"
if [[ "${VPFP_PHASE}" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=""
  VPFP_STAGE_EXIT=0
  VPFP_TESTS=(
    tests/test_vpfp_reference_refinement.py tests/test_vpfp_strong_audit_completion.py
    tests/test_vpfp_velocity_raw_reference.py tests/test_vpfp_strong_rho_continuation.py
    tests/test_vpfp_weak_block_diagnostic.py tests/test_vpfp_weak_block_driver.py
    tests/test_vpfp_coherent_velocity_diffusion.py tests/test_vpfp_velocity_dynamics_pilot.py
    tests/test_vpfp_dynamics_next_launch.py
    tests/test_vpfp_training_resume.py tests/test_vpfp_training_snapshots.py
    tests/test_vpfp_coherent_velocity_dynamics.py tests/test_vpfp_velocity_dynamics_preflight.py
    tests/test_vpfp_weak_dynamics_diagnostic.py tests/test_vpfp_weak_dynamics_driver.py
    tests/test_vpfp_independent_physics.py tests/test_vpfp_independent_physics_cli.py
    tests/test_vpfp_physical_controls.py tests/test_vpfp_coherent_diffusion.py
    tests/test_vpfp_diffusion_data.py tests/test_vpfp_initial_conditioned_diffusion.py
    tests/test_vpfp_strong_initial_layer_audit.py tests/test_vpfp_strong_regime_coherent.py
  )
  while IFS= read -r VPFP_TEST; do
    if [[ -n "${VPFP_TEST}" ]]; then VPFP_TESTS+=("${VPFP_TEST}"); fi
  done < <(rg --files tests -g 'test_vpfp_*moving_frame*.py' | sort)
  timeout --signal=TERM --kill-after=15s 1500 "${VPFP_PYTHON}" -m pytest -o addopts= -q -p no:cacheprovider \
    --basetemp "${SLURM_TMPDIR}/vpfp-next-cpu-tests" "${VPFP_TESTS[@]}" \
    > "${VPFP_OUTPUT}/mechanics-tests.log" 2>&1 || VPFP_STAGE_EXIT=$?
  if [[ "${VPFP_STAGE_EXIT}" == 0 ]]; then
    "${VPFP_PYTHON}" scripts/run_vpfp_strong_audit_completion.py --output "${VPFP_OUTPUT}/domain-preflight" \
      --device cpu --preflight-only --wall-seconds 90 > "${VPFP_OUTPUT}/domain-preflight.log" 2>&1 || VPFP_STAGE_EXIT=$?
  fi
else
  if [[ "${VPFP_NEXT_CPU_RECEIPT:-}" != /* ]]; then
    echo "Explicit same-source successful CPU receipt required." >&2; exit 2
  fi
  "${VPFP_PYTHON}" -c 'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.is_file() and not p.is_symlink(); rows=[l.split("\t") for l in p.read_text().splitlines()]; assert all(len(r)==2 for r in rows) and len(dict(rows))==len(rows); d=dict(rows); assert d["source_revision"]==sys.argv[3] and d["phase"]=="cpu" and d["experiment"]=="0" and d["launch"]=="0" and d["requested_gpus"]=="0"; import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print("same-source CPU and single CUDA device verified",torch.__version__,torch.version.cuda)' \
    "${VPFP_NEXT_CPU_RECEIPT}" "${VPFP_NEXT_CPU_SHA256:-}" "${VPFP_REVISION}" > "${VPFP_OUTPUT}/binding-gpu-gate.log" 2>&1
  VPFP_STAGE_EXIT=0
  timeout --signal=TERM --kill-after=15s 600 "${VPFP_PYTHON}" -m pytest -o addopts= -q -p no:cacheprovider \
    --basetemp "${SLURM_TMPDIR}/vpfp-next-cuda-tests" tests/test_vpfp_training_resume.py \
    tests/test_vpfp_independent_physics.py tests/test_vpfp_coherent_velocity_diffusion.py \
    tests/test_vpfp_velocity_dynamics_pilot.py \
    > "${VPFP_OUTPUT}/CUDA-mechanics.log" 2>&1 || VPFP_STAGE_EXIT=$?
  if [[ "${VPFP_STAGE_EXIT}" != 0 ]]; then exit "${VPFP_STAGE_EXIT}"; fi
  VPFP_LIMIT=21600
  if [[ "${VPFP_PHASE}" == rho_continuation ]]; then VPFP_LIMIT=14400; fi
  VPFP_REMAINING=$((VPFP_LIMIT-(SECONDS-VPFP_STARTED)-90))
  if ((VPFP_REMAINING<=60)); then exit 124; fi
  case "${VPFP_PHASE}" in
    strong)
      timeout --signal=TERM --kill-after=30s "${VPFP_REMAINING}" "${VPFP_PYTHON}" scripts/run_vpfp_strong_audit_completion.py \
        --output "${VPFP_OUTPUT}/audit" --device cuda --wall-seconds "$((VPFP_REMAINING-30))" \
        > "${VPFP_OUTPUT}/audit.log" 2>&1 || VPFP_STAGE_EXIT=$?
      ;;
    weak)
      # Raw-time reference is a representation diagnostic, not an automatic
      # scientific pass. Do not start new-model training in this allocation.
      timeout --signal=TERM --kill-after=30s "${VPFP_REMAINING}" "${VPFP_PYTHON}" scripts/run_vpfp_velocity_raw_reference.py \
        --output "${VPFP_OUTPUT}/raw-time-reference" --device cuda --wall-seconds "$((VPFP_REMAINING-30))" \
        > "${VPFP_OUTPUT}/raw-time-reference.log" 2>&1 || VPFP_STAGE_EXIT=$?
      # Independent block intervention is meaningful even if raw refinement
      # failed. Preserve both results and do not overwrite the failure code.
      VPFP_BLOCK_EXIT=0
      VPFP_REMAINING=$((VPFP_LIMIT-(SECONDS-VPFP_STARTED)-90))
      if ((VPFP_REMAINING>60)); then
        timeout --signal=TERM --kill-after=30s "${VPFP_REMAINING}" "${VPFP_PYTHON}" scripts/run_vpfp_weak_block_diagnostic.py \
          --output "${VPFP_OUTPUT}/blocks" --device cuda --wall-seconds "$((VPFP_REMAINING-30))" \
          > "${VPFP_OUTPUT}/blocks.log" 2>&1 || VPFP_BLOCK_EXIT=$?
      else VPFP_BLOCK_EXIT=124; fi
      if [[ "${VPFP_BLOCK_EXIT}" != 0 ]]; then VPFP_STAGE_EXIT=${VPFP_BLOCK_EXIT}; fi
      ;;
    velocity_pilot)
      if [[ ! "${VPFP_VELOCITY_PREFLIGHT_SHA256:-}" =~ ^[0-9a-f]{64}$ || "${VPFP_WEAK_RAW_REFERENCE:-}" != /* || ! "${VPFP_WEAK_RAW_REFERENCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then exit 2; fi
      timeout --signal=TERM --kill-after=30s "${VPFP_REMAINING}" "${VPFP_PYTHON}" scripts/run_vpfp_velocity_dynamics_pilot.py \
        --config configs/vpfp_unified_live_weak_collision.json --output "${VPFP_OUTPUT}/pilot" \
        --expected-preflight-sha256 "${VPFP_VELOCITY_PREFLIGHT_SHA256}" --wall-seconds "$((VPFP_REMAINING-30))" \
        --raw-reference-receipt "${VPFP_WEAK_RAW_REFERENCE}" --raw-reference-sha256 "${VPFP_WEAK_RAW_REFERENCE_SHA256}" \
        > "${VPFP_OUTPUT}/pilot.log" 2>&1 || VPFP_STAGE_EXIT=$?
      ;;
    rho_continuation)
      if [[ "${VPFP_COMPLETED_STRONG_AUDIT:-}" != /* || ! "${VPFP_COMPLETED_STRONG_AUDIT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then exit 2; fi
      timeout --signal=TERM --kill-after=30s "${VPFP_REMAINING}" "${VPFP_PYTHON}" scripts/run_vpfp_strong_rho_continuation.py \
        --output "${VPFP_OUTPUT}/continuation" --audit-completion "${VPFP_COMPLETED_STRONG_AUDIT}" \
        --audit-sha256 "${VPFP_COMPLETED_STRONG_AUDIT_SHA256}" --wall-seconds "$((VPFP_REMAINING-30))" \
        > "${VPFP_OUTPUT}/continuation.log" 2>&1 || VPFP_STAGE_EXIT=$?
      ;;
  esac
fi
if [[ "$(git rev-parse HEAD)" != "${VPFP_REVISION}" || -n "$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)" ]]; then
  echo "Source changed while computing; gate invalid." >&2; VPFP_STAGE_EXIT=2
fi
exit "${VPFP_STAGE_EXIT}"
