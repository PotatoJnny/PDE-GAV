#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=vpfp_strong_initial_layer_audit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# User-requested independent audit: no training, new population or deployment.
# Charges original audit/preflight8h stage: 5076s used, this allocation <=10800s.
# Root maintains campaign48h and <=2 simultaneous campaign GPUs; no child jobs.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then exit 2; fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-audit-analysis
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-strong-initial-layer-audit-${SLURM_JOB_ID}"
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
VPFP_STARTED_SECONDS=$SECONDS VPFP_REVISION=not_verified VPFP_TEST_EXIT=not_run VPFP_AUDIT_EXIT=not_run
vpfp_finish() {
  local status=$?
  trap - EXIT
  printf 'stage\texit_code\nsource_revision\t%s\nCUDA_mechanics\t%s\nindependent_audit\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t1\nprior_audit_preflight_gpu_seconds\t5076\naudit_preflight_cap_gpu_seconds\t28800\nnew_training\tfalse\nscientific_admission\tfalse\n' \
    "${VPFP_REVISION}" "${VPFP_TEST_EXIT}" "${VPFP_AUDIT_EXIT}" "${status}" "$((SECONDS-VPFP_STARTED_SECONDS))" \
    > "${VPFP_OUTPUT}/launch-receipt.tsv.partial"
  mv -- "${VPFP_OUTPUT}/launch-receipt.tsv.partial" "${VPFP_OUTPUT}/launch-receipt.tsv"
  exit "${status}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "${VPFP_STRONG_CPU_GATE_RECEIPT:-}" != /* ]]; then
  echo "Explicit successful same-source CPU gate path required." >&2; exit 2
fi
cd "${VPFP_CHECKOUT}"
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ || -n "$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)" ]]; then exit 2; fi
git ls-files --error-unmatch -- scripts/run_vpfp_strong_initial_layer_audit.py scripts/run_vpfp_strong_initial_layer_audit.slurm >/dev/null
# A narrower user override may not silently alter the diagnostic budget: fail
# before numerical work unless the explicitly registered 3-hour allocation holds.
VPFP_CONTROL="$(scontrol --oneliner show job "${SLURM_JOB_ID}")"
if [[ " ${VPFP_CONTROL} " != *" JobState=RUNNING "* || " ${VPFP_CONTROL} " != *" TimeLimit=03:00:00 "* ]]; then exit 2; fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using existing venv without installing packages." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-strong-initial-layer-pycache"
"${VPFP_PYTHON}" -c 'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.is_file() and not p.is_symlink(); rows=[line.split("\t") for line in p.read_text().splitlines()]; assert all(len(r)==2 for r in rows) and len(dict(rows))==len(rows); r=dict(rows); assert r["source_revision"]==sys.argv[3] and r["mechanics_tests"]=="0" and r["completed_audit_figures"]=="0" and r["launch"]=="0" and r["requested_gpus"]=="0"; import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1' \
  "${VPFP_STRONG_CPU_GATE_RECEIPT}" "${VPFP_STRONG_CPU_GATE_SHA256:-}" "${VPFP_REVISION}" > "${VPFP_OUTPUT}/binding-gpu-gate.log" 2>&1
VPFP_TEST_EXIT=0
timeout --signal=TERM --kill-after=15s 900 "${VPFP_PYTHON}" -m pytest -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-strong-initial-layer-tests" \
  tests/test_vpfp_independent_physics.py tests/test_vpfp_reference_refinement.py \
  tests/test_vpfp_independent_physics_cli.py tests/test_vpfp_strong_initial_layer_audit.py \
  > "${VPFP_OUTPUT}/CUDA-mechanics.log" 2>&1 || VPFP_TEST_EXIT=$?
if [[ "${VPFP_TEST_EXIT}" != 0 ]]; then exit "${VPFP_TEST_EXIT}"; fi
VPFP_REMAINING=$((10800-(SECONDS-VPFP_STARTED_SECONDS)-45))
if (( VPFP_REMAINING <= 30 )); then VPFP_AUDIT_EXIT=124; exit 124; fi
VPFP_AUDIT_EXIT=0
timeout --signal=TERM --kill-after=15s "${VPFP_REMAINING}" "${VPFP_PYTHON}" \
  scripts/run_vpfp_strong_initial_layer_audit.py --output "${VPFP_OUTPUT}/audit" \
  --device cuda --wall-seconds "${VPFP_REMAINING}" > "${VPFP_OUTPUT}/audit.log" 2>&1 || VPFP_AUDIT_EXIT=$?
if [[ "$(git rev-parse HEAD)" != "${VPFP_REVISION}" || -n "$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)" ]]; then
  echo "Source changed during independent audit." >&2; VPFP_AUDIT_EXIT=2
fi
exit "${VPFP_AUDIT_EXIT}"
