#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
# Manual, explicit SHA-bound conversion only. Never submits a child job.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=vpfp_portable_codec_export_cpu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_VENV, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then exit 2; fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-portable-guidance-20260908
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-portable-codec-export-${SLURM_JOB_ID}"
VPFP_STARTED=$SECONDS VPFP_REVISION=unverified VPFP_STAGE_EXIT=not_run VPFP_MODULES=not_run
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
vpfp_finish() {
  local code=$?
  trap - EXIT
  if [[ "${VPFP_REVISION}" != unverified ]]; then
    if [[ "$(git -C "${VPFP_CHECKOUT}" rev-parse HEAD)" != "${VPFP_REVISION}" || -n "$(git -C "${VPFP_CHECKOUT}" status --porcelain --untracked-files=all)" ]]; then code=1; fi
  fi
  printf 'stage\tvalue\nsource_revision\t%s\nexperiment\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t0\nmodules\t%s\nold_checkpoint_restored\tfalse\nguidance_ready\tfalse\nscientific_admission\tfalse\n' \
    "${VPFP_REVISION}" "${VPFP_STAGE_EXIT}" "${code}" "$((SECONDS-VPFP_STARTED))" "${VPFP_MODULES}" \
    > "${VPFP_OUTPUT}/launch.tsv.partial"
  mv -- "${VPFP_OUTPUT}/launch.tsv.partial" "${VPFP_OUTPUT}/launch.tsv"
  exit "${code}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "${VPFP_CHECKOUT}"
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Conversion requires a fully committed clean checkout." >&2; exit 2
fi
if [[ ! "${VPFP_PORTABLE_PROBE_RESULT:-}" =~ ^${PDE_GAV_RUN_ROOT}/vpfp-strong-portable-replay-probe-[0-9]+/result\.json$ || ! "${VPFP_PORTABLE_PROBE_RESULT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Explicit completed replay report path and SHA must be supplied at submission." >&2; exit 2
fi
set +eu
set +o pipefail
source /etc/profile
VPFP_PROFILE_STATUS=$?
set -euo pipefail
printf 'system_profile_return=%s\n' "${VPFP_PROFILE_STATUS}"
if module load StdEnv/2023 python/3.12 arrow/23.0.1 scipy-stack pytest; then
  VPFP_MODULES=standard
else
  VPFP_MODULES=unavailable_existing_venv
  echo "Cluster modules unavailable; using the existing unchanged venv." >&2
fi
source "${PDE_GAV_VENV}/bin/activate"
test -n "${VIRTUAL_ENV:-}"
case "$(command -v python)" in "${VIRTUAL_ENV}/"*) ;; *) exit 2 ;; esac
export PYTHONPATH="${VPFP_CHECKOUT}/src:${VPFP_CHECKOUT}/scripts" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-portable-codec-export-pycache"
unset PYTHONOPTIMIZE
VPFP_REMAINING=$((1740-(SECONDS-VPFP_STARTED)))
if (( VPFP_REMAINING <= 60 )); then echo "CPU setup exhausted conversion budget." >&2; exit 2; fi
VPFP_STAGE_EXIT=0
timeout --signal=TERM --kill-after=15s "${VPFP_REMAINING}" ${PDE_GAV_PYTHON} \
  scripts/export_vpfp_portable_codec_probe.py --probe-result "${VPFP_PORTABLE_PROBE_RESULT}" \
  --probe-result-sha256 "${VPFP_PORTABLE_PROBE_RESULT_SHA256}" --output "${VPFP_OUTPUT}/conversion" \
  --wall-seconds "$((VPFP_REMAINING-15))" > "${VPFP_OUTPUT}/conversion.log" 2>&1 || VPFP_STAGE_EXIT=$?
exit "${VPFP_STAGE_EXIT}"
