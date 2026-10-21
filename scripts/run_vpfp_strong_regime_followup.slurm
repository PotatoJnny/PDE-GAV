#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:01:00
#SBATCH --job-name=vpfp_strong_regime_followup
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# MANUAL ONLY after5268586 is COMPLETED0:0. Root must explicitly override
# --time to <=floor((28800-terminal sacct ElapsedRaw)/60) minutes and bind:
# VPFP_STRONG_FOLLOWUP_RECEIPT, VPFP_STRONG_FOLLOWUP_RECEIPT_SHA256.
# Default1min is intentionally unusable for audit; the driver verifies actual
# Slurm TimeLimit, parent allocation GPU count/elapsed and all14 assets.
# Root also coordinates total48 GPUh and <=2 GPUs; no child submission.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "A separate Slurm allocation and no positional arguments are required." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-independent-regime-confirmation
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-strong-regime-followup-${SLURM_JOB_ID}"
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then echo "Preserving existing output." >&2; exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
VPFP_STARTED_SECONDS=$SECONDS
VPFP_REVISION=not_verified VPFP_RUN_EXIT=not_run
vpfp_finish() {
  local status=$?
  trap - EXIT
  printf 'stage\texit_code\nsource_revision\t%s\nfollowup\t%s\nlaunch\t%s\nreceipt_sha256\t%s\nwall_seconds\t%s\npredecessor\t5268586\nstrong_total_cap_gpu_seconds\t28800\nscientific_admission\tfalse\n' \
    "${VPFP_REVISION}" "${VPFP_RUN_EXIT}" "${status}" "${VPFP_STRONG_FOLLOWUP_RECEIPT_SHA256:-not_provided}" \
    "$((SECONDS-VPFP_STARTED_SECONDS))" > "${VPFP_OUTPUT}/launch-receipt.tsv.partial"
  mv -- "${VPFP_OUTPUT}/launch-receipt.tsv.partial" "${VPFP_OUTPUT}/launch-receipt.tsv"
  echo "Strong followup exit=${status}; retained ${VPFP_OUTPUT}; no automatic further work."
  exit "${status}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "${VPFP_STRONG_FOLLOWUP_RECEIPT:-}" != /* || ! "${VPFP_STRONG_FOLLOWUP_RECEIPT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Explicit terminal receipt path and SHA256 required; never discover/choose latest assets." >&2
  exit 2
fi
cd "${VPFP_CHECKOUT}"
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then echo "Formal source commit required." >&2; exit 2; fi
git ls-files --error-unmatch -- scripts/run_vpfp_strong_regime_followup.py scripts/run_vpfp_strong_regime_followup.slurm \
  scripts/run_vpfp_regime_fidelity_coverage.py scripts/run_plasma_canonical_smoke.py tests/test_vpfp_strong_regime_followup.py >/dev/null
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" ]]; then echo "Freeze relevant source before manual submission." >&2; exit 2; fi
if [[ ! -x "${VPFP_PYTHON}" || -z "$(command -v sacct)" || -z "$(command -v scontrol)" \
    || -z "${SLURM_CONF:-}" || ! -r "${SLURM_CONF}" ]]; then
  echo "Existing venv/Slurm executables and inherited readable SLURM_CONF required; no installation/config rewrite." >&2
  exit 2
fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using explicit existing venv without installing packages." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-strong-followup-pycache"
VPFP_RUN_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_strong_regime_followup.py \
  --receipt "${VPFP_STRONG_FOLLOWUP_RECEIPT}" --receipt-sha256 "${VPFP_STRONG_FOLLOWUP_RECEIPT_SHA256}" \
  --output "${VPFP_OUTPUT}/followup" --device cuda > "${VPFP_OUTPUT}/followup.log" 2>&1 || VPFP_RUN_EXIT=$?
exit "${VPFP_RUN_EXIT}"
