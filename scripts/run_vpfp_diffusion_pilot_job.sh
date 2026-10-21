#!/usr/bin/env bash
# Shared body for the two plain, bounded Slurm submissions.
# Requires the site variables PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
VPFP_REGIME="${1:?expected weak or strong}"
case "${VPFP_REGIME}" in
  weak) VPFP_SEED=120260904 ;;
  strong) VPFP_SEED=220260904 ;;
  *) echo "Unknown VPFP regime" >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "This pipeline requires a Slurm compute allocation." >&2
  exit 2
fi
VPFP_CHECKOUT="${SLURM_SUBMIT_DIR:?missing submission checkout}"
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_RUN_ROOT="${PDE_GAV_RUN_ROOT}/vpfp-full-coefficient-edm-${VPFP_REGIME}-${SLURM_JOB_ID}"
cd "${VPFP_CHECKOUT}"
if [[ ! -x "${VPFP_PYTHON}" || ! -f scripts/train_vpfp_structured_diffusion.py || -e "${VPFP_RUN_ROOT}" ]]; then
  echo "Missing runtime or existing output; refusing ambiguous execution." >&2
  exit 2
fi
git diff --exit-code HEAD -- src/pde_agent_verifier_solver/plugins/vpfp scripts/train_vpfp_structured_diffusion.py scripts/run_plasma_canonical_smoke.py
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing explicit venv." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-diffusion-pycache"
VPFP_EXIT=0
"${VPFP_PYTHON}" scripts/train_vpfp_structured_diffusion.py \
  --config "configs/vpfp_unified_live_${VPFP_REGIME}_collision.json" \
  --output "${VPFP_RUN_ROOT}" --seed "${VPFP_SEED}" \
  --train-answers 512 --heldout-answers 128 --steps 2000 --train-seconds 5400 \
  --batch-size 4 --collect-batch-size 8 --sample-steps 64 || VPFP_EXIT=$?
# Even an audit-detected distribution failure must leave the research checkpoint
# and independently test its native product ABI, without enabling admission.
if [[ -f "${VPFP_RUN_ROOT}/checkpoints/best.pt" ]]; then
  read -r VPFP_SHA VPFP_UNUSED < <(sha256sum "${VPFP_RUN_ROOT}/checkpoints/best.pt")
  "${VPFP_PYTHON}" scripts/run_plasma_canonical_smoke.py \
    --physics vpfp --config "configs/vpfp_unified_live_${VPFP_REGIME}_collision.json" \
    --diffusion-checkpoint "${VPFP_RUN_ROOT}/checkpoints/best.pt" \
    --diffusion-checkpoint-sha256 "${VPFP_SHA}" \
    --output "${VPFP_RUN_ROOT}/canonical-campaign-smoke" --device cuda \
    > "${VPFP_RUN_ROOT}/canonical-campaign-smoke.log" 2>&1 || VPFP_EXIT=$?
fi
echo "VPFP ${VPFP_REGIME} diffusion pilot artifacts: ${VPFP_RUN_ROOT}; execution exit: ${VPFP_EXIT}; no automatic scientific admission."
exit "${VPFP_EXIT}"
