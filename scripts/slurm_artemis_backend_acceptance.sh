#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=artemis_backend_acceptance
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Backend cold-deployment acceptance only: no Agent, formal restart or credentials.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "A separate Slurm compute allocation is required." >&2
  exit 2
fi
ARTEMIS_DEPLOY="${PDE_GAV_WORKTREES_ROOT}/artemis-backend-integrated-20260907"
ARTEMIS_PILOT="${PDE_GAV_WORKTREES_ROOT}/artemis-eval-benchmark-20260907"
ARTEMIS_PHYSICAL="${PDE_GAV_WORKTREES_ROOT}/artemis-oracle-opt-20260907"
ARTEMIS_BASELINE="${PDE_GAV_WORKTREES_ROOT}/plasma-canonical-integration"
ARTEMIS_UPSTREAM="${PDE_GAV_WORKTREES_ROOT}/canonical-worker-cleanup-mainline-20260907"
ARTEMIS_PYTHON="${PDE_GAV_PYTHON}"
ARTEMIS_CODEX="${CODEX_BINARY}"
ARTEMIS_S2="${PDE_GAV_RUN_ROOT}/artemis_canonical_5round_20260905/job-5243774/campaign/agent_turns/round-0002-artemis-critical-refiner-593ae19a865e4808b90d659f75c94b0d/.controller-candidates/07540f9afad42fd504bf99fae71fbaafdec8510038921187ed184c9a72ad24c0"
ARTEMIS_OUTPUT="${ARTEMIS_DEPLOY}/runs/backend-acceptance-20260907/job-${SLURM_JOB_ID}"
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the existing venv." >&2
fi
for ARTEMIS_TREE in "${ARTEMIS_DEPLOY}" "${ARTEMIS_PILOT}" "${ARTEMIS_PHYSICAL}" "${ARTEMIS_UPSTREAM}"; do
  if [[ -n "$(git -C "${ARTEMIS_TREE}" status --porcelain --untracked-files=all)" ]]; then
    echo "Acceptance requires committed clean source: ${ARTEMIS_TREE}" >&2
    exit 2
  fi
done
if [[ -e "${ARTEMIS_OUTPUT}" || -L "${ARTEMIS_OUTPUT}" ]]; then
  echo "Refusing to reuse acceptance output." >&2
  exit 2
fi
mkdir -p -- "${ARTEMIS_OUTPUT}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/artemis-backend-pycache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${ARTEMIS_DEPLOY}/src"
cd -- "${ARTEMIS_DEPLOY}"

# Existing no-model/no-auth probe verifies workspace writes, outside denial,
# and loopback under the unchanged canonical sandbox policy on this host.
"${ARTEMIS_PYTHON}" scripts/check_canonical_codex_environment.py \
  --codex-binary "${ARTEMIS_CODEX}" --config scripts/canonical_codex.config.toml \
  --output "${ARTEMIS_OUTPUT}/environment-check"
PYTHONPATH="${ARTEMIS_UPSTREAM}/src" "${ARTEMIS_PYTHON}" -m pytest -q \
  "${ARTEMIS_UPSTREAM}/tests/test_canonical_codex_agent.py" \
  -o cache_dir="${SLURM_TMPDIR}/pytest-upstream-cleanup" \
  --junitxml="${ARTEMIS_OUTPUT}/upstream-worker-tests.xml"
"${ARTEMIS_PYTHON}" -m pytest -q \
  tests/test_canonical_codex_agent.py tests/test_canonical_codex_environment.py \
  tests/test_artemis_canonical.py tests/test_artemis_formal_runner.py \
  tests/test_artemis_backend_deployment.py \
  -o cache_dir="${SLURM_TMPDIR}/pytest-artemis-deployment" \
  --junitxml="${ARTEMIS_OUTPUT}/deployment-tests.xml"
PYTHONPATH="${ARTEMIS_PHYSICAL}/src" "${ARTEMIS_PYTHON}" -m pytest -q \
  "${ARTEMIS_PHYSICAL}/tests/test_vp_1d1v_artemis_compact_factorized_field_v4.py" \
  "${ARTEMIS_PHYSICAL}/tests/test_vp_1d1v_artemis_compact_v5_production_runtime.py" \
  -o cache_dir="${SLURM_TMPDIR}/pytest-artemis-physical" \
  --junitxml="${ARTEMIS_OUTPUT}/physical-tests.xml"
PYTHONPATH="${ARTEMIS_PILOT}/src" "${ARTEMIS_PYTHON}" -m pytest -q \
  "${ARTEMIS_PILOT}/tests/test_artemis_oracle_benchmark.py" \
  -o cache_dir="${SLURM_TMPDIR}/pytest-artemis-pilot" \
  --junitxml="${ARTEMIS_OUTPUT}/benchmark-tests.xml"
"${ARTEMIS_PYTHON}" scripts/run_canonical_artemis_campaign.py \
  --config configs/artemis_canonical_5round.json --validate-only

# Same fixed S2, fresh Development seed, full legal native horizon. The old
# and new canonical packages run in separate processes with their real pins.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
ARTEMIS_STATUS=0
for ARTEMIS_MODE in baseline optimized; do
  if [[ "${ARTEMIS_MODE}" == baseline ]]; then
    ARTEMIS_CANON="${ARTEMIS_BASELINE}"
  else
    ARTEMIS_CANON="${ARTEMIS_DEPLOY}"
  fi
  PYTHONPATH="${ARTEMIS_CANON}/src" "${ARTEMIS_PYTHON}" \
    "${ARTEMIS_PILOT}/scripts/benchmark_canonical_artemis_oracle.py" run \
    --label "${ARTEMIS_MODE}" --canonical-root "${ARTEMIS_CANON}" \
    --backend-config "${ARTEMIS_CANON}/configs/plasma_canonical_artemis.json" \
    --candidate "S2=${ARTEMIS_S2}" --parents 2 --seed 2026090702 \
    --point-count 32 --horizon native-end \
    --output-root "${ARTEMIS_OUTPUT}/${ARTEMIS_MODE}" || ARTEMIS_STATUS=1
done
"${ARTEMIS_PYTHON}" "${ARTEMIS_PILOT}/scripts/benchmark_canonical_artemis_oracle.py" compare \
  --baseline "${ARTEMIS_OUTPUT}/baseline/result.json" \
  --optimized "${ARTEMIS_OUTPUT}/optimized/result.json" \
  --output-root "${ARTEMIS_OUTPUT}/comparison" || ARTEMIS_STATUS=1
echo "Artemis Backend acceptance finished (status=${ARTEMIS_STATUS}): ${ARTEMIS_OUTPUT}"
exit "${ARTEMIS_STATUS}"
