#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=gkw_unified_runtime_smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Requires the site variables PDE_GAV_RUN_ROOT, PDE_GAV_DEPS_ROOT (see docs/SITE_CONFIGURATION.md).
set -euo pipefail

if [[ "${PDE_ENABLE_LEGACY_MACRO_REFERENCE:-0}" != "1" ]]; then
  echo "Legacy Macro history/control-arm execution is disabled; set PDE_ENABLE_LEGACY_MACRO_REFERENCE=1 to opt in." >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Selected GKW service smoke requires a Slurm compute node." >&2
  exit 2
fi
VENV="${FUSION_VENV:-${HOME}/environment/venv}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Required cluster venv is absent: ${VENV}" >&2
  exit 2
fi
source "${VENV}/bin/activate"
test -n "${VIRTUAL_ENV:-}"
case "$(command -v python)" in
  "${VIRTUAL_ENV}/"*) ;;
  *) echo "Python is outside the activated venv." >&2; exit 2 ;;
esac

SMOKE_COHORT="${SMOKE_COHORT:-development}"
case "${SMOKE_COHORT}" in
  development|selection) ;;
  *) echo "SMOKE_COHORT must be development or selection." >&2; exit 2 ;;
esac

RUN_STORAGE_ROOT="${RUN_STORAGE_ROOT:-${PDE_GAV_RUN_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${RUN_STORAGE_ROOT}/gkw-unified-runtime-smoke-${SLURM_JOB_ID}}"
case "${RUN_ROOT}" in
  "${RUN_STORAGE_ROOT}"/*) ;;
  *) echo "RUN_ROOT must be below RUN_STORAGE_ROOT." >&2; exit 2 ;;
esac
UNIFIED_ROOT="${GKW_UNIFIED_ROOT:-${SLURM_SUBMIT_DIR}}"
if [[ ! -f "${UNIFIED_ROOT}/scripts/smoke_gkw_unified_runtime.py" ]]; then
  echo "GKW_UNIFIED_ROOT does not contain the unified runtime smoke." >&2
  exit 2
fi
if [[ ! -e "${UNIFIED_ROOT}/.git" ]]; then
  echo "GKW_UNIFIED_ROOT must be a Git checkout." >&2
  exit 2
fi
QUALIFICATION_CERTIFICATE="${GKW_PERTURBATION_CERTIFICATE:-}"
QUALIFICATION_REPORT="${GKW_QUALIFICATION_REPORT:-}"
GENERATOR_MANIFEST="${GKW_QUALIFICATION_GENERATOR_MANIFEST:-}"
OPERATOR_MANIFEST="${GKW_QUALIFICATION_OPERATOR_MANIFEST:-}"
PHYSICAL_SOURCE_MANIFEST="${GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST:-}"
SEED_PLAN="${GKW_QUALIFICATION_SEED_PLAN:-}"
for path in \
  "${QUALIFICATION_CERTIFICATE}" "${QUALIFICATION_REPORT}" \
  "${GENERATOR_MANIFEST}" "${OPERATOR_MANIFEST}" \
  "${PHYSICAL_SOURCE_MANIFEST}" "${SEED_PLAN}"; do
  if [[ -z "${path}" || ! -f "${path}" || -L "${path}" ]]; then
    echo "All six GKW production qualification files are required." >&2
    exit 2
  fi
done
UNIFIED_COMMIT="$(git -C "${UNIFIED_ROOT}" rev-parse HEAD)"
if [[ -n "$(git -C "${UNIFIED_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "GKW_UNIFIED_ROOT must be clean for an auditable smoke." >&2
  exit 2
fi

PHYSICAL_COMMIT="${GKW_PHYSICAL_COMMIT:?Set the final pinned lean physical commit}"
PHYSICAL_ROOT="${GKW_PHYSICAL_ROOT:?Set the clean pinned physical checkout}"
if [[ ! -e "${PHYSICAL_ROOT}/.git" ]]; then
  echo "GKW_PHYSICAL_ROOT must be a Git checkout." >&2
  exit 2
fi
ACTUAL_PHYSICAL_COMMIT="$(git -C "${PHYSICAL_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_PHYSICAL_COMMIT}" != "${PHYSICAL_COMMIT}" ]]; then
  echo "GKW physical checkout is not the pinned perturbation commit." >&2
  exit 2
fi
if [[ -n "$(git -C "${PHYSICAL_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "GKW physical checkout must be clean." >&2
  exit 2
fi
if [[ ! -f "${PHYSICAL_ROOT}/src/agent_verifier/fusion_simulation_first/gkw_selected_service_factory.py" || ! -f "${PHYSICAL_ROOT}/src/agent_verifier/fusion_simulation_first/gkw_smooth_perturbation.py" ]]; then
  echo "Pinned GKW physical interface is incomplete." >&2
  exit 2
fi
EXTERNAL_ROOT="${NEURAL_GYROKINETICS_ROOT:-${PDE_GAV_DEPS_ROOT}/neural-gyrokinetics-568411b}"
EXTERNAL_COMMIT="568411b33be486262b8c5cfe1dfb3e0fe97f6419"
DEPS_ROOT="${GKW_QUALIFICATION_DEPS_ROOT:-${PDE_GAV_DEPS_ROOT}/gkw-perturbation-qualification-python-v1}"

KINETIC_EDM="${KINETIC_EDM:-${PDE_GAV_RUN_ROOT}/fusion-nested-c256-edm-capacity384-highnoise-20k-v1/hierarchical_edm.pt}"
KINETIC_CORPUS="${KINETIC_CORPUS:-${PDE_GAV_RUN_ROOT}/fusion-nested-hierarchy-c256-width512-depth3-12k-v1/nested_hierarchy_corpus.pt}"
POTENTIAL_EDM="${POTENTIAL_EDM:-${PDE_GAV_RUN_ROOT}/fusion-potential-edm-axial-kx-30k-reaudit-v1/potential_spectral_edm.pt}"
POTENTIAL_CORPUS="${POTENTIAL_CORPUS:-${PDE_GAV_RUN_ROOT}/fusion-potential-corpus-k17y17-centered-screen16x-v1/potential_spectral_corpus.pt}"
HIERARCHY_CHECKPOINT="${HIERARCHY_CHECKPOINT:-${PDE_GAV_RUN_ROOT}/fusion-nested-hierarchy-c256-width512-depth3-12k-v1/nested_hierarchy_codec.pt}"
PARENT_HIERARCHY_CHECKPOINT="${PARENT_HIERARCHY_CHECKPOINT:-${PDE_GAV_RUN_ROOT}/fusion-nested-aware-parent-decoder-cont-physical-1500-v1/hierarchical_decoder_repair.pt}"
GEOMETRY_SIDECAR="${GEOMETRY_SIDECAR:-${PDE_GAV_RUN_ROOT}/fusion-geometry-sidecar-4280615/gkw_field_geometry_sidecar.pt}"
VELOCITY_METRIC="${VELOCITY_METRIC:-${PDE_GAV_RUN_ROOT}/fusion-generated-velocity-metric-scratch-s2k-v1/velocity_metric_carrier.pt}"
CORRECTION_CALIBRATION="${CORRECTION_CALIBRATION:-${PDE_GAV_RUN_ROOT}/fusion-joint-exact-h-phi-calibration-scratch-learned-480-v1/generated_field_closure_report.json}"
TIME_REGISTRY="${TIME_REGISTRY:-${PDE_GAV_RUN_ROOT}/fusion-gkw-time-4306935/gkw_time_registry.npz}"
ENCODE_REPORT="${ENCODE_REPORT:-${PDE_GAV_RUN_ROOT}/fusion-axial-flow-4278689/encode_report.json}"
LATENT_CORPUS="${LATENT_CORPUS:-${PDE_GAV_RUN_ROOT}/fusion-axial-flow-4278689/latent_corpus.pt}"
EQUATION_CALIBRATION="${EQUATION_CALIBRATION:-${PDE_GAV_RUN_ROOT}/fusion-gkw-equation-first-calibration-32-v1/report.json}"
GYARADAX_ROOT="${GYARADAX_ROOT:-${PDE_GAV_DEPS_ROOT}/gyaradax-318e3d11}"
GYARADAX_COMMIT="318e3d11d7326d99e96f2bd5a8998a462261dcd4"

INPUT_FILES=(
  "${KINETIC_EDM}"
  "${KINETIC_CORPUS}"
  "${POTENTIAL_EDM}"
  "${POTENTIAL_CORPUS}"
  "${HIERARCHY_CHECKPOINT}"
  "${PARENT_HIERARCHY_CHECKPOINT}"
  "${GEOMETRY_SIDECAR}"
  "${VELOCITY_METRIC}"
  "${CORRECTION_CALIBRATION}"
  "${TIME_REGISTRY}"
  "${ENCODE_REPORT}"
  "${LATENT_CORPUS}"
  "${EQUATION_CALIBRATION}"
)
for path in "${INPUT_FILES[@]}"; do
  case "${path}" in
    /scratch/*) ;;
    *) echo "Selected GKW input must be under /scratch: ${path}" >&2; exit 2 ;;
  esac
  if [[ ! -f "${path}" ]]; then
    echo "Selected GKW input is absent: ${path}" >&2
    exit 2
  fi
done
for checkout in "${EXTERNAL_ROOT}" "${GYARADAX_ROOT}"; do
  case "${checkout}" in
    /scratch/*) ;;
    *) echo "Pinned external checkout must be under /scratch: ${checkout}" >&2; exit 2 ;;
  esac
  if [[ ! -d "${checkout}/.git" ]]; then
    echo "Pinned external checkout is absent: ${checkout}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${checkout}" status --porcelain --untracked-files=all)" ]]; then
    echo "Pinned external checkout is dirty: ${checkout}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${EXTERNAL_ROOT}" rev-parse HEAD)" != "${EXTERNAL_COMMIT}" ]]; then
  echo "neural-gyrokinetics checkout is not the pinned commit." >&2
  exit 2
fi
if [[ "$(git -C "${GYARADAX_ROOT}" rev-parse HEAD)" != "${GYARADAX_COMMIT}" ]]; then
  echo "Gyaradax checkout is not the pinned commit." >&2
  exit 2
fi
if [[ "${DEPS_ROOT}" != /scratch/* || ! -d "${DEPS_ROOT}" ]]; then
  echo "Frozen qualification Python dependency bundle is absent: ${DEPS_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${SLURM_TMPDIR}/pycache"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export HF_HOME="${SLURM_TMPDIR}/huggingface"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH="${UNIFIED_ROOT}/src:${PHYSICAL_ROOT}/src:${EXTERNAL_ROOT}:${DEPS_ROOT}:${PYTHONPATH:-}"
cd "${UNIFIED_ROOT}"

python -c '
import importlib.metadata as metadata

expected = {
    "kappamodules": "0.1.112",
    "einops": "0.8.1",
    "omegaconf": "2.3.0",
    "antlr4-python3-runtime": "4.9.3",
    "PyYAML": "6.0.2",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"frozen GKW smoke dependencies changed: {actual!r}")
'

python -m py_compile \
  scripts/smoke_gkw_unified_runtime.py \
  src/pde_agent_verifier_solver/plugins/gkw/selected_torch_runtime.py \
  "${PHYSICAL_ROOT}/src/agent_verifier/fusion_simulation_first/gkw_selected_service_factory.py" \
  "${PHYSICAL_ROOT}/src/agent_verifier/fusion_simulation_first/gkw_perturbation_qualification.py"

python scripts/smoke_gkw_unified_runtime.py \
  --run-root "${RUN_ROOT}" \
  --storage-root "${RUN_STORAGE_ROOT}" \
  --external-root "${EXTERNAL_ROOT}" \
  --gyaradax-root "${GYARADAX_ROOT}" \
  --physical-commit "${PHYSICAL_COMMIT}" \
  --physical-root "${PHYSICAL_ROOT}" \
  --unified-commit "${UNIFIED_COMMIT}" \
  --qualification-certificate "${QUALIFICATION_CERTIFICATE}" \
  --qualification-report "${QUALIFICATION_REPORT}" \
  --generator-manifest "${GENERATOR_MANIFEST}" \
  --operator-manifest "${OPERATOR_MANIFEST}" \
  --physical-source-manifest "${PHYSICAL_SOURCE_MANIFEST}" \
  --seed-plan "${SEED_PLAN}" \
  --kinetic-edm "${KINETIC_EDM}" \
  --kinetic-corpus "${KINETIC_CORPUS}" \
  --potential-edm "${POTENTIAL_EDM}" \
  --potential-corpus "${POTENTIAL_CORPUS}" \
  --hierarchy-checkpoint "${HIERARCHY_CHECKPOINT}" \
  --parent-hierarchy-checkpoint "${PARENT_HIERARCHY_CHECKPOINT}" \
  --geometry-sidecar "${GEOMETRY_SIDECAR}" \
  --velocity-metric "${VELOCITY_METRIC}" \
  --correction-calibration "${CORRECTION_CALIBRATION}" \
  --time-registry "${TIME_REGISTRY}" \
  --encode-report "${ENCODE_REPORT}" \
  --latent-corpus "${LATENT_CORPUS}" \
  --equation-calibration "${EQUATION_CALIBRATION}" \
  --seed "${SEED:-20260824}" \
  --lifecycle-root-count "${LIFECYCLE_ROOT_COUNT:-1}" \
  --cohort "${SMOKE_COHORT}"
