#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --job-name=vpfp_weak_compare
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Manual-only registered weak screen. Twenty hours include tests, extraction,
# six training arms, generation,128-attempt audit and two A plus two W0 smokes.
# No automatic followup, shortened comparison or scientific admission.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 0 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Requires a Slurm compute allocation, local scratch and no positional arguments." >&2
  exit 2
fi
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-moving-frame-campaign
VPFP_PYTHON=${PDE_GAV_PYTHON}
VPFP_SOURCE=${PDE_GAV_RUN_ROOT}/vpfp-scaleup-weak-5241246
VPFP_SELECTION="${VPFP_CHECKOUT}/configs/vpfp_weak_coherent_representation_selection_5267643.json"
VPFP_SELECTION_SHA=93e125134a9b044c140ae44e12eb4bb0d429b45a40d4228be2c033d9b8033b30
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-weak-regime-coherent-${SLURM_JOB_ID}"
VPFP_AUDIT_OUTPUT="${VPFP_OUTPUT}-audit"
VPFP_RECEIPT="${VPFP_OUTPUT}.launch-exit-codes.tsv"
VPFP_GPU_LOG="${VPFP_OUTPUT}.gpu-preflight.log"
VPFP_TEST_LOG="${VPFP_OUTPUT}.mechanics-tests.log"
VPFP_RUN_LOG="${VPFP_OUTPUT}.experiment.log"
VPFP_AUDIT_LOG="${VPFP_OUTPUT}.audit.log"
VPFP_SMOKE_OUTPUTS=("${VPFP_OUTPUT}-native-smoke-seed-0" "${VPFP_OUTPUT}-native-smoke-seed-1")
VPFP_SMOKE_LOGS=("${VPFP_OUTPUT}.native-smoke-seed-0.log" "${VPFP_OUTPUT}.native-smoke-seed-1.log")
VPFP_SMOKE_BIND_LOGS=("${VPFP_OUTPUT}.native-binding-seed-0.log" "${VPFP_OUTPUT}.native-binding-seed-1.log")
VPFP_W0_OUTPUTS=("${VPFP_OUTPUT}-W0-smoke-seed-0" "${VPFP_OUTPUT}-W0-smoke-seed-1")
VPFP_W0_LOGS=("${VPFP_OUTPUT}.W0-smoke-seed-0.log" "${VPFP_OUTPUT}.W0-smoke-seed-1.log")
VPFP_W0_BIND_LOGS=("${VPFP_OUTPUT}.W0-binding-seed-0.log" "${VPFP_OUTPUT}.W0-binding-seed-1.log")
for VPFP_PATH in "${VPFP_OUTPUT}" "${VPFP_AUDIT_OUTPUT}" "${VPFP_RECEIPT}" "${VPFP_GPU_LOG}" \
    "${VPFP_TEST_LOG}" "${VPFP_RUN_LOG}" "${VPFP_AUDIT_LOG}" "${VPFP_SMOKE_OUTPUTS[@]}" \
    "${VPFP_SMOKE_LOGS[@]}" "${VPFP_SMOKE_BIND_LOGS[@]}" "${VPFP_W0_OUTPUTS[@]}" \
    "${VPFP_W0_LOGS[@]}" "${VPFP_W0_BIND_LOGS[@]}"; do
  if [[ -e "${VPFP_PATH}" || -L "${VPFP_PATH}" ]]; then
    echo "Refusing to overwrite ${VPFP_PATH}" >&2
    exit 2
  fi
done
VPFP_SOURCE_EXIT=not_run VPFP_BINDING_EXIT=not_run VPFP_GPU_EXIT=not_run VPFP_TEST_EXIT=not_run
VPFP_RUN_EXIT=not_run VPFP_AUDIT_EXIT=not_run VPFP_REVISION=not_verified
VPFP_SMOKE_EXITS=(not_run not_run)
VPFP_MANIFEST_SHAS=(not_verified not_verified)
VPFP_W0_EXITS=(not_run not_run)
VPFP_W0_SHAS=(not_verified not_verified)
vpfp_finish() {
  local status=$?
  trap - EXIT
  printf 'stage\texit_code\nsource_freeze\t%s\nrepresentation_binding\t%s\ngpu_preflight\t%s\nmechanics_tests\t%s\nweak_comparisons\t%s\npaired_A_B_audit\t%s\nnative_A_seed_0\t%s\nnative_A_seed_1\t%s\nlaunch\t%s\nsource_revision\t%s\nrepresentation_selection_sha256\t%s\nseed_0_manifest_sha256\t%s\nseed_1_manifest_sha256\t%s\n' \
    "${VPFP_SOURCE_EXIT}" "${VPFP_BINDING_EXIT}" "${VPFP_GPU_EXIT}" "${VPFP_TEST_EXIT}" "${VPFP_RUN_EXIT}" \
    "${VPFP_AUDIT_EXIT}" "${VPFP_SMOKE_EXITS[0]}" "${VPFP_SMOKE_EXITS[1]}" "${status}" "${VPFP_REVISION}" \
    "${VPFP_SELECTION_SHA}" "${VPFP_MANIFEST_SHAS[0]}" "${VPFP_MANIFEST_SHAS[1]}" > "${VPFP_RECEIPT}"
  printf 'native_W0_seed_0\t%s\nnative_W0_seed_1\t%s\nseed_0_W0_checkpoint_sha256\t%s\nseed_1_W0_checkpoint_sha256\t%s\n' \
    "${VPFP_W0_EXITS[0]}" "${VPFP_W0_EXITS[1]}" "${VPFP_W0_SHAS[0]}" "${VPFP_W0_SHAS[1]}" >> "${VPFP_RECEIPT}"
  echo "Weak screen exit=${status}; all partial artifacts retained under ${VPFP_OUTPUT} and its named sibling outputs. No admission."
  exit "${status}"
}
trap vpfp_finish EXIT
cd "${VPFP_CHECKOUT}"
VPFP_SOURCE_EXIT=2
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then echo "A formal Git commit is required." >&2; exit 2; fi
git ls-files --error-unmatch -- scripts/run_vpfp_weak_compare.slurm scripts/run_vpfp_weak_coherent.py \
  scripts/run_vpfp_regime_fidelity_coverage.py scripts/run_vpfp_moving_frame_smoke.py scripts/run_plasma_canonical_smoke.py \
  configs/vpfp_weak_coherent_representation_selection_5267643.json >/dev/null
VPFP_DIRTY="$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)"
if [[ -n "${VPFP_DIRTY}" ]]; then echo "Commit/freeze all relevant source before manual launch." >&2; exit 2; fi
for VPFP_PATH in "${VPFP_CHECKOUT}/configs/vpfp_unified_live_weak_collision.json" \
    "${VPFP_SOURCE}/corpus/manifest.json" "${VPFP_SOURCE}/result.json" "${VPFP_SELECTION}"; do
  if [[ ! -f "${VPFP_PATH}" || -L "${VPFP_PATH}" ]]; then echo "Missing/nonregular input ${VPFP_PATH}" >&2; exit 2; fi
done
if [[ ! -x "${VPFP_PYTHON}" ]]; then echo "Explicit venv is unavailable." >&2; exit 2; fi
VPFP_SOURCE_EXIT=0
VPFP_BINDING_EXIT=2
if VPFP_SHA_LINE="$(sha256sum -- "${VPFP_SELECTION}")"; then
  if [[ "${VPFP_SHA_LINE%% *}" != "${VPFP_SELECTION_SHA}" ]]; then echo "Frozen weak representation selection changed." >&2; exit 2; fi
else
  VPFP_BINDING_EXIT=$?
  exit "${VPFP_BINDING_EXIT}"
fi
if ! module load scipy-stack pytest; then
  echo "Requested modules unavailable; using the explicit existing venv without installing packages." >&2
fi
export PYTHONPATH="${VPFP_CHECKOUT}/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-weak-compare-pycache"
# Validate the pinned review and its immutable real preflight before numerics.
"${VPFP_PYTHON}" -c 'import hashlib,json,sys; from pathlib import Path; p=json.load(open(sys.argv[1])); assert p["schema"] == "vpfp_moving_frame_representation_selection_v1" and p["mechanics_passed"] is True and p["selected_mode"] == 6 and p["selected_fit_grid"] == 33; q=Path(p["preflight_result"]); assert q.is_file() and not q.is_symlink(); r=json.loads(q.read_text()); assert r["numerical_mechanics_passed"] is True and r["source_manifest_sha256"] == p["corpus_manifest_sha256"]' "${VPFP_SELECTION}" || { VPFP_BINDING_EXIT=$?; exit "${VPFP_BINDING_EXIT}"; }
VPFP_BINDING_EXIT=0
VPFP_GPU_EXIT=0
"${VPFP_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))' \
  > "${VPFP_GPU_LOG}" 2>&1 || VPFP_GPU_EXIT=$?
if [[ "${VPFP_GPU_EXIT}" != 0 ]]; then exit "${VPFP_GPU_EXIT}"; fi
VPFP_TESTS=(tests/test_vpfp_coherent_diffusion.py tests/test_vpfp_weak_coherent_driver.py
  tests/test_vpfp_regime_fidelity_coverage.py tests/test_vpfp_regime_fidelity_coverage_cli.py
  tests/test_vpfp_campaign_launchers.py)
VPFP_DISCOVERED_TESTS="$(rg --files tests -g 'test_vpfp_*moving_frame*.py' | sort)"
while IFS= read -r VPFP_TEST; do
  if [[ -n "${VPFP_TEST}" ]]; then VPFP_TESTS+=("${VPFP_TEST}"); fi
done <<< "${VPFP_DISCOVERED_TESTS}"
for VPFP_TEST in tests/test_vpfp_moving_frame.py tests/test_vpfp_moving_frame_audit.py \
    tests/test_vpfp_moving_frame_backend.py "${VPFP_TESTS[@]}"; do
  if [[ ! -f "${VPFP_TEST}" ]]; then echo "Missing registered mechanics test ${VPFP_TEST}" >&2; exit 2; fi
done
VPFP_TEST_EXIT=0
timeout --signal=TERM --kill-after=30s 900 "${VPFP_PYTHON}" -m pytest -q -p no:cacheprovider \
  --basetemp "${SLURM_TMPDIR}/vpfp-weak-compare-tests" "${VPFP_TESTS[@]}" \
  > "${VPFP_TEST_LOG}" 2>&1 || VPFP_TEST_EXIT=$?
if [[ "${VPFP_TEST_EXIT}" != 0 ]]; then exit "${VPFP_TEST_EXIT}"; fi
VPFP_RUN_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_weak_coherent.py \
  --config "${VPFP_CHECKOUT}/configs/vpfp_unified_live_weak_collision.json" --source "${VPFP_SOURCE}" \
  --representation-selection "${VPFP_SELECTION}" --output "${VPFP_OUTPUT}" --device cuda \
  --replicas 2 --steps 20000 --batch-size 4 --generated-answers 128 --sample-steps 64 \
  --train-seconds 7200 --validate-every 1000 --width 256 --depth 4 --learning-rate 0.0002 --seed 1520260906 \
  > "${VPFP_RUN_LOG}" 2>&1 || VPFP_RUN_EXIT=$?
if [[ "${VPFP_RUN_EXIT}" != 0 ]]; then exit "${VPFP_RUN_EXIT}"; fi
# Exactly one reference extraction serves W0 and both views of each same-draw
# paired_AB. The audit explicitly decodes A from m/z, never from old full z.
VPFP_AUDIT_EXIT=0
"${VPFP_PYTHON}" scripts/run_vpfp_regime_fidelity_coverage.py --source "${VPFP_SOURCE}" \
  --generated-dir "${VPFP_OUTPUT}/seed-0/W0" --generated-dir "${VPFP_OUTPUT}/seed-0/paired_AB" \
  --generated-dir "${VPFP_OUTPUT}/seed-1/W0" --generated-dir "${VPFP_OUTPUT}/seed-1/paired_AB" \
  --output "${VPFP_AUDIT_OUTPUT}" --device cuda --sample-count 128 --bootstrap-repeats 49 \
  > "${VPFP_AUDIT_LOG}" 2>&1 || VPFP_AUDIT_EXIT=$?
# Independent interfaces still run after an audit or the other seed fails.
# A runtime-produced SHA must agree with the frozen training receipt, not just
# with whichever bytes happen to be present when the smoke starts.
for VPFP_REPLICA in 0 1; do
  VPFP_MANIFEST="${VPFP_OUTPUT}/seed-${VPFP_REPLICA}/native-moving-frame-model.json"
  if VPFP_MANIFEST_SHA="$("${VPFP_PYTHON}" -c 'import hashlib,json,sys; from pathlib import Path; root=Path(sys.argv[1]); i=int(sys.argv[2]); p=json.loads((root/"result.json").read_text()); assert p["execution_passed"] is True and len(p["training"]) == 2; item=p["training"][i]["native_A_asset"]; path=root/f"seed-{i}"/"native-moving-frame-model.json"; assert Path(item["manifest"]) == path and path.is_file() and not path.is_symlink(); print(item.get("manifest_sha256", ""))' "${VPFP_OUTPUT}" "${VPFP_REPLICA}" 2> "${VPFP_SMOKE_BIND_LOGS[VPFP_REPLICA]}")"; then
    VPFP_MANIFEST_SHAS[VPFP_REPLICA]="${VPFP_MANIFEST_SHA}"
    VPFP_SMOKE_EXITS[VPFP_REPLICA]=0
    "${VPFP_PYTHON}" scripts/run_vpfp_moving_frame_smoke.py --manifest "${VPFP_MANIFEST}" \
      --manifest-sha256 "${VPFP_MANIFEST_SHA}" --output "${VPFP_SMOKE_OUTPUTS[VPFP_REPLICA]}" --device cuda \
      > "${VPFP_SMOKE_LOGS[VPFP_REPLICA]}" 2>&1 || VPFP_SMOKE_EXITS[VPFP_REPLICA]=$?
  else
    VPFP_SMOKE_EXITS[VPFP_REPLICA]=$?
  fi
done
# W0 is the old native full-value baseline, NOT the B view of conditional z.
# Reuse its unchanged macro-only smoke, including the existing perturbation
# mechanics; do not enable online generation, guidance or any training here.
for VPFP_REPLICA in 0 1; do
  VPFP_W0_CHECKPOINT="${VPFP_OUTPUT}/seed-${VPFP_REPLICA}/W0/checkpoints/fixed-step-020000.pt"
  if VPFP_W0_SHA="$("${VPFP_PYTHON}" -c 'import hashlib,json,sys; from pathlib import Path; root=Path(sys.argv[1]); i=int(sys.argv[2]); p=json.loads((root/"result.json").read_text()); assert p["execution_passed"] is True and len(p["training"]) == 2; item=p["training"][i]["W0"]; path=root/f"seed-{i}"/"W0"/"checkpoints"/"fixed-step-020000.pt"; assert Path(item["checkpoint"]) == path and path.is_file() and not path.is_symlink(); print(item.get("checkpoint_sha256", ""))' "${VPFP_OUTPUT}" "${VPFP_REPLICA}" 2> "${VPFP_W0_BIND_LOGS[VPFP_REPLICA]}")"; then
    VPFP_W0_SHAS[VPFP_REPLICA]="${VPFP_W0_SHA}"
    VPFP_W0_EXITS[VPFP_REPLICA]=0
    "${VPFP_PYTHON}" scripts/run_plasma_canonical_smoke.py --physics vpfp \
      --config "${VPFP_CHECKOUT}/configs/vpfp_unified_live_weak_collision.json" \
      --diffusion-checkpoint "${VPFP_W0_CHECKPOINT}" --diffusion-checkpoint-sha256 "${VPFP_W0_SHA}" \
      --output "${VPFP_W0_OUTPUTS[VPFP_REPLICA]}" --device cuda \
      > "${VPFP_W0_LOGS[VPFP_REPLICA]}" 2>&1 || VPFP_W0_EXITS[VPFP_REPLICA]=$?
  else
    VPFP_W0_EXITS[VPFP_REPLICA]=$?
  fi
done
if [[ "${VPFP_AUDIT_EXIT}" != 0 ]]; then exit "${VPFP_AUDIT_EXIT}"; fi
if [[ "${VPFP_SMOKE_EXITS[0]}" != 0 ]]; then exit "${VPFP_SMOKE_EXITS[0]}"; fi
if [[ "${VPFP_SMOKE_EXITS[1]}" != 0 ]]; then exit "${VPFP_SMOKE_EXITS[1]}"; fi
if [[ "${VPFP_W0_EXITS[0]}" != 0 ]]; then exit "${VPFP_W0_EXITS[0]}"; fi
exit "${VPFP_W0_EXITS[1]}"
