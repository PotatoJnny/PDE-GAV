#!/usr/bin/env bash
# One explicit phase; the parent submits jobs and owns the two-GPU limit.
# Requires the site variables PDE_GAV_WORKTREES_ROOT, PDE_GAV_RUN_ROOT, PDE_GAV_VENV, PDE_GAV_PYTHON (see docs/SITE_CONFIGURATION.md).
set -euo pipefail
umask 077
if [[ "$#" != 1 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || -z "${SLURM_TMPDIR:-}" ]]; then exit 2; fi
VPFP_PHASE=$1
case "${VPFP_PHASE}" in cpu|preflight|train) ;; *) exit 2 ;; esac
VPFP_CHECKOUT=${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-portable-guidance-20260908
VPFP_CONFIG="${VPFP_STRONG_GUIDANCE_CONFIG:-${VPFP_CHECKOUT}/configs/vpfp_strong_solver_guidance_20260908.json}"
VPFP_OUTPUT="${PDE_GAV_RUN_ROOT}/vpfp-strong-solver-guidance-${VPFP_PHASE}-${SLURM_JOB_ID}"
VPFP_STARTED=$SECONDS VPFP_STAGE_EXIT=not_run VPFP_REVISION=unverified VPFP_MODULES=not_run
VPFP_GPU=1 VPFP_CAP=21600
if [[ "${VPFP_PHASE}" == cpu ]]; then VPFP_GPU=0; VPFP_CAP=1800; fi
if [[ "${VPFP_PHASE}" == preflight ]]; then VPFP_CAP=3600; fi
if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]; then exit 2; fi
mkdir -- "${VPFP_OUTPUT}"
vpfp_finish() {
  local code=$?
  trap - EXIT
  printf 'stage\tvalue\nsource_revision\t%s\nphase\t%s\nexperiment\t%s\nlaunch\t%s\nwall_seconds\t%s\nrequested_gpus\t%s\nmodules\t%s\nscientific_admission\tfalse\n' \
    "${VPFP_REVISION}" "${VPFP_PHASE}" "${VPFP_STAGE_EXIT}" "${code}" "$((SECONDS-VPFP_STARTED))" "${VPFP_GPU}" "${VPFP_MODULES}" \
    > "${VPFP_OUTPUT}/launch.tsv.partial"
  mv -- "${VPFP_OUTPUT}/launch.tsv.partial" "${VPFP_OUTPUT}/launch.tsv"
  exit "${code}"
}
trap vpfp_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "${VPFP_CHECKOUT}"
if [[ "${VPFP_CONFIG}" != /* || ! -f "${VPFP_CONFIG}" || -L "${VPFP_CONFIG}" ]]; then
  echo "Explicit guidance config must be an absolute regular file." >&2; exit 2
fi
VPFP_REVISION="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${VPFP_REVISION}" =~ ^[0-9a-f]{40}$ || -n "$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)" ]]; then
  echo "Freeze source before scheduled computation." >&2; exit 2
fi
# Established cluster fallback: no installation or environment replacement.
set +eu
set +o pipefail
source /etc/profile
VPFP_PROFILE_STATUS=$?
set -euo pipefail
printf 'system_profile_return=%s\n' "${VPFP_PROFILE_STATUS}"
if module load StdEnv/2023 python/3.12 arrow/23.0.1 scipy-stack pytest apptainer/1.4.5; then
  VPFP_MODULES=standard
else
  VPFP_MODULES=unavailable_existing_venv
  echo "Cluster modules unavailable; using the existing unchanged venv." >&2
fi
source "${PDE_GAV_VENV}/bin/activate"
test -n "${VIRTUAL_ENV:-}"
case "$(command -v python)" in "${VIRTUAL_ENV}/"*) ;; *) exit 2 ;; esac
VPFP_PYTHON=${PDE_GAV_PYTHON}
export PYTHONPATH="${VPFP_CHECKOUT}/src:${VPFP_CHECKOUT}/scripts" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Embedded prerequisite guards use assertions; inherited optimization must not disable them.
unset PYTHONOPTIMIZE
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/vpfp-strong-guidance-pycache"
VPFP_TESTS=(tests/test_vpfp_frozen_solver_reward.py tests/test_vpfp_strong_guidance_rollouts.py
  tests/test_vpfp_solver_error_guidance.py tests/test_vpfp_strong_solver_guidance_round.py
  tests/test_vpfp_initial_conditioned_diffusion.py tests/test_vpfp_known_initial.py
  tests/test_vpfp_training_resume.py tests/test_vpfp_training_snapshots.py
  tests/test_vpfp_strong_solver_guidance_launch.py tests/test_vpfp_portable_initial_conditioned.py
  tests/test_vpfp_strong_portable_binding.py)
if [[ "${VPFP_PHASE}" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=""
  VPFP_STAGE_EXIT=0
  timeout --signal=TERM --kill-after=15s 1500 "${VPFP_PYTHON}" -m pytest -o addopts= -q -p no:cacheprovider \
    --basetemp "${SLURM_TMPDIR}/vpfp-strong-guidance-cpu" "${VPFP_TESTS[@]}" \
    > "${VPFP_OUTPUT}/mechanics-tests.log" 2>&1 || VPFP_STAGE_EXIT=$?
else
  if [[ "${VPFP_STRONG_GUIDANCE_CPU_RECEIPT:-}" != /* ]]; then
    echo "Explicit same-source successful CPU receipt required." >&2; exit 2
  fi
  # CPU_RECEIPT_GUARD: also require an actual one-GPU runtime, not a label.
  "${VPFP_PYTHON}" -c 'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.is_file() and not p.is_symlink(); rows=[l.split("\t") for l in p.read_text().splitlines()]; assert all(len(r)==2 for r in rows) and len(dict(rows))==len(rows); d=dict(rows); assert d["source_revision"]==sys.argv[3] and d["phase"]=="cpu" and d["experiment"]=="0" and d["launch"]=="0" and d["requested_gpus"]=="0"; import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print("same-source CPU and one GPU verified",torch.__version__,torch.version.cuda)' \
    "${VPFP_STRONG_GUIDANCE_CPU_RECEIPT}" "${VPFP_STRONG_GUIDANCE_CPU_SHA256:-}" "${VPFP_REVISION}" \
    > "${VPFP_OUTPUT}/binding-gpu-gate.log" 2>&1
  cp -- "${VPFP_STRONG_GUIDANCE_CPU_RECEIPT}" "${VPFP_OUTPUT}/cpu-gate-receipt.tsv"
  if [[ "${VPFP_PHASE}" == train ]]; then
    if [[ "${VPFP_STRONG_GUIDANCE_PIPELINE:-}" != /* || ! "${VPFP_STRONG_GUIDANCE_PREFLIGHT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Explicit successful preflight pipeline and result SHA required." >&2; exit 2
    fi
    if [[ -n "${VPFP_STRONG_GUIDANCE_CONTINUATION_SHA256:-}" && ! "${VPFP_STRONG_GUIDANCE_CONTINUATION_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Continuation requires the exact current partial result SHA." >&2; exit 2
    fi
    # PREFLIGHT_RECEIPT_GUARD: immutable proof plus explicitly bound current state.
    "${VPFP_PYTHON}" -c 'import hashlib,json,pathlib,sys
def regular(path):
    path=pathlib.Path(path)
    assert path.is_file() and not path.is_symlink()
    return path
def file_sha(path):
    h=hashlib.sha256()
    with regular(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""): h.update(chunk)
    return h.hexdigest()
def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
pipeline=pathlib.Path(sys.argv[1])
assert not pipeline.is_symlink() and pipeline.is_dir()
pipeline=pipeline.resolve()
assert pipeline.is_relative_to(pathlib.Path("${PDE_GAV_RUN_ROOT}")) and pipeline.name=="pipeline"
preflight_sha,revision,destination,continuation_sha,config_path=sys.argv[2:]
destination=pathlib.Path(destination)
assert not destination.exists() and not destination.is_symlink()
current_payload=regular(pipeline/"result.json").read_bytes()
current_sha=hashlib.sha256(current_payload).hexdigest()
current=json.loads(current_payload)
proof_path=pipeline/"launch-preflight-result.json"
if continuation_sha:
    proof_payload=regular(proof_path).read_bytes()
else:
    proof_payload=current_payload
    if proof_path.exists() or proof_path.is_symlink(): assert regular(proof_path).read_bytes()==proof_payload
proof=json.loads(proof_payload)
assert proof["schema"]=="vpfp_frozen_ic40_actual_solver_guidance_round_v1" and proof["source_revision"]==revision
assert proof["requested_stage_execution_passed"] is True and proof["preflight_only"] is True and proof["frozen_assets_final_verified"] is True
assert proof["stage"]=="preflight_complete" and proof["execution_passed"] is False and proof["full_round_completed"] is False and proof["training_started"] is False
assert type(proof["preflight_finite_labels"]) is int and proof["preflight_finite_labels"]>0
history=proof["invocations"]
assert isinstance(history,list) and history and all(x["stage"]=="preflight" for x in history)
for key in ("schema","source_revision","config_sha256","frozen_plan_sha256"):
    assert current[key]==proof[key]
if continuation_sha:
    assert current["execution_passed"] is False and current["requested_stage_execution_passed"] is False and current["full_round_completed"] is False
    assert current["invocations"][:len(history)]==history
    following=current["invocations"][len(history):]
    assert following and all(x["stage"]=="all" and x["resume"] is True for x in following)
config=json.loads(regular(config_path).read_text())
plan_path=pipeline/"frozen-plan.json"
plan=json.loads(plan_path.read_text())
assert plan["schema"]==proof["schema"] and plan["source_revision"]==revision and plan["config"]==config
assets=plan["assets"]
assert assets["config_sha256"]==proof["config_sha256"]
assert pathlib.Path(assets["manifest_path"]).resolve()==pipeline/"frozen-assets/manifest.json"
assert json.loads(pathlib.Path(assets["manifest_path"]).read_text())=={k:v for k,v in assets.items() if k not in ("manifest_path","manifest_sha256")}
expected_roles={"base_checkpoint","retained_control_checkpoint","base_decoder","expanded_decoder","solver","native_config"}
if "portable_codec" in config:
    from vpfp_strong_portable_binding import validate_portable_proof
    expected_roles.update({"portable_codec","portable_codec_proof"})
    portable_binding=validate_portable_proof(config,frozen_assets=assets["assets"])
    assert proof["portable_codec_binding"]==current["portable_codec_binding"]==portable_binding
assert set(assets["assets"])==expected_roles
for asset in assets["assets"].values():
    assert pathlib.Path(asset["path"]).resolve().parent==pipeline/"frozen-assets"
    regular(asset["path"]); regular(asset["source_path"])
if not proof_path.exists():
    with proof_path.open("xb") as stream: stream.write(proof_payload)
    proof_path.chmod(0o444)
with destination.open("xb") as stream: stream.write(proof_payload)
state_copy=destination.with_name("continuation-input-result.json")
if continuation_sha:
    with state_copy.open("xb") as stream: stream.write(current_payload)
receipt={"schema":"vpfp_strong_guidance_launch_binding_v1","mode":"continuation" if continuation_sha else "promotion","pipeline":str(pipeline),"preflight_sha256":preflight_sha,"current_result_sha256":current_sha,"source_revision":revision,"config_sha256":proof["config_sha256"],"frozen_plan_sha256":proof["frozen_plan_sha256"],"partial_is_preflight_pass":False}
with destination.with_name("train-input-binding.json").open("x") as stream: json.dump(receipt,stream,sort_keys=True,indent=2)
print("verified",receipt["mode"],pipeline)' \
      "${VPFP_STRONG_GUIDANCE_PIPELINE}" "${VPFP_STRONG_GUIDANCE_PREFLIGHT_SHA256}" "${VPFP_REVISION}" \
      "${VPFP_OUTPUT}/preflight-result.json" "${VPFP_STRONG_GUIDANCE_CONTINUATION_SHA256:-}" \
      "${VPFP_CONFIG}" > "${VPFP_OUTPUT}/preflight-binding.log" 2>&1
    VPFP_PIPELINE="${VPFP_STRONG_GUIDANCE_PIPELINE}"
  else
    VPFP_PIPELINE="${VPFP_OUTPUT}/pipeline"
  fi
  VPFP_STAGE_EXIT=0
  timeout --signal=TERM --kill-after=15s 600 "${VPFP_PYTHON}" -m pytest -o addopts= -q -p no:cacheprovider \
    --basetemp "${SLURM_TMPDIR}/vpfp-strong-guidance-cuda" "${VPFP_TESTS[@]}" \
    > "${VPFP_OUTPUT}/CUDA-mechanics.log" 2>&1 || VPFP_STAGE_EXIT=$?
  if [[ "${VPFP_STAGE_EXIT}" != 0 ]]; then exit "${VPFP_STAGE_EXIT}"; fi
  VPFP_REMAINING=$((VPFP_CAP-(SECONDS-VPFP_STARTED)-90))
  if ((VPFP_REMAINING<=60)); then VPFP_STAGE_EXIT=124; exit 124; fi
  VPFP_ARGS=(--config "${VPFP_CONFIG}" --output "${VPFP_PIPELINE}" --wall-seconds "$((VPFP_REMAINING-30))")
  if [[ "${VPFP_PHASE}" == preflight ]]; then VPFP_ARGS+=(--stage preflight); else VPFP_ARGS+=(--stage all --resume); fi
  timeout --signal=TERM --kill-after=30s "${VPFP_REMAINING}" "${VPFP_PYTHON}" scripts/run_vpfp_strong_solver_guidance_round.py \
    "${VPFP_ARGS[@]}" > "${VPFP_OUTPUT}/pipeline.log" 2>&1 || VPFP_STAGE_EXIT=$?
  printf 'pipeline\t%s\n' "${VPFP_PIPELINE}" > "${VPFP_OUTPUT}/pipeline-location.tsv"
fi
if [[ "$(git rev-parse HEAD)" != "${VPFP_REVISION}" || -n "$(git status --porcelain --untracked-files=all -- src scripts configs tests legacy/vpfp)" ]]; then
  echo "Source changed during computation." >&2; VPFP_STAGE_EXIT=2
fi
exit "${VPFP_STAGE_EXIT}"
