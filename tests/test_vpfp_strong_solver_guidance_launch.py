"""Static launch and actual embedded-receipt guards; no scheduler/solver calls."""
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT/"scripts/run_vpfp_strong_solver_guidance_job.sh"


def _guard(marker):
    text = WRAPPER.read_text().split(marker, 1)[1]
    match = re.search(r'"\$\{VPFP_PYTHON\}" -c \'([^\']+)\'', text)
    assert match is not None
    return match.group(1)


@pytest.mark.parametrize("phase,gres,time", [
    ("cpu", "gpu:0", "00:30:00"),
    ("preflight", "gpu:l40s:1", "01:00:00"),
    ("train", "gpu:l40s:1", "06:00:00"),
])
def test_three_explicit_resources_and_bash_syntax(phase, gres, time):
    path = ROOT/f"scripts/run_vpfp_strong_solver_guidance_{phase}.slurm"
    text = path.read_text()
    options = dict(re.findall(r"^#SBATCH --([^=\s]+)=(.+)$", text, re.M))
    assert options["gres"] == gres and options["time"] == time
    assert options["cpus-per-task"] == "8" and options["mem"] == "64G"
    assert "account" not in options and options["qos"] == "normal"  # site options come from sbatch
    assert options["partition"] == "gpubase_l40s_b3"
    assert "exclude" not in options and "chdir" not in options
    assert f"scripts/{WRAPPER.name}" in text
    assert f"run_vpfp_strong_solver_guidance_job.sh {phase}" in text
    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)


def test_wrapper_no_child_submission_no_overwrite_and_exact_stage_flags():
    text = WRAPPER.read_text()
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True, capture_output=True)
    assert not re.search(r"^\s*(sbatch|scancel|scontrol|srun)\b", text, re.M)
    assert 'if [[ -e "${VPFP_OUTPUT}" || -L "${VPFP_OUTPUT}" ]]' in text
    assert "--untracked-files=all -- src scripts configs tests legacy/vpfp" in text
    assert 'VPFP_ARGS+=(--stage preflight)' in text
    assert 'VPFP_ARGS+=(--stage all --resume)' in text
    assert 'VPFP_PIPELINE="${VPFP_STRONG_GUIDANCE_PIPELINE}"' in text
    assert 'VPFP_PIPELINE="${VPFP_OUTPUT}/pipeline"' in text
    assert 'VPFP_REMAINING=$((VPFP_CAP-(SECONDS-VPFP_STARTED)-90))' in text
    assert '"${VPFP_OUTPUT}/launch.tsv"' in text and 'requested_gpus' in text
    assert "source /etc/profile" in text and "scipy-stack pytest" in text
    assert "${PDE_GAV_VENV}/bin/activate" in text
    assert "pip install" not in text and "git reset" not in text
    assert "unset PYTHONOPTIMIZE" in text
    assert "VPFP_STRONG_GUIDANCE_CONTINUATION_SHA256" in text
    assert "launch-preflight-result.json" in text
    for name in ("frozen_solver_reward", "strong_guidance_rollouts", "solver_error_guidance",
                 "strong_solver_guidance_round", "initial_conditioned_diffusion", "training_resume"):
        assert f"tests/test_vpfp_{name}.py" in text
    # Actual CUDA availability is checked before a suite containing CUDA tests.
    assert "torch.cuda.is_available() and torch.cuda.device_count()==1" in text
    assert '"${VPFP_TESTS[@]}"' in text.split("CUDA-mechanics.log")[0]


def test_every_scheduled_test_file_exists_in_frozen_checkout():
    block = WRAPPER.read_text().split("VPFP_TESTS=(", 1)[1].split(")", 1)[0]
    paths = re.findall(r"tests/test_[a-z0-9_]+\.py", block)
    expected = {f"tests/test_vpfp_{name}.py" for name in (
        "frozen_solver_reward", "strong_guidance_rollouts", "solver_error_guidance",
        "strong_solver_guidance_round", "initial_conditioned_diffusion", "known_initial",
        "training_resume", "training_snapshots", "strong_solver_guidance_launch",
        "portable_initial_conditioned", "strong_portable_binding")}
    assert len(paths) == 11 and set(paths) == expected
    assert all((ROOT/path).is_file() for path in paths)
    assert "tests/test_vpfp_known_initial.py" in paths


def test_explicit_config_path_is_shared_by_guard_and_driver_with_legacy_default():
    text = WRAPPER.read_text()
    assert 'VPFP_CONFIG="${VPFP_STRONG_GUIDANCE_CONFIG:-${VPFP_CHECKOUT}/configs/vpfp_strong_solver_guidance_20260908.json}"' in text
    assert '"${VPFP_CONFIG}" > "${VPFP_OUTPUT}/preflight-binding.log"' in text
    assert 'VPFP_ARGS=(--config "${VPFP_CONFIG}"' in text


def _cpu_receipt(path, **changes):
    rows = {"stage": "value", "source_revision": "a"*40, "phase": "cpu",
            "experiment": "0", "launch": "0", "requested_gpus": "0"}
    rows.update(changes)
    path.write_text("".join(f"{key}\t{value}\n" for key, value in rows.items()))
    return sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("change", [None, "revision", "failed", "phase", "gpu", "hash", "duplicate", "no_cuda"])
def test_actual_cpu_receipt_guard_rejects_wrong_or_failed_prerequisites(tmp_path, monkeypatch, change):
    path = tmp_path/"cpu.tsv"
    kwargs = {}
    if change == "revision": kwargs["source_revision"] = "b"*40
    if change == "failed": kwargs["experiment"] = "1"
    if change == "phase": kwargs["phase"] = "train"
    if change == "gpu": kwargs["requested_gpus"] = "1"
    digest = _cpu_receipt(path, **kwargs)
    if change == "duplicate":
        path.write_text(path.read_text()+"launch\t0\n")
        digest = sha256(path.read_bytes()).hexdigest()
    if change == "hash": digest = "0"*64
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: change != "no_cuda", device_count=lambda: 1),
        __version__="synthetic", version=SimpleNamespace(cuda="synthetic")))
    monkeypatch.setattr(sys, "argv", ["guard", str(path), digest, "a"*40])
    if change in (None, "hash"):
        exec(_guard("CPU_RECEIPT_GUARD"), {})
    else:
        with pytest.raises(AssertionError): exec(_guard("CPU_RECEIPT_GUARD"), {})


def _preflight():
    return {"schema": "vpfp_frozen_ic40_actual_solver_guidance_round_v1",
            "source_revision": "a"*40, "requested_stage_execution_passed": True,
            "preflight_only": True, "frozen_assets_final_verified": True,
            "preflight_finite_labels": 1, "stage": "preflight_complete",
            "execution_passed": False, "full_round_completed": False, "training_started": False,
            "invocations": [{"job_id": "11", "stage": "preflight", "resume": False}]}


def _binding_fixture(tmp_path):
    """Real file/SHA binding with tiny synthetic bytes; no native model loading."""
    pipeline = tmp_path/"pipeline"
    asset_dir = pipeline/"frozen-assets"
    asset_dir.mkdir(parents=True)
    config = {"synthetic_fixture": True, "unchanged_config": 1}
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(config))
    config_sha = sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    entries = {}
    for role in ("base_checkpoint", "retained_control_checkpoint", "base_decoder",
                 "expanded_decoder", "solver", "native_config"):
        source, copied = tmp_path/f"source-{role}", asset_dir/role
        data = ("synthetic bytes for "+role).encode()
        source.write_bytes(data)
        copied.write_bytes(data)
        entries[role] = {"source_path": str(source), "path": str(copied), "sha256": sha256(data).hexdigest()}
    manifest = {"config_sha256": config_sha, "assets": entries}
    manifest_path = asset_dir/"manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assets = {**manifest, "manifest_path": str(manifest_path),
              "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest()}
    plan_path = pipeline/"frozen-plan.json"
    plan_path.write_text(json.dumps({"schema": _preflight()["schema"], "source_revision": "a"*40,
                                     "config": config, "assets": assets}))
    proof = {**_preflight(), "config_sha256": config_sha,
             "frozen_plan_sha256": sha256(plan_path.read_bytes()).hexdigest()}
    return pipeline, config_path, proof


def _mock_scratch_scope(monkeypatch):
    # Only declared scratch ancestry is mocked, never receipt/hash/asset guards.
    monkeypatch.setattr(Path, "is_relative_to", lambda self, other: str(other) == "${PDE_GAV_RUN_ROOT}")


def _guard_arguments(monkeypatch, pipeline, config_path, destination, preflight_sha, continuation_sha=""):
    monkeypatch.setattr(sys, "argv", ["guard", str(pipeline), preflight_sha, "a"*40,
                                      str(destination), continuation_sha, str(config_path)])


def _portable_binding_fixture(tmp_path):
    from tests.test_vpfp_strong_portable_binding import proof_fixture
    from vpfp_strong_portable_binding import validate_portable_proof, portable_roles
    pipeline, config_path, proof = _binding_fixture(tmp_path)
    plan_path = pipeline/"frozen-plan.json"
    plan = json.loads(plan_path.read_text())
    config = json.loads(config_path.read_text())
    base = plan["assets"]["assets"]["base_checkpoint"]
    config.update(base_checkpoint=base["source_path"], base_checkpoint_sha256=base["sha256"])
    config, _ = proof_fixture(tmp_path, config=config)
    for role, (source, digest, name) in portable_roles(config).items():
        copied = pipeline/"frozen-assets"/name
        copied.write_bytes(Path(source).read_bytes())
        plan["assets"]["assets"][role] = {"source_path": source, "path": str(copied), "sha256": digest}
    config_path.write_text(json.dumps(config))
    config_sha = sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan["config"] = config
    plan["assets"]["config_sha256"] = config_sha
    manifest_path = Path(plan["assets"]["manifest_path"])
    manifest_path.write_text(json.dumps({k:v for k,v in plan["assets"].items() if k not in ("manifest_path", "manifest_sha256")}))
    plan["assets"]["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    plan_path.write_text(json.dumps(plan))
    proof.update(config_sha256=config_sha, frozen_plan_sha256=sha256(plan_path.read_bytes()).hexdigest(),
        portable_codec_binding=validate_portable_proof(config, frozen_assets=plan["assets"]["assets"]))
    return pipeline, config_path, proof, plan


@pytest.mark.parametrize("change", [None, "codec_copy", "codec_source", "proof_copy", "missing_binding", "wrong_normalizer"])
def test_actual_portable_preflight_promotion_checks_extra_assets_and_proof(tmp_path, monkeypatch, change):
    pipeline, config_path, proof, plan = _portable_binding_fixture(tmp_path)
    if change == "codec_copy": Path(plan["assets"]["assets"]["portable_codec"]["path"]).write_bytes(b"changed")
    if change == "codec_source": Path(plan["assets"]["assets"]["portable_codec"]["source_path"]).write_bytes(b"changed")
    if change == "proof_copy": Path(plan["assets"]["assets"]["portable_codec_proof"]["path"]).write_text("{}")
    if change == "missing_binding": proof.pop("portable_codec_binding")
    if change == "wrong_normalizer": proof["portable_codec_binding"]["normalizer_sha256"] = "0"*64
    source = pipeline/"result.json"
    source.write_text(json.dumps(proof))
    digest = sha256(source.read_bytes()).hexdigest()
    destination = tmp_path/"preflight-copy.json"
    _mock_scratch_scope(monkeypatch)
    _guard_arguments(monkeypatch, pipeline, config_path, destination, digest)
    if change in (None, "codec_copy", "codec_source"):
        exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
        assert destination.read_bytes() == source.read_bytes()
    else:
        with pytest.raises((AssertionError, ValueError, KeyError)):
            exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
        assert not destination.exists()


@pytest.mark.parametrize("changed", [False, True])
def test_portable_partial_resume_cannot_reuse_labels_under_another_codec_identity(tmp_path, monkeypatch, changed):
    pipeline, config_path, proof, _ = _portable_binding_fixture(tmp_path)
    source = pipeline/"result.json"
    source.write_text(json.dumps(proof))
    preflight_sha = sha256(source.read_bytes()).hexdigest()
    first_dir = tmp_path/"first-launch"
    first_dir.mkdir()
    first = first_dir/"first-copy.json"
    _mock_scratch_scope(monkeypatch)
    _guard_arguments(monkeypatch, pipeline, config_path, first, preflight_sha)
    exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
    current = json.loads(source.read_text())
    current.update(requested_stage_execution_passed=False, stage="guidance_training", training_started=True)
    current["invocations"].append({"stage":"all", "resume":True, "job_id":"12"})
    if changed: current["portable_codec_binding"]["codec_content_sha256"] = "0"*64
    source.write_text(json.dumps(current))
    resume_dir = tmp_path/"resume-launch"
    resume_dir.mkdir()
    _guard_arguments(monkeypatch, pipeline, config_path, resume_dir/"resume-copy.json", preflight_sha,
        sha256(source.read_bytes()).hexdigest())
    if changed:
        with pytest.raises(AssertionError): exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
    else:
        exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})


@pytest.mark.parametrize("change", [None, "schema", "revision", "requested", "only", "assets", "zero", "bool_count", "already_trained", "hash"])
def test_actual_preflight_guard_and_immutable_receipt_copy(tmp_path, monkeypatch, change):
    pipeline, config_path, report = _binding_fixture(tmp_path)
    updates = {"schema": ("schema", "unrelated"), "revision": ("source_revision", "b"*40),
        "requested": ("requested_stage_execution_passed", False), "only": ("preflight_only", False),
        "assets": ("frozen_assets_final_verified", False), "zero": ("preflight_finite_labels", 0),
        "bool_count": ("preflight_finite_labels", True), "already_trained": ("training_started", True)}
    if change in updates:
        key, value = updates[change]
        report[key] = value
    source = pipeline/"result.json"
    source.write_text(json.dumps(report))
    original = source.read_bytes()
    digest = sha256(original).hexdigest() if change != "hash" else "0"*64
    destination = tmp_path/"preflight-snapshot.json"
    _guard_arguments(monkeypatch, pipeline, config_path, destination, digest)
    # Only the /scratch location is mocked: execute the real receipt checks on
    # pytest temporary files rather than creating a campaign on shared scratch.
    _mock_scratch_scope(monkeypatch)
    if change in (None, "hash"):
        exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
        assert destination.read_bytes() == original and source.read_bytes() == original
        assert (pipeline/"launch-preflight-result.json").read_bytes() == original
        assert json.loads((tmp_path/"train-input-binding.json").read_text())["mode"] == "promotion"
        with pytest.raises(AssertionError): exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
    else:
        with pytest.raises(AssertionError): exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
        assert not destination.exists()


def test_preflight_scope_guard_requires_pipeline_under_declared_scratch(tmp_path, monkeypatch):
    pipeline, config_path, proof = _binding_fixture(tmp_path)
    source = pipeline/"result.json"
    source.write_text(json.dumps(proof))
    _guard_arguments(monkeypatch, pipeline, config_path, tmp_path/"copy.json", sha256(source.read_bytes()).hexdigest())
    with pytest.raises(AssertionError): exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})


@pytest.mark.parametrize("change", [None, "collecting_before_training", "wrong_current_sha", "wrong_proof_sha",
    "missing_proof", "no_all_invocation", "wrong_resume", "changed_prefix", "config_identity",
    "plan_identity", "plan_bytes", "config_bytes", "asset_bytes", "source_bytes", "manifest_bytes",
    "complete", "stage_passed", "pretend_first_promotion", "source_revision"])
def test_partial_continuation_requires_original_proof_current_sha_and_unchanged_assets(tmp_path, monkeypatch, change):
    pipeline, config_path, proof = _binding_fixture(tmp_path)
    source = pipeline/"result.json"
    source.write_text(json.dumps(proof))
    original = source.read_bytes()
    preflight_sha = sha256(original).hexdigest()
    first = tmp_path/"first-launch"
    first.mkdir()
    _mock_scratch_scope(monkeypatch)
    _guard_arguments(monkeypatch, pipeline, config_path, first/"preflight-result.json", preflight_sha)
    exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
    current = {**proof, "stage": "guidance_training", "requested_stage_execution_passed": False,
               "training_started": True, "failures": [{"type": "TimeoutError"}],
               "invocations": proof["invocations"]+[{"job_id": "12", "stage": "all", "resume": True}]}
    if change == "collecting_before_training":
        current.update(stage="fit/strength-0/fit-0002", training_started=False)
    if change == "no_all_invocation": current["invocations"] = proof["invocations"]
    if change == "wrong_resume": current["invocations"][-1]["resume"] = False
    if change == "changed_prefix": current["invocations"] = [{"stage": "preflight", "job_id": "another"}, current["invocations"][-1]]
    if change == "config_identity": current["config_sha256"] = "0"*64
    if change == "plan_identity": current["frozen_plan_sha256"] = "0"*64
    if change == "source_revision": current["source_revision"] = "b"*40
    if change == "complete": current["full_round_completed"] = True
    if change == "stage_passed": current["requested_stage_execution_passed"] = True
    if change == "plan_bytes": (pipeline/"frozen-plan.json").write_text("{}")
    if change == "config_bytes": config_path.write_text("{}")
    if change == "asset_bytes": (pipeline/"frozen-assets/base_checkpoint").write_bytes(b"changed")
    if change == "source_bytes": (tmp_path/"source-solver").write_bytes(b"changed")
    if change == "manifest_bytes": (pipeline/"frozen-assets/manifest.json").write_text("{}")
    if change == "missing_proof": (pipeline/"launch-preflight-result.json").rename(pipeline/"preserved-proof-for-test.json")
    source.write_text(json.dumps(current))
    partial_bytes = source.read_bytes()
    continuation_sha = sha256(partial_bytes).hexdigest()
    if change == "wrong_current_sha": continuation_sha = "0"*64
    if change == "wrong_proof_sha": preflight_sha = "0"*64
    if change == "pretend_first_promotion": continuation_sha = ""
    following = tmp_path/"next-launch"
    following.mkdir()
    destination = following/"preflight-result.json"
    _guard_arguments(monkeypatch, pipeline, config_path, destination, preflight_sha, continuation_sha)
    if change in (None, "collecting_before_training", "wrong_current_sha", "wrong_proof_sha", "asset_bytes", "source_bytes"):
        exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
        assert destination.read_bytes() == original
        assert (following/"continuation-input-result.json").read_bytes() == partial_bytes
        receipt = json.loads((following/"train-input-binding.json").read_text())
        assert receipt["mode"] == "continuation" and receipt["partial_is_preflight_pass"] is False
        assert receipt["preflight_sha256"] == preflight_sha
        assert receipt["current_result_sha256"] == sha256(partial_bytes).hexdigest()
        assert json.loads(source.read_text())["failures"] == [{"type": "TimeoutError"}]
    else:
        with pytest.raises((AssertionError, KeyError)): exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
        assert not destination.exists()
    assert source.read_bytes() == partial_bytes
    assert (first/"preflight-result.json").read_bytes() == original


def test_promotion_can_retry_before_any_all_invocation_without_replacing_proof(tmp_path, monkeypatch):
    pipeline, config_path, proof = _binding_fixture(tmp_path)
    source = pipeline/"result.json"
    source.write_text(json.dumps(proof))
    data = source.read_bytes()
    digest = sha256(data).hexdigest()
    _mock_scratch_scope(monkeypatch)
    for name in ("first", "retry"):
        output = tmp_path/name
        output.mkdir()
        _guard_arguments(monkeypatch, pipeline, config_path, output/"preflight-result.json", digest)
        exec(_guard("PREFLIGHT_RECEIPT_GUARD"), {})
    assert source.read_bytes() == data
    assert (pipeline/"launch-preflight-result.json").read_bytes() == data
