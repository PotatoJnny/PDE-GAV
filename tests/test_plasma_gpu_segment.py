"""CPU-only tests of frozen plasma deployment and safe allocation handoffs."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_plasma_gpu_segment as segment
from scripts import ns_slurm_continuation as continuation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def site_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(segment, "ROOT", source)
    monkeypatch.setattr(segment, "_git",
                        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "")
    entries = []
    for name in ("run_plasma_gpu_segment.py", "plasma_gpu_campaign.py", "ns_slurm_continuation.py"):
        path = source / "scripts" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# frozen source\n")
        entries.append({"path": str(path.relative_to(source)), "sha256": segment.file_sha(path)})
    manifest = write_json(tmp_path / "source-manifest.json", {"files": entries})
    config = write_json(tmp_path / "config.json", {"equation": "strong", "master_seed": 101})
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n")
    codex.chmod(0o700)
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    auth.chmod(0o600)
    wrapper = tmp_path / "run.slurm"
    wrapper.write_text("#!/bin/bash\nexit 0\n")
    site = {
        "schema": segment.SCHEMA, "source_root": str(source), "source_revision": "a" * 40,
        "source_manifest": str(manifest), "source_manifest_sha256": segment.file_sha(manifest),
        "config_path": str(config), "config_sha256": segment.file_sha(config),
        "run_root": str(tmp_path / "run"), "python": str(Path(segment.sys.executable)),
        "codex_binary": str(codex), "codex_sha256": segment.file_sha(codex),
        "auth_file": str(auth), "slurm_script": str(wrapper),
        "slurm_script_sha256": segment.file_sha(wrapper),
        "account": "${PDE_GAV_SLURM_ACCOUNT}", "partition": "gpubase_l40s_b2", "qos": "normal",
        "cpus": 24, "memory": "128G", "time": "12:00:00", "gpu_count": 4,
        "max_segments": 12, "shutdown_margin_seconds": 2400,
    }
    return write_json(tmp_path / "site.json", site), site


@pytest.fixture
def allocation(monkeypatch):
    monkeypatch.setattr(segment.socket, "gethostname", lambda: "kn001")
    for key in tuple(segment.os.environ):
        if key.startswith("NS_CONTINUATION_"):
            monkeypatch.delenv(key)
    for key, value in {"SLURM_JOB_ID": "100", "SLURM_CPUS_PER_TASK": "24",
                       "SLURM_JOB_GPUS": "0,1,2,3", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                       "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}.items():
        monkeypatch.setenv(key, value)


def run_mock(site_fixture, campaign, **kwargs):
    path, _ = site_fixture
    return segment.run(path, campaign_run=campaign, role_probe=lambda *a, **k: {"status": "passed"},
                       verify=segment.verify_site, remaining_seconds=lambda job: 40000, **kwargs)


def campaign_result(campaign, *, status="paused", progress=1, stage="before_development",
                    cleanup_complete=True, retry_safe=True):
    checkpoint = write_json(campaign / "checkpoints/checkpoint.json",
                            {"schema": "campaign_checkpoint_v1", "progress": {"round": progress}})
    return {"status": status, "stage": stage, "cleanup_complete": cleanup_complete,
            "retry_safe": retry_safe, "checkpoint_path": str(checkpoint)}


def test_site_is_read_only_and_binds_resources_source_config_and_runtime(site_fixture):
    path, site = site_fixture
    result, config, identity = segment.verify_site(path)
    assert result == site and config["equation"] == "strong"
    assert identity["resource_profile"] == "plasma_gpu_12h"
    assert identity["resources"]["cpus"] == 24
    assert identity["resources"]["gpu_count"] == 4
    assert not Path(site["run_root"]).exists()


@pytest.mark.parametrize("field,value", [
    ("gpu_count", 3), ("cpus", 32), ("memory", "160G"), ("time", "06:00:00"),
    ("max_segments", 13), ("shutdown_margin_seconds", 600), ("account", "other"),
    ("partition", "gpubase_l40s_b3"),
    ("partition", "gpubase_h100_b2"),
    ("partition", "gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3"),
])
def test_unapproved_resource_drift_rejected(site_fixture, field, value):
    path, site = site_fixture
    write_json(path, {**site, field: value})
    with pytest.raises(ValueError, match="resources"):
        segment.verify_site(path)


@pytest.mark.parametrize("field", ["config_path", "codex_binary", "source_manifest", "slurm_script"])
def test_changed_immutable_files_rejected(site_fixture, field):
    path, site = site_fixture
    Path(site[field]).write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        segment.verify_site(path)


def test_source_inventory_and_clean_revision_required(site_fixture, monkeypatch):
    path, site = site_fixture
    tracked = Path(site["source_root"]) / "scripts/plasma_gpu_campaign.py"
    tracked.write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        segment.verify_site(path)
    tracked.write_text("# frozen source\n")
    monkeypatch.setattr(segment, "_git", lambda *args: "a" * 40 if args[0] == "rev-parse" else " M file")
    with pytest.raises(ValueError, match="clean frozen"):
        segment.verify_site(path)


def test_manifest_cannot_omit_execution_entrypoint(site_fixture):
    path, site = site_fixture
    manifest = Path(site["source_manifest"])
    payload = json.loads(manifest.read_text())
    payload["files"].pop()
    write_json(manifest, payload)
    write_json(path, {**site, "source_manifest_sha256": segment.file_sha(manifest)})
    with pytest.raises(ValueError, match="omits"):
        segment.verify_site(path)


def test_completed_run_preserves_four_role_map_and_never_submits(site_fixture, allocation):
    path, site = site_fixture
    root = Path(site["run_root"])
    root.mkdir()
    (root / "slurm-100.out").write_text("scheduler header")
    calls = []

    def campaign(config_path, campaign_root, **options):
        assert segment.os.environ["CUDA_VISIBLE_DEVICES"] == "2,0,1"
        assert options["evaluation_device_map"] == {"2": "cuda:0", "0": "cuda:1", "1": "cuda:2"}
        assert options["worker_gpus"] == ("0", "1") and options["controller_gpu"] == "2"
        assert options["resume"] is False and options["allocation_seconds"] == 40000
        assert options["shutdown_margin_seconds"] == 2400
        assert options["binding_check"] is segment._binding_check
        return campaign_result(campaign_root, status="completed")

    result = run_mock(site_fixture, campaign, submitter=lambda *a, **k: calls.append(k))
    assert result["status"] == "completed" and calls == []
    assert continuation.read_state(root)["status"] == "completed"
    assert segment.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    with pytest.raises(FileExistsError):
        run_mock(site_fixture, campaign)


@pytest.mark.parametrize("change", [
    {"status": "failed"}, {"stage": "agent_round_interrupted"},
    {"cleanup_complete": False}, {"retry_safe": False},
])
def test_failure_never_restarts_an_agent_or_submits(site_fixture, allocation, change):
    _, site = site_fixture
    calls = []

    def campaign(config, root, **options):
        return {**campaign_result(root), **change}

    with pytest.raises(ValueError):
        run_mock(site_fixture, campaign, submitter=lambda *a, **k: calls.append(k))
    assert calls == []
    assert continuation.read_state(site["run_root"])["status"] == "failed"
    assert segment.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_exception_does_not_mask_failure_or_expand_cuda_visibility(site_fixture, allocation):
    _, site = site_fixture

    def campaign(*args, **options):
        raise RuntimeError("numerical failure")

    with pytest.raises(RuntimeError, match="numerical"):
        run_mock(site_fixture, campaign, submitter=lambda *a, **k: pytest.fail("must not submit"))
    assert continuation.read_state(site["run_root"])["status"] == "failed"
    assert segment.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


@pytest.mark.parametrize("partition", ["gpubase_l40s_b2", continuation.PLASMA_L40S_PARTITIONS])
def test_paused_run_submits_exact_afterok_and_successor_resumes_without_fresh_start(
        site_fixture, allocation, monkeypatch, partition):
    path, site = site_fixture
    site["partition"] = partition
    write_json(path, site)
    calls = []

    def sbatch(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout="200\n", stderr="")

    monkeypatch.setattr(continuation.subprocess, "run", sbatch)
    first = run_mock(site_fixture, lambda config, root, **kw: campaign_result(root))
    assert first["status"] == "paused" and first["continuation_status"] == "submitted"
    command, options = calls[0]
    for flag in ("--gres=gpu:l40s:4", "--cpus-per-task=24", "--mem=128G",
                 "--time=12:00:00", "--dependency=afterok:100", f"--partition={partition}"):
        assert flag in command
    for key, value in options["env"].items():
        if key.startswith("NS_CONTINUATION_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("SLURM_JOB_ID", "200")

    def resumed(config, root, **options):
        assert options["resume"] is True
        return campaign_result(root, status="completed", progress=2)

    second = run_mock(site_fixture, resumed)
    assert second["status"] == "completed" and len(calls) == 1
    assert continuation.read_state(site["run_root"])["segment_index"] == 2


def test_no_progress_pause_stops_successor_loop(site_fixture, allocation, monkeypatch):
    calls = []

    def sbatch(command, **options):
        calls.append(options["env"])
        return SimpleNamespace(returncode=0, stdout="200\n", stderr="")

    monkeypatch.setattr(continuation.subprocess, "run", sbatch)
    run_mock(site_fixture, lambda config, root, **kw: campaign_result(root))
    for key, value in calls[0].items():
        if key.startswith("NS_CONTINUATION_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("SLURM_JOB_ID", "200")
    with pytest.raises(ValueError, match="no scientific progress"):
        run_mock(site_fixture, lambda config, root, **kw: campaign_result(root))
    assert len(calls) == 1
    assert continuation.read_state(site_fixture[1]["run_root"])["status"] == "failed"


def test_uncertain_submission_is_not_retried_and_keeps_afterok_parent_success(
        site_fixture, allocation, monkeypatch):
    calls = []

    def sbatch(command, **options):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(continuation.subprocess, "run", sbatch)
    result = run_mock(site_fixture, lambda config, root, **kw: campaign_result(root))
    assert result["status"] == "paused" and result["continuation_status"] == "submission_unknown"
    assert continuation.read_state(site_fixture[1]["run_root"])["status"] == "submission_unknown"
    assert len(calls) == 1


@pytest.mark.parametrize("equation,profile,memory,hours,margin", [
    ("strong", "plasma_gpu_12h", "128G", "12:00:00", 2400),
    ("weak", "plasma_gpu_12h", "128G", "12:00:00", 2400),
    ("artemis", "plasma_gpu_6h", "64G", "06:00:00", 600),
])
@pytest.mark.parametrize("partition", ["gpubase_l40s_b2", continuation.PLASMA_L40S_PARTITIONS])
def test_equation_resource_profiles_and_ns_defaults_unchanged(
        site_fixture, equation, profile, memory, hours, margin, partition):
    path, site = site_fixture
    config = write_json(Path(site["config_path"]), {"equation": equation})
    site.update(memory=memory, time=hours, shutdown_margin_seconds=margin,
                config_sha256=segment.file_sha(config), partition=partition)
    write_json(path, site)
    identity = segment.verify_site(path)[2]
    suffix = "_b2_b3" if partition == continuation.PLASMA_L40S_PARTITIONS else ""
    assert identity["resource_profile"] == profile + suffix
    actual = identity["resources"]
    assert actual["partition"] == partition
    assert actual["memory"] == memory and actual["cpus"] == 24
    assert actual["wall_seconds"] == int(hours[:2]) * 3600
    legacy = continuation._submission_spec(site["slurm_script"], (), 3)
    assert (legacy["cpus"], legacy["memory"], legacy["wall_seconds"]) == (32, "160G", 43200)
    with pytest.raises(ValueError, match="2 or 3"):
        continuation._submission_spec(site["slurm_script"], (), 4)


def test_no_allocation_no_writes(site_fixture, allocation, monkeypatch):
    monkeypatch.setattr(segment.socket, "gethostname", lambda: "login01")
    with pytest.raises(RuntimeError, match="compute allocation"):
        run_mock(site_fixture, lambda *a, **kw: pytest.fail("must not run"))
    assert not Path(site_fixture[1]["run_root"]).exists()


def role_report(token, index):
    uuid = f"GPU-00000000-0000-0000-0000-{index:012x}"
    return {"status": "passed", "token": token, "nvidia_uuid": uuid,
            "cuda_metadata": {"schema": "candidate-cuda-execution-v1", "device": "cuda:0",
                "cuda_visible_devices": token, "uuid": uuid, "name": "mock GPU",
                "device_count": 1, "readiness_kernel_passed": True, "synchronized": True,
                "peak_memory_allocated_bytes": 1024, "peak_memory_reserved_bytes": 2048}}


def test_role_probe_validates_four_physical_uuid_and_preserves_parent(tmp_path, allocation, monkeypatch):
    original = segment.os.environ["CUDA_VISIBLE_DEVICES"]
    calls = []

    def child(command, **options):
        token = command[-1]
        calls.append(options["env"]["CUDA_VISIBLE_DEVICES"])
        options["stdout"].write(json.dumps(role_report(token, int(token))) + "\n")
        options["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(segment.subprocess, "run", child)
    receipt = segment.probe_allocated_roles(tmp_path / "roles")
    assert receipt["status"] == "passed"
    assert calls == ["0", "1", "2", "3"]
    assert len(set(receipt["physical_uuid_by_token"].values())) == 4
    assert segment.os.environ["CUDA_VISIBLE_DEVICES"] == original
    reports = {role: role_report(token, 0) for role, token in receipt["roles"].items()}
    with pytest.raises(ValueError, match="alias"):
        segment.validate_role_reports(receipt["roles"], reports)


def test_role_probe_failure_keeps_partial_receipt(tmp_path, allocation, monkeypatch):
    def child(command, **options):
        token = command[-1]
        if token == "1":
            raise subprocess.TimeoutExpired(command, 60)
        options["stdout"].write(json.dumps(role_report(token, int(token))) + "\n")
        options["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(segment.subprocess, "run", child)
    with pytest.raises(subprocess.TimeoutExpired):
        segment.probe_allocated_roles(tmp_path / "roles")
    saved = json.loads((tmp_path / "roles/result.json").read_text())
    assert saved["status"] == "failed" and list(saved["role_probes"]) == ["agent_broad"]
